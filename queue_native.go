//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// Queue handles command submission and data transfers.
//
// Queue is safe for concurrent use from multiple goroutines. All mutating
// operations (Submit, WriteBuffer, WriteTexture) are serialized via an
// internal mutex, matching Rust wgpu-core which uses RwLock::write() on
// device.fence + device.command_indices during submit (queue.rs:1171-1173).
type Queue struct {
	mu sync.Mutex // serializes Submit, WriteBuffer, WriteTexture (Rust wgpu parity)

	hal       hal.Queue
	halDevice hal.Device
	device    *Device
	pending   *pendingWrites

	// lastSubmissionIndex is the most recent submission index returned by
	// hal.Queue.Submit(). Used by DestroyQueue to conservatively defer
	// resource destruction until after the latest known submission completes.
	// Atomic: written by Submit (under mu), read by onZero callbacks which
	// may fire during Triage while Queue.mu is held. Using atomic avoids
	// deadlock: Submit→Triage→onZero→lastSubmissionIndex→mu (ADR-056).
	lastSubmissionIndex atomic.Uint64

	// barrierEncoder and barrierCBs hold the barrier encoder of the current
	// Submit and the command buffers it recorded, one per command buffer of
	// the submit that needs barriers (injectBarriers). Deferred recycling
	// happens in postSubmit with the ACTUAL submission index (not the stale
	// lastSubmissionIndex).
	barrierEncoder hal.CommandEncoder
	barrierCBs     []hal.CommandBuffer

	// Scratch for Submit, under mu: the HAL command buffer list, the barrier
	// command buffer of each submitted command buffer (nil for none), the
	// transitions of one command buffer and the tracker index resolvers for
	// barrier injection. Cleared after each use so they pin nothing between
	// submits.
	submitBufs      []hal.CommandBuffer
	cbBarriers      []hal.CommandBuffer
	texTransitions  []track.TexturePendingTransition
	zeroTransitions []track.TexturePendingTransition
	bufTransitions  []track.PendingTransition
	texResolver     map[track.TrackerIndex]hal.Texture
	bufResolver     map[track.TrackerIndex]hal.Buffer
	barrierScratch  core.BarrierScratch

	// Scratch for presentSurfaces, under mu: the surface textures the
	// flushed pending writes wrote, the tracker indices of the ones taken to
	// their present state, and the transitions and barriers there.
	writtenSurfaces    []*Texture
	presentIndices     []track.TrackerIndex
	presentTransitions []track.TexturePendingTransition
	presentBarriers    []hal.TextureBarrier

	// Scratch for WriteTexture, under mu: the HAL forms of its arguments,
	// which travel by pointer and would otherwise be heap copies every call.
	wtDst    hal.ImageCopyTexture
	wtLayout hal.ImageDataLayout
	wtSize   hal.Extent3D
}

// ResolveTexture implements core.BarrierResolver over the textures of the
// command buffers being submitted.
func (q *Queue) ResolveTexture(idx track.TrackerIndex) hal.Texture { return q.texResolver[idx] }

// ResolveBuffer implements core.BarrierResolver over the buffers of the
// command buffers being submitted.
func (q *Queue) ResolveBuffer(idx track.TrackerIndex) hal.Buffer { return q.bufResolver[idx] }

// Submit submits command buffers for execution. Non-blocking.
// Returns a submission index that can be used with Poll() to track completion.
// Command buffers are owned by the caller — free them after Poll confirms completion.
//
// If there are pending WriteBuffer/WriteTexture operations, they are flushed
// and run ahead of the barriers and the user command buffers in a single HAL
// submit.
func (q *Queue) Submit(commandBuffers ...*CommandBuffer) (_ uint64, err error) {
	defer q.device.reportError(&err)
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.hal == nil {
		return 0, fmt.Errorf("wgpu: queue not available")
	}
	if err := q.usable(); err != nil {
		return 0, err
	}
	if err := q.validateSubmit(commandBuffers); err != nil {
		return 0, err
	}

	flushed, zeros, err := q.flushWrites(commandBuffers)
	if err != nil {
		return 0, fmt.Errorf("wgpu: flush pending writes: %w", err)
	}

	// ADR-060: Submit-time texture barrier injection via DeviceTracker.
	// Merge each command buffer's trackers into the device-level tracker.
	// When the texture state changes (e.g., Resource -> ColorTarget), the
	// tracker produces PendingTransitions that are recorded as barriers in a
	// command buffer that runs right ahead of that user CB.
	//
	// Reference: wgpu-core device/queue.rs pre_submit_for_command_buffers
	if err := q.injectBarriers(commandBuffers, zeros); err != nil {
		return 0, fmt.Errorf("wgpu: submit barrier injection: %w", q.fail(flushed, err))
	}
	defer clear(q.cbBarriers)

	// Build combined command buffer list: the pending writes, then each
	// user buffer behind its barriers. The writes go first because the
	// merge above starts from the state they leave a texture in (one zeroed
	// or written this submit is Resource), so a barrier out of that state
	// has to run after them; Rust wgpu-core puts pending_execution at the
	// front of the list too. The list lives in the queue's scratch, the HAL
	// does not keep it.
	//
	// Reference: wgpu-core device/queue.rs, active_executions.insert(0, pending_execution)
	q.submitBufs = q.submitBufs[:0]
	if flushed.cmdBuf != nil {
		q.submitBufs = append(q.submitBufs, flushed.cmdBuf)
	}

	// Flatten multi-CB command buffers: each CommandBuffer may contain
	// multiple HAL command buffers from multi-CB recording (OpenPass/
	// CloseCB/CloseAndSwap/CloseAndPushFront). All CBs are submitted
	// together in submission order.
	//
	// Reference: Rust wgpu-core queue.rs submit — iterates encoder.list
	for i, cb := range commandBuffers {
		if i < len(q.cbBarriers) && q.cbBarriers[i] != nil {
			q.submitBufs = append(q.submitBufs, q.cbBarriers[i])
		}
		q.submitBufs = append(q.submitBufs, cb.halBufferList()...)
	}
	err = q.presentSurfaces(commandBuffers, q.writtenSurfaces)
	clear(q.writtenSurfaces)
	if err != nil {
		return 0, fmt.Errorf("wgpu: submit barrier injection: %w", q.fail(flushed, err))
	}
	allBuffers := q.submitBufs
	defer clear(allBuffers)

	// Nothing to run (no command buffers and no queued writes) is not a
	// submission: the HALs return index 0 for it, and a Release right after
	// would defer its destroy at 0, to run while earlier work is still on the
	// GPU. Keep the last real index and just do the housekeeping below.
	subIdx := q.lastSubmissionIndex.Load()
	var submitErr error
	if len(allBuffers) > 0 {
		var err error
		subIdx, err = q.hal.Submit(allBuffers)
		if err != nil && subIdx == 0 {
			return 0, fmt.Errorf("wgpu: submit failed: %w", q.fail(flushed, err))
		}
		if err != nil {
			submitErr = fmt.Errorf("wgpu: submit: %w", q.device.lose(err))
		}
		// Track the latest submission index for deferred resource destruction.
		q.lastSubmissionIndex.Store(subIdx)
	}

	// One completion query serves everything below: the belt, the destroy
	// queue and the pending maps all ask the same question.
	completed := q.hal.PollCompleted()
	q.trackFlushed(flushed, subIdx, completed)

	// Post-submit bookkeeping: track refs, recycle encoders, triage destroys.
	q.postSubmit(subIdx, completed, commandBuffers)

	// Auto-poll pending buffer map requests after each Submit. Mirrors
	// Rust wgpu-core queue.rs:1429 which calls maintain() at the tail
	// of queue_submit. Non-blocking — drains whatever has already
	// completed so beginner code paths that read a buffer right after a
	// Submit see the mapping resolve without having to call Device.Poll
	// explicitly. (FEAT-WGPU-MAPPING-001)
	if q.device != nil && q.device.core != nil && q.device.core.HasPendingMaps() {
		q.device.core.PollMaps(completed)
	}

	return subIdx, submitErr
}

// flushForPresent submits the queue writes ahead of the present of t, a
// surface texture, as wgpu-core submits them at present, and with them the
// zeroing of t when nothing set it.
func (q *Queue) flushForPresent(t *Texture) error {
	if q.pending == nil {
		return nil
	}
	if err := q.pending.zeroForPresent(t); err != nil {
		return err
	}
	if !q.pending.recording() {
		return nil
	}
	_, err := q.Submit()
	return err
}

// usable returns the error the queue refuses work with: ErrReleased once
// its device is released, the error that lost it once it is lost.
func (q *Queue) usable() error {
	if q.released() {
		return ErrReleased
	}
	if q.device != nil {
		return q.device.lostErr()
	}
	return nil
}

// validateSubmit checks the command buffers of a Submit before the pending
// writes are closed: a validation error leaves buffered writes intact for a
// later valid Submit and cannot strand an ended internal command buffer.
func (q *Queue) validateSubmit(commandBuffers []*CommandBuffer) error {
	for i, cb := range commandBuffers {
		if cb == nil {
			return fmt.Errorf("wgpu: command buffer at index %d is nil", i)
		}
		if q.device.foreign(cb.device) {
			return errForeign("Queue.Submit", fmt.Sprintf("command buffer %d", i))
		}
		if j := slices.Index(commandBuffers[:i], cb); j >= 0 {
			return fmt.Errorf("wgpu: Submit: command buffer at index %d is also at index %d: %w",
				i, j, ErrSubmitCommandBufferInvalid)
		}
		if err := validateCommandBufferForSubmit(cb, i); err != nil {
			return err
		}
	}
	return nil
}

// flushedWrites is the batch of queue writes a Submit closed.
type flushedWrites struct {
	cmdBuf   hal.CommandBuffer
	encoder  hal.CommandEncoder
	textures []hal.Texture
	buffers  []hal.Buffer
}

// flushWrites closes the batch of queue writes ahead of commandBuffers,
// and returns it with the textures to zero ahead of each of them.
func (q *Queue) flushWrites(commandBuffers []*CommandBuffer) (flushedWrites, [][]*Texture, error) {
	var f flushedWrites
	if q.pending == nil {
		return f, nil, nil
	}
	q.pending.mu.Lock()
	defer q.pending.mu.Unlock()
	q.pending.forgetClearsLocked(commandBuffers)
	q.writtenSurfaces = q.pending.writtenSurfacesLocked(q.writtenSurfaces[:0])
	var err error
	f.cmdBuf, f.encoder, f.textures, f.buffers, err = q.pending.flush()
	if err != nil {
		clear(q.writtenSurfaces)
		return f, nil, err
	}
	return f, q.pending.queueDiscardedLocked(commandBuffers), nil
}

// trackFlushed keeps the batch a Submit ran, as submission subIdx, until
// the GPU is done with it, and recycles what earlier ones used.
func (q *Queue) trackFlushed(f flushedWrites, subIdx, completed uint64) {
	if q.pending == nil {
		return
	}
	pw := q.pending
	pw.mu.Lock()
	defer pw.mu.Unlock()
	// dstTextures/dstBuffers prevent premature Release (BUG-DX12-006: use-after-free).
	if f.cmdBuf != nil || f.textures != nil {
		pw.inflight = append(pw.inflight, inflightSubmission{
			submissionIndex: subIdx,
			staging:         nil, // staging managed by belt
			cmdBuf:          f.cmdBuf,
			encoder:         f.encoder,
			dstTextures:     f.textures,
			dstBuffers:      f.buffers,
			views:           pw.flushedViews,
			destroys:        pw.flushedDestroys,
			refs:            pw.flushedRefs,
		})
	}
	pw.flushedViews = nil
	pw.flushedDestroys, pw.flushedRefs = nil, nil
	pw.forgetUndo()
	// Update the staging belt with the actual submission index
	// (belt.finish() was called during flush() before Submit).
	if pw.belt != nil {
		pw.belt.setLastSubmissionIndex(subIdx)
	}
	pw.maintain(completed)
}

// fail gives back what Submit set up before err, a failure of the backend,
// and loses the device, as wgpu-core does on any error of the backend's
// submit: the device tracker holds the states of command buffers that
// did not run. It returns the error the queue refuses work with now.
func (q *Queue) fail(f flushedWrites, err error) error {
	q.cancelSubmit(f)
	return q.device.lose(err)
}

// cancelSubmit gives back what Submit set up before it failed: the flushed
// pending writes, what they and the command buffers changed of the zero
// lists, and the barrier encoder, which the next Submit would otherwise
// overwrite without returning it to the pool.
func (q *Queue) cancelSubmit(f flushedWrites) {
	if q.pending != nil {
		q.pending.mu.Lock()
		if f.cmdBuf != nil {
			q.pending.cancelFlush(f.cmdBuf, f.encoder, f.textures)
		} else {
			q.pending.rollback()
		}
		q.pending.mu.Unlock()
	}
	if q.barrierEncoder != nil {
		q.barrierEncoder.ResetAll(q.barrierCBs)
		q.device.cmdEncoderPool.release(q.barrierEncoder)
		q.barrierEncoder = nil
		clear(q.barrierCBs)
		q.barrierCBs = q.barrierCBs[:0]
	}
}

// postSubmit handles bookkeeping after a successful HAL submit:
// 1. Tracks Clone'd ResourceRefs for Drop on GPU completion (Phase 2)
// 2. Schedules HAL encoder recycling via DestroyQueue (BUG-DX12-004)
// 3. Triages deferred resource destructions
func (q *Queue) postSubmit(subIdx, completed uint64, commandBuffers []*CommandBuffer) {
	// Mark all command buffers as submitted to prevent double-submit (VAL-A6),
	// and drop the encode-time reference sets. This runs before the DestroyQueue
	// lookup below: the HAL submit has already succeeded, so these buffers are
	// spent whether or not there is a queue to triage.
	for _, cb := range commandBuffers {
		if cb != nil {
			cb.submitted = true
			cb.dropUsedSets()
			// injectBarriers has merged the trackers; the next encoder can
			// have them.
			cb.core.ReleaseScopes()
		}
	}

	dq := q.destroyQueue()
	if dq == nil {
		return
	}

	// Hand each command buffer's tracked refs to the DestroyQueue for this
	// submission (Phase 2: per-command-buffer resource tracking). The refs
	// are Drop'd when the GPU completes, and the queue reuses the emptied
	// lists.
	for _, cb := range commandBuffers {
		if cb != nil && cb.trackedRefs != nil {
			dq.TrackSubmission(subIdx, cb.trackedRefs)
			cb.trackedRefs = nil
		}
	}

	// Schedule HAL encoder recycling after GPU completion (BUG-DX12-004).
	// Each command buffer carries the HAL encoder that produced it. After the
	// GPU finishes this submission, the encoder is reset via ResetAll (which
	// resets the DX12 ID3D12CommandAllocator or Vulkan VkCommandPool) and
	// returned to the device's encoder pool for reuse.
	//
	// For multi-CB encoders, ALL HAL command buffers must be passed to
	// ResetAll so the underlying pool/allocator can reclaim them all.
	//
	// Matches Rust wgpu-core's CommandAllocator::release_encoder pattern where
	// encoders travel: CommandEncoder -> CommandBuffer -> EncoderInFlight -> pool.
	for _, cb := range commandBuffers {
		if cb == nil || cb.halEncoder == nil {
			continue
		}
		// All the HAL command buffers from this encoder go back with it, and
		// the encoder object itself once the GPU is done.
		cb.recycle = encoderRecycle{enc: cb.halEncoder, bufs: cb.halBufferList(), pool: q.device.cmdEncoderPool, owner: cb.enc}
		cb.halEncoder = nil // ownership moves to the recycle the queue runs
		dq.DeferRelease(subIdx, "CmdEncoder", &cb.recycle)
	}

	// ADR-060: Defer barrier encoder recycling with the CURRENT submission
	// index, the actual subIdx, not the stale lastSubmissionIndex set before
	// hal.Submit. The recycle record comes from a pool, so a frame with
	// barriers allocates nothing.
	if q.barrierEncoder != nil && len(q.barrierCBs) > 0 {
		r := recyclePool.Get().(*encoderRecycle)
		r.enc, r.pool, r.pooled = q.barrierEncoder, q.device.cmdEncoderPool, true
		r.bufs = append(r.bufs[:0], q.barrierCBs...)
		dq.DeferRelease(subIdx, "BarrierEncoder", r)
		q.barrierEncoder = nil
		clear(q.barrierCBs)
		q.barrierCBs = q.barrierCBs[:0]
	}

	// Triage deferred resource destructions from the DestroyQueue.
	// Resources whose GPU submissions have completed are now safe to destroy.
	dq.Triage(completed)
}

// Poll returns the last completed submission index. Non-blocking.
// All submissions with index <= the returned value have been completed by the GPU.
func (q *Queue) Poll() uint64 {
	if q.hal == nil || q.released() {
		return 0
	}
	return q.hal.PollCompleted()
}

// released reports whether the queue's device has been released, taking its
// HAL device with it.
func (q *Queue) released() bool {
	return q.device != nil && q.device.released.Load()
}

// WriteBuffer writes data to a buffer.
// If PendingWrites batching is enabled (DX12/Vulkan/Metal), the write is
// recorded into a shared command encoder and flushed on the next Submit.
// For GLES/Software backends, the write is performed immediately.
//
// MapWrite buffers (upload heap on DX12, host-visible on Vulkan) are written
// directly via HAL without staging — GPU copy into upload heap is undefined
// behavior on DX12 (upload heap is GENERIC_READ, read-only to GPU).
// See BUG-DX12-003.
//
// Validation (VAL-A1, WebGPU spec §21.1):
//   - Buffer must not be currently mapped
//   - Buffer must have CopyDst usage
//   - offset must be 4-byte aligned
//   - data size must be 4-byte aligned
//   - offset + data size must not exceed buffer size
//
// Matches Rust wgpu-core queue.rs:647-672 (validate_write_buffer_impl).
func (q *Queue) WriteBuffer(buffer *Buffer, offset uint64, data []byte) (err error) {
	defer q.device.reportError(&err)
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.hal == nil || buffer == nil {
		return fmt.Errorf("wgpu: WriteBuffer: queue or buffer is nil")
	}
	if err := q.usable(); err != nil {
		return err
	}
	if q.device.foreign(buffer.device) {
		return errForeign("Queue.WriteBuffer", "the buffer")
	}

	// --- VAL-A1: bounds + state validation (before HAL access) ---

	// 1. Buffer must not be mapped (write to mapped buffer = data race).
	// Rust: !matches!(&*buffer.map_state.lock(), BufferMapState::Idle)
	if buffer.MapState() != MapStateUnmapped {
		return fmt.Errorf("wgpu: WriteBuffer: buffer is currently mapped")
	}

	// 2. Buffer must have COPY_DST usage.
	// Rust: buffer.check_usage(wgt::BufferUsages::COPY_DST)
	if buffer.Usage()&BufferUsageCopyDst == 0 {
		return fmt.Errorf("wgpu: WriteBuffer: buffer missing CopyDst usage")
	}

	// 3. Offset must be 4-byte aligned (COPY_BUFFER_ALIGNMENT = 4).
	// Rust: buffer_offset % wgt::COPY_BUFFER_ALIGNMENT != 0
	if offset%4 != 0 {
		return fmt.Errorf("wgpu: WriteBuffer: offset %d not 4-byte aligned", offset)
	}

	// 4. Data size must be 4-byte aligned.
	// Rust: buffer_size.get() % wgt::COPY_BUFFER_ALIGNMENT != 0
	dataSize := uint64(len(data))
	if dataSize%4 != 0 {
		return fmt.Errorf("wgpu: WriteBuffer: data size %d not 4-byte aligned", dataSize)
	}

	// 5. Write must not exceed buffer bounds.
	// Rust: buffer_offset + buffer_size.get() > buffer.size
	if offset+dataSize > buffer.Size() {
		return fmt.Errorf("wgpu: WriteBuffer: offset %d + size %d exceeds buffer size %d", offset, dataSize, buffer.Size())
	}

	// --- end VAL-A1 ---

	halBuffer := buffer.halBuffer()
	if halBuffer == nil {
		return fmt.Errorf("wgpu: WriteBuffer: no HAL buffer")
	}

	// Always route through PendingWrites staging belt when available.
	// Rust wgpu-core write_buffer() (queue.rs:549) ALWAYS creates a StagingBuffer,
	// even for MapWrite buffers. Data is immutable in staging until GPU completion.
	// This prevents data races when CPU overwrites while GPU reads (BUG-METAL-001).
	//
	// DX12: MapWrite buffers now use HEAP_TYPE_CUSTOM with WRITE_COMBINE + COMMON
	// state (matching Rust suballocation.rs:437), allowing CopyBufferRegion as dst.
	if q.pending != nil {
		return q.pending.writeBufferFor(buffer, halBuffer, buffer.Usage(), offset, data)
	}

	return q.hal.WriteBuffer(halBuffer, offset, data)
}

// WriteTexture writes data to a texture.
// If PendingWrites batching is enabled (DX12/Vulkan/Metal), the write is
// recorded into a shared command encoder and flushed on the next Submit.
// Resource barriers are computed from the texture's tracked CurrentUsage().
// For GLES/Software backends, the write is performed immediately via HAL.
func (q *Queue) WriteTexture(dst *ImageCopyTexture, data []byte, layout *ImageDataLayout, size *Extent3D) (err error) {
	defer q.device.reportError(&err)
	q.mu.Lock()
	defer q.mu.Unlock()

	if q.hal == nil || dst == nil {
		return fmt.Errorf("wgpu: WriteTexture: queue or destination is nil")
	}
	if err := q.usable(); err != nil {
		return err
	}
	if dst.Texture != nil && dst.Texture.resolveHAL() == nil {
		return ErrReleased
	}
	if dst.Texture == nil {
		return fmt.Errorf("wgpu: WriteTexture: destination texture is invalid")
	}
	if q.device.foreign(dst.Texture.device) {
		return errForeign("Queue.WriteTexture", "the texture")
	}
	// The texture was made, or the surface configured, with CopyDst.
	// Rust: dst.check_usage(wgt::TextureUsages::COPY_DST)
	if ct := dst.Texture.coreTexture; ct != nil && ct.Usage()&gputypes.TextureUsageCopyDst == 0 {
		return fmt.Errorf("wgpu: WriteTexture: destination texture is missing CopyDst usage")
	}
	if layout == nil {
		return fmt.Errorf("wgpu: WriteTexture: layout is nil")
	}
	if size == nil {
		return fmt.Errorf("wgpu: WriteTexture: size is nil")
	}
	copySize := gputypes.Extent3D(*size)
	if dst.Texture.coreTexture != nil {
		var err error
		if copySize, err = core.ValidateTextureBufferCopy(texelCopy(dst.Texture, dst), copySize, layout.toHAL(), uint64(len(data)), true, false); err != nil {
			return fmt.Errorf("wgpu: WriteTexture: %w", err)
		}
	}
	if emptyExtent(hal.Extent3D(copySize)) {
		return nil // valid, and nothing to write
	}

	// Into the queue's scratch: the HAL takes these by pointer and keeps
	// none of them.
	q.wtDst, q.wtLayout, q.wtSize = dst.halValue(), layout.toHAL(), hal.Extent3D(copySize)
	if ct := dst.Texture.coreTexture; ct != nil {
		q.wtDst.Aspect = core.CopyAspect(ct.Format(), dst.Aspect)
	}
	defer func() { q.wtDst = hal.ImageCopyTexture{} }()

	if q.pending != nil {
		return q.pending.writeTextureFor(dst.Texture, &q.wtDst, data, &q.wtLayout, &q.wtSize)
	}

	return q.hal.WriteTexture(&q.wtDst, data, &q.wtLayout, &q.wtSize)
}

// ExternalImage is a multi-planar image of the device made outside gowgpu,
// such as a video decoder's, which a timeline orders: its users wait for it
// to reach Value before they touch the image, and signal Value+1 when done.
// On Vulkan it is a VkImage in the VkImageLayout Layout, ordered by a timeline
// VkSemaphore; on Direct3D 12 an ID3D12Resource in the D3D12_RESOURCE_STATES
// Layout, COMMON, layer Slice of it when it is an array, and an ID3D12Fence;
// on Metal an IOSurfaceRef, such as a VideoToolbox CVPixelBuffer's, and an
// id<MTLSharedEvent>, Layout and Slice unused.
type ExternalImage struct {
	Image     uint64
	Layout    uint32
	Slice     uint32
	Semaphore uint64
	Value     uint64
}

// CopyExternalPlanes copies the planes of src into dsts, plane i into dsts[i],
// textures made with CopyDst, as much of the plane as they hold, in a format
// copies take the plane's texels to: R8 and RG8 for NV12's planes, on Vulkan
// and Metal RG8 and RGBA8 for P010's. The copy runs ahead of the next Submit,
// which waits for src's timeline to reach src.Value and has it signal
// src.Value+1 when done, so src must not be copied again before it. It
// returns the layout the copy leaves src in. The Vulkan, Direct3D 12 and
// Metal backends copy external images.
func (q *Queue) CopyExternalPlanes(src ExternalImage, dsts ...*Texture) (uint32, error) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if err := q.usable(); err != nil {
		return src.Layout, err
	}
	syncer, ok := q.hal.(externalSyncer)
	if !ok || q.pending == nil || len(dsts) == 0 || len(dsts) > 3 {
		return src.Layout, fmt.Errorf("wgpu: CopyExternalPlanes: needs the Vulkan, Direct3D 12 or Metal backend and 1 to 3 textures")
	}
	if syncer.ExternalSyncFull() { // the copy is not recorded without its sync
		return src.Layout, errExternalSyncFull
	}
	q.pending.mu.Lock()
	after, err := q.pending.copyExternalLocked(src.Image, src.Layout, src.Slice, dsts)
	q.pending.mu.Unlock()
	if err != nil {
		return src.Layout, err
	}
	if err := syncer.AddExternalSync(src.Semaphore, src.Value, src.Value+1); err != nil {
		return src.Layout, err
	}
	return after, nil
}

// externalSyncer is a backend queue that orders work with timelines of code
// sharing the device.
type externalSyncer interface {
	ExternalSyncFull() bool
	AddExternalSync(timeline, wait, signal uint64) error
}

var errExternalSyncFull = errors.New("wgpu: the next submit waits for as many external timelines as it can, submit first")

// ExternalBuffer is a buffer of the device made outside gowgpu that holds a
// picture another API draws, such as OpenGL through memory it shares, which
// a timeline orders as it does an ExternalImage: a VkBuffer and a timeline
// VkSemaphore on Vulkan, an ID3D12Resource and an ID3D12Fence on Direct3D 12,
// where BytesPerRow must be a multiple of 256, an id<MTLBuffer> and an
// id<MTLSharedEvent> on Metal.
type ExternalBuffer struct {
	Buffer      uint64
	BytesPerRow uint32
	Semaphore   uint64
	Value       uint64
}

// CopyExternalBuffer copies the picture in src into dst, a whole texture made
// with CopyDst whose rows src holds BytesPerRow apart, top row first. Like
// CopyExternalPlanes it runs ahead of the next Submit, which waits for src's
// semaphore to reach src.Value and signals src.Value+1 when done, so src must
// not be copied again before it. The Vulkan, Direct3D 12 and Metal backends
// copy external buffers.
func (q *Queue) CopyExternalBuffer(src ExternalBuffer, dst *Texture) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	if err := q.usable(); err != nil {
		return err
	}
	syncer, ok := q.hal.(externalSyncer)
	if !ok || q.pending == nil || dst == nil {
		return fmt.Errorf("wgpu: CopyExternalBuffer: needs the Vulkan, Direct3D 12 or Metal backend and a texture")
	}
	if syncer.ExternalSyncFull() { // the copy is not recorded without its sync
		return errExternalSyncFull
	}
	q.pending.mu.Lock()
	err := q.pending.copyExternalBufferLocked(src.Buffer, src.BytesPerRow, dst)
	q.pending.mu.Unlock()
	if err != nil {
		return err
	}
	return syncer.AddExternalSync(src.Semaphore, src.Value, src.Value+1)
}

// SetSwapchainSuppressed temporarily disables swapchain semaphore binding
// for subsequent Submit calls. Used for offscreen renders (e.g., RepaintBoundary)
// that must not consume acquire/present semaphores intended for the compositor
// submit.
//
// BUG-WGPU-VK-005: When rendering to an offscreen texture before compositing
// to the swapchain surface, the offscreen submit must not hijack swapchain
// semaphores. Without suppression, the compositor submit runs without GPU-side
// synchronization, causing a race condition (flickering).
//
// Call with true before offscreen submits, false after:
//
//	queue.SetSwapchainSuppressed(true)
//	defer queue.SetSwapchainSuppressed(false)
//	queue.Submit(offscreenCmds...)
//
// Only meaningful on Vulkan — other backends are no-ops.
func (q *Queue) SetSwapchainSuppressed(suppressed bool) {
	if q.hal != nil {
		q.hal.SetSwapchainSuppressed(suppressed)
	}
}

// LastSubmissionIndex returns the most recent submission index.
// Used by resource Release() and onZero callbacks to schedule deferred destruction.
// Lock-free via atomic — safe to call from onZero callbacks during Triage
// while Queue.mu is held by Submit. ADR-056 deadlock fix.
func (q *Queue) LastSubmissionIndex() uint64 {
	return q.lastSubmissionIndex.Load()
}

// destroyQueue returns the device's DestroyQueue, or nil if unavailable.
func (q *Queue) destroyQueue() *core.DestroyQueue {
	if q.device != nil && q.device.core != nil {
		return q.device.core.DestroyQueueRef()
	}
	return nil
}

// validateCommandBufferForSubmit checks that a command buffer and all its
// referenced resources are in a valid state for submission.
//
// This is the Go equivalent of Rust wgpu-core's validate_command_buffer
// (device/queue.rs:1764-1828). It checks:
//   - Command buffer has not already been submitted (double-submit prevention)
//   - All referenced buffers are not destroyed/released
//   - All referenced buffers are not mapped (mapped buffer in GPU commands = data race)
//   - All referenced textures are not destroyed/released
//
// The index parameter identifies which command buffer in the Submit() call
// failed validation, for clearer error messages.
func validateCommandBufferForSubmit(cb *CommandBuffer, index int) error {
	// 1. Check double-submit, and a handle whose encoder has been reused.
	if cb.submitted || cb.stale() {
		return fmt.Errorf("wgpu: Submit: command buffer at index %d: %w",
			index, ErrSubmitCommandBufferInvalid)
	}

	// 2. Check referenced buffers (matches Rust queue.rs:1780-1787).
	for _, buf := range cb.usedBuffers {
		if err := validateSubmitBuffer(buf, index); err != nil {
			return err
		}
	}

	// 3. Check referenced textures (matches Rust queue.rs:1791-1808).
	for _, tex := range cb.usedTextures {
		if err := validateSubmitTexture(tex, index); err != nil {
			return err
		}
	}

	// 4. Check the resources bound by every bind group. A bind group already
	// holds boundBuffers and boundTextures from CreateBindGroup, so passes
	// track only the group itself and this walk reaches the rest — no per-draw
	// fan-out needed. Release() never mutates those slices, so they stay
	// walkable after the group itself is released.
	//
	// This runs as its own pass, ahead of the bg.released pass below, so the
	// specific error always wins: a released buffer reports the buffer rather
	// than whichever bind group happens to reference it. Before the fan-out was
	// removed these resources sat in the flat usedBuffers/usedTextures sets,
	// checked in steps 2 and 3 ahead of any bind group, and folding the release
	// check into this loop would make the winner depend on map iteration order
	// whenever two bind groups are at fault.
	for _, bg := range cb.usedBindGroups {
		for _, buf := range bg.boundBuffers {
			if err := validateSubmitBuffer(buf, index); err != nil {
				return err
			}
		}
		for _, tex := range bg.boundTextures {
			if err := validateSubmitTexture(tex, index); err != nil {
				return err
			}
		}
	}

	// 5. Check the bind groups themselves (matches Rust queue.rs:1815-1817).
	for _, bg := range cb.usedBindGroups {
		if bg.released != nil && bg.released.Load() {
			return fmt.Errorf("wgpu: Submit: command buffer at index %d references released bind group: %w",
				index, ErrSubmitBindGroupDestroyed)
		}
	}

	// 6. Check the query sets ResolveQuerySet reads.
	for _, qs := range cb.usedQuerySets {
		if qs.released.Load() {
			return fmt.Errorf("wgpu: Submit: command buffer at index %d references released query set: %w",
				index, ErrReleased)
		}
	}

	return nil
}

// validateSubmitBuffer checks that a buffer is neither released nor mapped.
// Callers never pass nil: trackBuffer skips nil and collectBindGroupResources
// only records non-nil entries.
func validateSubmitBuffer(buf *Buffer, index int) error {
	// Check destroyed/released.
	if buf.released != nil && buf.released.Load() {
		return fmt.Errorf("wgpu: Submit: command buffer at index %d references released buffer %q: %w",
			index, buf.Label(), ErrSubmitBufferDestroyed)
	}
	// Check mapped state.
	// Rust: BufferMapState::Idle is the only valid state for submit.
	if buf.MapState() != MapStateUnmapped {
		return fmt.Errorf("wgpu: Submit: command buffer at index %d references mapped buffer %q: %w",
			index, buf.Label(), ErrSubmitBufferMapped)
	}
	return nil
}

// validateSubmitTexture checks that a texture has not been released.
// Callers never pass nil: trackTexture skips nil and collectBindGroupResources
// only records non-nil entries.
func validateSubmitTexture(tex *Texture, index int) error {
	if tex.resolveHAL() == nil {
		return fmt.Errorf("wgpu: Submit: command buffer at index %d references released texture: %w",
			index, ErrSubmitTextureDestroyed)
	}
	return nil
}

// injectBarriers merges each command buffer's texture and buffer trackers
// into the device-level DeviceTracker and records any resulting transitions
// into a new barrier command buffer. Returns the barrier CB (or nil if no
// transitions are needed). The textures of zeros[i] are zeroed first in
// the barrier command buffer of command buffer i: an earlier command buffer
// of the submit discarded their contents, which read as zeros.
//
// The barrier CB is prepended before user command buffers in the Submit call,
// ensuring correct resource state transitions before GPU execution.
//
// The HAL encoder used for the barrier CB is tracked via DestroyQueue for
// recycling after GPU completion, matching the encoder lifecycle of user
// command encoders.
//
// Reference: wgpu-core device/queue.rs pre_submit_for_command_buffers
func (q *Queue) injectBarriers(commandBuffers []*CommandBuffer, zeros [][]*Texture) error {
	q.cbBarriers = q.cbBarriers[:0]
	if q.device == nil || q.device.core == nil {
		return nil
	}
	tracker := q.device.core.Tracker()
	if tracker == nil {
		return nil
	}

	// Each command buffer gets its own barrier command buffer, run right
	// ahead of it: the merge takes the device state from the command buffer
	// before, so cb2's barriers start from where cb1 leaves things and have
	// to run after cb1. All of them are recorded on one encoder, taken from
	// the pool once there is something to record.
	//
	// Reference: wgpu-core device/queue.rs pre_submit_for_command_buffers
	// (one transit command buffer per baked command buffer, close_and_push_front)
	defer func() {
		clear(q.texResolver)
		clear(q.bufResolver)
	}()
	resolved := false
	for i, cb := range commandBuffers {
		q.cbBarriers = append(q.cbBarriers, nil)
		if cb == nil || cb.core == nil {
			continue
		}
		var cbZeros []*Texture
		if i < len(zeros) {
			cbZeros = zeros[i]
		}
		q.zeroTransitions = q.zeroTransitions[:0]
		for _, t := range cbZeros {
			if idx, ok := q.pending.trackedIndex(t); ok {
				q.zeroTransitions = tracker.SetTextureUsage(idx, t.coreTexture.Subresources(), track.TextureSelector{},
					core.ZeroUses(t.format), q.zeroTransitions)
			}
		}
		q.texTransitions = tracker.MergeTextureScopeInto(cb.core.TextureScope(), q.texTransitions[:0])
		q.bufTransitions = tracker.MergeBufferScopeInto(cb.core.BufferScope(), q.bufTransitions[:0])
		if len(cbZeros) == 0 && len(q.texTransitions) == 0 && len(q.bufTransitions) == 0 {
			continue
		}
		if q.device.cmdEncoderPool == nil {
			continue // no pool = cannot create barrier CB
		}

		// Build resolvers that map TrackerIndex -> hal resource from all
		// resources referenced by the submitted command buffers. Only now
		// that there are barriers to record, and into maps the queue keeps.
		if !resolved {
			if q.texResolver == nil {
				q.texResolver = make(map[track.TrackerIndex]hal.Texture)
				q.bufResolver = make(map[track.TrackerIndex]hal.Buffer)
			}
			fillTextureResolver(q.texResolver, commandBuffers)
			fillBufferResolver(q.bufResolver, commandBuffers)
			resolved = true
		}
		barrierCB, err := q.recordAndTrackBarriers(cbZeros, q.texTransitions, q.bufTransitions)
		if err != nil {
			return err
		}
		q.cbBarriers[i] = barrierCB
	}
	clear(q.zeroTransitions)
	return nil
}

// fillTextureResolver walks all textures referenced by the submitted
// command buffers and maps their TrackerIndex to their hal.Texture.
// Includes both directly used textures and those bound via bind groups.
func fillTextureResolver(resolver map[track.TrackerIndex]hal.Texture, commandBuffers []*CommandBuffer) {
	for _, cb := range commandBuffers {
		if cb == nil {
			continue
		}
		collectTextureIndices(resolver, cb.usedTextures)
		for _, bg := range cb.usedBindGroups {
			if bg == nil {
				continue
			}
			collectBoundTextureIndices(resolver, bg.boundTextures)
		}
	}
}

// collectTextureIndices adds TrackerIndex -> hal.Texture entries from
// the used textures.
func collectTextureIndices(resolver map[track.TrackerIndex]hal.Texture, used []*Texture) {
	for _, tex := range used {
		addTrackedTexture(resolver, tex)
	}
}

// collectBoundTextureIndices adds TrackerIndex -> hal.Texture entries from
// bind group bound textures.
func collectBoundTextureIndices(resolver map[track.TrackerIndex]hal.Texture, bound []*Texture) {
	for _, tex := range bound {
		addTrackedTexture(resolver, tex)
	}
}

// addTrackedTexture adds a single texture to the resolver if it has a
// valid coreTexture and TrackerIndex.
func addTrackedTexture(resolver map[track.TrackerIndex]hal.Texture, tex *Texture) {
	if tex == nil || tex.coreTexture == nil {
		return
	}
	td := tex.coreTexture.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return
	}
	if _, exists := resolver[td.Index()]; exists {
		return
	}
	if halTex := tex.resolveHAL(); halTex != nil {
		resolver[td.Index()] = halTex
	}
}

// fillBufferResolver walks all buffers referenced by the submitted command
// buffers and maps their TrackerIndex to their hal.Buffer.
func fillBufferResolver(resolver map[track.TrackerIndex]hal.Buffer, commandBuffers []*CommandBuffer) {
	for _, cb := range commandBuffers {
		if cb == nil {
			continue
		}
		for _, buf := range cb.usedBuffers {
			addTrackedBuffer(resolver, buf)
		}
		for _, bg := range cb.usedBindGroups {
			if bg == nil {
				continue
			}
			for _, buf := range bg.boundBuffers {
				addTrackedBuffer(resolver, buf)
			}
		}
	}
}

// addTrackedBuffer adds a single buffer to the resolver if it has a
// valid core buffer and TrackerIndex.
func addTrackedBuffer(resolver map[track.TrackerIndex]hal.Buffer, buf *Buffer) {
	if buf == nil || buf.core == nil {
		return
	}
	td := buf.core.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return
	}
	if _, exists := resolver[td.Index()]; exists {
		return
	}
	if halBuf := buf.halBuffer(); halBuf != nil {
		resolver[td.Index()] = halBuf
	}
}

// recordAndTrackBarriers records the texture and buffer barriers of one
// command buffer of the submit into a HAL command buffer of their own, on
// the submit's barrier encoder, taken from the pool for the first one,
// after the zeroing of zeros, whose barriers q.zeroTransitions holds. The
// encoder and its command buffers are recycled after GPU completion.
func (q *Queue) recordAndTrackBarriers(
	zeros []*Texture,
	texTransitions []track.TexturePendingTransition,
	bufTransitions []track.PendingTransition,
) (hal.CommandBuffer, error) {
	if q.barrierEncoder == nil {
		halEnc, err := q.device.cmdEncoderPool.acquire()
		if err != nil {
			return nil, fmt.Errorf("acquire barrier encoder: %w", err)
		}
		q.barrierEncoder = halEnc
	}
	enc := q.barrierEncoder
	if err := enc.BeginEncoding("barrier-inject"); err != nil {
		return nil, fmt.Errorf("record barriers: %w", err)
	}
	core.RecordBarriers(enc, q.zeroTransitions, nil, q, &q.barrierScratch)
	for _, t := range zeros {
		raw, dev := t.resolveHAL(), q.device.halDevice()
		if raw == nil || dev == nil {
			continue
		}
		if view, _ := t.sharedDefaultView(dev, raw); view != nil {
			core.ZeroWithPass(enc, view, t.format)
		}
	}
	core.RecordBarriers(enc, texTransitions, bufTransitions, q, &q.barrierScratch)
	barrierCB, err := enc.EndEncoding()
	if err != nil {
		enc.DiscardEncoding()
		return nil, fmt.Errorf("record barriers: %w", err)
	}

	// Kept for deferred recycling. The actual Defer call happens in
	// postSubmit with the CURRENT submission index (not the stale
	// lastSubmissionIndex that was set before hal.Submit).
	q.barrierCBs = append(q.barrierCBs, barrierCB)
	return barrierCB, nil
}

// presentSurfaces records the barriers that take the surface textures the
// submit uses, those its command buffers use and those its queue writes
// wrote, to their present state after the last of them, in a barrier command
// buffer of their own at the end of q.submitBufs, as wgpu-core does at the
// end of every submit: the present finds them there.
//
// Reference: wgpu-core device/queue.rs submit (used_surface_textures go
// to PRESENT in a final command buffer)
func (q *Queue) presentSurfaces(commandBuffers []*CommandBuffer, written []*Texture) error {
	if q.device == nil || q.device.core == nil || q.device.core.Tracker() == nil || q.device.cmdEncoderPool == nil {
		return nil
	}
	defer func() {
		clear(q.presentBarriers)
		q.presentIndices, q.presentTransitions, q.presentBarriers = q.presentIndices[:0], q.presentTransitions[:0], q.presentBarriers[:0]
	}()
	tracker := q.device.core.Tracker()
	for _, t := range written {
		q.presentSurface(tracker, t)
	}
	for _, cb := range commandBuffers {
		for _, t := range cb.usedTextures {
			q.presentSurface(tracker, t)
		}
		for _, g := range cb.usedBindGroups {
			for _, t := range g.boundTextures {
				q.presentSurface(tracker, t)
			}
		}
	}
	if len(q.presentBarriers) == 0 {
		return nil
	}
	if q.barrierEncoder == nil {
		enc, err := q.device.cmdEncoderPool.acquire()
		if err != nil {
			return fmt.Errorf("acquire barrier encoder: %w", err)
		}
		q.barrierEncoder = enc
	}
	if err := q.barrierEncoder.BeginEncoding("present-barrier"); err != nil {
		return fmt.Errorf("record present barriers: %w", err)
	}
	q.barrierEncoder.TransitionTextures(q.presentBarriers)
	cb, err := q.barrierEncoder.EndEncoding()
	if err != nil {
		q.barrierEncoder.DiscardEncoding()
		return fmt.Errorf("record present barriers: %w", err)
	}
	q.barrierCBs = append(q.barrierCBs, cb)
	q.submitBufs = append(q.submitBufs, cb)
	return nil
}

// presentSurface moves t, when it is a surface texture not moved yet, to its
// present state in tracker, and keeps the barriers there.
func (q *Queue) presentSurface(tracker *core.DeviceTracker, t *Texture) {
	if t == nil || t.surface == nil || t.coreTexture == nil {
		return
	}
	td := t.coreTexture.TrackingData()
	if td == nil || !td.Index().IsValid() || slices.Contains(q.presentIndices, td.Index()) {
		return
	}
	raw := t.resolveHAL()
	if raw == nil {
		return
	}
	q.presentIndices = append(q.presentIndices, td.Index())
	n := len(q.presentTransitions)
	q.presentTransitions = tracker.SetTextureUsage(td.Index(), t.coreTexture.Subresources(), track.TextureSelector{},
		track.TextureUsesPresent, q.presentTransitions)
	for _, tr := range q.presentTransitions[n:] {
		q.presentBarriers = append(q.presentBarriers, tr.IntoHAL(raw))
	}
}

// release cleans up queue resources.
func (q *Queue) release() {
	if q.pending != nil {
		q.pending.destroy()
		q.pending = nil
	}
}
