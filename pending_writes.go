//go:build !(js && wasm)

package wgpu

import (
	"fmt"
	"sync"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// pendingWrites accumulates WriteBuffer operations into a
// single HAL command encoder. The accumulated commands are flushed as a single
// batch when the user calls Queue.Submit(), prepended before user command
// buffers. This eliminates per-write GPU submits and matches Rust wgpu-core's
// PendingWrites architecture.
//
// Encoder lifecycle (matching Rust wgpu-core CommandAllocator):
// Encoders are acquired from the encoder pool and returned after GPU completion.
// While one encoder records new copy commands, the previous one may be in-flight
// on the GPU. When the GPU completes, the inflight encoder is reset via ResetAll
// and returned to the pool. This avoids creating new DX12 command allocators or
// Vulkan command pools every frame, enabling maxFramesInFlight=2 on DX12.
//
// For GLES and Software backends (which use direct API calls, not command
// encoders for writes), pendingWrites delegates directly to the HAL queue.
type pendingWrites struct {
	mu sync.Mutex

	// halDevice is used to create staging buffers.
	halDevice hal.Device

	// halQueue is the underlying HAL queue for direct-write backends.
	halQueue hal.Queue

	// pool is a reference to the shared device-level encoder pool.
	// Not owned by pendingWrites — owned by Device. destroy() clears the
	// reference without destroying the pool. nil for non-batching backends.
	pool *encoderPool

	// belt manages reusable staging buffer chunks for zero-allocation
	// buffer writes. nil for non-batching backends.
	belt *stagingBelt

	// tracker is the device tracker: the state the last submit left a
	// texture in, which is where a queue write's barrier starts, and where
	// the write's own barriers leave it. Set by the device after creation.
	tracker *core.DeviceTracker

	// zeroBuf is what zeroTextureLocked copies zeros from, see zeroBuffer.
	// zeros is its source where it can't copy (no batching, host-writable
	// textures, rows wider than the buffer), grown on demand up to a band.
	zeroBuf hal.Buffer
	zeros   []byte

	// encoder is the shared command encoder for recording copy commands.
	// nil until the first write in a batch. Acquired from pool by activate().
	encoder hal.CommandEncoder

	// isRecording is true when encoder has had BeginEncoding called.
	isRecording bool

	// staging holds staging buffers that must remain alive until the GPU
	// completes the submission that references them. Moved to inflight on flush.
	staging []hal.Buffer

	// copyRegion is a pre-allocated single-element BufferCopy slice used by
	// writeBuffer to avoid heap escape when calling CopyBufferToBuffer through
	// the hal.CommandEncoder interface. Reused across writes within a frame.
	copyRegion [1]hal.BufferCopy

	// dstBuffers tracks buffers that have pending writes, keyed by HAL buffer.
	// Usage is passed from the wgpu public API level (matching Rust wgpu-core
	// where usage lives on core::Buffer, not on hal::Buffer). Used by flush()
	// to compute COPY_DST → read-state barriers.
	dstBuffers map[hal.Buffer]dstBuffer

	// dstTextures tracks textures that have pending writes, and dstOwners the
	// public wrappers that contributed commands for them, so that every
	// public lifetime is still live when the batch is flushed. This matters
	// for swapchains, which may reuse the same HAL texture across
	// acquisitions. A flat list rather than a list per texture, so a steady
	// stream of glyph uploads allocates nothing.
	dstTextures map[hal.Texture]struct{}
	dstOwners   []dstOwner

	// Scratch for writeTextureLocked and writeBufferFor, under pw.mu: what
	// they hand the HAL by slice, which does not keep it.
	texBarrier [1]hal.TextureBarrier
	texCopy    [1]hal.BufferTextureCopy
	bufBarrier [1]hal.BufferBarrier

	// inflight tracks staging buffers and encoders from previous submissions,
	// keyed by submission index. Cleaned up when PollCompleted advances past them.
	inflight []inflightSubmission

	// usesBatching is true for DX12/Vulkan/Metal (command-buffer-based
	// copy backends). false for GLES/Software (direct API writes).
	// Set once at creation, never changes.
	usesBatching bool

	// alignBuf is a reusable scratch buffer for row-pitch alignment in writeTexture.
	// Grown on demand; never shrunk. Eliminates per-upload make() when texture row
	// stride is not a multiple of pendingWritesRowPitchAlignment.
	// Access is safe because writeTexture always holds pw.mu.
	alignBuf []byte

	// zero holds the textures created since the last flush that nothing has
	// written yet. WebGPU hands out textures zeroed, the backends hand out
	// whatever was in the memory before, so a partial write and flush zero
	// them first. A write or render pass that covers the whole texture takes
	// it off without zeroing.
	zero map[*Texture]struct{}

	// views are what the passes that zero textures in the batch render to
	// (clearTextureLocked). flush hands them to flushedViews, which Submit
	// puts in flight with the batch, and they are destroyed once the GPU is
	// done with it.
	views, flushedViews []hal.TextureView

	// destroys are the HAL destroys of textures released while the batch
	// copies into them (holdDestroy), and refs keep the buffers the batch
	// copies into alive, as a command buffer keeps what it binds. Both run
	// out once the batch's submission is complete; flush hands them to the
	// flushed ones, which Submit moves to the inflight record.
	destroys, flushedDestroys []heldDestroy
	refs, flushedRefs         []*core.ResourceRef

	// Scratch reused under pw.mu, so a steady stream of flushes allocates
	// nothing: flush's barrier list, maintain's one element command buffer
	// list for ResetAll, and the destination lists of completed inflight
	// submissions, emptied, for the next flushes to fill.
	barriers         []hal.BufferBarrier
	resetList        [1]hal.CommandBuffer
	spareBufferLists [][]hal.Buffer
	spareTexLists    [][]hal.Texture
}

// maxSpareDstLists caps how many emptied destination lists pendingWrites
// keeps of each kind.
const maxSpareDstLists = 32

// takeSpare returns a kept emptied list, or nil.
func takeSpare[T any](spares *[][]T) []T {
	n := len(*spares)
	if n == 0 {
		return nil
	}
	l := (*spares)[n-1]
	(*spares)[n-1] = nil
	*spares = (*spares)[:n-1]
	return l
}

// keepSpare clears a list nothing references any more and keeps it.
func keepSpare[T any](spares *[][]T, l []T) {
	if cap(l) == 0 || len(*spares) >= maxSpareDstLists {
		return
	}
	clear(l)
	*spares = append(*spares, l[:0])
}

// dstOwner is a public texture wrapper that wrote to a raw texture in the
// batch being recorded.
type dstOwner struct {
	raw   hal.Texture
	owner *Texture
}

// dstBuffer is a buffer the batch writes: its creation usage and the public
// wrapper that wrote it, nil from the ownerless writeBuffer.
type dstBuffer struct {
	usage gputypes.BufferUsage
	owner *Buffer
}

// inflightSubmission tracks resources from a single submission that must
// remain alive until the GPU completes that submission.
type inflightSubmission struct {
	submissionIndex uint64
	destroys        []heldDestroy
	refs            []*core.ResourceRef
	staging         []hal.Buffer
	cmdBuf          hal.CommandBuffer  // pending writes command buffer (consumed by ResetAll)
	encoder         hal.CommandEncoder // encoder to reset+release after completion (nil if not pooled)
	// dstTextures and dstBuffers hold references to destination resources used by
	// CopyBufferToTexture/CopyBufferToBuffer in this submission. This prevents
	// premature Release() from destroying D3D12 resources while the GPU is still
	// executing commands that reference them. Root cause of DX12 TDR (BUG-DX12-006).
	dstTextures []hal.Texture
	dstBuffers  []hal.Buffer
	// views are the views of the passes that zeroed textures, see
	// pendingWrites.views.
	views []hal.TextureView
}

// heldDestroy is a texture's HAL destroy held back while the batch being
// recorded copies into it (Texture.Release): now runs it once the batch's
// submission is complete, later defers it as a plain Release does, for a
// batch that is dropped.
type heldDestroy struct {
	now, later func()
}

// holdDestroy holds t's HAL destroy until the batch being recorded, which
// copies into t, has run. A Release between a WriteTexture and the Submit
// used to destroy the texture under the copy, which made flush drop the
// whole batch. It reports false when the batch does not write t, and the
// caller destroys as usual.
func (pw *pendingWrites) holdDestroy(t *Texture, now, later func()) bool {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	if !pw.isRecording {
		return false
	}
	for _, o := range pw.dstOwners {
		if o.owner == t {
			pw.destroys = append(pw.destroys, heldDestroy{now: now, later: later})
			return true
		}
	}
	return false
}

// dropHeld runs the held destroys as deferred and drops the refs of a batch
// that will not run.
func dropHeld(destroys []heldDestroy, refs []*core.ResourceRef) {
	for _, h := range destroys {
		h.later()
	}
	for _, r := range refs {
		r.Drop()
	}
}

// newPendingWrites creates a pendingWrites for the given HAL device and queue.
// For batching backends (DX12/Vulkan/Metal), pool is the shared device-level
// encoder pool (same pool used by CreateCommandEncoder). This matches Rust
// wgpu-core which uses a single device.command_allocator for both user command
// encoders and internal PendingWrites (queue.rs:1373). pool may be nil for
// non-batching backends (GLES/Software).
//
// The staging belt's maxStagingBufferSize is automatically queried from the HAL
// device via the optional hal.MaxStagingBufferSizer interface (BUG-VK-001).
// For Vulkan, this returns min(64MB, maxMemoryAllocationSize). For backends
// that do not implement the interface, the default 64MB cap is used.
func newPendingWrites(halDevice hal.Device, halQueue hal.Queue, pool *encoderPool) *pendingWrites {
	pw := &pendingWrites{
		halDevice:    halDevice,
		halQueue:     halQueue,
		dstBuffers:   make(map[hal.Buffer]dstBuffer),
		dstTextures:  make(map[hal.Texture]struct{}),
		usesBatching: halQueue.SupportsCommandBufferCopies(),
	}
	if pw.usesBatching {
		pw.pool = pool
		// Query the device for its maximum staging buffer allocation size.
		// Vulkan devices report min(64MB, maxMemoryAllocationSize) to prevent
		// vkAllocateMemory from silently failing on oversized requests (BUG-VK-001).
		// Other backends return 0 (use default 64MB), which is safe.
		var maxStaging uint64
		if sizer, ok := halDevice.(hal.MaxStagingBufferSizer); ok {
			maxStaging = sizer.MaxStagingBufferSize()
		}
		pw.belt = newStagingBelt(halDevice, halQueue, 0, 0, maxStaging)
	}
	return pw
}

// writeBuffer records a buffer write. For batching backends, creates a staging
// buffer, copies data via CPU, and records CopyBufferToBuffer into the shared
// encoder. For direct-write backends, delegates to halQueue.WriteBuffer.
//
// usage is the buffer's WebGPU creation usage flags, passed from the wgpu public
// API level (wgpu.Buffer → Queue.WriteBuffer → here). This matches Rust wgpu-core
// where usage lives on core::Buffer, not hal::Buffer. Used by flush() to compute
// correct COPY_DST → read-state transition barriers (BUG-DX12-010).
func (pw *pendingWrites) writeBuffer(buffer hal.Buffer, usage gputypes.BufferUsage, offset uint64, data []byte) error {
	return pw.writeBufferFor(nil, buffer, usage, offset, data)
}

// writeBufferFor is writeBuffer for the public buffer owner, whose state in
// the device tracker the write moves to CopyDst, Queue.WriteBuffer's form.
func (pw *pendingWrites) writeBufferFor(owner *Buffer, buffer hal.Buffer, usage gputypes.BufferUsage, offset uint64, data []byte) error {
	pw.mu.Lock()
	defer pw.mu.Unlock()

	if !pw.usesBatching {
		return pw.halQueue.WriteBuffer(buffer, offset, data)
	}

	dataLen := uint64(len(data))
	if dataLen == 0 {
		return nil
	}

	// Allocate from staging belt (ring-buffer of reusable chunks).
	// Zero heap allocations in steady state — chunks are pre-allocated
	// and recycled after GPU completion. Matches Rust wgpu StagingBelt.
	alloc, err := pw.belt.allocate(dataLen, data)
	if err != nil {
		return fmt.Errorf("wgpu: pending writes: staging belt allocate: %w", err)
	}

	// Activate encoder (lazy creation + BeginEncoding).
	enc, err := pw.activate()
	if err != nil {
		return fmt.Errorf("wgpu: pending writes: activate encoder: %w", err)
	}

	// The write is a CopyDst use of the buffer, after a barrier from what
	// the GPU used it as last, so it neither lands under the reads of
	// earlier submits nor races an earlier write, and the next submit's
	// barriers start from it.
	if idx, ok := pw.trackedBufferIndex(owner); ok {
		if transition, needed := pw.tracker.SetBufferUsage(idx, track.BufferUsesCopyDst); needed {
			pw.bufBarrier[0] = transition.IntoHAL(buffer)
			enc.TransitionBuffers(pw.bufBarrier[:])
			pw.bufBarrier[0] = hal.BufferBarrier{}
		}
	}

	// Check if the allocation was chunked (BUG-VK-001).
	// For chunked writes, we issue one CopyBufferToBuffer per chunk at
	// the correct destination offset. Each chunk has its own staging buffer.
	if len(pw.belt.chunkedAllocs) > 1 {
		dstOffset := offset
		for _, chunk := range pw.belt.chunkedAllocs {
			copyRegion := [1]hal.BufferCopy{{
				SrcOffset: chunk.offset,
				DstOffset: dstOffset,
				Size:      chunk.size,
			}}
			enc.CopyBufferToBuffer(chunk.buffer, buffer, copyRegion[:])
			dstOffset += chunk.size
		}
		pw.belt.chunkedAllocs = pw.belt.chunkedAllocs[:0]
	} else {
		// Normal (non-chunked) path: single CopyBufferToBuffer.
		// Use pre-allocated pw.copyRegion to avoid heap escape through interface call.
		pw.copyRegion[0] = hal.BufferCopy{
			SrcOffset: alloc.offset,
			DstOffset: offset,
			Size:      dataLen,
		}
		enc.CopyBufferToBuffer(alloc.buffer, buffer, pw.copyRegion[:])
	}

	// The batch keeps the buffer alive until its copy has run: a Release
	// before the Submit destroyed the buffer under the copy.
	if _, seen := pw.dstBuffers[buffer]; !seen && owner != nil && owner.core != nil && owner.core.Ref != nil {
		owner.core.Ref.Clone()
		pw.refs = append(pw.refs, owner.core.Ref)
	}

	// Track destination buffer for barrier computation at flush time.
	pw.dstBuffers[buffer] = dstBuffer{usage: usage, owner: owner}

	return nil
}

// pendingWritesRowPitchAlignment is the row pitch alignment used for staging
// buffer-to-texture copies. 256 bytes is safe for all batching backends:
// DX12 requires D3D12_TEXTURE_DATA_PITCH_ALIGNMENT (256), Vulkan benefits
// from optimalBufferCopyRowPitchAlignment (typically 256), and Metal has no
// requirement but 256 is recommended.
const pendingWritesRowPitchAlignment = 256

// writeTexture records a texture write. Creates a staging buffer with proper
// row pitch alignment, copies data via CPU, and records CopyBufferToTexture
// into the shared encoder with correct resource barriers.
//
// Barrier strategy (matching Rust wgpu-core queue.rs:935-947):
// - Before copy: transition texture from CurrentUsage() → CopyDst
// - After copy: transition texture CopyDst → TextureBinding (SHADER_RESOURCE)
//
// CurrentUsage() returns the DX12 texture's tracked currentState mapped to
// gputypes.TextureUsage. For newly created textures this is 0 (COMMON),
// for previously-written textures this is TextureBinding (SHADER_RESOURCE).
// On non-DX12 backends, CurrentUsage() returns 0 and TransitionTextures is a no-op.
func (pw *pendingWrites) writeTexture(
	dst *hal.ImageCopyTexture,
	data []byte,
	layout *hal.ImageDataLayout,
	size *hal.Extent3D,
) error {
	return pw.writeTextureFor(nil, dst, data, layout, size)
}

// writeTextureFor records a texture write and retains the public wrapper whose
// lifetime authorized it. Queue.WriteTexture supplies owner; the ownerless
// form above remains available to focused HAL-level tests.
func (pw *pendingWrites) writeTextureFor(
	owner *Texture,
	dst *hal.ImageCopyTexture,
	data []byte,
	layout *hal.ImageDataLayout,
	size *hal.Extent3D,
) error {
	pw.mu.Lock()
	defer pw.mu.Unlock()

	if _, ok := pw.zero[owner]; ok {
		delete(pw.zero, owner)
		if !coversTexture(owner, dst, size) {
			if err := pw.zeroTextureLocked(owner); err != nil {
				return err
			}
		}
	}
	return pw.writeTextureLocked(owner, dst, data, layout, size)
}

// writeTextureLocked is writeTextureFor with pw.mu held.
func (pw *pendingWrites) writeTextureLocked(
	owner *Texture,
	dst *hal.ImageCopyTexture,
	data []byte,
	layout *hal.ImageDataLayout,
	size *hal.Extent3D,
) error {
	if !pw.usesBatching {
		return pw.halQueue.WriteTexture(dst, data, layout, size)
	}

	if len(data) == 0 {
		return nil
	}

	// Host-writable textures (Metal Shared on Apple Silicon UMA) accept direct
	// CPU writes without staging buffers or GPU copy commands. Bypass the
	// staging belt to avoid allocating ~width*height*4 bytes per frame.
	if hw, ok := dst.Texture.(interface{ IsHostWritable() bool }); ok && hw.IsHostWritable() {
		return pw.halQueue.WriteTexture(dst, data, layout, size)
	}

	// Calculate aligned row pitch.
	bytesPerRow := layout.BytesPerRow
	if bytesPerRow == 0 {
		bytesPerRow = uint32(len(data)) //nolint:gosec // data length fits uint32 for textures
	}

	alignedBytesPerRow := alignUp(bytesPerRow, pendingWritesRowPitchAlignment)

	rowsPerImage := layout.RowsPerImage
	if rowsPerImage == 0 {
		rowsPerImage = size.Height
	}

	depthOrLayers := size.DepthOrArrayLayers
	if depthOrLayers == 0 {
		depthOrLayers = 1
	}

	stagingSize := uint64(alignedBytesPerRow) * uint64(rowsPerImage) * uint64(depthOrLayers)

	// CPU copy with row pitch alignment. Returns data directly when no padding needed,
	// otherwise writes into pw.alignBuf (grown on demand, never shrunk, reused each call).
	stagingData := alignTextureDataInto(&pw.alignBuf, data, layout.Offset, bytesPerRow, alignedBytesPerRow, rowsPerImage, depthOrLayers, stagingSize)
	// The belt writes all it is given: data that goes on past the copy, the
	// rest of an image a region is uploaded from, would run over the next
	// allocation of the chunk.
	if uint64(len(stagingData)) > stagingSize {
		stagingData = stagingData[:stagingSize]
	}

	// Allocate from staging belt (ring-buffer of reusable chunks).
	alloc, err := pw.belt.allocate(stagingSize, stagingData)
	if err != nil {
		return fmt.Errorf("wgpu: pending writes: staging belt allocate texture: %w", err)
	}
	return pw.copyToTextureLocked(owner, dst, alloc.buffer, alloc.offset, alignedBytesPerRow, rowsPerImage, size)
}

// copyToTextureLocked records a copy from src at offset into dst, between the
// barriers that make the texture a copy destination and take it back to what
// it is used as. pw.mu is held and usesBatching is set.
func (pw *pendingWrites) copyToTextureLocked(
	owner *Texture,
	dst *hal.ImageCopyTexture,
	src hal.Buffer,
	offset uint64,
	bytesPerRow, rowsPerImage uint32,
	size *hal.Extent3D,
) error {
	enc, err := pw.activate()
	if err != nil {
		return fmt.Errorf("wgpu: pending writes: activate encoder: %w", err)
	}
	w := pw.beginTextureWrite(enc, owner, dst, size)

	// Record copy command.
	pw.texCopy[0] = hal.BufferTextureCopy{
		BufferLayout: hal.ImageDataLayout{
			Offset:       offset,
			BytesPerRow:  bytesPerRow,
			RowsPerImage: rowsPerImage,
		},
		TextureBase: *dst,
		Size:        *size,
	}
	enc.CopyBufferToTexture(src, dst.Texture, pw.texCopy[:])
	pw.texCopy[0] = hal.BufferTextureCopy{}

	pw.endTextureWrite(enc, owner, dst, w)
	return nil
}

// textureWrite is what beginTextureWrite found out about a texture that
// endTextureWrite needs.
type textureWrite struct {
	copyRange hal.TextureRange
	idx       track.TrackerIndex
	tracked   bool
}

// beginTextureWrite records the barrier that makes dst a copy destination.
func (pw *pendingWrites) beginTextureWrite(enc hal.CommandEncoder, owner *Texture, dst *hal.ImageCopyTexture, size *hal.Extent3D) textureWrite {
	// Transition texture to COPY_DST from the state the GPU has it in: what
	// the last submit left (the device tracker), else what the backend tracks
	// itself (DX12), else undefined, which is right for a new texture and
	// nothing else: on Vulkan it lets the driver discard the contents. A
	// texture the tracker only knows as uninitialized (every texture is,
	// from creation) has no usage to name, and the backend's own state
	// stands.
	currentUsage := dst.Texture.CurrentUsage()
	var currentUses hal.TextureUses
	idx, tracked := pw.trackedIndex(owner)
	if tracked {
		state := pw.tracker.TextureUsage(idx)
		currentUses = state.ToHAL()
		if uses := state.ToTextureUsage(); uses != 0 {
			currentUsage = uses
		}
	}
	copyRange := textureCopyRange(owner, dst, size)
	// One the tracker knows as a copy destination already was written by an
	// earlier copy, which this one has to wait for.
	if currentUsage != gputypes.TextureUsageCopyDst || currentUses&hal.TextureUsesCopyDst != 0 {
		pw.texBarrier[0] = hal.TextureBarrier{
			Texture: dst.Texture,
			Range:   copyRange,
			Usage: hal.TextureUsageTransition{
				OldUsage: currentUsage,
				NewUsage: gputypes.TextureUsageCopyDst,
				Old:      currentUses,
				New:      hal.TextureUsesCopyDst,
			},
		}
		enc.TransitionTextures(pw.texBarrier[:])
		pw.texBarrier[0] = hal.TextureBarrier{}
	}
	return textureWrite{copyRange: copyRange, idx: idx, tracked: tracked}
}

// endTextureWrite records what follows a copy into dst: the barrier taking it
// to what it is used as, and its new state in the tracker.
func (pw *pendingWrites) endTextureWrite(enc hal.CommandEncoder, owner *Texture, dst *hal.ImageCopyTexture, w textureWrite) {
	// A texture that is never sampled, a surface texture (presented or
	// rendered over) or one made without TextureBinding, stays a copy
	// destination, and the present, the render pass or the next copy takes
	// it from there: a barrier into TextureBinding names a sampled layout
	// the image has no usage for (VUID-VkImageMemoryBarrier-oldLayout-01211
	// on Vulkan).
	if owner.textureUsage()&gputypes.TextureUsageTextureBinding == 0 {
		if w.tracked {
			pw.tracker.InsertTexture(w.idx, track.TextureUsesCopyDst)
		}
	} else {
		// Transition texture to SHADER_RESOURCE for rendering.
		// ADR-060: The centralized TextureTracker now handles render pass -> submit barriers.
		// These eager barriers remain for the pending_writes -> render path until the tracker
		// also covers queue write operations (pending_writes integration, future work).
		pw.texBarrier[0] = hal.TextureBarrier{
			Texture: dst.Texture,
			Range:   w.copyRange,
			Usage: hal.TextureUsageTransition{
				OldUsage: gputypes.TextureUsageCopyDst,
				NewUsage: gputypes.TextureUsageTextureBinding,
				Old:      hal.TextureUsesCopyDst,
				New:      hal.TextureUsesResource,
			},
		}
		enc.TransitionTextures(pw.texBarrier[:])
		pw.texBarrier[0] = hal.TextureBarrier{}

		// The write leaves the texture ready to sample; the barriers of the
		// submits that follow start from there.
		if w.tracked {
			pw.tracker.InsertTexture(w.idx, track.TextureUsesResource)
		}
	}

	// Staging buffer is managed by the belt (not tracked individually).
	pw.trackDst(owner, dst.Texture)
}

// copyExternalLocked records copies of the planes of src, an image made
// outside gowgpu in layout, layer slice of it, into dsts, plane i into dsts[i]
// whole, between the barriers a write has, and returns the layout the copy
// leaves src in. pw.mu is held and usesBatching is set.
func (pw *pendingWrites) copyExternalLocked(src uintptr, layout, slice uint32, owners []*Texture) (uint32, error) {
	enc, err := pw.activate()
	if err != nil {
		return layout, fmt.Errorf("wgpu: pending writes: activate encoder: %w", err)
	}
	copier, ok := enc.(interface {
		CopyExternalPlanes(src uintptr, layout, slice uint32, dsts []hal.Texture) uint32
	})
	if !ok {
		return layout, fmt.Errorf("wgpu: copying from external images is not supported on this backend")
	}
	var (
		dsts   [3]hal.ImageCopyTexture
		raws   [3]hal.Texture
		writes [3]textureWrite
	)
	for i, owner := range owners {
		raws[i] = owner.resolveHAL()
		if raws[i] == nil || owner.coreTexture == nil {
			return layout, ErrReleased
		}
		delete(pw.zero, owner) // the copy fills it, where zeroing at flush would clear it after
		dsts[i] = hal.ImageCopyTexture{Texture: raws[i]}
		s := owner.coreTexture.Size()
		writes[i] = pw.beginTextureWrite(enc, owner, &dsts[i], &hal.Extent3D{Width: s.Width, Height: s.Height, DepthOrArrayLayers: 1})
	}
	after := copier.CopyExternalPlanes(src, layout, slice, raws[:len(owners)])
	for i, owner := range owners {
		pw.endTextureWrite(enc, owner, &dsts[i], writes[i])
	}
	return after, nil
}

// copyExternalBufferLocked records a copy of src, a buffer made outside
// gowgpu with rows bytesPerRow apart, into owner whole, between the barriers
// a write has. pw.mu is held and usesBatching is set.
func (pw *pendingWrites) copyExternalBufferLocked(src uintptr, bytesPerRow uint32, owner *Texture) error {
	enc, err := pw.activate()
	if err != nil {
		return fmt.Errorf("wgpu: pending writes: activate encoder: %w", err)
	}
	copier, ok := enc.(interface {
		CopyExternalBuffer(src uintptr, bytesPerRow uint32, dst hal.Texture)
	})
	if !ok {
		return fmt.Errorf("wgpu: copying from external buffers is not supported on this backend")
	}
	raw := owner.resolveHAL()
	if raw == nil || owner.coreTexture == nil {
		return ErrReleased
	}
	delete(pw.zero, owner) // the copy fills it, where zeroing at flush would clear it after
	dst := hal.ImageCopyTexture{Texture: raw}
	s := owner.coreTexture.Size()
	write := pw.beginTextureWrite(enc, owner, &dst, &hal.Extent3D{Width: s.Width, Height: s.Height, DepthOrArrayLayers: 1})
	copier.CopyExternalBuffer(src, bytesPerRow, raw)
	pw.endTextureWrite(enc, owner, &dst, write)
	return nil
}

// trackDst keeps raw, a texture the batch writes, alive until the GPU is done
// with the batch: AddPendingRef prevents premature Destroy (BUG-DX12-006). And
// it lists owner for the check at flush.
func (pw *pendingWrites) trackDst(owner *Texture, raw hal.Texture) {
	if _, already := pw.dstTextures[raw]; !already {
		raw.AddPendingRef()
		pw.dstTextures[raw] = struct{}{}
	}
	if owner != nil && !pw.hasOwner(raw, owner) {
		pw.dstOwners = append(pw.dstOwners, dstOwner{raw: raw, owner: owner})
	}
}

// textureUsage is the usage the texture was made with, or the surface
// configured with; every usage for a wrapper without core state, so a
// test stub is treated as before.
func (t *Texture) textureUsage() gputypes.TextureUsage {
	if t == nil || t.coreTexture == nil {
		return ^gputypes.TextureUsage(0)
	}
	return t.coreTexture.Usage()
}

// hasOwner reports whether owner is already listed for raw in this batch.
func (pw *pendingWrites) hasOwner(raw hal.Texture, owner *Texture) bool {
	for _, o := range pw.dstOwners {
		if o.raw == raw && o.owner == owner {
			return true
		}
	}
	return false
}

// trackedBufferIndex is owner's index in the device tracker, when there is
// a tracker and the buffer has an index.
func (pw *pendingWrites) trackedBufferIndex(owner *Buffer) (track.TrackerIndex, bool) {
	if pw.tracker == nil || owner == nil || owner.core == nil {
		return 0, false
	}
	td := owner.core.TrackingData()
	if td == nil || !td.Index().IsValid() || td.IsReleased() {
		return 0, false
	}
	return td.Index(), true
}

// trackedIndex is owner's index in the device tracker, when there is a
// tracker and the texture has an index.
func (pw *pendingWrites) trackedIndex(owner *Texture) (track.TrackerIndex, bool) {
	if pw.tracker == nil || owner == nil || owner.coreTexture == nil {
		return 0, false
	}
	td := owner.coreTexture.TrackingData()
	if td == nil || !td.Index().IsValid() || td.IsReleased() {
		return 0, false
	}
	return td.Index(), true
}

func textureCopyRange(owner *Texture, dst *hal.ImageCopyTexture, size *hal.Extent3D) hal.TextureRange {
	baseLayer := uint32(0)
	layerCount := uint32(1)
	if owner != nil && owner.coreTexture != nil && owner.coreTexture.Dimension() != gputypes.TextureDimension3D {
		baseLayer = dst.Origin.Z
		layerCount = size.DepthOrArrayLayers
		if layerCount == 0 {
			layerCount = 1
		}
	}
	return hal.TextureRange{
		Aspect:          dst.Aspect,
		BaseMipLevel:    dst.MipLevel,
		MipLevelCount:   1,
		BaseArrayLayer:  baseLayer,
		ArrayLayerCount: layerCount,
	}
}

// zeroableTexture reports whether a texture made from desc gets zeroed through
// the pending writes, and whether by a pass (zeroByPass) rather than by
// copies, which take a color format with a copy size and one sample.
// ponytail: GL writes to textures straight away instead of recording the
// writes, so it zeroes uncompressed color textures from the CPU and leaves
// depth, multisampled and compressed ones with old memory: the GL HAL has no
// depth-only passes and no compressed uploads, and a clear pass would need
// an encoder of its own, submitted ahead of the rest.
func zeroableTexture(desc *hal.TextureDescriptor, batching bool) (zero, pass bool) {
	f := desc.Format
	color := !f.IsDepthStencil() && desc.SampleCount <= 1 && f.BlockCopySize() != 0
	pass = zeroByPass(f, desc.Dimension, desc.SampleCount, desc.Usage)
	switch {
	case !batching:
		return color && f <= gputypes.TextureFormatRGBA32Sint, pass // the uncompressed ones, in webgpu.h order
	case pass:
		return desc.Dimension == gputypes.TextureDimension2D, true
	}
	return color, false
}

// zeroByPass reports whether a new texture is zeroed by a render pass that
// clears it and stores, as Rust wgpu-core clears render attachments: depth
// and stencil, which a copy takes an aspect at a time if at all, multisampled
// textures, which no copy reaches, and 2D render targets nothing samples or
// copies into, which then needn't become copy destinations. Anything else is
// zeroed by copies.
func zeroByPass(f gputypes.TextureFormat, dim gputypes.TextureDimension, samples uint32, usage gputypes.TextureUsage) bool {
	return f.IsDepthStencil() || samples > 1 ||
		dim == gputypes.TextureDimension2D && usage&gputypes.TextureUsageRenderAttachment != 0 &&
			usage&(gputypes.TextureUsageTextureBinding|gputypes.TextureUsageCopyDst) == 0
}

// queueZero puts a new texture on the list of textures to zero.
func (pw *pendingWrites) queueZero(t *Texture) {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	if pw.zero == nil {
		pw.zero = make(map[*Texture]struct{})
	}
	pw.zero[t] = struct{}{}
}

// forgetZero takes t off the list: it was released, or a pass cleared all of it.
func (pw *pendingWrites) forgetZero(t *Texture) {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	delete(pw.zero, t)
}

// coversTexture reports whether a write of size at dst sets every texel of t.
// A write sets one aspect of a depth/stencil format, never both.
func coversTexture(t *Texture, dst *hal.ImageCopyTexture, size *hal.Extent3D) bool {
	s, f := t.coreTexture.Size(), t.coreTexture.Format()
	return t.coreTexture.MipLevelCount() <= 1 && dst.MipLevel == 0 && dst.Origin == (hal.Origin3D{}) &&
		size.Width == s.Width && size.Height == s.Height &&
		max(size.DepthOrArrayLayers, 1) == max(s.DepthOrArrayLayers, 1) &&
		(!f.HasDepth() || !f.HasStencil())
}

// zeroBand is how many bytes of a texture are zeroed at a time, some rows of
// it: the size of the zero buffer, and the cap on the CPU zeros. It holds a
// row of a 16384 texel wide texture of the widest format.
const zeroBand = 256 << 10

// zeroTextureLocked writes zeros over every mip and layer of t. pw.mu is held.
func (pw *pendingWrites) zeroTextureLocked(t *Texture) error {
	raw := t.resolveHAL()
	if raw == nil {
		return nil
	}
	// A host-writable texture takes writes on the CPU straight away, which a
	// copy made at the submit would land on top of, and GL writes nothing any
	// other way. Both zero a color render target from the CPU too.
	// ponytail: a depth texture gets its clear pass even when it is
	// host-writable (Metal on Apple GPUs), so a WriteTexture into part of
	// one before its first submit lands under the clear.
	hw, ok := raw.(interface{ IsHostWritable() bool })
	cpu := !pw.usesBatching || ok && hw.IsHostWritable()
	ct := t.coreTexture
	f := ct.Format()
	if zeroByPass(f, ct.Dimension(), ct.SampleCount(), ct.Usage()) && (!cpu || f.IsDepthStencil()) {
		return pw.clearTextureLocked(t, raw)
	}
	s := ct.Size()
	// The copies go by rows of blocks, which a compressed format makes 4 or
	// more texels high, and stop at the edge of the mip level, where a block
	// may stick out.
	bw, bh := f.BlockDimensions()
	blockSize := f.BlockCopySize()
	for mip := range max(ct.MipLevelCount(), 1) {
		w, h, layers := max(s.Width>>mip, 1), max(s.Height>>mip, 1), max(s.DepthOrArrayLayers, 1)
		if ct.Dimension() == gputypes.TextureDimension3D {
			layers = max(layers>>mip, 1)
		}
		blockRows := (h + bh - 1) / bh
		bytesPerRow := alignUp((w+bw-1)/bw*blockSize, pendingWritesRowPitchAlignment)
		rows := min(blockRows, max(zeroBand/bytesPerRow, 1))
		var zeroBuf hal.Buffer
		if !cpu && bytesPerRow <= zeroBand {
			var err error
			if zeroBuf, err = pw.zeroBuffer(); err != nil {
				return fmt.Errorf("wgpu: zero new texture: %w", err)
			}
		} else if n := int(bytesPerRow * rows); len(pw.zeros) < n {
			pw.zeros = make([]byte, n)
		}
		for z := range layers {
			// The tracker knows the texture as a whole, which the copies of
			// the levels and layers before moved on, while this one is still
			// as the texture was made, where its first copy starts. A level
			// of a 3D texture is one subresource, every slice of it.
			if idx, ok := pw.trackedIndex(t); ok && !cpu && (z == 0 || ct.Dimension() != gputypes.TextureDimension3D) {
				pw.tracker.InsertTexture(idx, track.TextureUsesUninitialized)
			}
			for r := uint32(0); r < blockRows; r += rows {
				n, y := min(rows, blockRows-r), r*bh
				dst := hal.ImageCopyTexture{Texture: raw, MipLevel: mip, Origin: hal.Origin3D{Y: y, Z: z}, Aspect: gputypes.TextureAspectAll}
				size := hal.Extent3D{Width: w, Height: min(n*bh, h-y), DepthOrArrayLayers: 1}
				var err error
				if zeroBuf != nil {
					// RowsPerImage 0: the copy is one image, and the HALs
					// don't count the rows of a compressed format alike.
					err = pw.copyToTextureLocked(t, &dst, zeroBuf, 0, bytesPerRow, 0, &size)
				} else {
					layout := hal.ImageDataLayout{BytesPerRow: bytesPerRow, RowsPerImage: n}
					err = pw.writeTextureLocked(t, &dst, pw.zeros[:bytesPerRow*n], &layout, &size)
				}
				if err != nil {
					return fmt.Errorf("wgpu: zero new texture: %w", err)
				}
			}
		}
	}
	return nil
}

// clearTextureLocked zeroes t, a 2D texture, with a render pass per mip level
// and layer that clears it (color 0, depth 0, stencil 0) and stores, as Rust
// wgpu-core's clear_texture_via_render_passes does. A texture on the zero
// list was made after the last submit and nothing in the batch touched it, so
// the passes start from the state it was made in. pw.mu is held and
// usesBatching is set.
func (pw *pendingWrites) clearTextureLocked(t *Texture, raw hal.Texture) error {
	enc, err := pw.activate()
	if err != nil {
		return fmt.Errorf("wgpu: pending writes: activate encoder: %w", err)
	}
	ct := t.coreTexture
	f := ct.Format()
	layers := max(ct.Size().DepthOrArrayLayers, 1)
	dim := gputypes.TextureViewDimension2D
	if layers > 1 {
		dim = gputypes.TextureViewDimension2DArray // a 2D view is the first layer on DX12
	}
	for mip := range max(ct.MipLevelCount(), 1) {
		for layer := range layers {
			view, err := pw.halDevice.CreateTextureView(raw, &hal.TextureViewDescriptor{
				Label:     "(wgpu internal) clear view",
				Dimension: dim, BaseMipLevel: mip, MipLevelCount: 1, BaseArrayLayer: layer, ArrayLayerCount: 1,
			})
			if err != nil {
				return fmt.Errorf("wgpu: zero new texture: %w", err)
			}
			pw.views = append(pw.views, view)
			desc := &hal.RenderPassDescriptor{Label: "(wgpu internal) clear pass"}
			if f.IsDepthStencil() {
				ds := &hal.RenderPassDepthStencilAttachment{View: view}
				if f.HasDepth() {
					ds.DepthLoadOp, ds.DepthStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
				}
				if f.HasStencil() {
					ds.StencilLoadOp, ds.StencilStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
				}
				desc.DepthStencilAttachment = ds
			} else {
				desc.ColorAttachments = []hal.RenderPassColorAttachment{{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}}
			}
			if pass := enc.BeginRenderPass(desc); pass != nil {
				pass.End()
			}
		}
	}
	// The passes leave it an attachment, which is where the barriers of the
	// writes and submits that follow start.
	if idx, ok := pw.trackedIndex(t); ok {
		uses := track.TextureUsesColorTarget
		if f.IsDepthStencil() {
			uses = track.TextureUsesDepthStencilWrite
		}
		pw.tracker.InsertTexture(idx, uses)
	}
	pw.trackDst(t, raw)
	return nil
}

// destroyViews destroys views the GPU is done with, or never got.
func (pw *pendingWrites) destroyViews(views []hal.TextureView) {
	for _, v := range views {
		pw.halDevice.DestroyTextureView(v)
	}
}

// zeroBuffer returns the zeroBand bytes of zeros that textures are zeroed
// from, made on first use and kept. Zeroing a texture by copies from it keeps
// a texture's worth of zeros out of the Go heap and out of the staging belt,
// which would keep a buffer that size mapped for good.
func (pw *pendingWrites) zeroBuffer() (hal.Buffer, error) {
	if pw.zeroBuf != nil {
		return pw.zeroBuf, nil
	}
	buf, err := pw.halDevice.CreateBuffer(&hal.BufferDescriptor{
		Label:            "(wgpu internal) zero buffer",
		Size:             zeroBand,
		Usage:            gputypes.BufferUsageCopySrc | gputypes.BufferUsageMapWrite,
		MappedAtCreation: true,
	})
	if err != nil {
		return nil, err
	}
	// Mapped memory comes with whatever was in it.
	if err := pw.halQueue.WriteBuffer(buf, 0, make([]byte, zeroBand)); err != nil {
		pw.halDevice.DestroyBuffer(buf)
		return nil, err
	}
	pw.zeroBuf = buf
	return buf, nil
}

// copyTextureDataAligned copies texture data with row pitch alignment padding.
// If no padding is needed (alignedBytesPerRow == bytesPerRow), returns data directly.
// Callers that can supply a reusable scratch buffer should use alignTextureDataInto
// to avoid per-call heap allocation.
func copyTextureDataAligned(data []byte, srcOffset uint64, bytesPerRow, alignedBytesPerRow, rowsPerImage, depthOrLayers uint32, stagingSize uint64) []byte {
	if alignedBytesPerRow == bytesPerRow {
		return dataFrom(data, srcOffset)
	}
	aligned := make([]byte, stagingSize)
	alignTextureRowsInto(aligned, data, srcOffset, bytesPerRow, alignedBytesPerRow, rowsPerImage, depthOrLayers)
	return aligned
}

// alignTextureDataInto is the zero-allocation variant of copyTextureDataAligned.
// If no padding is needed it returns data directly (no copy). Otherwise it grows
// *buf to stagingSize (never shrinks) and copies rows with padding into it.
func alignTextureDataInto(buf *[]byte, data []byte, srcOffset uint64, bytesPerRow, alignedBytesPerRow, rowsPerImage, depthOrLayers uint32, stagingSize uint64) []byte {
	if alignedBytesPerRow == bytesPerRow {
		return dataFrom(data, srcOffset)
	}
	if uint64(cap(*buf)) < stagingSize {
		*buf = make([]byte, stagingSize)
	} else {
		*buf = (*buf)[:stagingSize]
	}
	alignTextureRowsInto(*buf, data, srcOffset, bytesPerRow, alignedBytesPerRow, rowsPerImage, depthOrLayers)
	return *buf
}

// dataFrom is data from offset on, what a WriteTexture copies when its rows
// need no padding: the offset used to be dropped on that path.
func dataFrom(data []byte, offset uint64) []byte {
	if offset >= uint64(len(data)) {
		return nil
	}
	return data[offset:]
}

// alignTextureRowsInto does the inner row-copy loop shared by both aligned functions.
func alignTextureRowsInto(aligned, data []byte, srcOffset uint64, bytesPerRow, alignedBytesPerRow, rowsPerImage, depthOrLayers uint32) {
	for layer := uint32(0); layer < depthOrLayers; layer++ {
		for row := uint32(0); row < rowsPerImage; row++ {
			dstRowStart := uint64(layer)*uint64(rowsPerImage)*uint64(alignedBytesPerRow) +
				uint64(row)*uint64(alignedBytesPerRow)
			srcRowStart := srcOffset + uint64(layer)*uint64(rowsPerImage)*uint64(bytesPerRow) +
				uint64(row)*uint64(bytesPerRow)
			if srcRowStart >= uint64(len(data)) {
				break
			}
			// WebGPU's data may stop at the copy's width in the last row; it
			// used to be dropped whole.
			n := min(uint64(bytesPerRow), uint64(len(data))-srcRowStart)
			copy(aligned[dstRowStart:dstRowStart+n], data[srcRowStart:srcRowStart+n])
		}
	}
}

// alignUp rounds n up to the nearest multiple of alignment.
func alignUp(n, alignment uint32) uint32 {
	if alignment == 0 {
		return n
	}
	return (n + alignment - 1) / alignment * alignment
}

// bufferReadUsage extracts the read-state usage from a buffer's creation flags.
// Strips write/transfer usages (CopyDst, CopySrc, MapWrite, MapRead, Storage)
// to get the usage the buffer will be in during render (VERTEX, INDEX, UNIFORM).
// Returns 0 if no read usage is set (buffer only used for copies/storage).
func bufferReadUsage(usage gputypes.BufferUsage) gputypes.BufferUsage {
	// Keep only read-state flags relevant for DX12 transition barriers.
	// These map to DX12 read states: VERTEX_AND_CONSTANT_BUFFER, INDEX_BUFFER,
	// INDIRECT_ARGUMENT. Combined read states are valid in DX12.
	readMask := gputypes.BufferUsageVertex | gputypes.BufferUsageIndex |
		gputypes.BufferUsageUniform | gputypes.BufferUsageIndirect
	return usage & readMask
}

// writesTo reports whether the batch being recorded writes buffer.
func (pw *pendingWrites) writesTo(buffer hal.Buffer) bool {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	_, ok := pw.dstBuffers[buffer]
	return ok
}

// bufferReadUses is a bufferReadUsage in the device tracker's terms.
func bufferReadUses(usage gputypes.BufferUsage) track.BufferUses {
	var uses track.BufferUses
	for _, m := range [...]struct {
		usage gputypes.BufferUsage
		uses  track.BufferUses
	}{
		{gputypes.BufferUsageVertex, track.BufferUsesVertex},
		{gputypes.BufferUsageIndex, track.BufferUsesIndex},
		{gputypes.BufferUsageUniform, track.BufferUsesUniform},
		{gputypes.BufferUsageIndirect, track.BufferUsesIndirect},
	} {
		if usage&m.usage != 0 {
			uses |= m.uses
		}
	}
	return uses
}

// activate lazily begins encoding on the shared command encoder.
// Acquires an encoder from the pool if none exists. Must be called with pw.mu held.
func (pw *pendingWrites) activate() (hal.CommandEncoder, error) {
	if pw.isRecording {
		return pw.encoder, nil
	}

	if pw.encoder == nil {
		enc, err := pw.pool.acquire()
		if err != nil {
			return nil, fmt.Errorf("acquire encoder from pool: %w", err)
		}
		pw.encoder = enc
	}

	if err := pw.encoder.BeginEncoding("(wgpu internal) pending writes"); err != nil {
		return nil, fmt.Errorf("begin encoding: %w", err)
	}

	pw.isRecording = true
	return pw.encoder, nil
}

// flush closes the pending command encoder and returns a command buffer to
// prepend before user command buffers. Returns nil if no writes were recorded.
// The encoder is detached for inflight tracking — after GPU completion,
// maintain() calls ResetAll and returns it to the pool.
// Must be called with pw.mu held.
func (pw *pendingWrites) flush() (hal.CommandBuffer, hal.CommandEncoder, []hal.Texture, []hal.Buffer, error) {
	// Textures nothing wrote since they were made are zeroed now, in the
	// batch that runs ahead of the command buffers that could read them.
	for t := range pw.zero {
		if err := pw.zeroTextureLocked(t); err != nil {
			Logger().Warn("wgpu: texture keeps its old memory", "err", err)
		}
	}
	clear(pw.zero)

	if !pw.isRecording {
		return nil, nil, nil, nil, nil
	}

	// Queue writes are recorded lazily, so a surface acquisition can retire
	// between WriteTexture and Submit. Revalidate every contributing wrapper
	// before any command buffer reaches HAL. The raw texture comparison also
	// protects against a wrapper being rebound to a different resource. A
	// released texture of any other kind is held alive for the batch
	// (holdDestroy).
	for _, o := range pw.dstOwners {
		if o.owner.hal != o.raw || o.owner.surface != nil && o.owner.resolveHAL() != o.raw {
			pw.discardRecording()
			return nil, nil, nil, nil, fmt.Errorf("wgpu: pending writes: destination texture retired before submit: %w", ErrReleased)
		}
	}

	// Transition destination buffers from COPY_DEST to their primary read usage
	// before closing the encoder. Without this barrier, buffers remain in COPY_DEST
	// state when the render pass tries to use them as VERTEX/INDEX/UNIFORM —
	// undefined behavior per DX12 spec (BUG-DX12-010).
	//
	// Microsoft docs: "if promoted from COMMON to COPY_DEST, a barrier is still
	// required to transition from COPY_DEST to RENDER_TARGET."
	//
	// Usage comes from the wgpu public API level (core.Buffer.usage), matching
	// Rust wgpu-core where DeviceTracker computes barriers using core::Buffer.usage.
	// We extract read-only flags (VERTEX|INDEX|UNIFORM|INDIRECT) from the creation
	// usage to determine the target state.
	if len(pw.dstBuffers) > 0 {
		barriers := pw.barriers[:0]
		for buf, dst := range pw.dstBuffers {
			readUsage := bufferReadUsage(dst.usage)
			if readUsage != 0 {
				barriers = append(barriers, hal.BufferBarrier{
					Buffer: buf,
					Usage: hal.BufferUsageTransition{
						OldUsage: gputypes.BufferUsageCopyDst,
						NewUsage: readUsage,
					},
				})
				// The device tracker follows the buffer into the read state.
				if idx, ok := pw.trackedBufferIndex(dst.owner); ok {
					pw.tracker.InsertBuffer(idx, bufferReadUses(readUsage))
				}
			}
		}
		if len(barriers) > 0 {
			pw.encoder.TransitionBuffers(barriers)
		}
		clear(barriers)
		pw.barriers = barriers[:0]
	}

	cmdBuf, err := pw.encoder.EndEncoding()
	if err != nil {
		pw.discardRecording()
		return nil, nil, nil, nil, fmt.Errorf("wgpu: pending writes: end encoding: %w", err)
	}

	// Move resources out for inflight tracking.
	// dstTextures/dstBuffers hold references to prevent premature Release (BUG-DX12-006).
	flushedEncoder := pw.encoder

	// Finish the staging belt — moves active chunks to closed.
	// Belt is the sole owner of all staging resources (chunks + oversized).
	// recall() handles destruction after GPU completion — no double-destroy.
	if pw.belt != nil {
		pw.belt.finish(0) // submissionIndex set by caller via setLastSubmissionIndex
	}

	// Both stay nil when empty: Submit treats a nil flushedDstTextures as
	// nothing in flight.
	var flushedDstTextures []hal.Texture
	if len(pw.dstTextures) > 0 {
		flushedDstTextures = takeSpare(&pw.spareTexLists)
	}
	for tex := range pw.dstTextures {
		flushedDstTextures = append(flushedDstTextures, tex)
	}
	var flushedDstBuffers []hal.Buffer
	if len(pw.dstBuffers) > 0 {
		flushedDstBuffers = takeSpare(&pw.spareBufferLists)
	}
	for buf := range pw.dstBuffers {
		flushedDstBuffers = append(flushedDstBuffers, buf)
	}
	pw.isRecording = false
	pw.encoder = nil

	// PERF-PW-001: reuse map buckets instead of allocating new maps each flush.
	// clear() zeros values and deletes all keys but keeps allocated hash table
	// buckets, so subsequent inserts in the next batch don't trigger growth allocations.
	clear(pw.dstBuffers)
	clear(pw.dstTextures)
	pw.clearOwners()
	pw.flushedViews, pw.views = pw.views, nil
	pw.flushedDestroys, pw.destroys = pw.destroys, nil
	pw.flushedRefs, pw.refs = pw.refs, nil

	return cmdBuf, flushedEncoder, flushedDstTextures, flushedDstBuffers, nil
}

// clearOwners empties the owner list, keeping its room.
func (pw *pendingWrites) clearOwners() {
	clear(pw.dstOwners)
	pw.dstOwners = pw.dstOwners[:0]
}

// discardRecording abandons the current batch and restores every resource to
// its pre-batch ownership state. Must be called with pw.mu held.
func (pw *pendingWrites) discardRecording() {
	if pw.encoder != nil {
		if pw.isRecording {
			pw.encoder.DiscardEncoding()
		}
		pw.encoder.ResetAll(nil)
		if pw.pool != nil {
			pw.pool.release(pw.encoder)
		} else {
			pw.encoder.Destroy()
		}
	}

	pw.destroyViews(pw.views)
	pw.views = nil
	dropHeld(pw.destroys, pw.refs)
	pw.destroys, pw.refs = nil, nil
	for raw := range pw.dstTextures {
		raw.DecPendingRef()
	}
	if pw.belt != nil {
		pw.belt.discardActive()
	}

	pw.isRecording = false
	pw.encoder = nil
	clear(pw.dstBuffers)
	clear(pw.dstTextures)
	pw.clearOwners()
}

// cancelFlush releases a batch that was closed but rejected by HAL Submit.
// Must be called with pw.mu held.
func (pw *pendingWrites) cancelFlush(cmdBuf hal.CommandBuffer, encoder hal.CommandEncoder, dstTextures []hal.Texture) {
	if pw.belt != nil {
		pw.belt.discardLastClosed()
	}
	for _, texture := range dstTextures {
		texture.DecPendingRef()
	}
	if encoder != nil {
		encoder.ResetAll([]hal.CommandBuffer{cmdBuf})
		if pw.pool != nil {
			pw.pool.release(encoder)
		} else {
			encoder.Destroy()
		}
	} else if cmdBuf != nil {
		pw.halDevice.FreeCommandBuffer(cmdBuf)
	}
	pw.destroyViews(pw.flushedViews)
	pw.flushedViews = nil
	dropHeld(pw.flushedDestroys, pw.flushedRefs)
	pw.flushedDestroys, pw.flushedRefs = nil, nil
}

// maintain frees staging buffers and returns encoders to the pool from
// completed submissions. Must be called with pw.mu held.
func (pw *pendingWrites) maintain(completedIndex uint64) {
	// Recycle staging belt chunks from completed submissions.
	if pw.belt != nil {
		pw.belt.recall(completedIndex)
	}

	// Find the cutoff point — all submissions with index <= completedIndex are done.
	cutoff := 0
	for i := range pw.inflight {
		sub := &pw.inflight[i]
		if sub.submissionIndex > completedIndex {
			break
		}
		cutoff = i + 1
		// Destroy oversized staging buffers from completed submission.
		for _, buf := range sub.staging {
			pw.halDevice.DestroyBuffer(buf)
		}
		pw.destroyViews(sub.views)
		for _, h := range sub.destroys {
			h.now()
		}
		for _, r := range sub.refs {
			r.Drop()
		}
		// Release pending refs on destination textures/buffers (BUG-DX12-006).
		// This allows deferred Destroy to proceed now that GPU is done.
		for _, tex := range sub.dstTextures {
			tex.DecPendingRef()
		}
		// Reset the encoder and return it to the pool.
		if sub.encoder != nil && sub.cmdBuf != nil {
			pw.resetList[0] = sub.cmdBuf
			sub.encoder.ResetAll(pw.resetList[:])
			pw.resetList[0] = nil
			pw.pool.release(sub.encoder)
		} else if sub.cmdBuf != nil {
			pw.halDevice.FreeCommandBuffer(sub.cmdBuf)
		}
		keepSpare(&pw.spareTexLists, sub.dstTextures)
		keepSpare(&pw.spareBufferLists, sub.dstBuffers)
	}

	if cutoff > 0 {
		// Compact in place, so the list does not creep along its backing
		// array and reallocate every few submits.
		n := copy(pw.inflight, pw.inflight[cutoff:])
		clear(pw.inflight[n:])
		pw.inflight = pw.inflight[:n]
	}
}

// HasPendingWork returns true if there are buffered writes waiting to be flushed.
func (pw *pendingWrites) HasPendingWork() bool {
	pw.mu.Lock()
	defer pw.mu.Unlock()
	return pw.isRecording
}

// destroy releases all resources held by pendingWrites.
func (pw *pendingWrites) destroy() {
	pw.mu.Lock()
	defer pw.mu.Unlock()

	// Discard any in-progress encoding and balance its pending references.
	pw.discardRecording()

	// Destroy pending staging buffers.
	for _, buf := range pw.staging {
		pw.halDevice.DestroyBuffer(buf)
	}
	pw.staging = nil

	// Destroy inflight staging buffers, command buffers, and encoders.
	for i := range pw.inflight {
		sub := &pw.inflight[i]
		for _, buf := range sub.staging {
			pw.halDevice.DestroyBuffer(buf)
		}
		if sub.encoder != nil {
			// Encoder owns the command buffer's resources. Destroy releases both.
			sub.encoder.Destroy()
		} else if sub.cmdBuf != nil {
			pw.halDevice.FreeCommandBuffer(sub.cmdBuf)
		}
		pw.destroyViews(sub.views)
		for _, h := range sub.destroys {
			h.now()
		}
		for _, r := range sub.refs {
			r.Drop()
		}
		for _, tex := range sub.dstTextures {
			tex.DecPendingRef()
		}
	}
	pw.inflight = nil

	// Destroy the staging belt (releases all chunk buffers).
	if pw.belt != nil {
		pw.belt.destroy()
	}
	if pw.zeroBuf != nil {
		pw.halDevice.DestroyBuffer(pw.zeroBuf)
		pw.zeroBuf = nil
	}

	// Clear pool reference — pool is owned by Device, not PendingWrites.
	// Device.Release() destroys the shared pool after PendingWrites.destroy().
	pw.pool = nil

	pw.dstBuffers = nil
	pw.dstTextures = nil
	pw.dstOwners = nil
}
