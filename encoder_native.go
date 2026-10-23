//go:build !(js && wasm)

package wgpu

import (
	"fmt"
	"maps"
	"slices"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// CommandEncoder records GPU commands for later submission.
//
// A command encoder is single-use. After calling Finish(), the encoder
// cannot be used again. Call Device.CreateCommandEncoder() to create a new one.
//
// NOT thread-safe - do not use from multiple goroutines.
type CommandEncoder struct {
	core     *core.CoreCommandEncoder
	device   *Device
	released bool
	// serial identifies this encoder in the encoderMarks of what it tracks;
	// see id. A new one every time the object is reused, so a command buffer
	// handed out before the reuse can tell it is stale.
	serial uint64

	// coreEnc, pass, cpass and cb are the core encoder, the render pass, the
	// compute pass and the command buffer this encoder hands out: one of each,
	// so a frame allocates none of them. core points at coreEnc, except on the
	// pool-less path. pooled encoders go back to cmdEncoderObjPool once the
	// command buffer is spent, see recycle; after that every pointer handed
	// out for them is stale, as the API says they are after Finish and Submit.
	coreEnc core.CoreCommandEncoder
	pass    RenderPassEncoder
	cpass   ComputePassEncoder
	cb      CommandBuffer
	pooled  bool
	// trackedRefs accumulates Clone'd ResourceRefs from render/compute passes,
	// one per resource. Transferred to the CommandBuffer on Finish(), then to
	// the DestroyQueue on Submit(). Phase 2: per-command-buffer resource tracking.
	trackedRefs []*core.ResourceRef

	// halEncoder is the HAL command encoder acquired from the Device's pool.
	// On Finish(), ownership transfers to the CommandBuffer for post-GPU recycling.
	// On DiscardEncoding(), the encoder is reset and returned to the pool immediately.
	halEncoder hal.CommandEncoder

	// usedBuffers tracks root-level buffers referenced during encoding for
	// submit-time validation (VAL-A6). At Submit, each buffer is checked for
	// destroyed/mapped state. Each buffer is listed once (see encoderMark);
	// one a bind group already brought in is reached through the group.
	usedBuffers []*Buffer

	// usedTextures tracks root-level textures referenced during encoding for
	// submit-time validation (VAL-A6). At Submit, each texture is checked for
	// destroyed state.
	usedTextures []*Texture

	// passSerial identifies the render pass being recorded in the passMarks
	// of the bind groups it binds, so a group's textures and buffers go in
	// each pass's usage scope once. Zero outside a render pass: a compute
	// pass's groups go in per dispatch instead.
	passSerial uint64

	// copyReq, copyTex, copyBuf, the transitions and the barrier lists are
	// the scratch of one copy command's usage scope: the subresources each
	// region of the copy reads or writes, and its buffers.
	copyReq        []copyTextureUsage
	copyTex        []preparedCopyTextureUsage
	copyBuf        [2]preparedCopyBufferUsage
	texTransitions []track.TexturePendingTransition
	texBarriers    []hal.TextureBarrier
	bufBarriers    [2]hal.BufferBarrier

	// usedBindGroups tracks bind groups referenced during encoding for
	// submit-time validation (VAL-B5). At Submit, each bind group is checked
	// for destroyed state. Matches Rust wgpu-core's cmd_buf_data.trackers.bind_groups
	// (device/queue.rs:1815-1817).
	usedBindGroups []*BindGroup

	// usedQuerySets tracks the query sets ResolveQuerySet reads and render
	// passes write timestamps into, for submit-time validation: a released
	// one fails the submit.
	usedQuerySets []*QuerySet

	// clearsFirst are the textures still to be zeroed that a render pass of
	// this encoder clears whole before the encoder uses them any other way.
	clearsFirst []*Texture

	// discards are the textures whose contents a pass of the encoder has
	// discarded so far, which read as zeros until something sets them:
	// textures of one subresource the queue can zero.
	discards map[*Texture]struct{}
}

// encoderSerials hands out CommandEncoder serials. Zero is never handed out,
// so a zero encoderMark matches no encoder.
var encoderSerials atomic.Uint64

// cmdEncoderObjPool keeps spent CommandEncoder objects, with their core
// encoder, pass and command buffer inside, for the next CreateCommandEncoder.
var cmdEncoderObjPool = sync.Pool{New: func() any { return &CommandEncoder{} }}

// recycle puts a spent pooled encoder back for reuse. Its lists and refs
// have been moved out or dropped by then; the serial changes on reuse.
func (e *CommandEncoder) recycle() {
	if e == nil || !e.pooled {
		return
	}
	e.pooled = false
	e.core, e.device, e.halEncoder = nil, nil, nil
	e.trackedRefs, e.usedBuffers, e.usedTextures, e.usedBindGroups, e.usedQuerySets = nil, nil, nil, nil, nil
	e.clearsFirst = nil
	clear(e.discards)
	e.passSerial = 0
	// The command buffer handed out reads as spent, and as stale once the
	// object is reused, until the next Finish.
	e.cb = CommandBuffer{enc: e, serial: e.cb.serial, submitted: true}
	cmdEncoderObjPool.Put(e)
}

// encoderMark remembers the last command encoder that tracked a resource, so
// an encoder clones a resource's ref and lists it for Submit once, not on
// every SetPipeline or SetBindGroup a frame makes: fyne's painter binds the
// same few pipelines and bind groups hundreds of times a frame.
//
// Serials are never reused, so a mark equal to an encoder's serial means that
// encoder tracked the resource and holds it until its command buffer is
// spent. Encoders recording on other goroutines just overwrite the mark,
// which costs a repeated entry at most, never a missing one.
type encoderMark struct {
	last atomic.Uint64
}

// claim marks the resource as tracked by the encoder with this serial and
// reports whether it was not already.
func (m *encoderMark) claim(serial uint64) bool {
	if m.last.Load() == serial {
		return false
	}
	m.last.Store(serial)
	return true
}

// usedLists are the submit-time validation lists of one command buffer.
type usedLists struct {
	buffers    []*Buffer
	textures   []*Texture
	bindGroups []*BindGroup
	querySets  []*QuerySet
}

// usedListPool keeps the emptied validation lists of spent command buffers,
// so the next command encoder fills them instead of growing new ones.
type usedListPool struct {
	mu   sync.Mutex
	free []usedLists
}

// maxFreeUsedLists caps how many sets of lists a device keeps for reuse.
const maxFreeUsedLists = 32

func (p *usedListPool) get() usedLists {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := len(p.free)
	if n == 0 {
		return usedLists{}
	}
	l := p.free[n-1]
	p.free[n-1] = usedLists{}
	p.free = p.free[:n-1]
	return l
}

// put clears the lists, so they stop pinning resources, and keeps them.
func (p *usedListPool) put(l usedLists) {
	if cap(l.buffers)+cap(l.textures)+cap(l.bindGroups)+cap(l.querySets) == 0 {
		return
	}
	clear(l.buffers)
	clear(l.textures)
	clear(l.bindGroups)
	clear(l.querySets)
	l.buffers, l.textures, l.bindGroups, l.querySets = l.buffers[:0], l.textures[:0], l.bindGroups[:0], l.querySets[:0]
	p.mu.Lock()
	if len(p.free) < maxFreeUsedLists {
		p.free = append(p.free, l)
	}
	p.mu.Unlock()
}

// id returns the encoder's serial, handing one out on first use.
func (e *CommandEncoder) id() uint64 {
	if e.serial == 0 {
		e.serial = encoderSerials.Add(1)
	}
	return e.serial
}

// fallbackLimits are the limits of an encoder without a device, WebGPU's.
var fallbackLimits = gputypes.DefaultLimits()

// limits returns the limits of the encoder's device.
func (e *CommandEncoder) limits() *gputypes.Limits {
	if e.device != nil && e.device.core != nil {
		return &e.device.core.Limits
	}
	return &fallbackLimits
}

// setError records a deferred error on the underlying command encoder.
// This implements the WebGPU deferred error pattern: encoding-phase errors
// are collected and surfaced when Finish() is called.
func (e *CommandEncoder) setError(err error) {
	if e.core != nil {
		e.core.SetError(err)
	}
}

// hasDownlevel reports whether the encoder's adapter has flag. An encoder
// without a device has every one.
func (e *CommandEncoder) hasDownlevel(flag gputypes.DownlevelFlags) bool {
	return e.device == nil || e.device.core == nil || e.device.core.DownlevelCapabilitiesRef().Flags.Contains(flag)
}

// foreign records an error for what, an object made on owner, passed to op
// when owner is another device than e's, and reports whether it did.
func (e *CommandEncoder) foreign(op string, owner *Device, what string) bool {
	if !e.device.foreign(owner) {
		return false
	}
	e.setError(errForeign(op, what))
	return true
}

// keepAlive Clone()'s a ResourceRef and accumulates it for transfer to the
// CommandBuffer on Finish(). This keeps the resource alive until the GPU
// completes the submission. Callers go through an encoderMark first, so a
// resource is cloned once per encoder.
func (e *CommandEncoder) keepAlive(ref *core.ResourceRef) {
	if ref != nil {
		ref.Clone()
		e.trackedRefs = append(e.trackedRefs, ref)
	}
}

// trackPipeline keeps a render or compute pipeline alive until the GPU is
// done with this encoder's commands.
func (e *CommandEncoder) trackPipeline(mark *encoderMark, ref *core.ResourceRef) {
	if mark.claim(e.id()) {
		e.keepAlive(ref)
	}
}

// trackBuffer keeps a buffer alive until the GPU is done with this encoder's
// commands and records it for submit-time validation (VAL-A6).
func (e *CommandEncoder) trackBuffer(buf *Buffer) {
	if buf == nil || !buf.mark.claim(e.id()) {
		return
	}
	if buf.core != nil {
		e.keepAlive(buf.core.Ref)
	}
	e.usedBuffers = append(e.usedBuffers, buf)
}

// trackQuerySet records a query set for submit-time validation.
func (e *CommandEncoder) trackQuerySet(qs *QuerySet) {
	if qs == nil || !qs.mark.claim(e.id()) {
		return
	}
	e.usedQuerySets = append(e.usedQuerySets, qs)
}

// trackView keeps a texture view alive until the GPU is done with this
// encoder's commands and records its texture for submit-time validation.
func (e *CommandEncoder) trackView(v *TextureView) {
	if v == nil {
		return
	}
	if v.ref != nil && v.mark.claim(e.id()) {
		e.keepAlive(v.ref)
	}
	e.trackTexture(v.texture)
}

// trackTexture records a texture reference for submit-time validation (VAL-A6).
func (e *CommandEncoder) trackTexture(tex *Texture) {
	if tex == nil || !tex.mark.claim(e.id()) {
		return
	}
	e.usedTextures = append(e.usedTextures, tex)
}

// trackBindGroup keeps a bind group and the buffers it binds alive until the
// GPU is done with this encoder's commands, and records the group for
// submit-time validation (VAL-B5, and VAL-A6 through its boundBuffers and
// boundTextures). Matches Rust wgpu-core's cmd_buf_data.trackers.bind_groups
// (device/queue.rs:1815-1817) and merge_bind_group's
// ResourceMetadata.insert(Arc<Buffer>): without the buffer refs,
// Buffer.Release() before Submit could HAL-destroy a buffer a pending command
// buffer still binds. A bound buffer is not added to usedBuffers: Submit
// reaches it through the group, which stays listed.
func (e *CommandEncoder) trackBindGroup(bg *BindGroup) {
	if bg == nil {
		return
	}
	if bg.mark.claim(e.id()) {
		e.keepAlive(bg.ref)
		for _, buf := range bg.boundBuffers {
			if buf.core != nil && buf.mark.claim(e.id()) {
				e.keepAlive(buf.core.Ref)
			}
		}
		e.usedBindGroups = append(e.usedBindGroups, bg)
	}
}

// recordPassBindGroup puts the textures and buffers a bind group binds in
// the usage scope of the render pass being recorded, once per pass however
// often the pass binds the group, so that each is transitioned into the
// pass: from what an earlier command of this encoder left it in (a render
// target sampled after, a copy destination sampled after, a storage buffer
// a dispatch wrote), ahead of the pass, or from what the last submit left it
// in, at submit. Within the pass the usage has to agree with the resource's
// other uses in it: an attachment that is also sampled is a conflict.
//
// Reference: wgpu-core command/render.rs set_bind_group (merge_bind_group
// into the pass scope)
func (e *CommandEncoder) recordPassBindGroup(bg *BindGroup) {
	if bg == nil || e.passSerial == 0 || !bg.passMark.claim(e.passSerial) {
		return
	}
	e.recordBindGroupUsages(bg, "SetBindGroup")
}

// recordDispatchBindGroup is recordPassBindGroup for a compute dispatch:
// each dispatch is a usage scope of its own, so the groups it runs with go
// in every time, and a group bound and replaced before the dispatch never
// does.
//
// Reference: wgpu-core command/compute.rs State::flush_states
func (e *CommandEncoder) recordDispatchBindGroup(bg *BindGroup, op string) {
	if bg == nil {
		return
	}
	e.recordBindGroupUsages(bg, op)
}

// recordBindGroupUsages records a group's textures and buffers in the usage
// scope being recorded; op names the command for the conflict error.
func (e *CommandEncoder) recordBindGroupUsages(bg *BindGroup, op string) {
	if e.core == nil {
		return
	}
	for i, tex := range bg.boundTextures {
		if i < len(bg.boundTextureUses) {
			var sel track.TextureSelector
			if i < len(bg.boundTextureRanges) {
				sel = bg.boundTextureRanges[i]
			}
			e.recordBoundTextureUsage(tex, sel, bg.boundTextureUses[i], op)
		}
	}
	for i, buf := range bg.boundBuffers {
		if i < len(bg.boundBufferUses) {
			e.recordBoundBufferUsage(buf, bg.boundBufferUses[i], op)
		}
	}
}

// recordBoundTextureUsage puts the usage of the subresources sel selects of
// a bound texture in the usage scope being recorded.
func (e *CommandEncoder) recordBoundTextureUsage(tex *Texture, sel track.TextureSelector, usage track.TextureUses, op string) {
	if tex == nil || tex.coreTexture == nil {
		return
	}
	if err := e.core.RecordPassTextureUsage(tex.coreTexture, tex.resolveHAL(), sel, usage); err != nil {
		e.setError(fmt.Errorf("wgpu: %s: texture usage conflict: %w", op, err))
	}
}

// recordBoundBufferUsage puts a bound buffer's usage in the usage scope
// being recorded.
func (e *CommandEncoder) recordBoundBufferUsage(buf *Buffer, usage track.BufferUses, op string) {
	if buf == nil || buf.core == nil {
		return
	}
	if err := e.core.RecordPassBufferUsage(buf.core, buf.halBuffer(), usage); err != nil {
		e.setError(fmt.Errorf("wgpu: %s: buffer usage conflict: %w", op, err))
	}
}

// copyTextureUsage describes the subresources of a texture one region of a
// copy command reads or writes.
type copyTextureUsage struct {
	texture *Texture
	sel     track.TextureSelector
	usage   track.TextureUses
}

type preparedCopyTextureUsage struct {
	texture *Texture
	index   track.TrackerIndex
	full    track.TextureSelector
	sel     track.TextureSelector
	usage   track.TextureUses
}

// copyBufferUsage describes one buffer endpoint of a copy command.
type copyBufferUsage struct {
	buffer *Buffer
	usage  track.BufferUses
}

type preparedCopyBufferUsage struct {
	buffer *Buffer
	index  track.TrackerIndex
	usage  track.BufferUses
}

// recordCopyUsages records the resource usages of ONE copy command.
//
// A copy command is its own WebGPU usage scope. Its endpoints are validated
// against each other only (a texture or buffer copied onto itself); usages
// by earlier commands of this encoder, a render pass that drew into the
// source, an upload into a texture sampled later, belong to other usage
// scopes and are never conflicts. After validation each usage is applied to
// the command buffer tracker, and the transitions from the previous usages
// are recorded on raw before the copy.
//
// Validation finishes before any state changes, so a rejected copy leaves
// the tracker, the HAL encoder and the submit validation sets untouched.
//
// Reference: wgpu-core command/transfer.rs (tracker.textures.set_single /
// tracker.buffers.set_single, then transition_textures/buffers before the copy)
func (e *CommandEncoder) recordCopyUsages(raw hal.CommandEncoder, textures []copyTextureUsage, buffers []copyBufferUsage) bool {
	if e.core == nil {
		return true
	}
	defer e.clearCopyScratch()
	preparedTextures, err := prepareCopyTextureUsagesInto(e.copyTex[:0], textures)
	e.copyTex = preparedTextures
	if err != nil {
		e.setError(fmt.Errorf("wgpu: texture usage conflict: %w", err))
		return false
	}
	preparedBuffers, err := prepareCopyBufferUsagesInto(e.copyBuf[:0], buffers)
	if err != nil {
		e.setError(fmt.Errorf("wgpu: buffer usage conflict: %w", err))
		return false
	}
	e.transitionCopyTextures(raw, preparedTextures)
	e.transitionCopyBuffers(raw, preparedBuffers)
	return true
}

// clearCopyScratch empties the scratch of a copy command, so it pins no
// texture or buffer until the next one.
func (e *CommandEncoder) clearCopyScratch() {
	clear(e.copyReq)
	clear(e.copyTex)
	clear(e.copyBuf[:])
	e.copyReq, e.copyTex = e.copyReq[:0], e.copyTex[:0]
}

// copyRegionUsage adds to the copy scratch the use of the subresources of t
// a copy region at base of size reads or writes.
func (e *CommandEncoder) copyRegionUsage(t *Texture, base hal.ImageCopyTexture, size hal.Extent3D, usage track.TextureUses) {
	e.copyReq = append(e.copyReq, copyTextureUsage{texture: t, sel: t.copySubresources(&base, &size), usage: usage})
}

// transitionCopyTextures applies a copy command's texture usages to the
// command buffer tracker and records the needed barriers on raw.
func (e *CommandEncoder) transitionCopyTextures(raw hal.CommandEncoder, requests []preparedCopyTextureUsage) {
	tracker := e.core.Mutable().TextureScope()
	if tracker == nil {
		return
	}
	transitions, barriers := e.texTransitions[:0], e.texBarriers[:0]
	for _, request := range requests {
		n := len(transitions)
		transitions = tracker.SetSubresourceUsage(request.index, request.full, request.sel, request.usage, transitions)
		if halTexture := request.texture.resolveHAL(); halTexture != nil {
			for _, transition := range transitions[n:] {
				barriers = append(barriers, transition.IntoHAL(halTexture))
			}
		}
	}
	if len(barriers) > 0 && raw != nil {
		raw.TransitionTextures(barriers)
	}
	clear(barriers)
	e.texTransitions, e.texBarriers = transitions[:0], barriers[:0]
}

// transitionCopyBuffers applies a copy command's buffer usages to the command
// buffer tracker and records the needed barriers on raw.
func (e *CommandEncoder) transitionCopyBuffers(raw hal.CommandEncoder, requests []preparedCopyBufferUsage) {
	tracker := e.core.Mutable().BufferScope()
	if tracker == nil {
		return
	}
	barriers := e.bufBarriers[:0]
	for _, request := range requests {
		transition, needed := tracker.SetSingle(request.index, request.usage)
		if !needed {
			continue
		}
		if halBuffer := request.buffer.halBuffer(); halBuffer != nil {
			barriers = append(barriers, transition.IntoHAL(halBuffer))
		}
	}
	if len(barriers) > 0 && raw != nil {
		raw.TransitionBuffers(barriers)
	}
	clear(e.bufBarriers[:])
}

// prepareCopyBufferUsages builds the buffer part of one copy command's usage
// scope: multiple roles of the same buffer are merged, and incompatible roles
// (a buffer as both copy source and destination) are rejected.
//
// Reference: wgpu-core command/transfer.rs SameSourceDestinationBuffer
func prepareCopyBufferUsages(requests []copyBufferUsage) ([]preparedCopyBufferUsage, error) {
	return prepareCopyBufferUsagesInto(nil, requests)
}

// prepareCopyBufferUsagesInto is prepareCopyBufferUsages appending to prepared.
func prepareCopyBufferUsagesInto(prepared []preparedCopyBufferUsage, requests []copyBufferUsage) ([]preparedCopyBufferUsage, error) {
	for _, request := range requests {
		if request.buffer == nil || request.buffer.core == nil {
			continue
		}
		td := request.buffer.core.TrackingData()
		if td == nil || !td.Index().IsValid() {
			continue
		}
		index := td.Index()
		position := -1
		for i := range prepared {
			if prepared[i].index == index {
				position = i
				break
			}
		}
		if position < 0 {
			prepared = append(prepared, preparedCopyBufferUsage{
				buffer: request.buffer, index: index, usage: request.usage,
			})
			continue
		}
		existing := prepared[position].usage
		if !existing.IsCompatible(request.usage) {
			return nil, &track.UsageConflictError{Index: index, Existing: existing, New: request.usage}
		}
		prepared[position].usage = existing | request.usage
	}
	return prepared, nil
}

// prepareCopyTextureUsages builds the texture part of one copy command's
// usage scope, the subresources each of its regions reads or writes. Roles
// of the same subresources are merged; a subresource used in incompatible
// roles (as both copy source and destination, a copy within one
// subresource) is rejected, a copy between two subresources of one
// texture is not.
//
// Reference: wgpu-core command/transfer.rs validate_copy_within_same_texture
func prepareCopyTextureUsages(requests []copyTextureUsage) ([]preparedCopyTextureUsage, error) {
	return prepareCopyTextureUsagesInto(nil, requests)
}

// prepareCopyTextureUsagesInto is prepareCopyTextureUsages appending to prepared.
func prepareCopyTextureUsagesInto(prepared []preparedCopyTextureUsage, requests []copyTextureUsage) ([]preparedCopyTextureUsage, error) {
	for _, request := range requests {
		if request.texture == nil || request.texture.coreTexture == nil {
			continue
		}
		td := request.texture.coreTexture.TrackingData()
		if td == nil || !td.Index().IsValid() {
			continue
		}
		index := td.Index()
		merged := false
		for i, p := range prepared {
			if p.index != index || !p.sel.Overlaps(request.sel) {
				continue
			}
			combined := p.usage | request.usage
			if !combined.IsCompatible(combined) {
				return prepared, &track.TextureUsageConflictError{
					Index: index, Selector: request.sel, Existing: p.usage, New: request.usage,
				}
			}
			if p.sel == request.sel {
				prepared[i].usage, merged = combined, true
			}
		}
		if !merged {
			prepared = append(prepared, preparedCopyTextureUsage{
				texture: request.texture, index: index, full: request.texture.coreTexture.Subresources(),
				sel: request.sel, usage: request.usage,
			})
		}
	}
	return prepared, nil
}

// BeginRenderPass begins a render pass.
// The returned RenderPassEncoder records draw commands.
// Call RenderPassEncoder.End() when done.
func (e *CommandEncoder) BeginRenderPass(desc *RenderPassDescriptor) (*RenderPassEncoder, error) {
	if e.released {
		return nil, ErrReleased
	}
	if err := validateRenderPassTextureViews(desc); err != nil {
		return nil, err
	}
	if desc != nil {
		if err := validateTimestampWrites(desc.TimestampWrites); err != nil {
			e.setError(err)
			return nil, err
		}
		if e.attachmentsForeign(desc) {
			return nil, errForeign("BeginRenderPass", "an attachment or the timestamp query set")
		}
	}
	e.noteClears(desc)
	trackRenderPassTextureViews(e, desc)

	// Core reads the descriptor and keeps none of it, so it comes from a pool
	// and goes straight back.
	s := passDescPool.Get().(*corePassDescriptor)
	coreDesc := s.fill(desc)
	e.clearDiscardedLoads(desc, coreDesc)
	corePass, err := e.core.BeginRenderPass(coreDesc)
	s.release()
	if err != nil {
		return nil, err
	}
	e.noteDiscards(desc)

	// The pass is a usage scope of its own: the bind groups it binds put
	// their textures in it once each.
	e.passSerial = encoderSerials.Add(1)
	e.pass.reset(corePass, e)
	return &e.pass, nil
}

// noteClears adds to clearsFirst the textures still to be zeroed that the
// pass of desc clears whole, a color attachment or every aspect of a
// depth/stencil one, when the encoder hasn't used them before. Submit
// takes them off the zero list when the command buffer is the first of the
// submit to use them, so a clear decides nothing for a command buffer that
// is dropped or runs after another that reads the texture. A clear whose
// store op discards leaves the texture to noteDiscards.
//
// Reference: wgpu-core command/render.rs (Clear with Store is an implicit
// init) and device/queue.rs (init actions applied in submission order)
func (e *CommandEncoder) noteClears(desc *RenderPassDescriptor) {
	d := e.device
	if desc == nil || d == nil || d.queue == nil || d.queue.pending == nil {
		return
	}
	note := func(t *Texture) {
		ct := t.coreTexture
		if ct == nil || ct.MipLevelCount() > 1 || ct.Size().DepthOrArrayLayers > 1 || t.mark.last.Load() == e.id() ||
			!d.queue.pending.zeroing(t) {
			return
		}
		for _, g := range e.usedBindGroups {
			if slices.Contains(g.boundTextures, t) {
				return
			}
		}
		e.clearsFirst = append(e.clearsFirst, t)
	}
	for _, ca := range desc.ColorAttachments {
		if ca.LoadOp == gputypes.LoadOpClear && ca.View != nil && ca.View.texture != nil {
			note(ca.View.texture)
		}
	}
	if ds := desc.DepthStencilAttachment; ds != nil && ds.View != nil && ds.View.texture != nil {
		f := ds.View.texture.format
		if (!f.HasDepth() || ds.DepthLoadOp == gputypes.LoadOpClear) && (!f.HasStencil() || ds.StencilLoadOp == gputypes.LoadOpClear) {
			note(ds.View.texture)
		}
	}
}

// clearDiscardedLoads turns a load of an attachment whose contents an
// earlier pass of the encoder discarded into a clear to zero in coreDesc,
// the core descriptor of desc: a discarded texture reads as zeros.
//
// Reference: Dawn CommandBuffer.cpp LazyClearRenderPassAttachments
func (e *CommandEncoder) clearDiscardedLoads(desc *RenderPassDescriptor, coreDesc *core.RenderPassDescriptor) {
	if desc == nil || len(e.discards) == 0 {
		return
	}
	discarded := func(v *TextureView) bool {
		if v == nil {
			return false
		}
		_, ok := e.discards[v.texture]
		return ok
	}
	for i, ca := range desc.ColorAttachments {
		if ca.LoadOp == gputypes.LoadOpLoad && discarded(ca.View) {
			coreDesc.ColorAttachments[i].LoadOp = gputypes.LoadOpClear
			coreDesc.ColorAttachments[i].ClearValue = gputypes.Color{}
		}
	}
	if ds := desc.DepthStencilAttachment; ds != nil && discarded(ds.View) {
		cds := coreDesc.DepthStencilAttachment
		if ds.DepthLoadOp == gputypes.LoadOpLoad {
			cds.DepthLoadOp, cds.DepthClearValue = gputypes.LoadOpClear, 0
		}
		if ds.StencilLoadOp == gputypes.LoadOpLoad {
			cds.StencilLoadOp, cds.StencilClearValue = gputypes.LoadOpClear, 0
		}
	}
}

// noteDiscards records which attachments of desc's pass end with their
// contents discarded, every aspect they have, and which with them set: a
// texture of one subresource the queue can zero. What is still discarded
// when the command buffer runs goes back on the zero list after its Submit.
//
// Reference: wgpu-core command/render.rs (StoreOp::Discard registers a
// discard), Dawn CommandBuffer.cpp (SetIsSubresourceContentInitialized
// false on discard)
func (e *CommandEncoder) noteDiscards(desc *RenderPassDescriptor) {
	if desc == nil {
		return
	}
	note := func(v *TextureView, discarded bool) {
		if v == nil || v.texture == nil || !v.texture.zeroable {
			return
		}
		t := v.texture
		if ct := t.coreTexture; ct == nil || ct.MipLevelCount() > 1 || ct.Size().DepthOrArrayLayers > 1 {
			return
		}
		if !discarded {
			delete(e.discards, t)
			return
		}
		if e.discards == nil {
			e.discards = make(map[*Texture]struct{})
		}
		e.discards[t] = struct{}{}
	}
	for _, ca := range desc.ColorAttachments {
		note(ca.View, ca.StoreOp == gputypes.StoreOpDiscard)
		note(ca.ResolveTarget, false)
	}
	if ds := desc.DepthStencilAttachment; ds != nil && ds.View != nil && ds.View.texture != nil {
		f := ds.View.texture.format
		note(ds.View, (!f.HasDepth() || ds.DepthStoreOp == gputypes.StoreOpDiscard) &&
			(!f.HasStencil() || ds.StencilStoreOp == gputypes.StoreOpDiscard))
	}
}

// zeroDiscarded clears t when a pass of the encoder discarded its contents,
// before a copy reads it or writes part of it.
func (e *CommandEncoder) zeroDiscarded(t *Texture) {
	if _, ok := e.discards[t]; !ok {
		return
	}
	delete(e.discards, t)
	view, err := e.device.CreateTextureView(t, nil)
	if err != nil {
		e.setError(err)
		return
	}
	defer view.Release()
	desc := &RenderPassDescriptor{}
	if f := t.format; f.IsDepthStencil() {
		ds := &RenderPassDepthStencilAttachment{View: view}
		if f.HasDepth() {
			ds.DepthLoadOp, ds.DepthStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
		}
		if f.HasStencil() {
			ds.StencilLoadOp, ds.StencilStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
		}
		desc.DepthStencilAttachment = ds
	} else {
		desc.ColorAttachments = []RenderPassColorAttachment{{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}}
	}
	if pass, err := e.BeginRenderPass(desc); err == nil {
		_ = pass.End()
	}
}

// BeginComputePass begins a compute pass.
// The returned ComputePassEncoder records dispatch commands.
// Call ComputePassEncoder.End() when done.
func (e *CommandEncoder) BeginComputePass(desc *ComputePassDescriptor) (*ComputePassEncoder, error) {
	if e.released {
		return nil, ErrReleased
	}

	var coreDesc *core.CoreComputePassDescriptor
	if desc != nil {
		coreDesc = &core.CoreComputePassDescriptor{Label: desc.Label}
	}

	corePass, err := e.core.BeginComputePass(coreDesc)
	if err != nil {
		return nil, err
	}

	// No pass serial: a compute pass's bind groups enter a usage scope per
	// dispatch (ComputePassEncoder.recordDispatch), not once per pass.
	e.passSerial = 0
	e.cpass.reset(corePass, e)
	return &e.cpass, nil
}

// CopyBufferToBuffer copies data between buffers.
func (e *CommandEncoder) CopyBufferToBuffer(src *Buffer, srcOffset uint64, dst *Buffer, dstOffset uint64, size uint64) {
	if e.released {
		return
	}
	if src == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToBuffer: source buffer is nil"))
		return
	}
	if dst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToBuffer: destination buffer is nil"))
		return
	}
	if e.foreign("CommandEncoder.CopyBufferToBuffer", src.device, "the source buffer") ||
		e.foreign("CommandEncoder.CopyBufferToBuffer", dst.device, "the destination buffer") {
		return
	}
	raw := e.core.RecordingEncoder()
	if raw == nil {
		return
	}
	halSrc := src.halBuffer()
	halDst := dst.halBuffer()
	if halSrc == nil || halDst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToBuffer: source or destination buffer is released: %w", ErrReleased))
		return
	}
	// WebGPU's copyBufferToBuffer checks, which a backend would otherwise
	// get as a copy its API calls undefined.
	switch {
	case src.Usage()&BufferUsageCopySrc == 0:
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToBuffer: source buffer missing CopySrc usage"))
		return
	case dst.Usage()&BufferUsageCopyDst == 0:
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToBuffer: destination buffer missing CopyDst usage"))
		return
	case srcOffset%4 != 0 || dstOffset%4 != 0 || size%4 != 0:
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToBuffer: offsets %d and %d and size %d must be multiples of 4", srcOffset, dstOffset, size))
		return
	case srcOffset > src.Size() || size > src.Size()-srcOffset:
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToBuffer: source offset %d + size %d exceeds buffer size %d", srcOffset, size, src.Size()))
		return
	case dstOffset > dst.Size() || size > dst.Size()-dstOffset:
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToBuffer: destination offset %d + size %d exceeds buffer size %d", dstOffset, size, dst.Size()))
		return
	case size == 0:
		return // nothing to copy, and Vulkan takes no empty copy
	}
	if !e.recordCopyUsages(raw, nil, []copyBufferUsage{
		{buffer: src, usage: track.BufferUsesCopySrc},
		{buffer: dst, usage: track.BufferUsesCopyDst},
	}) {
		return
	}
	e.trackBuffer(src)
	e.trackBuffer(dst)
	raw.CopyBufferToBuffer(halSrc, halDst, []hal.BufferCopy{
		{SrcOffset: srcOffset, DstOffset: dstOffset, Size: size},
	})
}

// CopyTextureToBuffer copies data from a texture to a buffer.
// This is used for GPU-to-CPU readback of rendered content.
func (e *CommandEncoder) CopyTextureToBuffer(src *Texture, dst *Buffer, regions []BufferTextureCopy) {
	if e.released {
		return
	}
	if src == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToBuffer: source texture is nil"))
		return
	}
	if dst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToBuffer: destination buffer is nil"))
		return
	}
	if e.foreign("CommandEncoder.CopyTextureToBuffer", src.device, "the source texture") ||
		e.foreign("CommandEncoder.CopyTextureToBuffer", dst.device, "the destination buffer") {
		return
	}
	halSrc := src.resolveHAL()
	if halSrc == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToBuffer: source texture is released: %w", ErrReleased))
		return
	}
	for _, region := range regions {
		if region.TextureBase.Texture != nil && region.TextureBase.Texture.resolveHAL() == nil {
			e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToBuffer: region texture is released: %w", ErrReleased))
			return
		}
	}
	halDst := dst.halBuffer()
	if halDst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToBuffer: destination buffer is released: %w", ErrReleased))
		return
	}
	if err := checkCopyTextureUsage("CopyTextureToBuffer", "source", src, gputypes.TextureUsageCopySrc); err != nil {
		e.setError(err)
		return
	}
	if dst.Usage()&BufferUsageCopyDst == 0 {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToBuffer: destination buffer missing CopyDst usage"))
		return
	}
	if err := checkCopyRowPitch("CopyTextureToBuffer", regions); err != nil {
		e.setError(err)
		return
	}
	halRegions, err := bufferTextureCopies("CopyTextureToBuffer", src, halSrc, dst, regions, false)
	if err != nil {
		e.setError(err)
		return
	}
	e.zeroDiscarded(src)
	raw := e.core.RecordingEncoder()
	if raw == nil {
		return
	}
	for _, r := range halRegions {
		e.copyRegionUsage(src, r.TextureBase, r.Size, track.TextureUsesCopySrc)
	}
	if !e.recordCopyUsages(raw, e.copyReq,
		[]copyBufferUsage{{buffer: dst, usage: track.BufferUsesCopyDst}},
	) {
		return
	}
	for _, region := range regions {
		e.trackTexture(region.TextureBase.Texture)
	}
	e.trackTexture(src)
	e.trackBuffer(dst)
	if len(halRegions) > 0 {
		raw.CopyTextureToBuffer(halSrc, halDst, halRegions)
	}
}

// CopyTextureToTexture copies data between textures using DMA hardware copy.
// WebGPU spec: GPUCommandEncoder.copyTextureToTexture()
func (e *CommandEncoder) CopyTextureToTexture(src, dst *Texture, regions []TextureCopy) {
	if e.released {
		return
	}
	if src == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToTexture: source texture is nil"))
		return
	}
	if dst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToTexture: destination texture is nil"))
		return
	}
	if e.foreign("CommandEncoder.CopyTextureToTexture", src.device, "the source texture") ||
		e.foreign("CommandEncoder.CopyTextureToTexture", dst.device, "the destination texture") {
		return
	}
	halSrc := src.resolveHAL()
	halDst := dst.resolveHAL()
	if halSrc == nil || halDst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToTexture: texture is released: %w", ErrReleased))
		return
	}
	for _, region := range regions {
		if (region.Source.Texture != nil && region.Source.Texture.resolveHAL() == nil) ||
			(region.Destination.Texture != nil && region.Destination.Texture.resolveHAL() == nil) {
			e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToTexture: region texture is released: %w", ErrReleased))
			return
		}
	}
	if err := checkCopyTextureUsage("CopyTextureToTexture", "source", src, gputypes.TextureUsageCopySrc); err != nil {
		e.setError(err)
		return
	}
	if err := checkCopyTextureUsage("CopyTextureToTexture", "destination", dst, gputypes.TextureUsageCopyDst); err != nil {
		e.setError(err)
		return
	}
	halRegions := make([]hal.TextureCopy, len(regions))
	for i := range regions {
		r := &regions[i]
		halRegions[i] = r.toHAL(halSrc, halDst)
		if src.coreTexture == nil || dst.coreTexture == nil {
			continue
		}
		size, err := core.ValidateTextureToTextureCopy(texelCopy(src, &r.Source), texelCopy(dst, &r.Destination), gputypes.Extent3D(r.Size))
		if err != nil {
			e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyTextureToTexture: region %d: %w", i, err))
			return
		}
		halRegions[i].Size = hal.Extent3D(size)
		halRegions[i].SrcBase.Aspect = core.CopyAspect(src.coreTexture.Format(), r.Source.Aspect)
		halRegions[i].DstBase.Aspect = core.CopyAspect(dst.coreTexture.Format(), r.Destination.Aspect)
	}
	// A region with nothing in it is valid and copies nothing, which no
	// backend takes.
	halRegions = slices.DeleteFunc(halRegions, func(r hal.TextureCopy) bool { return emptyExtent(r.Size) })
	e.zeroDiscarded(src)
	e.zeroDiscarded(dst)
	raw := e.core.RecordingEncoder()
	if raw == nil {
		return
	}
	for _, r := range halRegions {
		e.copyRegionUsage(src, r.SrcBase, r.Size, track.TextureUsesCopySrc)
		e.copyRegionUsage(dst, r.DstBase, r.Size, track.TextureUsesCopyDst)
	}
	if !e.recordCopyUsages(raw, e.copyReq, nil) {
		return
	}
	for _, region := range regions {
		e.trackTexture(region.Source.Texture)
		e.trackTexture(region.Destination.Texture)
	}
	e.trackTexture(src)
	e.trackTexture(dst)
	if len(halRegions) > 0 {
		raw.CopyTextureToTexture(halSrc, halDst, halRegions)
	}
}

// TransitionTextures records explicit texture barriers.
//
// Copy commands and render passes already transition their textures from
// the state left by earlier commands of the same encoder, so explicit
// barriers are only needed for states the encoder cannot see. Each explicit
// barrier also updates the encoder's tracker: later commands transition from
// the barrier's NewUsage instead of repeating it. On Metal, GLES, and
// software backends the barrier itself is a no-op.
func (e *CommandEncoder) TransitionTextures(barriers []TextureBarrier) {
	if e.released {
		return
	}
	raw := e.core.RecordingEncoder()
	if raw == nil {
		return
	}
	halBarriers := make([]hal.TextureBarrier, 0, len(barriers))
	for _, b := range barriers {
		if b.Texture != nil && b.Texture.resolveHAL() == nil {
			e.setError(fmt.Errorf("wgpu: CommandEncoder.TransitionTextures: texture is released: %w", ErrReleased))
			return
		}
		if b.Texture == nil {
			continue
		}
		e.trackTexture(b.Texture)
		halBarriers = append(halBarriers, b.toHAL())
	}
	if len(halBarriers) == 0 {
		return
	}
	raw.TransitionTextures(halBarriers)
	for _, b := range barriers {
		if b.Texture != nil {
			e.core.RecordTextureTransition(b.Texture.coreTexture, hal.TextureRange(b.Range), b.Usage.OldUsage, b.Usage.NewUsage)
		}
	}
}

// CopyBufferToTexture copies data from a buffer to a texture.
// WebGPU spec: GPUCommandEncoder.copyBufferToTexture.
// copyBytesPerRowAlignment is what WebGPU asks of bytesPerRow on the buffer
// side of a copy between a buffer and a texture in a command encoder.
const copyBytesPerRowAlignment = 256

// checkCopyTextureUsage is the usage check of a copy: the texture was made,
// or the surface configured, with the usage the copy needs. A texture
// without core state (a test stub) is not checked.
//
// Rust: texture.check_usage(COPY_SRC / COPY_DST) in command/transfer.rs
func checkCopyTextureUsage(op, role string, t *Texture, usage gputypes.TextureUsage) error {
	if t == nil || t.coreTexture == nil || t.coreTexture.Usage()&usage != 0 {
		return nil
	}
	name := "CopyDst"
	if usage == gputypes.TextureUsageCopySrc {
		name = "CopySrc"
	}
	return fmt.Errorf("wgpu: CommandEncoder.%s: %s texture is missing %s usage", op, role, name)
}

// checkCopyRowPitch rejects a bytesPerRow that isn't a multiple of 256. DX12
// can't copy any other pitch and used to drop such a copy without an error,
// while the other backends took it.
func checkCopyRowPitch(op string, regions []BufferTextureCopy) error {
	for _, r := range regions {
		if bpr := r.BufferLayout.BytesPerRow; bpr%copyBytesPerRowAlignment != 0 {
			return fmt.Errorf("wgpu: CommandEncoder.%s: bytesPerRow %d is not a multiple of %d", op, bpr, copyBytesPerRowAlignment)
		}
	}
	return nil
}

// bufferTextureCopies checks each region of a copy between t and buf and
// returns the regions for the backend, on halTex, sized to the texels the
// texture has, without the regions with nothing in them, which are valid
// and copy nothing but no backend takes. A texture without core state (a
// test stub) is not checked.
func bufferTextureCopies(op string, t *Texture, halTex hal.Texture, buf *Buffer, regions []BufferTextureCopy, toTexture bool) ([]hal.BufferTextureCopy, error) {
	halRegions := make([]hal.BufferTextureCopy, len(regions))
	for i := range regions {
		r := &regions[i]
		halRegions[i] = r.toHAL(halTex)
		if t.coreTexture == nil {
			continue
		}
		size, err := core.ValidateTextureBufferCopy(texelCopy(t, &r.TextureBase), gputypes.Extent3D(r.Size), r.BufferLayout.toHAL(), buf.Size(), toTexture, true)
		if err != nil {
			return nil, fmt.Errorf("wgpu: CommandEncoder.%s: region %d: %w", op, i, err)
		}
		halRegions[i].Size = hal.Extent3D(size)
		halRegions[i].TextureBase.Aspect = core.CopyAspect(t.coreTexture.Format(), r.TextureBase.Aspect)
	}
	return slices.DeleteFunc(halRegions, func(r hal.BufferTextureCopy) bool { return emptyExtent(r.Size) }), nil
}

func emptyExtent(e hal.Extent3D) bool {
	return e.Width == 0 || e.Height == 0 || e.DepthOrArrayLayers == 0
}

// texelCopy is the core side of a copy of base on t, whatever base.Texture
// holds.
func texelCopy(t *Texture, base *ImageCopyTexture) *core.TexelCopyTexture {
	return &core.TexelCopyTexture{Texture: t.coreTexture, MipLevel: base.MipLevel, Origin: gputypes.Origin3D(base.Origin), Aspect: base.Aspect}
}

func (e *CommandEncoder) CopyBufferToTexture(src *Buffer, dst *Texture, regions []BufferTextureCopy) {
	if e.released {
		return
	}
	if src == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToTexture: source buffer is nil"))
		return
	}
	if dst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToTexture: destination texture is nil"))
		return
	}
	if e.foreign("CommandEncoder.CopyBufferToTexture", src.device, "the source buffer") ||
		e.foreign("CommandEncoder.CopyBufferToTexture", dst.device, "the destination texture") {
		return
	}
	halDst := dst.resolveHAL()
	if halDst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToTexture: destination texture is released: %w", ErrReleased))
		return
	}
	halSrc := src.halBuffer()
	if halSrc == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToTexture: source buffer is released: %w", ErrReleased))
		return
	}
	if err := checkCopyTextureUsage("CopyBufferToTexture", "destination", dst, gputypes.TextureUsageCopyDst); err != nil {
		e.setError(err)
		return
	}
	if src.Usage()&BufferUsageCopySrc == 0 {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToTexture: source buffer missing CopySrc usage"))
		return
	}
	if err := checkCopyRowPitch("CopyBufferToTexture", regions); err != nil {
		e.setError(err)
		return
	}
	halRegions, err := bufferTextureCopies("CopyBufferToTexture", dst, halDst, src, regions, true)
	if err != nil {
		e.setError(err)
		return
	}
	e.zeroDiscarded(dst)
	raw := e.core.RecordingEncoder()
	if raw == nil {
		return
	}
	for _, r := range halRegions {
		e.copyRegionUsage(dst, r.TextureBase, r.Size, track.TextureUsesCopyDst)
	}
	if !e.recordCopyUsages(raw, e.copyReq,
		[]copyBufferUsage{{buffer: src, usage: track.BufferUsesCopySrc}},
	) {
		return
	}
	e.trackTexture(dst)
	e.trackBuffer(src)
	if len(halRegions) > 0 {
		raw.CopyBufferToTexture(halSrc, halDst, halRegions)
	}
}

func validateRenderPassTextureViews(desc *RenderPassDescriptor) error {
	if desc == nil {
		return nil
	}
	for _, attachment := range desc.ColorAttachments {
		if attachment.View != nil && attachment.View.resolveHAL() == nil {
			return fmt.Errorf("wgpu: BeginRenderPass: color attachment view is released: %w", ErrReleased)
		}
		if attachment.ResolveTarget != nil && attachment.ResolveTarget.resolveHAL() == nil {
			return fmt.Errorf("wgpu: BeginRenderPass: resolve target view is released: %w", ErrReleased)
		}
	}
	if attachment := desc.DepthStencilAttachment; attachment != nil && attachment.View != nil && attachment.View.resolveHAL() == nil {
		return fmt.Errorf("wgpu: BeginRenderPass: depth/stencil attachment view is released: %w", ErrReleased)
	}
	return nil
}

// validateTimestampWrites checks a render pass's timestamp writes: a live
// timestamp query set, at least one index, each inside the set, and not the
// same one twice.
//
// Reference: Dawn CommandEncoder.cpp ValidatePassTimestampWrites
func validateTimestampWrites(w *RenderPassTimestampWrites) error {
	if w == nil {
		return nil
	}
	begin, end := w.BeginningOfPassWriteIndex, w.EndOfPassWriteIndex
	outside := func(i *uint32) bool { return i != nil && *i >= w.QuerySet.Count() }
	var err error
	switch {
	case w.QuerySet == nil:
		err = fmt.Errorf("no query set")
	case w.QuerySet.halQuerySet() == nil:
		err = fmt.Errorf("query set is released: %w", ErrReleased)
	case w.QuerySet.Type() != QueryTypeTimestamp:
		err = fmt.Errorf("the query set is not a timestamp query set")
	case begin == nil && end == nil:
		err = fmt.Errorf("neither a beginning nor an end of pass index")
	case outside(begin) || outside(end):
		err = fmt.Errorf("an index past the %d queries of the set", w.QuerySet.Count())
	case begin != nil && end != nil && *begin == *end:
		err = fmt.Errorf("query %d for both the beginning and the end of the pass", *begin)
	}
	if err != nil {
		return fmt.Errorf("wgpu: BeginRenderPass: timestamp writes: %w", err)
	}
	return nil
}

// attachmentsForeign records an error for an attachment or timestamp query
// set of another device than e's in desc, and reports whether it did.
func (e *CommandEncoder) attachmentsForeign(desc *RenderPassDescriptor) bool {
	foreign := func(v *TextureView) bool {
		return v != nil && e.foreign("BeginRenderPass", v.device, "an attachment")
	}
	for _, ca := range desc.ColorAttachments {
		if foreign(ca.View) || foreign(ca.ResolveTarget) {
			return true
		}
	}
	if ds := desc.DepthStencilAttachment; ds != nil && foreign(ds.View) {
		return true
	}
	w := desc.TimestampWrites
	return w != nil && w.QuerySet != nil && e.foreign("BeginRenderPass", w.QuerySet.device, "the timestamp query set")
}

func trackRenderPassTextureViews(e *CommandEncoder, desc *RenderPassDescriptor) {
	if e == nil || desc == nil {
		return
	}
	for _, attachment := range desc.ColorAttachments {
		e.trackView(attachment.View)
		e.trackView(attachment.ResolveTarget)
	}
	if attachment := desc.DepthStencilAttachment; attachment != nil {
		e.trackView(attachment.View)
	}
	if writes := desc.TimestampWrites; writes != nil {
		e.trackQuerySet(writes.QuerySet)
	}
}

// ClearBuffer zeroes size bytes of buffer from offset on, or the rest of the
// buffer from offset on when size is 0, as clearBuffer without a size does.
// The buffer needs CopyDst, and offset and size are multiples of 4.
// WebGPU spec: GPUCommandEncoder.clearBuffer.
func (e *CommandEncoder) ClearBuffer(buffer *Buffer, offset, size uint64) {
	if e.released {
		return
	}
	if buffer == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ClearBuffer: buffer is nil"))
		return
	}
	if e.foreign("CommandEncoder.ClearBuffer", buffer.device, "the buffer") {
		return
	}
	raw := e.core.RecordingEncoder()
	if raw == nil {
		return
	}
	halBuffer := buffer.halBuffer()
	if halBuffer == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ClearBuffer: buffer is released: %w", ErrReleased))
		return
	}
	if size == 0 && offset <= buffer.Size() {
		size = buffer.Size() - offset
	}
	switch {
	case buffer.Usage()&BufferUsageCopyDst == 0:
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ClearBuffer: buffer missing CopyDst usage"))
		return
	case offset%4 != 0 || size%4 != 0:
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ClearBuffer: offset %d and size %d must be multiples of 4", offset, size))
		return
	case offset > buffer.Size() || size > buffer.Size()-offset:
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ClearBuffer: offset %d + size %d exceeds buffer size %d", offset, size, buffer.Size()))
		return
	case size == 0:
		return // offset is the end of the buffer, nothing to clear
	}
	if !e.recordCopyUsages(raw, nil, []copyBufferUsage{{buffer: buffer, usage: track.BufferUsesCopyDst}}) {
		return
	}
	e.trackBuffer(buffer)
	raw.ClearBuffer(halBuffer, offset, size)
}

// ResolveQuerySet copies query results into a buffer.
func (e *CommandEncoder) ResolveQuerySet(querySet *QuerySet, firstQuery, queryCount uint32, dst *Buffer, dstOffset uint64) {
	if e.released {
		return
	}

	if dst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ResolveQuerySet: destination buffer is nil"))
		return
	}
	if dst.released == nil || dst.released.Load() {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ResolveQuerySet: destination buffer is released: %w", ErrReleased))
		return
	}
	if querySet != nil && e.foreign("CommandEncoder.ResolveQuerySet", querySet.device, "the query set") ||
		e.foreign("CommandEncoder.ResolveQuerySet", dst.device, "the destination buffer") {
		return
	}

	raw := e.core.RecordingEncoder()
	if raw == nil {
		return
	}

	halDestination := dst.halBuffer()
	if halDestination == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ResolveQuerySet: destination buffer is released: %w", ErrReleased))
		return
	}

	halQuerySet := querySet.halQuerySet()
	if halQuerySet == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ResolveQuerySet: query set is released: %w", ErrReleased))
		return
	}
	// WebGPU's resolveQuerySet checks, as Dawn's ValidateQuerySetResolve
	// makes them.
	var err error
	switch count := querySet.Count(); {
	case firstQuery >= count || queryCount > count-firstQuery:
		err = fmt.Errorf("queries %d to %d are not all among the %d of the set", firstQuery, uint64(firstQuery)+uint64(queryCount), count)
	case dst.Usage()&BufferUsageQueryResolve == 0:
		err = fmt.Errorf("destination buffer missing QueryResolve usage")
	case dstOffset%256 != 0:
		err = fmt.Errorf("destination offset %d is not a multiple of 256", dstOffset)
	case dstOffset > dst.Size() || uint64(queryCount)*8 > dst.Size()-dstOffset:
		err = fmt.Errorf("%d results at offset %d exceed the destination buffer's %d bytes", queryCount, dstOffset, dst.Size())
	}
	if err != nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.ResolveQuerySet: %w", err))
		return
	}

	if !e.recordCopyUsages(raw, nil, []copyBufferUsage{{buffer: dst, usage: track.BufferUsesCopyDst}}) {
		return
	}
	e.trackBuffer(dst)
	e.trackQuerySet(querySet)
	raw.ResolveQuerySet(halQuerySet, firstQuery, queryCount, halDestination, dstOffset)
}

// DiscardEncoding discards the encoder without producing a command buffer.
// Use this to abandon an in-progress encoding when an error occurs.
// If the encoder was acquired from the pool, it is returned for reuse.
func (e *CommandEncoder) DiscardEncoding() {
	if e.released {
		return
	}
	e.released = true
	// Drop all tracked refs since no submission will happen.
	for _, ref := range e.trackedRefs {
		ref.Drop()
	}
	e.trackedRefs = nil
	e.dropUsedLists()
	e.core.Discard()
	// Return pooled encoder immediately — no GPU work was submitted.
	e.returnEncoderToPool()
	e.recycle()
}

// returnEncoderToPool resets and returns the HAL encoder to the device's pool.
// Called when the encoder will not be submitted (error or discard).
// ResetAll must be called before release to ensure the command pool is reset
// and all command buffers return to initial state — otherwise the next
// BeginEncoding will fail with VUID-vkBeginCommandBuffer-commandBuffer-00049
// because CBs from an unreset pool may still be in executable/recording state.
// Matches Rust wgpu-core InnerCommandEncoder::drop (command/mod.rs:726-738).
func (e *CommandEncoder) returnEncoderToPool() {
	if e.halEncoder == nil || e.device == nil || e.device.cmdEncoderPool == nil {
		return
	}
	// The command buffers the encoder closed before the discard go back
	// with it, so the HAL can reuse them.
	var closed []hal.CommandBuffer
	if e.core != nil {
		closed = e.core.ClosedBuffers()
	}
	e.halEncoder.ResetAll(closed)
	e.device.cmdEncoderPool.release(e.halEncoder)
	e.halEncoder = nil
}

// Finish completes command recording and returns a CommandBuffer.
// After calling Finish(), the encoder cannot be used again.
//
// The HAL encoder ownership transfers from the CommandEncoder to the
// CommandBuffer. After GPU completion, Submit() schedules the encoder
// to be reset via ResetAll and returned to the Device's encoder pool.
func (e *CommandEncoder) Finish() (_ *CommandBuffer, err error) {
	defer e.device.reportError(&err)
	if e.released {
		return nil, ErrReleased
	}
	e.released = true

	coreCmdBuffer, err := e.core.Finish()
	if err != nil {
		// On error, drop all tracked refs since no submission will happen.
		for _, ref := range e.trackedRefs {
			ref.Drop()
		}
		e.trackedRefs = nil
		e.dropUsedLists()
		// An encoder in the error state never gets to EndEncoding, so the
		// HAL encoder is still recording. Discard that like DiscardEncoding
		// does before it goes back to the pool: otherwise Vulkan loses a
		// command buffer every time, and Metal and DX12 hand out an encoder
		// that is still recording, so each later BeginEncoding on it fails.
		if raw := e.core.RawEncoder(); raw != nil {
			raw.DiscardEncoding()
		}
		// Return the pooled encoder on error — it won't be submitted.
		e.returnEncoderToPool()
		e.recycle()
		return nil, err
	}

	// Transfer HAL encoder ownership to the CommandBuffer for post-GPU recycling.
	// Extract the HAL encoder from the core encoder's Snatchable so the pool
	// will own the only reference after recycling. Without this, the Snatchable
	// would hold a dangling reference after ResetAll.
	if e.halEncoder != nil {
		e.core.TakeHALEncoder()
	}

	cb := &e.cb
	*cb = CommandBuffer{
		core:           coreCmdBuffer,
		device:         e.device,
		trackedRefs:    e.trackedRefs,
		halEncoder:     e.halEncoder,
		usedBuffers:    e.usedBuffers,
		usedTextures:   e.usedTextures,
		usedBindGroups: e.usedBindGroups,
		usedQuerySets:  e.usedQuerySets,
		clearsFirst:    e.clearsFirst,
		discards:       slices.Collect(maps.Keys(e.discards)),
		enc:            e,
		serial:         e.id(),
	}
	e.trackedRefs = nil
	e.halEncoder = nil     // ownership transferred
	e.usedBuffers = nil    // ownership transferred
	e.usedTextures = nil   // ownership transferred
	e.usedBindGroups = nil // ownership transferred
	e.usedQuerySets = nil  // ownership transferred
	e.clearsFirst = nil
	clear(e.discards)
	return cb, nil
}

// corePassDescriptor is the core descriptor of one render pass, with room for
// the views of its attachments, kept in a pool between passes.
type corePassDescriptor struct {
	desc   core.RenderPassDescriptor
	color  []core.RenderPassColorAttachment
	views  []core.TextureView
	depth  core.RenderPassDepthStencilAttachment
	stamps core.RenderPassTimestampWrites
	label  string
}

var passDescPool = sync.Pool{New: func() any { return new(corePassDescriptor) }}

// fill converts desc into the core descriptor.
func (c *corePassDescriptor) fill(desc *RenderPassDescriptor) *core.RenderPassDescriptor {
	c.desc = core.RenderPassDescriptor{}
	if desc == nil {
		return &c.desc
	}
	// A copy, so nothing of desc itself reaches the heap: storing its label
	// would move the caller's attachment slice there too.
	if c.label != desc.Label {
		c.label = strings.Clone(desc.Label)
	}
	c.desc.Label = c.label

	// The attachments point into views, so it gets its full length before the
	// first one is taken: growing it later would leave them behind.
	if n := 2*len(desc.ColorAttachments) + 1; cap(c.views) < n {
		c.views = make([]core.TextureView, 0, n)
	}
	c.views = c.views[:0]

	c.color = c.color[:0]
	for _, ca := range desc.ColorAttachments {
		coreCA := core.RenderPassColorAttachment{
			LoadOp:     ca.LoadOp,
			StoreOp:    ca.StoreOp,
			ClearValue: ca.ClearValue,
		}
		if ca.View != nil {
			coreCA.View = c.view(ca.View)
		}
		if ca.ResolveTarget != nil {
			coreCA.ResolveTarget = c.view(ca.ResolveTarget)
		}
		c.color = append(c.color, coreCA)
	}
	c.desc.ColorAttachments = c.color

	if ds := desc.DepthStencilAttachment; ds != nil {
		c.depth = core.RenderPassDepthStencilAttachment{
			DepthLoadOp:       ds.DepthLoadOp,
			DepthStoreOp:      ds.DepthStoreOp,
			DepthClearValue:   ds.DepthClearValue,
			DepthReadOnly:     ds.DepthReadOnly,
			StencilLoadOp:     ds.StencilLoadOp,
			StencilStoreOp:    ds.StencilStoreOp,
			StencilClearValue: ds.StencilClearValue,
			StencilReadOnly:   ds.StencilReadOnly,
		}
		if ds.View != nil {
			c.depth.View = c.view(ds.View)
		}
		c.desc.DepthStencilAttachment = &c.depth
	}

	if writes := desc.TimestampWrites; writes != nil {
		if writes.QuerySet.halQuerySet() != nil {
			c.stamps = core.RenderPassTimestampWrites{
				QuerySet:                  writes.QuerySet.core,
				BeginningOfPassWriteIndex: writes.BeginningOfPassWriteIndex,
				EndOfPassWriteIndex:       writes.EndOfPassWriteIndex,
			}
			c.desc.TimestampWrites = &c.stamps
		}
	}

	return &c.desc
}

// view adds the core view of v to views and returns it.
func (c *corePassDescriptor) view(v *TextureView) *core.TextureView {
	c.views = append(c.views, coreTextureViewFrom(v))
	return &c.views[len(c.views)-1]
}

// release puts c back in the pool, holding on to nothing.
func (c *corePassDescriptor) release() {
	clear(c.color)
	clear(c.views)
	c.depth = core.RenderPassDepthStencilAttachment{}
	c.stamps = core.RenderPassTimestampWrites{}
	c.desc = core.RenderPassDescriptor{}
	passDescPool.Put(c)
}

// coreTextureViewFrom makes the core.TextureView of a public TextureView,
// wiring the Parent to the texture's coreTexture for TrackerIndex access, so
// core records the attachment in the render pass's usage scope, and the HAL
// texture the barriers into the pass are recorded against: the public
// Texture is the authority for the current handle, a borrowed surface
// texture shares one core Texture across the swapchain images.
func coreTextureViewFrom(v *TextureView) core.TextureView {
	cv := core.TextureView{
		HAL:             v.resolveHAL(),
		BaseMipLevel:    v.desc.BaseMipLevel,
		BaseArrayLayer:  v.desc.BaseArrayLayer,
		Format:          v.desc.Format,
		Dimension:       v.desc.Dimension,
		Aspect:          v.desc.Aspect,
		MipLevelCount:   v.desc.MipLevelCount,
		ArrayLayerCount: v.desc.ArrayLayerCount,
	}
	if v.texture != nil && v.texture.coreTexture != nil {
		cv.Parent = v.texture.coreTexture
		cv.Texture = v.texture.resolveHAL()
	}
	return cv
}

// CommandBuffer holds recorded GPU commands ready for submission.
// Created by CommandEncoder.Finish().
//
// It is submitted once. After Submit or Release it is spent and a Submit of
// it fails, but don't keep it: the memory behind it is reused for the command
// buffer of a later encoder.
type CommandBuffer struct {
	core   *core.CoreCommandBuffer
	device *Device
	// trackedRefs holds Clone'd ResourceRefs from encoding. Transferred to
	// the DestroyQueue on Submit() so refs are Drop'd when GPU completes.
	// Phase 2: per-command-buffer resource tracking.
	trackedRefs []*core.ResourceRef

	// halEncoder is the HAL command encoder that produced this command buffer.
	// Ownership transfers from CommandEncoder to CommandBuffer on Finish(),
	// then to the DestroyQueue on Submit() for recycling after GPU completion.
	// After GPU completion, the encoder is reset via ResetAll and returned to
	// the Device's encoder pool. This avoids creating new DX12 command allocators
	// (~64KB each) or Vulkan command pools every frame.
	//
	// Matches Rust wgpu-core where the encoder travels:
	// CommandEncoder -> CommandBuffer -> EncoderInFlight -> GPU done -> pool
	halEncoder hal.CommandEncoder

	// usedBuffers tracks the buffers referenced during encoding (VAL-A6),
	// besides those its bind groups bind.
	// Validated at Submit time: destroyed or mapped buffers cause an error.
	// Matches Rust wgpu-core's cmd_buf_data.trackers.buffers.used_resources()
	// (device/queue.rs:1780-1787).
	usedBuffers []*Buffer

	// usedTextures tracks all textures referenced during encoding (VAL-A6).
	// Validated at Submit time: destroyed textures cause an error.
	// Matches Rust wgpu-core's cmd_buf_data.trackers.textures.used_resources()
	// (device/queue.rs:1791-1808).
	usedTextures []*Texture

	// usedBindGroups tracks all bind groups referenced during encoding (VAL-B5).
	// Validated at Submit time: destroyed bind groups cause an error.
	// Matches Rust wgpu-core's cmd_buf_data.trackers.bind_groups
	// (device/queue.rs:1815-1817).
	usedBindGroups []*BindGroup

	// usedQuerySets tracks the query sets ResolveQuerySet reads and render
	// passes write timestamps into, validated at Submit time: a released one
	// causes an error.
	usedQuerySets []*QuerySet

	// clearsFirst is the encoder's clearsFirst, and discards the textures
	// it left discarded, for Submit.
	clearsFirst, discards []*Texture

	// submitted is set to true after this command buffer has been submitted
	// to a queue. A command buffer cannot be submitted twice.
	// Matches Rust wgpu-core's CommandBuffer::take_finished() which consumes
	// the buffer, preventing reuse.
	submitted bool

	// enc is the encoder this command buffer lives in and serial the serial
	// the encoder had when it was handed out: once they differ the encoder
	// has been reused and this handle is stale.
	enc    *CommandEncoder
	serial uint64

	// recycle gives the HAL encoder back to the pool after the GPU is done
	// with this command buffer; Submit fills it.
	recycle encoderRecycle
}

// stale reports whether cb was handed out by an encoder that has since been
// reused, in which case it is someone else's command buffer now.
func (cb *CommandBuffer) stale() bool {
	return cb.enc != nil && cb.enc.serial != cb.serial
}

// encoderRecycle resets a submitted command buffer's HAL encoder and gives it
// back to the pool. It lives in the CommandBuffer, or comes from recyclePool
// for the barrier encoder, so the destroy queue holds it without a closure.
type encoderRecycle struct {
	enc    hal.CommandEncoder
	bufs   []hal.CommandBuffer
	pool   *encoderPool
	owner  *CommandEncoder // the encoder to recycle with it, if pooled
	pooled bool            // from recyclePool, goes back there
}

// recyclePool keeps encoderRecycle records for the barrier encoders.
var recyclePool = sync.Pool{New: func() any { return &encoderRecycle{bufs: make([]hal.CommandBuffer, 0, 1)} }}

// ReleaseDeferred implements core.Releaser.
func (r *encoderRecycle) ReleaseDeferred() {
	r.enc.ResetAll(r.bufs)
	r.pool.release(r.enc)
	if r.pooled {
		clear(r.bufs)
		r.enc, r.pool, r.bufs = nil, nil, r.bufs[:0]
		recyclePool.Put(r)
		return
	}
	owner := r.owner
	*r = encoderRecycle{}
	owner.recycle() // r lives in owner.cb and is gone with it
}

// Release releases a CommandBuffer that will NOT be submitted to the GPU.
// This returns the HAL encoder to the device pool and drops tracked resource refs.
//
// In normal flow, Submit() takes ownership of the encoder and handles recycling
// after GPU completion. Release() is for error paths and canceled operations
// where the CommandBuffer is discarded without submitting.
//
// Matches Rust wgpu-core InnerCommandEncoder::Drop (command/mod.rs:726-738)
// which always calls reset_all + release_encoder regardless of whether the
// command buffer was submitted.
//
// A CommandBuffer MUST be either Submit()'d or Release()'d. Failing to do
// either leaks the HAL encoder (DX12 ~64KB allocator, Vulkan VkCommandPool).
func (cb *CommandBuffer) Release() {
	// A submitted command buffer is the queue's: it recycles everything
	// once the GPU is done, and a stale handle is someone else's buffer.
	if cb == nil || cb.submitted || cb.stale() {
		return
	}
	// Return encoder to pool (reset native allocator).
	// For multi-CB encoders, pass all HAL command buffers to ResetAll
	// so the underlying pool/allocator can reclaim them.
	if cb.halEncoder != nil && cb.device != nil && cb.device.cmdEncoderPool != nil {
		cb.halEncoder.ResetAll(cb.halBufferList())
		cb.device.cmdEncoderPool.release(cb.halEncoder)
		cb.halEncoder = nil
	}
	// Drop tracked resource refs.
	for _, ref := range cb.trackedRefs {
		ref.Drop()
	}
	cb.trackedRefs = nil
	cb.dropUsedSets()
	cb.core.ReleaseScopes()
	cb.enc.recycle()
}

// dropUsedSets releases the encode-time validation sets. usedBuffers,
// usedTextures and usedBindGroups exist only for
// validateCommandBufferForSubmit; once a command buffer is spent — submitted
// or released — they are hard references pinning every resource the frame
// touched for as long as the command buffer stays reachable. The emptied
// lists go back to the device for the next encoder.
func (cb *CommandBuffer) dropUsedSets() {
	if cb.device != nil {
		cb.device.usedLists.put(usedLists{cb.usedBuffers, cb.usedTextures, cb.usedBindGroups, cb.usedQuerySets})
	}
	cb.usedBuffers = nil
	cb.usedTextures = nil
	cb.usedBindGroups = nil
	cb.usedQuerySets = nil
}

// dropUsedLists is dropUsedSets for an encoder that will not produce a
// command buffer.
func (e *CommandEncoder) dropUsedLists() {
	if e.device != nil {
		e.device.usedLists.put(usedLists{e.usedBuffers, e.usedTextures, e.usedBindGroups, e.usedQuerySets})
	}
	e.usedBuffers = nil
	e.usedTextures = nil
	e.usedBindGroups = nil
	e.usedQuerySets = nil
}

// halBufferList returns all HAL command buffers in submission order.
// For single-CB recording (the common case), returns a single-element slice.
// For multi-CB recording (via OpenPass/CloseCB/CloseAndSwap/CloseAndPushFront),
// returns all accumulated CBs.
//
// Reference: Rust wgpu-core BakedCommands.encoder.list (command/mod.rs:742-749)
func (cb *CommandBuffer) halBufferList() []hal.CommandBuffer {
	if cb.core == nil {
		return nil
	}
	return cb.core.HalBufferList()
}
