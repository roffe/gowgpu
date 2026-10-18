//go:build !(js && wasm)

package wgpu

import (
	"fmt"
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
	// see id.
	serial uint64
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

	// explicitTextureTransitions records the most recent explicit transition
	// for each texture. When a following command uses that exact state, its
	// usage replaces (rather than conflicts with) the pre-transition scope
	// state because the caller already encoded the intervening barrier.
	explicitTextureTransitions map[*Texture]gputypes.TextureUsage

	// usedBindGroups tracks bind groups referenced during encoding for
	// submit-time validation (VAL-B5). At Submit, each bind group is checked
	// for destroyed state. Matches Rust wgpu-core's cmd_buf_data.trackers.bind_groups
	// (device/queue.rs:1815-1817).
	usedBindGroups []*BindGroup
}

// encoderSerials hands out CommandEncoder serials. Zero is never handed out,
// so a zero encoderMark matches no encoder.
var encoderSerials atomic.Uint64

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
}

// usedListPool keeps the emptied validation lists of spent command buffers,
// so the next command encoder fills them instead of growing new ones.
type usedListPool struct {
	mu   sync.Mutex
	free []usedLists
}

// maxFreeUsedLists caps how many sets of lists a device keeps for reuse.
const maxFreeUsedLists = 8

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
	if cap(l.buffers)+cap(l.textures)+cap(l.bindGroups) == 0 {
		return
	}
	clear(l.buffers)
	clear(l.textures)
	clear(l.bindGroups)
	l.buffers, l.textures, l.bindGroups = l.buffers[:0], l.textures[:0], l.bindGroups[:0]
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

// setError records a deferred error on the underlying command encoder.
// This implements the WebGPU deferred error pattern: encoding-phase errors
// are collected and surfaced when Finish() is called.
func (e *CommandEncoder) setError(err error) {
	if e.core != nil {
		e.core.SetError(err)
	}
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
	if bg == nil || !bg.mark.claim(e.id()) {
		return
	}
	e.keepAlive(bg.ref)
	for _, buf := range bg.boundBuffers {
		if buf.core != nil && buf.mark.claim(e.id()) {
			e.keepAlive(buf.core.Ref)
		}
	}
	e.usedBindGroups = append(e.usedBindGroups, bg)
}

// copyTextureUsage describes one texture endpoint of a copy command.
type copyTextureUsage struct {
	texture *Texture
	usage   track.TextureUses
}

type preparedCopyTextureUsage struct {
	texture *Texture
	index   track.TrackerIndex
	usage   track.TextureUses
}

type copyBufferUsage struct {
	buffer *core.Buffer
	usage  track.BufferUses
}

type preparedCopyBufferUsage struct {
	index track.TrackerIndex
	usage track.BufferUses
}

// recordCopyBufferUsages preflights every buffer endpoint before committing
// any usage. This keeps failed multi-buffer copies atomic, matching the mixed
// texture/buffer copy paths below.
func (e *CommandEncoder) recordCopyBufferUsages(requests []copyBufferUsage) bool {
	if e.core == nil {
		return true
	}

	prepared := make([]preparedCopyBufferUsage, 0, len(requests))
	positions := make(map[track.TrackerIndex]int, len(requests))
	for _, request := range requests {
		if request.buffer == nil {
			continue
		}
		td := request.buffer.TrackingData()
		if td == nil || !td.Index().IsValid() {
			continue
		}

		index := td.Index()
		if position, exists := positions[index]; exists {
			existing := prepared[position].usage
			if !existing.IsCompatible(request.usage) {
				e.setError(fmt.Errorf("wgpu: buffer usage conflict: %w", &track.UsageConflictError{
					Index: index, Existing: existing, New: request.usage,
				}))
				return false
			}
			prepared[position].usage = existing | request.usage
			continue
		}

		positions[index] = len(prepared)
		prepared = append(prepared, preparedCopyBufferUsage{index: index, usage: request.usage})
	}

	scope := e.core.Mutable().BufferScope()
	for i := range prepared {
		request := &prepared[i]
		if !scope.IsUsed(request.index) {
			continue
		}
		existing := scope.GetUsage(request.index)
		if !existing.IsCompatible(request.usage) {
			e.setError(fmt.Errorf("wgpu: buffer usage conflict: %w", &track.UsageConflictError{
				Index: request.index, Existing: existing, New: request.usage,
			}))
			return false
		}
		request.usage |= existing
	}

	for _, request := range prepared {
		scope.ReplaceUsage(request.index, request.usage)
	}
	return true
}

// recordCopyUsages preflights every resource scope update before committing any
// of them. Copy commands span multiple independently tracked resources, so a
// conflict on one endpoint must not leave another endpoint recorded.
func (e *CommandEncoder) recordCopyUsages(textures []copyTextureUsage, buffer *core.Buffer, bufferUsage track.BufferUses) bool {
	if e.core == nil {
		return true
	}
	prepared, err := prepareCopyTextureUsages(textures)
	if err != nil {
		e.setError(fmt.Errorf("wgpu: texture usage conflict: %w", err))
		return false
	}
	if err := e.preflightCopyTextureUsages(prepared); err != nil {
		e.setError(fmt.Errorf("wgpu: texture usage conflict: %w", err))
		return false
	}
	bufferIndex, finalBufferUsage, trackedBuffer, err := e.preflightCopyBufferUsage(buffer, bufferUsage)
	if err != nil {
		e.setError(fmt.Errorf("wgpu: buffer usage conflict: %w", err))
		return false
	}

	// All validation is complete. ReplaceUsage cannot fail, so the commit has no
	// partial-failure path after the first scope mutation.
	textureScope := e.core.Mutable().TextureScope()
	for _, request := range prepared {
		textureScope.ReplaceUsage(request.index, request.usage)
	}
	if trackedBuffer {
		e.core.Mutable().BufferScope().ReplaceUsage(bufferIndex, finalBufferUsage)
	}
	return true
}

// prepareCopyTextureUsages groups multiple roles for the same texture and
// rejects incompatible roles before consulting or changing the command scope.
func prepareCopyTextureUsages(requests []copyTextureUsage) ([]preparedCopyTextureUsage, error) {
	prepared := make([]preparedCopyTextureUsage, 0, len(requests))
	positions := make(map[track.TrackerIndex]int, len(requests))
	for _, request := range requests {
		if request.texture == nil || request.texture.coreTexture == nil {
			continue
		}
		td := request.texture.coreTexture.TrackingData()
		if td == nil || !td.Index().IsValid() {
			continue
		}
		index := td.Index()
		position, exists := positions[index]
		if !exists {
			positions[index] = len(prepared)
			prepared = append(prepared, preparedCopyTextureUsage{
				texture: request.texture, index: index, usage: request.usage,
			})
			continue
		}
		existing := prepared[position].usage
		combined := existing | request.usage
		if !combined.IsCompatible(combined) {
			return nil, &track.TextureUsageConflictError{
				Index: index, Existing: existing, New: request.usage,
			}
		}
		prepared[position].usage = combined
	}
	return prepared, nil
}

func (e *CommandEncoder) preflightCopyTextureUsages(requests []preparedCopyTextureUsage) error {
	scope := e.core.Mutable().TextureScope()
	for i := range requests {
		request := &requests[i]
		if !scope.IsUsed(request.index) {
			continue
		}
		existing := scope.GetUsage(request.index)
		combined := existing | request.usage
		if combined.IsCompatible(combined) {
			request.usage = combined
			continue
		}
		if transitioned, ok := e.explicitTextureTransitions[request.texture]; ok &&
			transitioned == request.usage.ToTextureUsage() {
			continue
		}
		return &track.TextureUsageConflictError{
			Index: request.index, Existing: existing, New: request.usage,
		}
	}
	return nil
}

func (e *CommandEncoder) preflightCopyBufferUsage(buffer *core.Buffer, usage track.BufferUses) (track.TrackerIndex, track.BufferUses, bool, error) {
	if buffer == nil {
		return 0, track.BufferUsesNone, false, nil
	}
	td := buffer.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return 0, track.BufferUsesNone, false, nil
	}
	index := td.Index()
	scope := e.core.Mutable().BufferScope()
	if !scope.IsUsed(index) {
		return index, usage, true, nil
	}
	existing := scope.GetUsage(index)
	if existing.IsCompatible(usage) {
		return index, existing | usage, true, nil
	}
	return 0, track.BufferUsesNone, false, &track.UsageConflictError{
		Index: index, Existing: existing, New: usage,
	}
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
	trackRenderPassTextureViews(e, desc)

	coreDesc := convertRenderPassDesc(desc)

	corePass, err := e.core.BeginRenderPass(coreDesc)
	if err != nil {
		return nil, err
	}
	forgetClearedTextures(e.device, desc)

	return &RenderPassEncoder{core: corePass, encoder: e}, nil
}

// forgetClearedTextures takes the textures this pass clears whole off the list
// of new textures to zero: the clear does it.
// ponytail: decided when the pass is recorded, not when it runs, so a command
// buffer submitted before this one that samples the texture still sees old
// memory.
func forgetClearedTextures(d *Device, desc *RenderPassDescriptor) {
	if desc == nil || d == nil || d.queue == nil || d.queue.pending == nil {
		return
	}
	for _, ca := range desc.ColorAttachments {
		if ca.LoadOp != gputypes.LoadOpClear || ca.View == nil || ca.View.texture == nil {
			continue
		}
		if ct := ca.View.texture.coreTexture; ct != nil && ct.MipLevelCount() <= 1 && ct.Size().DepthOrArrayLayers <= 1 {
			d.queue.pending.forgetZero(ca.View.texture)
		}
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

	return &ComputePassEncoder{core: corePass, encoder: e}, nil
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
	raw := e.core.RawEncoder()
	if raw == nil {
		return
	}
	halSrc := src.halBuffer()
	halDst := dst.halBuffer()
	if halSrc == nil || halDst == nil {
		e.setError(fmt.Errorf("wgpu: CommandEncoder.CopyBufferToBuffer: source or destination buffer is released: %w", ErrReleased))
		return
	}
	if !e.recordCopyBufferUsages([]copyBufferUsage{
		{buffer: src.core, usage: track.BufferUsesCopySrc},
		{buffer: dst.core, usage: track.BufferUsesCopyDst},
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
	raw := e.core.RawEncoder()
	if raw == nil {
		return
	}
	if !e.recordCopyUsages(
		[]copyTextureUsage{{texture: src, usage: track.TextureUsesCopySrc}},
		dst.core, track.BufferUsesCopyDst,
	) {
		return
	}
	for _, region := range regions {
		e.trackTexture(region.TextureBase.Texture)
	}
	e.trackTexture(src)
	e.trackBuffer(dst)
	halRegions := make([]hal.BufferTextureCopy, len(regions))
	for i, r := range regions {
		halRegions[i] = r.toHAL()
	}
	raw.CopyTextureToBuffer(halSrc, halDst, halRegions)
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
	raw := e.core.RawEncoder()
	if raw == nil {
		return
	}
	if !e.recordCopyUsages([]copyTextureUsage{
		{texture: src, usage: track.TextureUsesCopySrc},
		{texture: dst, usage: track.TextureUsesCopyDst},
	}, nil, track.BufferUsesNone) {
		return
	}
	for _, region := range regions {
		e.trackTexture(region.Source.Texture)
		e.trackTexture(region.Destination.Texture)
	}
	e.trackTexture(src)
	e.trackTexture(dst)
	halRegions := make([]hal.TextureCopy, len(regions))
	for i, r := range regions {
		halRegions[i] = r.toHAL()
	}
	raw.CopyTextureToTexture(halSrc, halDst, halRegions)
}

// TransitionTextures transitions texture states for synchronization.
// This is needed on Vulkan for layout transitions between render pass
// and copy operations (e.g., after MSAA resolve before CopyTextureToBuffer).
// On Metal, GLES, and software backends this is a no-op.
func (e *CommandEncoder) TransitionTextures(barriers []TextureBarrier) {
	if e.released {
		return
	}
	raw := e.core.RawEncoder()
	if raw == nil {
		return
	}
	halBarriers := make([]hal.TextureBarrier, 0, len(barriers))
	validBarriers := make([]TextureBarrier, 0, len(barriers))
	for _, b := range barriers {
		if b.Texture != nil && b.Texture.resolveHAL() == nil {
			e.setError(fmt.Errorf("wgpu: CommandEncoder.TransitionTextures: texture is released: %w", ErrReleased))
			return
		}
		if b.Texture == nil || b.Texture.resolveHAL() == nil {
			continue
		}
		e.trackTexture(b.Texture)
		halBarriers = append(halBarriers, b.toHAL())
		validBarriers = append(validBarriers, b)
	}
	if len(halBarriers) > 0 {
		raw.TransitionTextures(halBarriers)
		if e.explicitTextureTransitions == nil {
			e.explicitTextureTransitions = make(map[*Texture]gputypes.TextureUsage)
		}
		for _, b := range validBarriers {
			e.explicitTextureTransitions[b.Texture] = b.Usage.NewUsage
		}
	}
}

// CopyBufferToTexture copies data from a buffer to a texture.
// WebGPU spec: GPUCommandEncoder.copyBufferToTexture.
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
	raw := e.core.RawEncoder()
	if raw == nil {
		return
	}
	if !e.recordCopyUsages(
		[]copyTextureUsage{{texture: dst, usage: track.TextureUsesCopyDst}},
		src.core, track.BufferUsesCopySrc,
	) {
		return
	}
	e.trackTexture(dst)
	e.trackBuffer(src)
	halRegions := make([]hal.BufferTextureCopy, len(regions))
	for i, r := range regions {
		halRegions[i] = hal.BufferTextureCopy{
			BufferLayout: hal.ImageDataLayout{
				Offset:       r.BufferLayout.Offset,
				BytesPerRow:  r.BufferLayout.BytesPerRow,
				RowsPerImage: r.BufferLayout.RowsPerImage,
			},
			TextureBase: hal.ImageCopyTexture{
				Texture:  halDst,
				MipLevel: r.TextureBase.MipLevel,
				Origin:   hal.Origin3D(r.TextureBase.Origin),
			},
			Size: hal.Extent3D(r.Size),
		}
	}
	raw.CopyBufferToTexture(halSrc, halDst, halRegions)
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

func trackRenderPassTextureViews(e *CommandEncoder, desc *RenderPassDescriptor) {
	if e == nil || desc == nil {
		return
	}
	for _, attachment := range desc.ColorAttachments {
		if attachment.View != nil {
			e.trackTexture(attachment.View.texture)
		}
		if attachment.ResolveTarget != nil {
			e.trackTexture(attachment.ResolveTarget.texture)
		}
	}
	if attachment := desc.DepthStencilAttachment; attachment != nil && attachment.View != nil {
		e.trackTexture(attachment.View.texture)
	}
}

// ClearBuffer clears a buffer region to zero.
// WebGPU spec: GPUCommandEncoder.clearBuffer.
func (e *CommandEncoder) ClearBuffer(buffer *Buffer, offset, size uint64) {
	if e.released || buffer == nil {
		return
	}
	raw := e.core.RawEncoder()
	if raw == nil {
		return
	}
	raw.ClearBuffer(buffer.halBuffer(), offset, size)
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

	raw := e.core.RawEncoder()
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
	raw := e.core.RawEncoder()
	if raw != nil {
		raw.DiscardEncoding()
	}
	// Return pooled encoder immediately — no GPU work was submitted.
	e.returnEncoderToPool()
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
	e.halEncoder.ResetAll(nil)
	e.device.cmdEncoderPool.release(e.halEncoder)
	e.halEncoder = nil
}

// Finish completes command recording and returns a CommandBuffer.
// After calling Finish(), the encoder cannot be used again.
//
// The HAL encoder ownership transfers from the CommandEncoder to the
// CommandBuffer. After GPU completion, Submit() schedules the encoder
// to be reset via ResetAll and returned to the Device's encoder pool.
func (e *CommandEncoder) Finish() (*CommandBuffer, error) {
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
		return nil, err
	}

	// Transfer HAL encoder ownership to the CommandBuffer for post-GPU recycling.
	// Extract the HAL encoder from the core encoder's Snatchable so the pool
	// will own the only reference after recycling. Without this, the Snatchable
	// would hold a dangling reference after ResetAll.
	if e.halEncoder != nil {
		e.core.TakeHALEncoder()
	}

	cb := &CommandBuffer{
		core:           coreCmdBuffer,
		device:         e.device,
		trackedRefs:    e.trackedRefs,
		halEncoder:     e.halEncoder,
		usedBuffers:    e.usedBuffers,
		usedTextures:   e.usedTextures,
		usedBindGroups: e.usedBindGroups,
	}
	e.trackedRefs = nil
	e.halEncoder = nil     // ownership transferred
	e.usedBuffers = nil    // ownership transferred
	e.usedTextures = nil   // ownership transferred
	e.usedBindGroups = nil // ownership transferred
	return cb, nil
}

// convertRenderPassDesc converts a public descriptor to core descriptor.
// The conversion wires core.TextureView.Parent from the public TextureView's
// parent Texture coreTexture, enabling TrackerIndex-based usage tracking in
// populateTextureScope for submit-time barrier injection.
func convertRenderPassDesc(desc *RenderPassDescriptor) *core.RenderPassDescriptor {
	if desc == nil {
		return &core.RenderPassDescriptor{}
	}

	coreDesc := &core.RenderPassDescriptor{
		Label: desc.Label,
	}

	for _, ca := range desc.ColorAttachments {
		coreCA := core.RenderPassColorAttachment{
			LoadOp:     ca.LoadOp,
			StoreOp:    ca.StoreOp,
			ClearValue: ca.ClearValue,
		}
		if ca.View != nil {
			coreCA.View = coreTextureViewFrom(ca.View)
		}
		if ca.ResolveTarget != nil {
			coreCA.ResolveTarget = coreTextureViewFrom(ca.ResolveTarget)
		}
		coreDesc.ColorAttachments = append(coreDesc.ColorAttachments, coreCA)
	}

	if desc.DepthStencilAttachment != nil {
		ds := desc.DepthStencilAttachment
		coreDSA := &core.RenderPassDepthStencilAttachment{
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
			coreDSA.View = coreTextureViewFrom(ds.View)
		}
		coreDesc.DepthStencilAttachment = coreDSA
	}

	if writes := desc.TimestampWrites; writes != nil {
		halQuerySet := writes.QuerySet.halQuerySet()
		if halQuerySet != nil {
			coreDesc.TimestampWrites = &core.RenderPassTimestampWrites{
				QuerySet:                  writes.QuerySet.core,
				BeginningOfPassWriteIndex: writes.BeginningOfPassWriteIndex,
				EndOfPassWriteIndex:       writes.EndOfPassWriteIndex,
			}
		}
	}

	return coreDesc
}

// coreTextureViewFrom creates a core.TextureView from a public TextureView,
// wiring the Parent to the texture's coreTexture for TrackerIndex access.
// The Parent reference enables populateTextureScope to record per-texture
// usage in the command buffer's TextureUsageScope.
func coreTextureViewFrom(v *TextureView) *core.TextureView {
	cv := &core.TextureView{HAL: v.resolveHAL()}
	if v.texture != nil && v.texture.coreTexture != nil {
		cv.Parent = v.texture.coreTexture
	}
	return cv
}

// CommandBuffer holds recorded GPU commands ready for submission.
// Created by CommandEncoder.Finish().
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

	// submitted is set to true after this command buffer has been submitted
	// to a queue. A command buffer cannot be submitted twice.
	// Matches Rust wgpu-core's CommandBuffer::take_finished() which consumes
	// the buffer, preventing reuse.
	submitted bool
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
	if cb == nil {
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
}

// dropUsedSets releases the encode-time validation sets. usedBuffers,
// usedTextures and usedBindGroups exist only for
// validateCommandBufferForSubmit; once a command buffer is spent — submitted
// or released — they are hard references pinning every resource the frame
// touched for as long as the command buffer stays reachable. The emptied
// lists go back to the device for the next encoder.
func (cb *CommandBuffer) dropUsedSets() {
	if cb.device != nil {
		cb.device.usedLists.put(usedLists{cb.usedBuffers, cb.usedTextures, cb.usedBindGroups})
	}
	cb.usedBuffers = nil
	cb.usedTextures = nil
	cb.usedBindGroups = nil
}

// dropUsedLists is dropUsedSets for an encoder that will not produce a
// command buffer.
func (e *CommandEncoder) dropUsedLists() {
	if e.device != nil {
		e.device.usedLists.put(usedLists{e.usedBuffers, e.usedTextures, e.usedBindGroups})
	}
	e.usedBuffers = nil
	e.usedTextures = nil
	e.usedBindGroups = nil
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
