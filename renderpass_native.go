//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"fmt"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
)

// RenderPassEncoder records draw commands within a render pass.
//
// Created by CommandEncoder.BeginRenderPass().
// Must be ended with End() before the CommandEncoder can be finished.
//
// NOT thread-safe.
type RenderPassEncoder struct {
	core    *core.CoreRenderPassEncoder
	encoder *CommandEncoder
	// currentPipelineBindGroupCount tracks the bind group count of the
	// currently set pipeline. Used by SetBindGroup to validate that the
	// group index is within the pipeline layout bounds. Zero means no
	// pipeline has been set yet.
	currentPipelineBindGroupCount uint32
	// pipelineSet tracks whether SetPipeline has been called.
	// Draw commands require a pipeline to be set first.
	pipelineSet bool
	// binder tracks bind group assignments and validates compatibility
	// at draw time, matching Rust wgpu-core's Binder pattern.
	binder binder
	// vertexSteps is how the current pipeline reads each vertex buffer, and
	// vertexBuffers what SetVertexBuffer bound at each slot.
	vertexSteps   []vertexStep
	vertexBuffers []boundVertexBuffer
	// indexBufferSet tracks whether SetIndexBuffer has been called.
	// DrawIndexed and DrawIndexedIndirect require an index buffer.
	indexBufferSet bool
	// indexBufferFormat stores the format passed to the most recent SetIndexBuffer call.
	// Used to validate against the pipeline's StripIndexFormat at DrawIndexed/DrawIndexedIndirect time.
	// Matches Rust wgpu-core State.index.buffer_format (render.rs:568-582).
	indexBufferFormat gputypes.IndexFormat
	// indexCount is how many indices the index buffer holds past its offset.
	indexCount uint64
	// currentStripIndexFormat stores the pipeline's StripIndexFormat for draw-time validation.
	// Set by SetPipeline from RenderPipeline.stripIndexFormat.
	currentStripIndexFormat *gputypes.IndexFormat
	// blendConstantRequired is true if the current pipeline uses
	// BlendFactorConstant or BlendFactorOneMinusConstant.
	// Set by SetPipeline from RenderPipeline.blendConstantRequired.
	blendConstantRequired bool
	// blendConstantSet tracks whether SetBlendConstant has been called.
	// Matches Rust wgpu-core OptionalState for blend_constant.
	blendConstantSet bool
}

// reset readies p, which lives in its encoder, for a new pass. The binder's
// lists keep their room.
func (p *RenderPassEncoder) reset(corePass *core.CoreRenderPassEncoder, enc *CommandEncoder) {
	b := p.binder
	*p = RenderPassEncoder{core: corePass, encoder: enc, vertexBuffers: p.vertexBuffers[:0]}
	b.reset()
	p.binder = b
}

// SetPipeline sets the active render pipeline.
func (p *RenderPassEncoder) SetPipeline(pipeline *RenderPipeline) {
	if pipeline == nil {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.SetPipeline: pipeline is nil"))
		return
	}
	if pipeline.released {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.SetPipeline: pipeline is released: %w", ErrReleased))
		return
	}
	// A pipeline CreateRenderPipeline did not make has no attachments to
	// check: it draws with no HAL pipeline either.
	if pipeline.passContext.SampleCount != 0 {
		if err := p.core.CheckPipeline(&pipeline.passContext, pipeline.writesDepth, pipeline.writesStencil, pipeline.label); err != nil {
			p.encoder.setError(fmt.Errorf("wgpu: RenderPass.SetPipeline: %w", err))
			return
		}
	}
	p.currentPipelineBindGroupCount = pipeline.bindGroupCount
	p.pipelineSet = true
	p.vertexSteps = pipeline.vertexSteps
	p.currentStripIndexFormat = pipeline.stripIndexFormat
	if pipeline.blendConstantRequired {
		p.blendConstantRequired = true
	}
	p.binder.updateExpectations(pipeline.bindGroupLayouts)
	p.binder.updateLateBufferBindingsFromPipeline(pipeline.lateSizedBufferGroups)
	p.encoder.trackPipeline(&pipeline.mark, pipeline.ref)
	raw := p.core.RawPass()
	if raw != nil && pipeline.hal != nil {
		raw.SetPipeline(pipeline.hal)
	}
}

// SetBindGroup sets a bind group for the given index.
func (p *RenderPassEncoder) SetBindGroup(index uint32, group *BindGroup, offsets []uint32) {
	if err := validateSetBindGroup("RenderPass", index, group, offsets, p.currentPipelineBindGroupCount, p.encoder.limits()); err != nil {
		p.encoder.setError(err)
		return
	}
	p.binder.assign(index, group.layout)
	p.binder.assignBindGroup(index, group)
	// Keeps the group and its buffers alive and lists the group for
	// submit-time validation, once per encoder; and puts its textures and
	// buffers in the pass's usage scope, once per pass.
	p.encoder.trackBindGroup(group)
	p.encoder.recordPassBindGroup(group)
	raw := p.core.RawPass()
	if raw != nil && group.hal != nil {
		raw.SetBindGroup(index, group.hal, offsets)
	}
}

// boundVertexBuffer is the vertex buffer bound at a slot: the bytes past its
// offset.
type boundVertexBuffer struct {
	set  bool
	size uint64
}

// SetVertexBuffer sets a vertex buffer for the given slot.
//
// Reference: Dawn RenderEncoderBase::APISetVertexBuffer
func (p *RenderPassEncoder) SetVertexBuffer(slot uint32, buffer *Buffer, offset uint64) {
	if buffer == nil {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.SetVertexBuffer: buffer is nil"))
		return
	}
	var err error
	switch limit := p.encoder.limits().MaxVertexBuffers; {
	case limit > 0 && slot >= limit:
		err = fmt.Errorf("slot %d is not below the limit of %d vertex buffers", slot, limit)
	case buffer.Usage()&BufferUsageVertex == 0:
		err = fmt.Errorf("buffer %q lacks BufferUsageVertex", buffer.Label())
	case offset%4 != 0:
		err = fmt.Errorf("offset %d is not a multiple of 4", offset)
	case offset > buffer.Size():
		err = fmt.Errorf("offset %d is past the end of buffer %q (%d bytes)", offset, buffer.Label(), buffer.Size())
	}
	if err != nil {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.SetVertexBuffer: %w", err))
		return
	}
	for uint32(len(p.vertexBuffers)) <= slot { //nolint:gosec // slot count bounded by MaxVertexBuffers
		p.vertexBuffers = append(p.vertexBuffers, boundVertexBuffer{})
	}
	p.vertexBuffers[slot] = boundVertexBuffer{set: true, size: buffer.Size() - offset}
	p.encoder.trackBuffer(buffer)
	p.core.SetVertexBuffer(slot, buffer.coreBuffer(), offset)
}

// SetIndexBuffer sets the index buffer.
//
// Reference: Dawn RenderEncoderBase::APISetIndexBuffer
func (p *RenderPassEncoder) SetIndexBuffer(buffer *Buffer, format gputypes.IndexFormat, offset uint64) {
	if buffer == nil {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.SetIndexBuffer: buffer is nil"))
		return
	}
	var err error
	indexSize := uint64(format.Size())
	switch {
	case buffer.Usage()&BufferUsageIndex == 0:
		err = fmt.Errorf("buffer %q lacks BufferUsageIndex", buffer.Label())
	case indexSize == 0:
		err = fmt.Errorf("index format %v is not Uint16 or Uint32", format)
	case offset%indexSize != 0:
		err = fmt.Errorf("offset %d is not a multiple of the %v index size", offset, format)
	case offset > buffer.Size():
		err = fmt.Errorf("offset %d is past the end of buffer %q (%d bytes)", offset, buffer.Label(), buffer.Size())
	}
	if err != nil {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.SetIndexBuffer: %w", err))
		return
	}
	p.indexBufferSet = true
	p.indexBufferFormat = format
	p.indexCount = (buffer.Size() - offset) / indexSize
	p.encoder.trackBuffer(buffer)
	p.core.SetIndexBuffer(buffer.coreBuffer(), format, offset)
}

// SetViewport sets the viewport transformation.
func (p *RenderPassEncoder) SetViewport(vp gputypes.Viewport) {
	p.core.SetViewport(vp)
}

// SetScissorRect sets the scissor rectangle for clipping.
func (p *RenderPassEncoder) SetScissorRect(rect gputypes.ScissorRect) {
	p.core.SetScissorRect(rect)
}

// SetBlendConstant sets the blend constant color.
func (p *RenderPassEncoder) SetBlendConstant(color *gputypes.Color) {
	p.blendConstantSet = true
	p.core.SetBlendConstant(color)
}

// SetStencilReference sets the stencil reference value.
func (p *RenderPassEncoder) SetStencilReference(reference uint32) {
	p.core.SetStencilReference(reference)
}

// validateDrawState checks that a pipeline has been set, all bind groups
// are compatible, and enough vertex buffers have been set before a draw call.
// Returns true if validation passes, false if an error was recorded.
//
// Each validation failure wraps a typed sentinel error so that callers can
// use errors.Is() to identify the failure category programmatically.
// Matches Rust wgpu-core State::is_ready (command/render.rs:542-593).
func (p *RenderPassEncoder) validateDrawState(method string) bool {
	if !p.pipelineSet {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.%s: no pipeline set (call SetPipeline first): %w",
			method, ErrDrawMissingPipeline))
		return false
	}
	if err := p.binder.checkCompatibility(); err != nil {
		// Wrap the binder error with the appropriate draw-time sentinel
		// so errors.Is works for both the specific binder cause and the
		// public draw-time category.
		sentinel := ErrDrawMissingBindGroup
		if errors.Is(err, errBindGroupIncompatible) {
			sentinel = ErrDrawIncompatibleBindGroup
		}
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.%s: %w: %w", method, sentinel, err))
		return false
	}
	for slot, step := range p.vertexSteps {
		if step.mode != gputypes.VertexStepModeVertexBufferNotUsed && (slot >= len(p.vertexBuffers) || !p.vertexBuffers[slot].set) {
			p.encoder.setError(fmt.Errorf("wgpu: RenderPass.%s: the pipeline reads vertex buffer slot %d, which has none: %w",
				method, slot, ErrDrawMissingVertexBuffer))
			return false
		}
	}
	if p.blendConstantRequired && !p.blendConstantSet {
		p.encoder.setError(fmt.Errorf(
			"wgpu: RenderPass.%s: %w",
			method, ErrDrawMissingBlendConstant,
		))
		return false
	}
	// Late buffer binding size validation: check that bound buffers are large enough
	// for bindings with MinBindingSize == 0. Matches Rust wgpu-core's is_ready()
	// call to check_late_buffer_bindings before draw (render.rs:542-545).
	if err := p.binder.checkLateBufferBindings(); err != nil {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.%s: %w: %w", method, ErrDrawLateBufferTooSmall, err))
		return false
	}
	return true
}

// checkVertexRanges checks that the vertex buffers hold the elements a draw
// reads: up to lastVertex for the per-vertex ones and up to lastInstance for
// the per-instance ones. An indexed draw passes a lastVertex of 0, the
// vertices it reads depend on the indices.
//
// Reference: wgpu-core command/draw.rs VertexLimits
func (p *RenderPassEncoder) checkVertexRanges(method string, lastVertex, lastInstance uint64) bool {
	for slot, step := range p.vertexSteps {
		last, what := lastVertex, "vertex"
		switch step.mode {
		case gputypes.VertexStepModeVertexBufferNotUsed:
			continue
		case gputypes.VertexStepModeInstance:
			last, what = lastInstance, "instance"
		}
		size := p.vertexBuffers[slot].size
		if last == 0 || step.stride == 0 && size >= step.lastStride {
			continue
		}
		var limit uint64
		if size >= step.lastStride {
			limit = (size-step.lastStride)/step.stride + 1
		}
		if last > limit {
			p.encoder.setError(fmt.Errorf("wgpu: RenderPass.%s: draws up to %s %d, the vertex buffer at slot %d holds %d",
				method, what, last, slot, limit))
			return false
		}
	}
	return true
}

// Draw draws primitives. A non-zero FirstInstance needs no feature: WebGPU
// only gates it for indirect draws (indirect-first-instance), and every
// registered backend passes it to the native draw.
func (p *RenderPassEncoder) Draw(args gputypes.DrawArgs) {
	if !p.validateDrawState("Draw") ||
		!p.checkVertexRanges("Draw", uint64(args.FirstVertex)+uint64(args.VertexCount), uint64(args.FirstInstance)+uint64(args.InstanceCount)) {
		return
	}
	p.core.Draw(args)
}

// DrawIndexed draws indexed primitives.
func (p *RenderPassEncoder) DrawIndexed(args gputypes.DrawIndexedArgs) {
	if !p.validateDrawState("DrawIndexed") {
		return
	}
	if !p.indexBufferSet {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.DrawIndexed: no index buffer set (call SetIndexBuffer first): %w",
			ErrDrawMissingIndexBuffer))
		return
	}
	// VAL-B2: Validate index format matches pipeline's strip index format.
	// Matches Rust wgpu-core render.rs:568-582 (UnmatchedIndexFormats).
	if p.currentStripIndexFormat != nil && p.indexBufferFormat != *p.currentStripIndexFormat {
		p.encoder.setError(fmt.Errorf(
			"wgpu: RenderPass.DrawIndexed: index buffer format %v does not match pipeline strip index format %v: %w",
			p.indexBufferFormat, *p.currentStripIndexFormat, ErrDrawIndexFormatMismatch))
		return
	}
	if last := uint64(args.FirstIndex) + uint64(args.IndexCount); last > p.indexCount {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.DrawIndexed: draws up to index %d, the index buffer holds %d", last, p.indexCount))
		return
	}
	if !p.checkVertexRanges("DrawIndexed", 0, uint64(args.FirstInstance)+uint64(args.InstanceCount)) {
		return
	}
	p.core.DrawIndexed(args)
}

// DrawIndirect draws primitives with GPU-generated parameters.
func (p *RenderPassEncoder) DrawIndirect(buffer *Buffer, offset uint64) {
	p.MultiDrawIndirect(buffer, offset, 1)
}

// MultiDrawIndirect draws consecutive primitives with GPU-generated parameters.
// Each argument record is 16 bytes.
func (p *RenderPassEncoder) MultiDrawIndirect(buffer *Buffer, offset uint64, drawCount uint32) {
	if drawCount == 0 {
		return
	}
	if !p.validateDrawState("DrawIndirect") {
		return
	}
	// VAL-C1: Multi-draw indirect requires FeatureMultiDrawIndirect when drawCount > 1.
	// Reference: wgpu-core command/render.rs multi-draw feature check.
	if drawCount > 1 {
		if err := core.RequireFeature(
			p.encoder.device.Features(),
			gputypes.FeatureMultiDrawIndirect,
			"MultiDrawIndirect",
		); err != nil {
			p.encoder.setError(err)
			return
		}
	}
	if buffer == nil {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.DrawIndirect: buffer is nil"))
		return
	}
	// VAL-B3: Validate indirect buffer has INDIRECT usage.
	// Matches Rust wgpu-core render.rs:2763 (check_usage(BufferUsages::INDIRECT)).
	if buffer.Usage()&BufferUsageIndirect == 0 {
		p.encoder.setError(fmt.Errorf(
			"wgpu: RenderPass.DrawIndirect: buffer %q missing BufferUsageIndirect usage: %w",
			buffer.Label(), ErrDrawIndirectBufferUsage))
		return
	}
	// VAL-B3: Validate indirect buffer offset is 4-byte aligned.
	// Matches Rust wgpu-core render.rs:2766 (offset % 4 != 0).
	if offset%4 != 0 {
		p.encoder.setError(fmt.Errorf(
			"wgpu: RenderPass.DrawIndirect: offset %d is not 4-byte aligned: %w",
			offset, ErrDrawIndirectOffsetAlignment))
		return
	}
	// VAL-B3: Validate indirect args fit within buffer.
	// DrawIndirect args: 4 × uint32 = 16 bytes. Matches Rust render.rs:2772-2779.
	if !drawIndirectRangeFits(buffer.Size(), offset, drawCount) {
		p.encoder.setError(fmt.Errorf(
			"wgpu: RenderPass.MultiDrawIndirect: offset %d + %d draw(s) exceeds buffer size %d: %w",
			offset, drawCount, buffer.Size(), ErrDrawIndirectBufferOverrun))
		return
	}
	p.encoder.trackBuffer(buffer)
	p.core.MultiDrawIndirect(buffer.coreBuffer(), offset, drawCount)
}

// DrawIndexedIndirect draws indexed primitives with GPU-generated parameters.
func (p *RenderPassEncoder) DrawIndexedIndirect(buffer *Buffer, offset uint64) {
	p.MultiDrawIndexedIndirect(buffer, offset, 1)
}

// MultiDrawIndexedIndirect draws consecutive indexed primitives with
// GPU-generated parameters. Each argument record is 20 bytes.
func (p *RenderPassEncoder) MultiDrawIndexedIndirect(buffer *Buffer, offset uint64, drawCount uint32) {
	if drawCount == 0 {
		return
	}
	if !p.validateDrawState("DrawIndexedIndirect") {
		return
	}
	// VAL-C1: Multi-draw indexed indirect requires FeatureMultiDrawIndirect when drawCount > 1.
	// Reference: wgpu-core command/render.rs multi-draw feature check.
	if drawCount > 1 {
		if err := core.RequireFeature(
			p.encoder.device.Features(),
			gputypes.FeatureMultiDrawIndirect,
			"MultiDrawIndexedIndirect",
		); err != nil {
			p.encoder.setError(err)
			return
		}
	}
	if !p.indexBufferSet {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.DrawIndexedIndirect: no index buffer set (call SetIndexBuffer first): %w",
			ErrDrawMissingIndexBuffer))
		return
	}
	// VAL-B2: Validate index format matches pipeline's strip index format.
	// Matches Rust wgpu-core render.rs:568-582 (UnmatchedIndexFormats).
	if p.currentStripIndexFormat != nil && p.indexBufferFormat != *p.currentStripIndexFormat {
		p.encoder.setError(fmt.Errorf(
			"wgpu: RenderPass.DrawIndexedIndirect: index buffer format %v does not match pipeline strip index format %v: %w",
			p.indexBufferFormat, *p.currentStripIndexFormat, ErrDrawIndexFormatMismatch))
		return
	}
	if buffer == nil {
		p.encoder.setError(fmt.Errorf("wgpu: RenderPass.DrawIndexedIndirect: buffer is nil"))
		return
	}
	// VAL-B3: Validate indirect buffer has INDIRECT usage.
	// Matches Rust wgpu-core render.rs:2763 (check_usage(BufferUsages::INDIRECT)).
	if buffer.Usage()&BufferUsageIndirect == 0 {
		p.encoder.setError(fmt.Errorf(
			"wgpu: RenderPass.DrawIndexedIndirect: buffer %q missing BufferUsageIndirect usage: %w",
			buffer.Label(), ErrDrawIndirectBufferUsage))
		return
	}
	// VAL-B3: Validate indirect buffer offset is 4-byte aligned.
	// Matches Rust wgpu-core render.rs:2766 (offset % 4 != 0).
	if offset%4 != 0 {
		p.encoder.setError(fmt.Errorf(
			"wgpu: RenderPass.DrawIndexedIndirect: offset %d is not 4-byte aligned: %w",
			offset, ErrDrawIndirectOffsetAlignment))
		return
	}
	// VAL-B3: Validate all indirect args fit within the buffer without allowing
	// offset+drawCount*20 to wrap around uint64.
	if !indexedIndirectRangeFits(buffer.Size(), offset, drawCount) {
		p.encoder.setError(fmt.Errorf(
			"wgpu: RenderPass.MultiDrawIndexedIndirect: offset %d + %d draw(s) exceeds buffer size %d: %w",
			offset, drawCount, buffer.Size(), ErrDrawIndirectBufferOverrun))
		return
	}
	p.encoder.trackBuffer(buffer)
	p.core.MultiDrawIndexedIndirect(buffer.coreBuffer(), offset, drawCount)
}

// ExecuteBundles executes reusable render command bundles in this render pass.
func (p *RenderPassEncoder) ExecuteBundles(bundles ...*RenderBundle) {
	raw := p.core.RawPass()
	if raw == nil {
		return
	}

	for _, bundle := range bundles {
		if bundle == nil || bundle.released.Load() || bundle.hal == nil {
			continue
		}

		raw.ExecuteBundle(bundle.hal)
	}
}

// MultiDrawIndirectCount draws primitives with a GPU-provided draw count.
// countBuffer must hold a uint32 count at countOffset. VAL-C2: requires FeatureMultiDrawIndirectCount.
func (p *RenderPassEncoder) MultiDrawIndirectCount(
	indirectBuffer *Buffer, indirectOffset uint64,
	countBuffer *Buffer, countOffset uint64, maxDrawCount uint32,
) {
	p.executeIndirectCountDraw(indirectCountDrawConfig{
		validateDrawOp:  "DrawIndirect",
		featureResource: "MultiDrawIndirectCount",
		indirectBuffer:  indirectBuffer,
		indirectOffset:  indirectOffset,
		countBuffer:     countBuffer,
		countOffset:     countOffset,
		maxDrawCount:    maxDrawCount,
		recordStride:    drawIndirectRecordSize,
		record: func() {
			p.core.DrawIndirectCount(
				indirectBuffer.coreBuffer(), indirectOffset,
				countBuffer.coreBuffer(), countOffset, maxDrawCount,
			)
		},
	})
}

// MultiDrawIndexedIndirectCount draws indexed primitives with a GPU-provided draw count.
func (p *RenderPassEncoder) MultiDrawIndexedIndirectCount(
	indirectBuffer *Buffer, indirectOffset uint64,
	countBuffer *Buffer, countOffset uint64, maxDrawCount uint32,
) {
	p.executeIndirectCountDraw(indirectCountDrawConfig{
		validateDrawOp:  "DrawIndexedIndirect",
		featureResource: "MultiDrawIndexedIndirectCount",
		indirectBuffer:  indirectBuffer,
		indirectOffset:  indirectOffset,
		countBuffer:     countBuffer,
		countOffset:     countOffset,
		maxDrawCount:    maxDrawCount,
		recordStride:    drawIndexedIndirectRecordSize,
		preValidate:     p.validateIndexedIndirectCountPreconditions,
		record: func() {
			p.core.DrawIndexedIndirectCount(
				indirectBuffer.coreBuffer(), indirectOffset,
				countBuffer.coreBuffer(), countOffset, maxDrawCount,
			)
		},
	})
}

func (p *RenderPassEncoder) validateIndirectCountBuffers(
	indirectBuffer *Buffer, indirectOffset uint64,
	countBuffer *Buffer, countOffset uint64,
	maxDrawCount uint32, recordStride uint64,
) error {
	if indirectBuffer == nil {
		return fmt.Errorf("wgpu: RenderPass.DrawIndirect: indirect buffer is nil")
	}
	if countBuffer == nil {
		return fmt.Errorf("wgpu: RenderPass.DrawIndirect: count buffer is nil")
	}
	if indirectBuffer.Usage()&BufferUsageIndirect == 0 {
		return fmt.Errorf(
			"wgpu: RenderPass.DrawIndirect: buffer %q missing BufferUsageIndirect usage: %w",
			indirectBuffer.Label(), ErrDrawIndirectBufferUsage)
	}
	if countBuffer.Usage()&BufferUsageIndirect == 0 {
		return fmt.Errorf(
			"wgpu: RenderPass.DrawIndirect: count buffer %q missing BufferUsageIndirect usage: %w",
			countBuffer.Label(), ErrDrawIndirectBufferUsage)
	}
	if indirectOffset%4 != 0 {
		return fmt.Errorf(
			"wgpu: RenderPass.DrawIndirect: offset %d is not 4-byte aligned: %w",
			indirectOffset, ErrDrawIndirectOffsetAlignment)
	}
	if countOffset%4 != 0 {
		return fmt.Errorf(
			"wgpu: RenderPass.DrawIndirect: count offset %d is not 4-byte aligned: %w",
			countOffset, ErrDrawIndirectOffsetAlignment)
	}
	if countOffset+4 > countBuffer.Size() {
		return fmt.Errorf(
			"wgpu: RenderPass.DrawIndirect: count offset %d + 4 exceeds count buffer size %d: %w",
			countOffset, countBuffer.Size(), ErrDrawIndirectBufferOverrun)
	}
	if recordStride == drawIndirectRecordSize {
		if !drawIndirectRangeFits(indirectBuffer.Size(), indirectOffset, maxDrawCount) {
			return fmt.Errorf(
				"wgpu: RenderPass.MultiDrawIndirectCount: offset %d + max %d draw(s) exceeds buffer size %d: %w",
				indirectOffset, maxDrawCount, indirectBuffer.Size(), ErrDrawIndirectBufferOverrun)
		}
	} else if !indexedIndirectRangeFits(indirectBuffer.Size(), indirectOffset, maxDrawCount) {
		return fmt.Errorf(
			"wgpu: RenderPass.MultiDrawIndexedIndirectCount: offset %d + max %d draw(s) exceeds buffer size %d: %w",
			indirectOffset, maxDrawCount, indirectBuffer.Size(), ErrDrawIndirectBufferOverrun)
	}
	return nil
}

// End ends the render pass.
// After this call, the encoder cannot be used again.
func (p *RenderPassEncoder) End() error {
	return p.core.End()
}
