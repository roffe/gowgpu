// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"fmt"
	"math/bits"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
	"github.com/roffe/gowgpu/internal/indirect"
)

// CommandAllocator wraps a D3D12 command allocator.
type CommandAllocator struct {
	raw *d3d12.ID3D12CommandAllocator
}

// CommandBuffer holds a recorded D3D12 command list.
// Allocators are managed by Device frame tracking, not by CommandBuffer.
type CommandBuffer struct {
	cmdList          *d3d12.ID3D12GraphicsCommandList
	stateSummaries   []commandStateSummary
	preambleBarriers []stateBarrierPlan // populated by Queue.Submit for diagnostics/replay
}

// Destroy releases the command buffer's command list.
func (c *CommandBuffer) Destroy() {
	if c.cmdList != nil {
		c.cmdList.Release()
		c.cmdList = nil
	}
}

// CommandEncoder implements hal.CommandEncoder for DirectX 12.
//
// Matches Rust wgpu-hal architecture: each encoder OWNS its ID3D12CommandAllocator
// permanently. Command lists are pooled in freeLists and reused via Reset.
// The allocator is only Reset when the encoder is returned to the pool after
// GPU completion (via ResetAll). This avoids per-frame allocator churn that
// causes TDR on Intel Iris Xe.
type CommandEncoder struct {
	device      *Device
	allocator   *d3d12.ID3D12CommandAllocator // Owned permanently by this encoder
	cmdList     *d3d12.ID3D12GraphicsCommandList
	freeLists   []*d3d12.ID3D12GraphicsCommandList // Pool of reusable command lists
	label       string
	isRecording bool

	// descriptorHeaps is a pre-allocated array for SetDescriptorHeaps calls,
	// avoiding a slice allocation per render/compute pass. At most 2 heaps
	// are needed: one for CBV/SRV/UAV views and one for samplers.
	descriptorHeaps     [2]*d3d12.ID3D12DescriptorHeap
	descriptorHeapCount int

	stateTracker commandStateTracker

	// scratch are the buffers recorded copies go through, kept until the
	// GPU is done with the command lists (ResetAll).
	scratch []*Buffer

	// err is the first command that could not be recorded; EndEncoding
	// returns it in place of a command buffer that lacks the command.
	err error
}

func (e *CommandEncoder) fail(err error) {
	if e.err == nil {
		e.err = err
	}
}

// scratchBuffer is a new zeroed buffer of size bytes for a copy to go
// through.
func (e *CommandEncoder) scratchBuffer(size uint64) (*Buffer, error) {
	buf, err := e.device.CreateBuffer(&hal.BufferDescriptor{
		Label: "(wgpu internal) copy scratch",
		Size:  size,
		Usage: gputypes.BufferUsageCopySrc | gputypes.BufferUsageCopyDst,
	})
	if err != nil {
		return nil, err
	}
	b := buf.(*Buffer)
	e.scratch = append(e.scratch, b)
	return b, nil
}

func (e *CommandEncoder) releaseScratch() {
	for _, b := range e.scratch {
		b.Destroy()
	}
	e.scratch = e.scratch[:0]
}

// BeginEncoding begins command recording.
// Reuses a command list from freeLists (Reset with owned allocator) or creates new.
// The allocator is permanently owned by this encoder — not acquired per-call.
func (e *CommandEncoder) BeginEncoding(label string) error {
	e.label = label
	e.err = nil

	// Try reusing a command list from the free pool.
	if len(e.freeLists) > 0 {
		list := e.freeLists[len(e.freeLists)-1]
		e.freeLists = e.freeLists[:len(e.freeLists)-1]
		if err := list.Reset(e.allocator, nil); err == nil {
			e.cmdList = list
			e.isRecording = true
			e.stateTracker.reset()
			return nil
		}
		// Reset failed — discard this list, try next or create new.
		list.Release()
	}

	// No reusable lists — create a new one with our owned allocator.
	cmdList, err := e.device.raw.CreateCommandList(0, d3d12.D3D12_COMMAND_LIST_TYPE_DIRECT, e.allocator, nil)
	if err != nil {
		return fmt.Errorf("dx12: CreateCommandList failed: %w", err)
	}
	e.cmdList = cmdList
	e.isRecording = true
	e.stateTracker.reset()
	return nil
}

// EndEncoding finishes command recording and returns a command buffer.
// The command list is detached from the encoder — it will be returned
// to freeLists when ResetAll is called after GPU completion.
func (e *CommandEncoder) EndEncoding() (hal.CommandBuffer, error) {
	if !e.isRecording {
		return nil, fmt.Errorf("dx12: command encoder is not recording")
	}
	if e.err != nil {
		e.DiscardEncoding()
		return nil, e.err
	}

	if hal.DebugEnabled() {
		hal.Logger().Debug("dx12: command list close", "label", e.label)
	}
	if err := e.cmdList.Close(); err != nil {
		return nil, fmt.Errorf("dx12: command list close failed: %w", err)
	}

	e.isRecording = false
	cb := &CommandBuffer{cmdList: e.cmdList, stateSummaries: e.stateTracker.summary()}
	e.cmdList = nil // Detach — owned by CommandBuffer until ResetAll returns it.
	return cb, nil
}

// DiscardEncoding discards the encoder without creating a command buffer.
// The closed list goes back on freeLists for BeginEncoding to reset, since
// BeginEncoding overwrites cmdList without releasing what was there.
func (e *CommandEncoder) DiscardEncoding() {
	if e.isRecording {
		// Close the command list even though we're discarding it
		_ = e.cmdList.Close()
		e.isRecording = false
	}
	if e.cmdList != nil {
		e.freeLists = append(e.freeLists, e.cmdList)
		e.cmdList = nil
	}
}

// ResetAll resets the encoder's allocator and returns command lists to the free pool.
// Called after GPU confirms completion of all commands recorded by this encoder.
// Matches Rust wgpu-hal CommandEncoder::reset_all (command.rs:442-450).
func (e *CommandEncoder) ResetAll(commandBuffers []hal.CommandBuffer) {
	// Return command lists to the free pool for reuse.
	for _, cb := range commandBuffers {
		if dx12CB, ok := cb.(*CommandBuffer); ok && dx12CB.cmdList != nil {
			e.freeLists = append(e.freeLists, dx12CB.cmdList)
			dx12CB.cmdList = nil // Prevent double-free via CommandBuffer.Destroy.
		}
	}
	// Reset the owned allocator. This frees the internal memory used by
	// previously recorded commands. Safe because GPU is done with them.
	if e.allocator != nil {
		if err := e.allocator.Reset(); err != nil {
			hal.Logger().Error("dx12: ID3D12CommandAllocator::Reset failed", "err", err)
		}
	}
	e.releaseScratch()
}

// Destroy releases the allocator and all pooled command lists.
// Must be called when the encoder is permanently retired (e.g., device shutdown).
func (e *CommandEncoder) Destroy() {
	for _, list := range e.freeLists {
		list.Release()
	}
	e.freeLists = nil
	if e.cmdList != nil {
		e.cmdList.Release()
		e.cmdList = nil
	}
	if e.allocator != nil {
		e.allocator.Release()
		e.allocator = nil
	}
	e.releaseScratch()
}

// TransitionBuffers transitions buffer states for synchronization.
func (e *CommandEncoder) TransitionBuffers(barriers []hal.BufferBarrier) {
	if !e.isRecording || len(barriers) == 0 {
		return
	}

	plans := make([]stateBarrierPlan, 0, len(barriers))
	for _, b := range barriers {
		buf, ok := b.Buffer.(*Buffer)
		if !ok || buf.raw == nil {
			hal.Logger().Warn("TransitionBuffers: skipping invalid buffer (nil or destroyed)")
			continue
		}

		afterState := bufferBarrierState(b.Usage.NewUsage, b.Usage.New)
		beforeState, needsBarrier := e.stateTracker.transitionBuffer(buf, afterState)
		if needsBarrier || keepsUnorderedAccess(beforeState, afterState) {
			plans = append(plans, stateBarrierPlan{resource: buf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: beforeState, after: afterState})
		}
	}
	e.emitStateBarrierPlans(plans)
}

// TransitionTextures transitions texture states for synchronization.
func (e *CommandEncoder) TransitionTextures(barriers []hal.TextureBarrier) {
	if !e.isRecording || len(barriers) == 0 {
		return
	}

	plans := make([]stateBarrierPlan, 0, len(barriers))
	for _, b := range barriers {
		tex, ok := b.Texture.(*Texture)
		if !ok || tex.raw == nil {
			hal.Logger().Warn("TransitionTextures: skipping invalid texture (nil or destroyed)")
			continue
		}

		afterState := textureBarrierState(tex, b.Usage.NewUsage, b.Usage.New)
		plans = append(plans, e.stateTracker.transitionTextureRange(tex, textureRangeSubresources(tex, b.Range), afterState)...)
	}
	e.emitStateBarrierPlans(plans)
}

func (e *CommandEncoder) emitStateBarrierPlans(plans []stateBarrierPlan) {
	if !e.isRecording || len(plans) == 0 {
		return
	}
	d3dBarriers := make([]d3d12.D3D12_RESOURCE_BARRIER, 0, len(plans))
	for _, plan := range plans {
		var raw *d3d12.ID3D12Resource
		switch resource := plan.resource.(type) {
		case *Buffer:
			raw = resource.raw
		case *Texture:
			raw = resource.raw
		}
		if raw == nil {
			continue
		}
		switch {
		case keepsUnorderedAccess(plan.before, plan.after):
			d3dBarriers = append(d3dBarriers, d3d12.NewUAVBarrier(raw))
		case plan.before != plan.after:
			d3dBarriers = append(d3dBarriers, d3d12.NewTransitionBarrier(raw, plan.before, plan.after, plan.subresource))
		}
	}
	if len(d3dBarriers) > 0 {
		if hal.DebugEnabled() {
			hal.Logger().Debug("dx12: resource barrier", "label", e.label, "count", len(d3dBarriers))
		}
		e.cmdList.ResourceBarrier(uint32(len(d3dBarriers)), &d3dBarriers[0])
	}
}

// ClearBuffer clears a buffer region to zero. D3D12 has no fill for a buffer
// that isn't a UAV, so it copies from the device's zero buffer, a
// zeroBufferSize piece at a time, as Rust's wgpu-hal does.
func (e *CommandEncoder) ClearBuffer(buffer hal.Buffer, offset, size uint64) {
	if !e.isRecording {
		return
	}
	buf, ok := buffer.(*Buffer)
	if !ok {
		return
	}
	zero, err := e.device.zeroBuffer()
	if err != nil {
		hal.Logger().Warn("dx12: ClearBuffer: no zero buffer, the buffer keeps its contents", "err", err)
		return
	}
	regions := make([]hal.BufferCopy, 0, (size+zeroBufferSize-1)/zeroBufferSize)
	for end := offset + size; offset < end; offset += zeroBufferSize {
		regions = append(regions, hal.BufferCopy{DstOffset: offset, Size: min(end-offset, zeroBufferSize)})
	}
	e.CopyBufferToBuffer(zero, buf, regions)
}

// CopyBufferToBuffer copies data between buffers.
// Inserts D3D12_RESOURCE_BARRIER transitions when buffers are not already in
// the required state (COPY_SOURCE for src, COPY_DEST for dst). This is the
// DX12-specific fix for BUG-DX12-012: missing UNORDERED_ACCESS -> COPY_SOURCE
// barrier after compute dispatch causes DEVICE_REMOVED.
func (e *CommandEncoder) CopyBufferToBuffer(src, dst hal.Buffer, regions []hal.BufferCopy) {
	if !e.isRecording {
		return
	}

	srcBuf, srcOk := src.(*Buffer)
	dstBuf, dstOk := dst.(*Buffer)
	if !srcOk || !dstOk {
		return
	}

	plans := make([]stateBarrierPlan, 0, 2)
	if before, needsBarrier := e.stateTracker.transitionBuffer(srcBuf, d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE); needsBarrier {
		plans = append(plans, stateBarrierPlan{resource: srcBuf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE})
	}
	if before, needsBarrier := e.stateTracker.transitionBuffer(dstBuf, d3d12.D3D12_RESOURCE_STATE_COPY_DEST); needsBarrier {
		plans = append(plans, stateBarrierPlan{resource: dstBuf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_DEST})
	}
	e.emitStateBarrierPlans(plans)

	for _, r := range regions {
		if hal.DebugEnabled() {
			hal.Logger().Debug("dx12: copy buffer region", "label", e.label, "offset", r.DstOffset, "size", r.Size)
		}
		e.cmdList.CopyBufferRegion(dstBuf.raw, r.DstOffset, srcBuf.raw, r.SrcOffset, r.Size)
	}
}

// CopyBufferToTexture copies data from a buffer to a texture.
// Inserts a transition barrier if the source buffer is not in COPY_SOURCE state.
func (e *CommandEncoder) CopyBufferToTexture(src hal.Buffer, dst hal.Texture, regions []hal.BufferTextureCopy) {
	if !e.isRecording {
		return
	}
	srcBuf, srcOK := src.(*Buffer)
	dstTex, dstOK := dst.(*Texture)
	if !srcOK || !dstOK {
		return
	}
	e.copyBufferToTexture(srcBuf, dstTex, regions)
}

// CopyTextureToBuffer copies data from a texture to a buffer.
// Inserts a transition barrier if the destination buffer is not in COPY_DEST state.
func (e *CommandEncoder) CopyTextureToBuffer(src hal.Texture, dst hal.Buffer, regions []hal.BufferTextureCopy) {
	if !e.isRecording {
		return
	}
	srcTex, srcOK := src.(*Texture)
	dstBuf, dstOK := dst.(*Buffer)
	if !srcOK || !dstOK {
		return
	}
	e.copyTextureToBuffer(srcTex, dstBuf, regions)
}

// CopyTextureToTexture copies data between textures.
func (e *CommandEncoder) CopyTextureToTexture(src, dst hal.Texture, regions []hal.TextureCopy) {
	if !e.isRecording {
		return
	}
	srcTex, srcOK := src.(*Texture)
	dstTex, dstOK := dst.(*Texture)
	if !srcOK || !dstOK {
		return
	}
	e.copyTextureToTexture(srcTex, dstTex, regions)
}

// ResolveQuerySet copies query results from a query set into a destination buffer.
// Each timestamp result is a uint64 (8 bytes).
// Rust wgpu-hal reference: dx12/command.rs copy_query_results.
func (e *CommandEncoder) ResolveQuerySet(querySet hal.QuerySet, firstQuery, queryCount uint32, destination hal.Buffer, destinationOffset uint64) {
	if !e.isRecording {
		return
	}
	qs, ok := querySet.(*QuerySet)
	if !ok || qs == nil || qs.raw == nil {
		return
	}
	buf, ok := destination.(*Buffer)
	if !ok || buf == nil || buf.raw == nil {
		return
	}
	if before, needsBarrier := e.stateTracker.transitionBuffer(buf, d3d12.D3D12_RESOURCE_STATE_COPY_DEST); needsBarrier {
		e.emitStateBarrierPlans([]stateBarrierPlan{{resource: buf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_DEST}})
	}
	e.cmdList.ResolveQueryData(qs.raw, qs.rawTy, firstQuery, queryCount, buf.raw, destinationOffset)
}

// timestampWrites is a common interface for render/compute pass timestamp writes.
// Both hal.RenderPassTimestampWrites and hal.ComputePassTimestampWrites share
// the same fields; this interface avoids duplicating extraction logic.
type timestampWrites interface {
	querySet() hal.QuerySet
	beginIndex() *uint32
	endIndex() *uint32
}

type renderTSW struct {
	w *hal.RenderPassTimestampWrites
}

func (r renderTSW) querySet() hal.QuerySet { return r.w.QuerySet }
func (r renderTSW) beginIndex() *uint32    { return r.w.BeginningOfPassWriteIndex }
func (r renderTSW) endIndex() *uint32      { return r.w.EndOfPassWriteIndex }

type computeTSW struct {
	w *hal.ComputePassTimestampWrites
}

func (c computeTSW) querySet() hal.QuerySet { return c.w.QuerySet }
func (c computeTSW) beginIndex() *uint32    { return c.w.BeginningOfPassWriteIndex }
func (c computeTSW) endIndex() *uint32      { return c.w.EndOfPassWriteIndex }

// writeBeginTimestamp writes the beginning-of-pass timestamp (if requested) and
// returns the query heap + index for the end-of-pass timestamp write.
// DX12 uses EndQuery for timestamps (NOT begin/end pair).
// Rust wgpu-hal reference: dx12/command.rs begin_render_pass / begin_compute_pass.
func (e *CommandEncoder) writeBeginTimestamp(halQS hal.QuerySet, tw timestampWrites) (*d3d12.ID3D12QueryHeap, uint32) {
	qs, ok := halQS.(*QuerySet)
	if !ok || qs == nil || qs.raw == nil {
		return nil, 0
	}

	if idx := tw.beginIndex(); idx != nil {
		e.cmdList.EndQuery(qs.raw, d3d12.D3D12_QUERY_TYPE_TIMESTAMP, *idx)
	}

	if idx := tw.endIndex(); idx != nil {
		return qs.raw, *idx
	}
	return nil, 0
}

// BeginRenderPass begins a render pass.
func (e *CommandEncoder) BeginRenderPass(desc *hal.RenderPassDescriptor) hal.RenderPassEncoder {
	rpe := &RenderPassEncoder{
		encoder: e,
		desc:    desc,
	}

	if hal.DebugEnabled() {
		hal.Logger().Debug("dx12: begin render pass", "label", e.label, "attachments", len(desc.ColorAttachments))
	}

	if !e.isRecording {
		return rpe
	}

	// Render attachments are first-use requirements in the command-local
	// tracker. Surface views share the same owner as their swapchain back
	// buffer, so PRESENT is captured from that owner rather than guessed here.
	attachmentPlans := make([]stateBarrierPlan, 0)
	for _, ca := range desc.ColorAttachments {
		view, ok := ca.View.(*TextureView)
		if !ok || view.texture == nil || view.texture.raw == nil {
			continue
		}
		for _, subresource := range textureViewSubresources(view) {
			if before, needsBarrier := e.stateTracker.transitionTexture(view.texture, subresource, d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET); needsBarrier {
				attachmentPlans = append(attachmentPlans, stateBarrierPlan{resource: view.texture, subresource: subresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET})
			}
		}
	}
	e.emitStateBarrierPlans(attachmentPlans)

	// Set render targets
	rtvHandles := make([]d3d12.D3D12_CPU_DESCRIPTOR_HANDLE, 0, len(desc.ColorAttachments))
	for _, ca := range desc.ColorAttachments {
		view, ok := ca.View.(*TextureView)
		if !ok || !view.hasRTV {
			continue
		}
		rtvHandles = append(rtvHandles, view.rtvHandle)

		// Clear if needed
		if ca.LoadOp == gputypes.LoadOpClear {
			clearColor := [4]float32{
				float32(ca.ClearValue.R),
				float32(ca.ClearValue.G),
				float32(ca.ClearValue.B),
				float32(ca.ClearValue.A),
			}
			e.cmdList.ClearRenderTargetView(view.rtvHandle, &clearColor, 0, nil)
		}
	}

	// Handle depth/stencil attachment using helper method to reduce nesting
	dsvHandle := e.setupDepthStencilAttachment(desc.DepthStencilAttachment)

	// Set render targets
	if len(rtvHandles) > 0 {
		e.cmdList.OMSetRenderTargets(uint32(len(rtvHandles)), &rtvHandles[0], 0, dsvHandle)
	} else if dsvHandle != nil {
		e.cmdList.OMSetRenderTargets(0, nil, 0, dsvHandle)
	}

	// Set default viewport and scissor based on first color attachment or depth attachment
	var width, height uint32
	if len(desc.ColorAttachments) > 0 {
		if view, ok := desc.ColorAttachments[0].View.(*TextureView); ok {
			width, height = view.mipExtent()
		}
	} else if desc.DepthStencilAttachment != nil {
		if view, ok := desc.DepthStencilAttachment.View.(*TextureView); ok {
			width, height = view.mipExtent()
		}
	}

	if width > 0 && height > 0 {
		viewport := d3d12.D3D12_VIEWPORT{
			TopLeftX: 0,
			TopLeftY: 0,
			Width:    float32(width),
			Height:   float32(height),
			MinDepth: 0,
			MaxDepth: 1,
		}
		e.cmdList.RSSetViewports(1, &viewport)

		scissor := d3d12.D3D12_RECT{
			Left:   0,
			Top:    0,
			Right:  int32(width),
			Bottom: int32(height),
		}
		e.cmdList.RSSetScissorRects(1, &scissor)
	}

	// Write beginning-of-pass timestamp and store end-of-pass for later.
	if desc.TimestampWrites != nil {
		tw := renderTSW{desc.TimestampWrites}
		rpe.endOfPassTimerHeap, rpe.endOfPassTimerIndex = e.writeBeginTimestamp(tw.querySet(), tw)
	}

	return rpe
}

// BeginComputePass begins a compute pass.
func (e *CommandEncoder) BeginComputePass(desc *hal.ComputePassDescriptor) hal.ComputePassEncoder {
	cpe := &ComputePassEncoder{
		encoder: e,
	}

	if e.isRecording && desc != nil && desc.TimestampWrites != nil {
		tw := computeTSW{desc.TimestampWrites}
		cpe.endOfPassTimerHeap, cpe.endOfPassTimerIndex = e.writeBeginTimestamp(tw.querySet(), tw)
	}

	return cpe
}

// RenderPassEncoder implements hal.RenderPassEncoder for DirectX 12.
type RenderPassEncoder struct {
	encoder            *CommandEncoder
	desc               *hal.RenderPassDescriptor
	pipeline           *RenderPipeline
	indexFormat        gputypes.IndexFormat
	descriptorHeapsSet bool // Tracks whether descriptor heaps have been bound
	groups             boundGroups

	// vertexBuffers are the views SetVertexBuffer made, with the strides of
	// the pipeline set last. A draw sets the ones dirtyVertexBuffers marks
	// first, so the stride is the drawing pipeline's even when the buffer
	// came before it, as WebGPU allows. Rust's wgpu-hal does the same.
	vertexBuffers      [32]d3d12.D3D12_VERTEX_BUFFER_VIEW // D3D12_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT
	dirtyVertexBuffers uint32

	// endOfPassTimerQuery stores the query heap and index for the end-of-pass
	// timestamp write. Set during BeginRenderPass, consumed in End().
	// Rust wgpu-hal reference: dx12/mod.rs end_of_pass_timer_query field.
	endOfPassTimerHeap  *d3d12.ID3D12QueryHeap
	endOfPassTimerIndex uint32
}

// End finishes the render pass.
// Handles MSAA resolve (if ResolveTarget is set), writes end-of-pass
// timestamp, and transitions surface textures back to PRESENT state.
func (e *RenderPassEncoder) End() {
	if e.desc == nil || e.encoder == nil || !e.encoder.isRecording {
		return
	}

	// Write end-of-pass timestamp before state transitions.
	// Rust wgpu-hal reference: dx12/command.rs end_render_pass calls
	// write_pass_end_timestamp_if_requested after resolves but before end_pass.
	if e.endOfPassTimerHeap != nil {
		e.encoder.cmdList.EndQuery(e.endOfPassTimerHeap, d3d12.D3D12_QUERY_TYPE_TIMESTAMP, e.endOfPassTimerIndex)
		e.endOfPassTimerHeap = nil
	}

	for _, ca := range e.desc.ColorAttachments {
		msaaView, ok := ca.View.(*TextureView)
		if !ok || msaaView.texture == nil || msaaView.texture.raw == nil {
			continue
		}

		// Check for MSAA resolve target.
		resolveView, _ := ca.ResolveTarget.(*TextureView)
		if resolveView != nil && resolveView.texture != nil && resolveView.texture.raw != nil && msaaView.texture.samples > 1 {
			// Determine resolve target's resting state based on ownership.
			// Surface (swapchain) textures live in PRESENT between frames.
			// Internal (offscreen) textures live in RENDER_TARGET, matching
			// the initial state set by CreateTexture for RenderAttachment usage.
			resolveRestState := d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET
			if resolveView.texture.isExternal {
				resolveRestState = d3d12.D3D12_RESOURCE_STATE_PRESENT
			}

			// MSAA resolve: render target → resolve source, resolve target → resolve dest.
			msaaSubresources := textureViewSubresources(msaaView)
			resolveSubresources := textureViewSubresources(resolveView)
			plans := make([]stateBarrierPlan, 0, len(msaaSubresources)+len(resolveSubresources))
			for _, subresource := range msaaSubresources {
				if before, needsBarrier := e.encoder.stateTracker.transitionTexture(msaaView.texture, subresource, d3d12.D3D12_RESOURCE_STATE_RESOLVE_SOURCE); needsBarrier {
					plans = append(plans, stateBarrierPlan{resource: msaaView.texture, subresource: subresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_RESOLVE_SOURCE})
				}
			}
			for _, subresource := range resolveSubresources {
				if before, needsBarrier := e.encoder.stateTracker.transitionTexture(resolveView.texture, subresource, d3d12.D3D12_RESOURCE_STATE_RESOLVE_DEST); needsBarrier {
					plans = append(plans, stateBarrierPlan{resource: resolveView.texture, subresource: subresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_RESOLVE_DEST})
				}
			}
			e.encoder.emitStateBarrierPlans(plans)

			// Resolve MSAA → single-sample.
			format := textureFormatToD3D12(msaaView.texture.format)
			e.encoder.cmdList.ResolveSubresource(
				resolveView.texture.raw, resolveSubresourceIndex(resolveView),
				msaaView.texture.raw, resolveSubresourceIndex(msaaView),
				format,
			)

			// Return the resources to their command-local resting states.
			plans = plans[:0]
			for _, subresource := range msaaSubresources {
				if before, needsBarrier := e.encoder.stateTracker.transitionTexture(msaaView.texture, subresource, d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET); needsBarrier {
					plans = append(plans, stateBarrierPlan{resource: msaaView.texture, subresource: subresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET})
				}
			}
			for _, subresource := range resolveSubresources {
				if before, needsBarrier := e.encoder.stateTracker.transitionTexture(resolveView.texture, subresource, resolveRestState); needsBarrier {
					plans = append(plans, stateBarrierPlan{resource: resolveView.texture, subresource: subresource, before: before, after: resolveRestState})
				}
			}
			e.encoder.emitStateBarrierPlans(plans)
			continue
		}

		// No resolve — just transition external surface back to PRESENT.
		if msaaView.texture.isExternal {
			for _, subresource := range textureViewSubresources(msaaView) {
				if before, needsBarrier := e.encoder.stateTracker.transitionTexture(msaaView.texture, subresource, d3d12.D3D12_RESOURCE_STATE_PRESENT); needsBarrier {
					e.encoder.emitStateBarrierPlans([]stateBarrierPlan{{resource: msaaView.texture, subresource: subresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_PRESENT}})
				}
			}
		}
	}
}

// SetPipeline sets the render pipeline.
func (e *RenderPassEncoder) SetPipeline(pipeline hal.RenderPipeline) {
	p, ok := pipeline.(*RenderPipeline)
	if !ok || !e.encoder.isRecording {
		return
	}
	prev := e.pipeline
	e.pipeline = p

	e.encoder.cmdList.SetPipelineState(p.pso)
	if p.rootSignature != nil {
		e.encoder.cmdList.SetGraphicsRootSignature(p.rootSignature)
	}
	e.encoder.cmdList.IASetPrimitiveTopology(p.topology)
	if prev == nil || prev.rootSignature != p.rootSignature {
		e.groups.bind(e.encoder, false, p.groupMappings)
	}
	for slot, stride := range p.vertexStrides {
		if slot < len(e.vertexBuffers) && e.vertexBuffers[slot].StrideInBytes != stride {
			e.vertexBuffers[slot].StrideInBytes = stride
			e.dirtyVertexBuffers |= 1 << slot
		}
	}

	// Bind the global sampler descriptor heap to the sampler root parameter.
	// This is the ONE root parameter that covers all 2048+2048 sampler slots.
	if p.samplerRootIndex >= 0 && e.encoder.device.samplerHeap != nil {
		e.encoder.ensureDescriptorHeapsBound()
		e.descriptorHeapsSet = true
		e.encoder.cmdList.SetGraphicsRootDescriptorTable(
			uint32(p.samplerRootIndex),
			e.encoder.device.samplerHeap.gpuStart,
		)
	}
}

// SetBindGroup sets a bind group for graphics operations.
// index is the bind group slot (0-3 typically).
// group contains the GPU descriptor handles for resources.
// offsets are dynamic buffer offsets (used for dynamic uniform/storage buffers).
func (e *RenderPassEncoder) SetBindGroup(index uint32, group hal.BindGroup, offsets []uint32) {
	bg, ok := group.(*BindGroup)
	if !ok || !e.encoder.isRecording {
		return
	}

	// Ensure descriptor heaps are bound before setting descriptor tables.
	if !e.descriptorHeapsSet {
		e.encoder.ensureDescriptorHeapsBound()
		e.descriptorHeapsSet = true
	}

	e.groups.set(index, bg, offsets)
	if e.pipeline != nil {
		e.encoder.bindGroupToRoot(index, bg, offsets, false, e.pipeline.groupMappings)
	}
	e.encoder.trackBindGroupState(bg)
}

// SetVertexBuffer sets a vertex buffer. Its view gets the stride of the
// pipeline that draws with it, see vertexBuffers.
func (e *RenderPassEncoder) SetVertexBuffer(slot uint32, buffer hal.Buffer, offset uint64) {
	buf, ok := buffer.(*Buffer)
	if !ok || !e.encoder.isRecording || slot >= uint32(len(e.vertexBuffers)) {
		return
	}
	if before, target, needsBarrier := e.encoder.stateTracker.transitionBufferRead(buf, d3d12.D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER); needsBarrier {
		e.encoder.emitStateBarrierPlans([]stateBarrierPlan{{resource: buf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: target}})
	}

	vbv := &e.vertexBuffers[slot]
	vbv.BufferLocation = buf.gpuVA + offset
	vbv.SizeInBytes = uint32(buf.size - offset)
	e.dirtyVertexBuffers |= 1 << slot
}

// setVertexBuffers sets the vertex buffer views that changed since the
// last draw.
func (e *RenderPassEncoder) setVertexBuffers() {
	for e.dirtyVertexBuffers != 0 {
		slot := bits.TrailingZeros32(e.dirtyVertexBuffers)
		e.dirtyVertexBuffers &^= 1 << slot
		e.encoder.cmdList.IASetVertexBuffers(uint32(slot), 1, &e.vertexBuffers[slot])
	}
}

// SetIndexBuffer sets the index buffer.
func (e *RenderPassEncoder) SetIndexBuffer(buffer hal.Buffer, format gputypes.IndexFormat, offset uint64) {
	buf, ok := buffer.(*Buffer)
	if !ok || !e.encoder.isRecording {
		return
	}
	if before, target, needsBarrier := e.encoder.stateTracker.transitionBufferRead(buf, d3d12.D3D12_RESOURCE_STATE_INDEX_BUFFER); needsBarrier {
		e.encoder.emitStateBarrierPlans([]stateBarrierPlan{{resource: buf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: target}})
	}

	e.indexFormat = format
	dxgiFormat := d3d12.DXGI_FORMAT_R16_UINT
	if format == gputypes.IndexFormatUint32 {
		dxgiFormat = d3d12.DXGI_FORMAT_R32_UINT
	}

	ibv := d3d12.D3D12_INDEX_BUFFER_VIEW{
		BufferLocation: buf.gpuVA + offset,
		SizeInBytes:    uint32(buf.size - offset),
		Format:         dxgiFormat,
	}

	e.encoder.cmdList.IASetIndexBuffer(&ibv)
}

// SetViewport sets the viewport.
func (e *RenderPassEncoder) SetViewport(vp gputypes.Viewport) {
	if !e.encoder.isRecording {
		return
	}

	viewport := d3d12.D3D12_VIEWPORT{
		TopLeftX: vp.X,
		TopLeftY: vp.Y,
		Width:    vp.Width,
		Height:   vp.Height,
		MinDepth: vp.MinDepth,
		MaxDepth: vp.MaxDepth,
	}

	e.encoder.cmdList.RSSetViewports(1, &viewport)
}

// SetScissorRect sets the scissor rectangle.
func (e *RenderPassEncoder) SetScissorRect(rect gputypes.ScissorRect) {
	if !e.encoder.isRecording {
		return
	}

	scissor := d3d12.D3D12_RECT{
		Left:   int32(rect.X),
		Top:    int32(rect.Y),
		Right:  int32(rect.X + rect.Width),
		Bottom: int32(rect.Y + rect.Height),
	}

	e.encoder.cmdList.RSSetScissorRects(1, &scissor)
}

// SetBlendConstant sets the blend constant.
func (e *RenderPassEncoder) SetBlendConstant(color *gputypes.Color) {
	if !e.encoder.isRecording || color == nil {
		return
	}

	blendFactor := [4]float32{
		float32(color.R),
		float32(color.G),
		float32(color.B),
		float32(color.A),
	}

	e.encoder.cmdList.OMSetBlendFactor(&blendFactor)
}

// SetStencilReference sets the stencil reference value.
func (e *RenderPassEncoder) SetStencilReference(ref uint32) {
	if !e.encoder.isRecording {
		return
	}

	e.encoder.cmdList.OMSetStencilRef(ref)
}

// Draw draws primitives.
func (e *RenderPassEncoder) Draw(args gputypes.DrawArgs) {
	if !e.encoder.isRecording {
		return
	}

	e.setVertexBuffers()
	e.encoder.cmdList.DrawInstanced(args.VertexCount, args.InstanceCount, args.FirstVertex, args.FirstInstance)
}

// DrawIndexed draws indexed primitives.
func (e *RenderPassEncoder) DrawIndexed(args gputypes.DrawIndexedArgs) {
	if !e.encoder.isRecording {
		return
	}

	e.setVertexBuffers()
	e.encoder.cmdList.DrawIndexedInstanced(args.IndexCount, args.InstanceCount, args.FirstIndex, args.BaseVertex, args.FirstInstance)
}

// DrawIndirect draws primitives with GPU-generated parameters.
// The buffer must contain a D3D12_DRAW_ARGUMENTS struct at the given offset
// (vertexCountPerInstance, instanceCount, startVertexLocation, startInstanceLocation — 16 bytes).
func (e *RenderPassEncoder) DrawIndirect(buffer hal.Buffer, offset uint64, drawCount uint32) {
	buf, ok := buffer.(*Buffer)
	if !ok || !e.encoder.isRecording || drawCount == 0 {
		return
	}
	if !indirect.RangeFits(buf.size, offset, 16, drawCount) {
		return
	}
	if _, ok := indirect.RecordOffset(offset, 16, drawCount-1); !ok {
		return
	}
	if before, target, needsBarrier := e.encoder.stateTracker.transitionBufferRead(buf, d3d12.D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT); needsBarrier {
		e.encoder.emitStateBarrierPlans([]stateBarrierPlan{{resource: buf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: target}})
	}

	e.setVertexBuffers()
	e.encoder.cmdList.ExecuteIndirect(
		e.encoder.device.cmdSignatures.draw,
		drawCount, buf.raw, offset, nil, 0,
	)
}

// DrawIndexedIndirect draws indexed primitives with GPU-generated parameters.
// The buffer must contain a D3D12_DRAW_INDEXED_ARGUMENTS struct at the given offset
// (indexCountPerInstance, instanceCount, startIndexLocation, baseVertexLocation, startInstanceLocation — 20 bytes).
func (e *RenderPassEncoder) DrawIndexedIndirect(buffer hal.Buffer, offset uint64, drawCount uint32) {
	buf, ok := buffer.(*Buffer)
	if !ok || !e.encoder.isRecording || drawCount == 0 {
		return
	}
	if !indirect.RangeFits(buf.size, offset, 20, drawCount) {
		return
	}
	if _, ok := indirect.RecordOffset(offset, 20, drawCount-1); !ok {
		return
	}
	if before, target, needsBarrier := e.encoder.stateTracker.transitionBufferRead(buf, d3d12.D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT); needsBarrier {
		e.encoder.emitStateBarrierPlans([]stateBarrierPlan{{resource: buf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: target}})
	}

	e.setVertexBuffers()
	e.encoder.cmdList.ExecuteIndirect(
		e.encoder.device.cmdSignatures.drawIndexed,
		drawCount, buf.raw, offset, nil, 0,
	)
}

// DrawIndirectCount draws using maxDrawCount when GPU count buffer is unavailable.
func (e *RenderPassEncoder) DrawIndirectCount(
	buffer hal.Buffer, offset uint64,
	countBuffer hal.Buffer, countOffset uint64,
	maxDrawCount uint32,
) {
	hal.RecordIndirectCountMax(e.DrawIndirect, buffer, offset, countBuffer, countOffset, maxDrawCount)
}

// DrawIndexedIndirectCount draws using maxDrawCount when GPU count buffer is unavailable.
func (e *RenderPassEncoder) DrawIndexedIndirectCount(
	buffer hal.Buffer, offset uint64,
	countBuffer hal.Buffer, countOffset uint64,
	maxDrawCount uint32,
) {
	hal.RecordIndirectCountMax(e.DrawIndexedIndirect, buffer, offset, countBuffer, countOffset, maxDrawCount)
}

// ExecuteBundle executes a pre-recorded render bundle.
func (e *RenderPassEncoder) ExecuteBundle(bundle hal.RenderBundle) {
	// Note: DX12 bundles use ID3D12GraphicsCommandList created with D3D12_COMMAND_LIST_TYPE_BUNDLE.
	_ = bundle
}

// ComputePassEncoder implements hal.ComputePassEncoder for DirectX 12.
type ComputePassEncoder struct {
	encoder            *CommandEncoder
	pipeline           *ComputePipeline
	descriptorHeapsSet bool // Tracks whether descriptor heaps have been bound
	groups             boundGroups

	// boundStorageBuffers tracks storage buffers from SetBindGroup calls.
	// Dispatch records them as UNORDERED_ACCESS in the command-local tracker so
	// a later copy in the same command list can transition from the UAV state.
	//
	// Matches Rust wgpu-core pattern: compute pass tracks buffer usage per
	// dispatch via BufferUsageScope, then drains barriers on state transitions
	// (wgpu-core/src/command/compute.rs:326-377).
	boundStorageBuffers []*Buffer

	// endOfPassTimerQuery stores the query heap and index for the end-of-pass
	// timestamp write. Set during BeginComputePass, consumed in End().
	endOfPassTimerHeap  *d3d12.ID3D12QueryHeap
	endOfPassTimerIndex uint32
}

// End finishes the compute pass.
// Writes end-of-pass timestamp if requested.
// Rust wgpu-hal reference: dx12/command.rs end_compute_pass.
func (e *ComputePassEncoder) End() {
	if e.encoder == nil || !e.encoder.isRecording {
		return
	}
	if e.endOfPassTimerHeap != nil {
		e.encoder.cmdList.EndQuery(e.endOfPassTimerHeap, d3d12.D3D12_QUERY_TYPE_TIMESTAMP, e.endOfPassTimerIndex)
		e.endOfPassTimerHeap = nil
	}
}

// SetPipeline sets the compute pipeline.
func (e *ComputePassEncoder) SetPipeline(pipeline hal.ComputePipeline) {
	p, ok := pipeline.(*ComputePipeline)
	if !ok || !e.encoder.isRecording {
		return
	}
	prev := e.pipeline
	e.pipeline = p

	e.encoder.cmdList.SetPipelineState(p.pso)
	if p.rootSignature != nil {
		e.encoder.cmdList.SetComputeRootSignature(p.rootSignature)
	}
	if prev == nil || prev.rootSignature != p.rootSignature {
		e.groups.bind(e.encoder, true, p.groupMappings)
	}

	// Bind the global sampler descriptor heap to the sampler root parameter.
	if p.samplerRootIndex >= 0 && e.encoder.device.samplerHeap != nil {
		e.encoder.ensureDescriptorHeapsBound()
		e.descriptorHeapsSet = true
		e.encoder.cmdList.SetComputeRootDescriptorTable(
			uint32(p.samplerRootIndex),
			e.encoder.device.samplerHeap.gpuStart,
		)
	}
}

// SetBindGroup sets a bind group for compute operations.
// index is the bind group slot (0-3 typically).
// group contains the GPU descriptor handles for resources.
// offsets are dynamic buffer offsets (used for dynamic uniform/storage buffers).
func (e *ComputePassEncoder) SetBindGroup(index uint32, group hal.BindGroup, offsets []uint32) {
	bg, ok := group.(*BindGroup)
	if !ok || !e.encoder.isRecording {
		return
	}

	// Ensure descriptor heaps are bound before setting descriptor tables.
	if !e.descriptorHeapsSet {
		e.encoder.ensureDescriptorHeapsBound()
		e.descriptorHeapsSet = true
	}

	e.groups.set(index, bg, offsets)
	if e.pipeline != nil {
		e.encoder.bindGroupToRoot(index, bg, offsets, true, e.pipeline.groupMappings)
	}
	e.encoder.trackBindGroupState(bg)

	// Track storage buffers from this bind group for state tracking.
	// After Dispatch(), these buffers will be marked as UNORDERED_ACCESS
	// so that subsequent copy commands can insert correct transition barriers.
	// storageBuffers is populated at CreateBindGroup time by matching layout
	// entry types (BindingTypeStorageBuffer) with buffer binding handles.
	e.boundStorageBuffers = append(e.boundStorageBuffers, bg.storageBuffers...)
}

// Dispatch dispatches compute work.
func (e *ComputePassEncoder) Dispatch(x, y, z uint32) {
	if !e.encoder.isRecording {
		return
	}

	plans := make([]stateBarrierPlan, 0, len(e.boundStorageBuffers))
	for _, buf := range e.boundStorageBuffers {
		if before, needsBarrier := e.encoder.stateTracker.transitionBuffer(buf, d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS); needsBarrier {
			plans = append(plans, stateBarrierPlan{resource: buf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS})
		}
	}
	e.encoder.emitStateBarrierPlans(plans)
	e.encoder.cmdList.Dispatch(x, y, z)
	e.insertUAVBarrier()

	// Mark all bound storage buffers as UNORDERED_ACCESS. After a compute
	// dispatch, storage buffers are left in UAV state. This enables correct
	// transition barriers when subsequent commands (e.g., CopyBufferToBuffer)
	// need these buffers in a different state (e.g., COPY_SOURCE).
	// Matches Rust wgpu-core pattern: flush_bindings sets BufferUses::STORAGE_READ_WRITE
	// on bound storage buffers, then drain_barriers emits transitions.
	// Clear for next dispatch (same pattern as Rust per-dispatch usage scope).
	e.boundStorageBuffers = e.boundStorageBuffers[:0]
}

// DispatchIndirect dispatches compute work with GPU-generated parameters.
// The buffer must contain a D3D12_DISPATCH_ARGUMENTS struct at the given offset
// (threadGroupCountX, threadGroupCountY, threadGroupCountZ — 12 bytes).
func (e *ComputePassEncoder) DispatchIndirect(buffer hal.Buffer, offset uint64) {
	buf, ok := buffer.(*Buffer)
	if !ok || !e.encoder.isRecording {
		return
	}
	plans := make([]stateBarrierPlan, 0, 1+len(e.boundStorageBuffers))
	if before, target, needsBarrier := e.encoder.stateTracker.transitionBufferRead(buf, d3d12.D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT); needsBarrier {
		plans = append(plans, stateBarrierPlan{resource: buf, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: target})
	}
	for _, storage := range e.boundStorageBuffers {
		if before, needsBarrier := e.encoder.stateTracker.transitionBuffer(storage, d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS); needsBarrier {
			plans = append(plans, stateBarrierPlan{resource: storage, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS})
		}
	}
	e.encoder.emitStateBarrierPlans(plans)

	e.encoder.cmdList.ExecuteIndirect(
		e.encoder.device.cmdSignatures.dispatch,
		1, buf.raw, offset, nil, 0,
	)
	e.insertUAVBarrier()

	// Mark bound storage buffers as UNORDERED_ACCESS, matching Dispatch() pattern.
	// After an indirect dispatch, storage buffers are left in UAV state for correct
	// transition barriers on subsequent commands.
	e.boundStorageBuffers = e.boundStorageBuffers[:0]
}

// insertUAVBarrier inserts a global UAV barrier after a dispatch.
// VAL-008: ensures UAV writes from one dispatch are visible to subsequent
// dispatches. NULL resource = global barrier across all UAV resources.
func (e *ComputePassEncoder) insertUAVBarrier() {
	barrier := d3d12.NewUAVBarrier(nil)
	e.encoder.cmdList.ResourceBarrier(1, &barrier)
}

// --- Buffer state transition helpers ---

// needsExplicitBarrier returns true if transitioning from the given current state
// to the target state requires an explicit D3D12_RESOURCE_BARRIER.
//
// DX12 implicit promotion rules (D3D12 spec, "Using Resource Barriers"):
//   - COMMON state allows implicit promotion to: COPY_DEST, VERTEX_AND_CONSTANT_BUFFER,
//     INDEX_BUFFER, NON_PIXEL_SHADER_RESOURCE, PIXEL_SHADER_RESOURCE, INDIRECT_ARGUMENT.
//   - COMMON does NOT allow implicit promotion to: COPY_SOURCE, UNORDERED_ACCESS.
//   - Same state -> same state: no barrier needed.
func needsExplicitBarrier(current, target d3d12.D3D12_RESOURCE_STATES) bool {
	if current == target {
		return false
	}
	// COMMON state allows implicit promotion to several read states and COPY_DEST,
	// but NOT to COPY_SOURCE or UNORDERED_ACCESS.
	if current == d3d12.D3D12_RESOURCE_STATE_COMMON {
		switch target {
		case d3d12.D3D12_RESOURCE_STATE_COPY_DEST,
			d3d12.D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
			d3d12.D3D12_RESOURCE_STATE_INDEX_BUFFER,
			d3d12.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
			d3d12.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
			d3d12.D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT:
			return false // Implicit promotion allowed
		}
	}
	return true
}

func resolveSubresourceIndex(view *TextureView) uint32 {
	if view == nil || view.texture == nil {
		return 0
	}
	if subresources := textureViewSubresources(view); len(subresources) > 0 {
		return subresources[0]
	}
	return view.texture.subresourceIndex(view.baseMip, view.baseLayer)
}

// --- Helper functions ---

// ensureDescriptorHeapsBound binds the shader-visible descriptor heaps to the command list.
// D3D12 requires descriptor heaps to be bound before setting root descriptor tables.
// This must be called before any SetBindGroup operations.
func (e *CommandEncoder) ensureDescriptorHeapsBound() {
	e.descriptorHeapCount = 0

	// Add shader-visible heaps (viewHeap for CBV/SRV/UAV, samplerHeap for samplers)
	if e.device.viewHeap != nil && e.device.viewHeap.raw != nil {
		e.descriptorHeaps[e.descriptorHeapCount] = e.device.viewHeap.raw
		e.descriptorHeapCount++
	}
	if e.device.samplerHeap != nil && e.device.samplerHeap.raw != nil {
		e.descriptorHeaps[e.descriptorHeapCount] = e.device.samplerHeap.raw
		e.descriptorHeapCount++
	}

	if e.descriptorHeapCount > 0 {
		e.cmdList.SetDescriptorHeaps(uint32(e.descriptorHeapCount), &e.descriptorHeaps[0])
	}
}

// boundGroups are the bind groups a pass has set, by index, with their
// dynamic offsets. A root descriptor table needs a root signature to go
// in, and a new one drops every root parameter set before it, so
// SetPipeline binds them again under its own: WebGPU lets a group be set
// before the pipeline and keeps it across pipeline changes. Rust's
// wgpu-hal keeps its root elements the same way.
type boundGroups [4]boundGroup // MaxBindGroups

type boundGroup struct {
	group   *BindGroup
	offsets []uint32 // a copy, the caller may reuse its slice
}

func (g *boundGroups) set(index uint32, bg *BindGroup, offsets []uint32) {
	if int(index) < len(g) {
		g[index] = boundGroup{bg, append(g[index].offsets[:0], offsets...)}
	}
}

func (g *boundGroups) bind(e *CommandEncoder, isCompute bool, mappings []rootParamMapping) {
	for i, b := range g {
		if b.group != nil {
			e.bindGroupToRoot(uint32(i), b.group, b.offsets, isCompute, mappings)
		}
	}
}

// bindGroupToRoot binds a BindGroup's CBV/SRV/UAV descriptor table to root
// parameters, and the root CBVs and root constants of its buffers with a
// dynamic offset. Samplers are handled separately via the global sampler
// heap (bound in SetPipeline). isCompute determines whether to use compute
// or graphics root parameters. groupMappings provides the actual root
// parameter indices.
func (e *CommandEncoder) bindGroupToRoot(bindGroupIndex uint32, bg *BindGroup, offsets []uint32, isCompute bool, groupMappings []rootParamMapping) {
	// A group the pipeline's layout has no slot for has no root parameter
	// to go in: a group left from an earlier pipeline, bound again by
	// boundGroups, would land on whatever parameter has its index.
	if int(bindGroupIndex) >= len(groupMappings) {
		return
	}
	m := groupMappings[bindGroupIndex]

	// Set CBV/SRV/UAV descriptor table (includes sampler index buffer SRV).
	if bg.gpuDescHandle.Ptr != 0 && m.cbvSrvUavIndex >= 0 {
		if isCompute {
			e.cmdList.SetComputeRootDescriptorTable(uint32(m.cbvSrvUavIndex), bg.gpuDescHandle)
		} else {
			e.cmdList.SetGraphicsRootDescriptorTable(uint32(m.cbvSrvUavIndex), bg.gpuDescHandle)
		}
	}

	// The offsets come in binding order, as dynamicBuffers do: a uniform
	// buffer's root CBV points at its binding plus the offset, a storage
	// buffer's offset goes in the root constants. A group left from an
	// earlier pipeline may have other dynamic buffers than this layout,
	// so it only fills the parameters the layout has; drawing with it is
	// a validation error anyway.
	var uniforms, storage int
	for i, d := range bg.dynamicBuffers {
		var offset uint32
		if i < len(offsets) {
			offset = offsets[i]
		}
		switch {
		case d.uniform && uniforms < m.dynamicUniforms:
			index := uint32(m.dynamicIndex + uniforms)
			if isCompute {
				e.cmdList.SetComputeRootConstantBufferView(index, d.gpuVA+uint64(offset))
			} else {
				e.cmdList.SetGraphicsRootConstantBufferView(index, d.gpuVA+uint64(offset))
			}
			uniforms++
		case !d.uniform && storage < m.dynamicStorage:
			index := uint32(m.dynamicIndex + m.dynamicUniforms)
			if isCompute {
				e.cmdList.SetComputeRoot32BitConstant(index, offset, uint32(storage))
			} else {
				e.cmdList.SetGraphicsRoot32BitConstant(index, offset, uint32(storage))
			}
			storage++
		}
	}
}

func (e *CommandEncoder) trackBindGroupState(bg *BindGroup) {
	if bg == nil {
		return
	}
	plans := make([]stateBarrierPlan, 0)
	for _, buffer := range bg.uniformBuffers {
		if before, target, needsBarrier := e.stateTracker.transitionBufferRead(buffer, d3d12.D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER); needsBarrier {
			plans = append(plans, stateBarrierPlan{resource: buffer, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: target})
		}
	}
	for _, buffer := range bg.readOnlyStorageBuffers {
		if before, target, needsBarrier := e.stateTracker.transitionBufferRead(buffer, d3d12.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|d3d12.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); needsBarrier {
			plans = append(plans, stateBarrierPlan{resource: buffer, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: target})
		}
	}
	for _, buffer := range bg.storageBuffers {
		if before, needsBarrier := e.stateTracker.transitionBuffer(buffer, d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS); needsBarrier {
			plans = append(plans, stateBarrierPlan{resource: buffer, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS})
		}
	}
	for _, view := range bg.sampledTextures {
		for _, subresource := range textureViewSubresources(view) {
			if before, target, needsBarrier := e.stateTracker.transitionTextureRead(view.texture, subresource, d3d12.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|d3d12.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE); needsBarrier {
				plans = append(plans, stateBarrierPlan{resource: view.texture, subresource: subresource, before: before, after: target})
			}
		}
	}
	for _, view := range bg.storageTextures {
		for _, subresource := range textureViewSubresources(view) {
			if before, needsBarrier := e.stateTracker.transitionTexture(view.texture, subresource, d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS); needsBarrier {
				plans = append(plans, stateBarrierPlan{resource: view.texture, subresource: subresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS})
			}
		}
	}
	e.emitStateBarrierPlans(plans)
}

// mipExtent is the size of the view's first mip level, at least one texel
// each way.
func (v *TextureView) mipExtent() (width, height uint32) {
	return max(1, v.texture.size.Width>>v.baseMip), max(1, v.texture.size.Height>>v.baseMip)
}

func textureViewSubresources(view *TextureView) []uint32 {
	if view == nil || view.texture == nil {
		return nil
	}
	view.subresOnce.Do(func() {
		view.subresources = textureRangeSubresources(view.texture, hal.TextureRange{
			Aspect:          view.aspect,
			BaseMipLevel:    view.baseMip,
			MipLevelCount:   view.mipCount,
			BaseArrayLayer:  view.baseLayer,
			ArrayLayerCount: view.layerCount,
		})
	})
	return view.subresources
}

func textureViewSubresourcePlanes(view *TextureView) []textureSubresource {
	if view == nil || view.texture == nil {
		return nil
	}
	return textureRangeSubresourcePlanes(view.texture, hal.TextureRange{
		Aspect:          view.aspect,
		BaseMipLevel:    view.baseMip,
		MipLevelCount:   view.mipCount,
		BaseArrayLayer:  view.baseLayer,
		ArrayLayerCount: view.layerCount,
	})
}

func textureViewPhysicalSubresourcePlanes(view *TextureView) []textureSubresource {
	if view == nil || view.texture == nil {
		return nil
	}
	planes := []uint32{0}
	if view.texture.planeCount() > 1 {
		planes = []uint32{0, 1}
	}
	return textureRangeSubresourcePlanesForPlanes(view.texture, hal.TextureRange{
		BaseMipLevel:    view.baseMip,
		MipLevelCount:   view.mipCount,
		BaseArrayLayer:  view.baseLayer,
		ArrayLayerCount: view.layerCount,
	}, planes)
}

// setupDepthStencilAttachment configures depth/stencil attachment for a render pass.
// Returns the DSV handle if valid, nil otherwise.
func (e *CommandEncoder) setupDepthStencilAttachment(dsa *hal.RenderPassDepthStencilAttachment) *d3d12.D3D12_CPU_DESCRIPTOR_HANDLE {
	if dsa == nil {
		return nil
	}

	view, ok := dsa.View.(*TextureView)
	if !ok || !view.hasDSV {
		return nil
	}

	depthReadOnly, stencilReadOnly := depthStencilReadOnlyFlags(view.format, view.aspect, dsa.DepthReadOnly, dsa.StencilReadOnly)
	dsvFlags := d3d12.D3D12_DSV_FLAG_NONE
	if depthReadOnly {
		dsvFlags |= d3d12.D3D12_DSV_FLAG_READ_ONLY_DEPTH
	}
	if stencilReadOnly {
		dsvFlags |= d3d12.D3D12_DSV_FLAG_READ_ONLY_STENCIL
	}
	_, srvPlane := textureFormatToSRV(view.format, view.aspect)
	plans := make([]stateBarrierPlan, 0)
	for _, subresource := range textureViewPhysicalSubresourcePlanes(view) {
		shaderReadable := view.hasSRV && subresource.plane == srvPlane
		state := depthStencilPlaneState(subresource.plane, depthReadOnly, stencilReadOnly, shaderReadable)
		if before, needsBarrier := e.stateTracker.transitionTexture(view.texture, subresource.subresource, state); needsBarrier {
			plans = append(plans, stateBarrierPlan{resource: view.texture, subresource: subresource.subresource, before: before, after: state})
		}
	}
	e.emitStateBarrierPlans(plans)

	// Determine clear flags
	var clearFlags d3d12.D3D12_CLEAR_FLAGS
	if dsa.DepthLoadOp == gputypes.LoadOpClear {
		clearFlags |= d3d12.D3D12_CLEAR_FLAG_DEPTH
	}
	if dsa.StencilLoadOp == gputypes.LoadOpClear {
		clearFlags |= d3d12.D3D12_CLEAR_FLAG_STENCIL
	}

	dsvHandle := view.dsvHandleForFlags(dsvFlags)

	// Clear if needed. Use the same DSV variant as the pass so an unexposed or
	// explicitly read-only companion plane stays protected during the clear.
	if clearFlags != 0 {
		e.cmdList.ClearDepthStencilView(
			*dsvHandle,
			clearFlags,
			dsa.DepthClearValue,
			uint8(dsa.StencilClearValue),
			0, nil,
		)
	}

	return dsvHandle
}

// bufferUsageToD3D12State converts buffer usage to D3D12 resource state.
func bufferUsageToD3D12State(usage gputypes.BufferUsage) d3d12.D3D12_RESOURCE_STATES {
	var state d3d12.D3D12_RESOURCE_STATES

	if usage&gputypes.BufferUsageCopySrc != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE
	}
	if usage&gputypes.BufferUsageCopyDst != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_COPY_DEST
	}
	if usage&gputypes.BufferUsageVertex != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
	}
	if usage&gputypes.BufferUsageIndex != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_INDEX_BUFFER
	}
	if usage&gputypes.BufferUsageUniform != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
	}
	if usage&gputypes.BufferUsageStorage != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS
	}
	if usage&gputypes.BufferUsageIndirect != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
	}

	if state == 0 {
		state = d3d12.D3D12_RESOURCE_STATE_COMMON
	}

	return state
}

// bufferBarrierState is the D3D12 state of a barrier's new buffer usage:
// from the fine-grained uses when the barrier has them, which tell
// read-only storage, read through an SRV, from read-write storage, written
// through a UAV, and from the coarse usage otherwise.
//
// Reference: wgpu-hal dx12/conv.rs map_buffer_usage_to_state
func bufferBarrierState(usage gputypes.BufferUsage, uses hal.BufferUses) d3d12.D3D12_RESOURCE_STATES {
	if uses == 0 {
		return bufferUsageToD3D12State(usage)
	}
	var state d3d12.D3D12_RESOURCE_STATES
	if uses&hal.BufferUsesCopySrc != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE
	}
	if uses&(hal.BufferUsesCopyDst|hal.BufferUsesQueryResolve) != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_COPY_DEST
	}
	if uses&hal.BufferUsesIndex != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_INDEX_BUFFER
	}
	if uses&(hal.BufferUsesVertex|hal.BufferUsesUniform) != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
	}
	if uses&hal.BufferUsesStorageWrite != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS
	} else if uses&hal.BufferUsesStorageRead != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE | d3d12.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE
	}
	if uses&hal.BufferUsesIndirect != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
	}
	if state == 0 {
		state = d3d12.D3D12_RESOURCE_STATE_COMMON
	}
	return state
}

// textureUsageToD3D12State converts texture usage to D3D12 resource state.
// d3d12StateToTextureUsage maps a D3D12 resource state back to gputypes.TextureUsage.
// Used by Texture.CurrentUsage() for PendingWrites barrier computation.
func d3d12StateToTextureUsage(state d3d12.D3D12_RESOURCE_STATES) gputypes.TextureUsage {
	var usage gputypes.TextureUsage

	if state&d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE != 0 {
		usage |= gputypes.TextureUsageCopySrc
	}
	if state&d3d12.D3D12_RESOURCE_STATE_COPY_DEST != 0 {
		usage |= gputypes.TextureUsageCopyDst
	}
	if state&(d3d12.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE|d3d12.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE) != 0 {
		usage |= gputypes.TextureUsageTextureBinding
	}
	if state&d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS != 0 {
		usage |= gputypes.TextureUsageStorageBinding
	}
	if state&d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET != 0 {
		usage |= gputypes.TextureUsageRenderAttachment
	}
	if state&(d3d12.D3D12_RESOURCE_STATE_DEPTH_READ|d3d12.D3D12_RESOURCE_STATE_DEPTH_WRITE) != 0 {
		usage |= gputypes.TextureUsageRenderAttachment
	}

	// COMMON (0) maps to 0 (no specific usage)
	return usage
}

// textureBarrierState is the state a usage of tex means in an explicit
// barrier (TransitionTextures). A depth/stencil texture cannot be in
// RENDER_TARGET: its attachment usage is DEPTH_WRITE, or DEPTH_READ when the
// fine-grained usage says the attachment is read-only (the render pass
// narrows DEPTH_WRITE to DEPTH_READ itself otherwise).
func textureBarrierState(tex *Texture, usage gputypes.TextureUsage, uses hal.TextureUses) d3d12.D3D12_RESOURCE_STATES {
	state := textureUsageToD3D12State(usage)
	if state&d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET != 0 && tex != nil && isDepthFormat(tex.format) {
		depth := d3d12.D3D12_RESOURCE_STATE_DEPTH_WRITE
		if uses&hal.TextureUsesDepthStencilRead != 0 && uses&hal.TextureUsesDepthStencilWrite == 0 {
			depth = d3d12.D3D12_RESOURCE_STATE_DEPTH_READ
		}
		state = state&^d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET | depth
	}
	return state
}

func textureUsageToD3D12State(usage gputypes.TextureUsage) d3d12.D3D12_RESOURCE_STATES {
	var state d3d12.D3D12_RESOURCE_STATES

	if usage&gputypes.TextureUsageCopySrc != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE
	}
	if usage&gputypes.TextureUsageCopyDst != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_COPY_DEST
	}
	if usage&gputypes.TextureUsageTextureBinding != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | d3d12.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
	}
	if usage&gputypes.TextureUsageStorageBinding != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_UNORDERED_ACCESS
	}
	if usage&gputypes.TextureUsageRenderAttachment != 0 {
		state |= d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET
	}

	if state == 0 {
		state = d3d12.D3D12_RESOURCE_STATE_COMMON
	}

	return state
}

// --- Compile-time interface assertions ---

var (
	_ hal.CommandEncoder     = (*CommandEncoder)(nil)
	_ hal.CommandBuffer      = (*CommandBuffer)(nil)
	_ hal.RenderPassEncoder  = (*RenderPassEncoder)(nil)
	_ hal.ComputePassEncoder = (*ComputePassEncoder)(nil)
)
