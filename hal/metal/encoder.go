// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin && !(js && wasm)

package metal

import (
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gogpu/naga/ir"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/internal/indirect"
)

// CommandEncoder implements hal.CommandEncoder for Metal.
//
// Recording state is determined by the presence of cmdBuffer (cmdBuffer != 0).
// This follows the wgpu-rs pattern where Option<CommandBuffer> presence
// indicates recording state, rather than a separate boolean flag.
type CommandEncoder struct {
	device    *Device
	cmdBuffer ID
	label     string
	icbOwners []*indexedICBOwnership
	external  []externalCopy
	finished  *CommandBuffer
	passState renderPassPendingState
	recordErr error
	// openPass is the pass begun and not ended yet, which DiscardEncoding
	// and EndEncoding end: Metal wants every encoder ended.
	openPass interface{ close() }
}

// IsRecording returns true if the encoder has an active command buffer.
// This is the canonical way to check recording state.
func (e *CommandEncoder) IsRecording() bool {
	return e.cmdBuffer != 0
}

// BeginEncoding begins command recording with an optional label.
// After successful call, IsRecording() returns true.
func (e *CommandEncoder) BeginEncoding(label string) error {
	if e.cmdBuffer != 0 {
		return fmt.Errorf("metal: encoder is already recording")
	}
	e.label = label
	e.recordErr = nil
	e.external = nil

	if n := e.device.uncommitted.Add(1); n >= maxCommandBuffers {
		e.device.uncommitted.Add(-1)
		return fmt.Errorf("metal: %d command buffers are recorded and not submitted, the queue holds %d", n-1, maxCommandBuffers)
	}

	// Scoped autorelease pool — drain immediately after creating the command buffer.
	// The command buffer is Retained so it survives the pool drain.
	// This prevents LIFO violations when pools from different frames overlap
	// on the ObjC autorelease pool stack (macOS Tahoe SIGABRT fix).
	pool := NewAutoreleasePool()
	defer pool.Drain()
	e.cmdBuffer = MsgSend(e.device.commandQueue, Sel("commandBuffer"))
	if e.cmdBuffer == 0 {
		e.device.uncommitted.Add(-1)
		return fmt.Errorf("metal: failed to create command buffer")
	}
	Retain(e.cmdBuffer)
	if label != "" {
		nsLabel := NSString(label)
		_ = MsgSend(e.cmdBuffer, Sel("setLabel:"), uintptr(nsLabel))
		Release(nsLabel)
	}
	if hal.DebugEnabled() {
		hal.Logger().Debug("metal: encoding started", "label", label)
	}
	return nil
}

// EndEncoding finishes command recording and returns a command buffer.
// After successful call, IsRecording() returns false.
func (e *CommandEncoder) EndEncoding() (hal.CommandBuffer, error) {
	if e.cmdBuffer == 0 {
		return nil, fmt.Errorf("metal: command encoder is not recording")
	}
	e.closeOpenPass()
	if e.recordErr != nil {
		err := e.recordErr
		e.recordErr = nil
		e.releaseCommandBuffer()
		e.releaseICBOwners()
		return nil, err
	}
	cb := &CommandBuffer{raw: e.cmdBuffer, device: e.device, icbOwners: e.icbOwners, external: e.external}
	cb.uncommitted.Store(true)
	e.cmdBuffer = 0 // Recording state becomes false
	e.icbOwners = nil
	e.external = nil
	e.finished = cb
	hal.Logger().Debug("metal: encoding ended")
	return cb, nil
}

// DiscardEncoding discards the encoder without creating a command buffer.
// After call, IsRecording() returns false.
func (e *CommandEncoder) DiscardEncoding() {
	e.closeOpenPass()
	if e.cmdBuffer != 0 {
		hal.Logger().Debug("metal: encoding discarded")
		e.releaseCommandBuffer()
	}
	e.releaseICBOwners()
	e.recordErr = nil
}

// ResetAll resets command buffers for reuse.
func (e *CommandEncoder) ResetAll(commandBuffers []hal.CommandBuffer) {
	if len(commandBuffers) == 0 {
		if e.finished != nil {
			e.finished.Destroy()
			e.finished = nil
		}
		return
	}
	for _, raw := range commandBuffers {
		if cb, ok := raw.(*CommandBuffer); ok && cb != nil {
			cb.Destroy()
			if e.finished == cb {
				e.finished = nil
			}
		}
	}
}

// Destroy releases recording, finished, and private ICB state.
func (e *CommandEncoder) Destroy() {
	if e == nil {
		return
	}
	e.closeOpenPass()
	if e.cmdBuffer != 0 {
		e.releaseCommandBuffer()
	}
	e.releaseICBOwners()
	e.recordErr = nil
	if e.finished != nil {
		e.finished.Destroy()
		e.finished = nil
	}
}

func (e *CommandEncoder) closeOpenPass() {
	if e.openPass != nil {
		e.openPass.close()
		e.openPass = nil
	}
}

// releaseCommandBuffer releases the command buffer being recorded, which
// will not be committed.
func (e *CommandEncoder) releaseCommandBuffer() {
	Release(e.cmdBuffer)
	e.cmdBuffer = 0
	e.device.uncommitted.Add(-1)
}

func (e *CommandEncoder) failRecording(err error) {
	if e != nil && err != nil && e.recordErr == nil {
		e.recordErr = err
	}
}

func (e *CommandEncoder) releaseICBOwners() {
	for _, owner := range e.icbOwners {
		owner.release()
	}
	e.icbOwners = nil
}

// TransitionBuffers transitions buffer states for synchronization.
func (e *CommandEncoder) TransitionBuffers(_ []hal.BufferBarrier) {}

// TransitionTextures transitions texture states for synchronization.
func (e *CommandEncoder) TransitionTextures(_ []hal.TextureBarrier) {}

// ClearBuffer clears a buffer region to zero.
func (e *CommandEncoder) ClearBuffer(buffer hal.Buffer, offset, size uint64) {
	if e.cmdBuffer == 0 {
		return
	}
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil {
		return
	}
	pool := NewAutoreleasePool()
	defer pool.Drain()
	blitEncoder := MsgSend(e.cmdBuffer, Sel("blitCommandEncoder"))
	if blitEncoder == 0 {
		return
	}
	_ = MsgSend(blitEncoder, Sel("fillBuffer:range:value:"), uintptr(buf.raw), uintptr(offset), uintptr(size), uintptr(0))
	_ = MsgSend(blitEncoder, Sel("endEncoding"))
}

// dropEmpty drops the regions that copy nothing, which WebGPU allows and
// Metal's validation doesn't, as Dawn skips them.
func dropEmpty[T any](regions []T, empty func(T) bool) []T {
	if !slices.ContainsFunc(regions, empty) {
		return regions
	}
	return slices.DeleteFunc(slices.Clone(regions), empty)
}

func emptyExtent(size hal.Extent3D) bool {
	return size.Width == 0 || size.Height == 0 || size.DepthOrArrayLayers == 0
}

func emptyBufferTextureCopy(r hal.BufferTextureCopy) bool { return emptyExtent(r.Size) }

// CopyBufferToBuffer copies data between buffers.
func (e *CommandEncoder) CopyBufferToBuffer(src, dst hal.Buffer, regions []hal.BufferCopy) {
	regions = dropEmpty(regions, func(r hal.BufferCopy) bool { return r.Size == 0 })
	if e.cmdBuffer == 0 || len(regions) == 0 {
		return
	}
	srcBuf, ok := src.(*Buffer)
	if !ok || srcBuf == nil {
		return
	}
	dstBuf, ok := dst.(*Buffer)
	if !ok || dstBuf == nil {
		return
	}
	pool := NewAutoreleasePool()
	defer pool.Drain()
	blitEncoder := MsgSend(e.cmdBuffer, Sel("blitCommandEncoder"))
	if blitEncoder == 0 {
		return
	}
	for _, region := range regions {
		_ = MsgSend(blitEncoder, Sel("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
			uintptr(srcBuf.raw), uintptr(region.SrcOffset), uintptr(dstBuf.raw), uintptr(region.DstOffset), uintptr(region.Size))
	}
	_ = MsgSend(blitEncoder, Sel("endEncoding"))
}

// CopyBufferToTexture copies data from a buffer to a texture.
func (e *CommandEncoder) CopyBufferToTexture(src hal.Buffer, dst hal.Texture, regions []hal.BufferTextureCopy) {
	regions = dropEmpty(regions, emptyBufferTextureCopy)
	if e.cmdBuffer == 0 || len(regions) == 0 {
		return
	}
	srcBuf, ok := src.(*Buffer)
	dstTex := metalTexture(dst)
	if !ok || srcBuf == nil || dstTex == nil {
		e.failRecording(fmt.Errorf("metal: CopyBufferToTexture: invalid buffer or texture"))
		return
	}
	for _, region := range regions {
		if _, _, ok := validateMetalBufferTextureCopyPlan(dstTex.format, region.TextureBase.Aspect, dstTex.dimension, region.BufferLayout, region.TextureBase.Origin, region.Size); !ok {
			e.failRecording(fmt.Errorf("metal: CopyBufferToTexture: invalid copy into %s texture", dstTex.format))
			return
		}
	}
	pool := NewAutoreleasePool()
	defer pool.Drain()
	blitEncoder := MsgSend(e.cmdBuffer, Sel("blitCommandEncoder"))
	if blitEncoder == 0 {
		return
	}
	for _, region := range regions {
		aspect := region.TextureBase.Aspect
		plan, bytesPerImage, _ := validateMetalBufferTextureCopyPlan(dstTex.format, aspect, dstTex.dimension, region.BufferLayout, region.TextureBase.Origin, region.Size)
		blockWidth, blockHeight, blockSize, _ := metalCopyBlockInfo(dstTex.format, aspect)
		for operation := uint32(0); operation < plan.operationCount; operation++ {
			destination, _ := plan.textureRegion(dstTex.dimension, region.TextureBase.Origin, region.Size, operation)
			sourceOffset, _ := plan.bufferOffset(region.BufferLayout.Offset, bytesPerImage, operation)
			for _, blit := range splitMetalBufferBlit(destination, dstTex.dimension, sourceOffset, srcBuf.size,
				uint64(region.BufferLayout.BytesPerRow), bytesPerImage, blockWidth, blockHeight, blockSize) {
				msgSendVoid(blitEncoder, Sel("copyFromBuffer:sourceOffset:sourceBytesPerRow:sourceBytesPerImage:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:options:"),
					argPointer(uintptr(srcBuf.raw)),
					argUint64(blit.offset),
					argUint64(blit.bytesPerRow),
					argUint64(blit.bytesPerImage),
					argStruct(blit.region.size, mtlSizeType),
					argPointer(uintptr(dstTex.raw)),
					argUint64(uint64(blit.region.slice)),
					argUint64(uint64(region.TextureBase.MipLevel)),
					argStruct(blit.region.origin, mtlOriginType),
					argUint64(uint64(metalBlitOption(dstTex.format, aspect))),
				)
			}
		}
	}
	_ = MsgSend(blitEncoder, Sel("endEncoding"))
}

// CopyTextureToBuffer copies data from a texture to a buffer.
func (e *CommandEncoder) CopyTextureToBuffer(src hal.Texture, dst hal.Buffer, regions []hal.BufferTextureCopy) {
	regions = dropEmpty(regions, emptyBufferTextureCopy)
	if e.cmdBuffer == 0 || len(regions) == 0 {
		return
	}
	srcTex := metalTexture(src)
	dstBuf, ok := dst.(*Buffer)
	if !ok || srcTex == nil || dstBuf == nil {
		e.failRecording(fmt.Errorf("metal: CopyTextureToBuffer: invalid texture or buffer"))
		return
	}
	for _, region := range regions {
		if _, _, ok := validateMetalBufferTextureCopyPlan(srcTex.format, region.TextureBase.Aspect, srcTex.dimension, region.BufferLayout, region.TextureBase.Origin, region.Size); !ok {
			e.failRecording(fmt.Errorf("metal: CopyTextureToBuffer: invalid copy out of %s texture", srcTex.format))
			return
		}
	}
	pool := NewAutoreleasePool()
	defer pool.Drain()
	blitEncoder := MsgSend(e.cmdBuffer, Sel("blitCommandEncoder"))
	if blitEncoder == 0 {
		return
	}
	for _, region := range regions {
		aspect := region.TextureBase.Aspect
		plan, bytesPerImage, _ := validateMetalBufferTextureCopyPlan(srcTex.format, aspect, srcTex.dimension, region.BufferLayout, region.TextureBase.Origin, region.Size)
		blockWidth, blockHeight, blockSize, _ := metalCopyBlockInfo(srcTex.format, aspect)
		for operation := uint32(0); operation < plan.operationCount; operation++ {
			source, _ := plan.textureRegion(srcTex.dimension, region.TextureBase.Origin, region.Size, operation)
			destinationOffset, _ := plan.bufferOffset(region.BufferLayout.Offset, bytesPerImage, operation)
			for _, blit := range splitMetalBufferBlit(source, srcTex.dimension, destinationOffset, dstBuf.size,
				uint64(region.BufferLayout.BytesPerRow), bytesPerImage, blockWidth, blockHeight, blockSize) {
				msgSendVoid(blitEncoder, Sel("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:"),
					argPointer(uintptr(srcTex.raw)),
					argUint64(uint64(blit.region.slice)),
					argUint64(uint64(region.TextureBase.MipLevel)),
					argStruct(blit.region.origin, mtlOriginType),
					argStruct(blit.region.size, mtlSizeType),
					argPointer(uintptr(dstBuf.raw)),
					argUint64(blit.offset),
					argUint64(blit.bytesPerRow),
					argUint64(blit.bytesPerImage),
					argUint64(uint64(metalBlitOption(srcTex.format, aspect))),
				)
			}
		}
	}
	_ = MsgSend(blitEncoder, Sel("endEncoding"))
}

// CopyTextureToTexture copies data between textures.
func (e *CommandEncoder) CopyTextureToTexture(src, dst hal.Texture, regions []hal.TextureCopy) {
	regions = dropEmpty(regions, func(r hal.TextureCopy) bool { return emptyExtent(r.Size) })
	if e.cmdBuffer == 0 || len(regions) == 0 {
		return
	}
	srcTex, dstTex := metalTexture(src), metalTexture(dst)
	if srcTex == nil || dstTex == nil {
		e.failRecording(fmt.Errorf("metal: CopyTextureToTexture: invalid texture"))
		return
	}
	for _, region := range regions {
		if _, ok := validateMetalTextureCopyPlan(srcTex.dimension, dstTex.dimension, region.SrcBase.Origin, region.DstBase.Origin, region.Size); !ok {
			e.failRecording(fmt.Errorf("metal: CopyTextureToTexture: invalid copy"))
			return
		}
	}
	pool := NewAutoreleasePool()
	defer pool.Drain()
	// Metal copies between textures of one pixel format; WebGPU between
	// formats that differ only in being sRGB, so the copy writes a view of
	// the destination in the source's format.
	target := dstTex.raw
	if srcFormat := e.device.adapter.mapTextureFormat(srcTex.format); srcFormat != e.device.adapter.mapTextureFormat(dstTex.format) {
		target = MsgSend(dstTex.raw, Sel("newTextureViewWithPixelFormat:"), uintptr(srcFormat))
		if target == 0 {
			e.failRecording(fmt.Errorf("metal: CopyTextureToTexture: no %s view of the destination", srcTex.format))
			return
		}
		defer Release(target)
	}
	blitEncoder := MsgSend(e.cmdBuffer, Sel("blitCommandEncoder"))
	if blitEncoder == 0 {
		return
	}
	for _, region := range regions {
		plan, _ := validateMetalTextureCopyPlan(srcTex.dimension, dstTex.dimension, region.SrcBase.Origin, region.DstBase.Origin, region.Size)
		for operation := uint32(0); operation < plan.operationCount; operation++ {
			source, _ := plan.textureRegion(srcTex.dimension, region.SrcBase.Origin, region.Size, operation)
			destination, _ := plan.textureRegion(dstTex.dimension, region.DstBase.Origin, region.Size, operation)
			msgSendVoid(blitEncoder, Sel("copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toTexture:destinationSlice:destinationLevel:destinationOrigin:"),
				argPointer(uintptr(srcTex.raw)),
				argUint64(uint64(source.slice)),
				argUint64(uint64(region.SrcBase.MipLevel)),
				argStruct(source.origin, mtlOriginType),
				argStruct(source.size, mtlSizeType),
				argPointer(uintptr(target)),
				argUint64(uint64(destination.slice)),
				argUint64(uint64(region.DstBase.MipLevel)),
				argStruct(destination.origin, mtlOriginType),
			)
		}
	}
	_ = MsgSend(blitEncoder, Sel("endEncoding"))
}

// ResolveQuerySet copies query results from a query set into a destination buffer.
func (e *CommandEncoder) ResolveQuerySet(querySet hal.QuerySet, firstQuery, queryCount uint32, destination hal.Buffer, destinationOffset uint64) {
	if e.cmdBuffer == 0 || queryCount == 0 {
		return
	}
	q, ok := querySet.(*QuerySet)
	dst, dstOK := destination.(*Buffer)
	if !ok || q == nil || !dstOK || dst == nil || uint64(firstQuery)+uint64(queryCount) > uint64(q.count) ||
		destinationOffset > dst.size || uint64(queryCount)*8 > dst.size-destinationOffset {
		e.failRecording(fmt.Errorf("metal: ResolveQuerySet: invalid query set or destination range"))
		return
	}
	pool := NewAutoreleasePool()
	defer pool.Drain()
	blitEncoder := MsgSend(e.cmdBuffer, Sel("blitCommandEncoder"))
	if blitEncoder == 0 {
		return
	}
	_ = MsgSend(blitEncoder, Sel("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"),
		uintptr(q.buffer), uintptr(firstQuery)*8, uintptr(dst.raw), uintptr(destinationOffset), uintptr(queryCount)*8)
	_ = MsgSend(blitEncoder, Sel("endEncoding"))
}

// BuildAccelerationStructures builds one or more acceleration structures.
//
// Delegates to buildAccelerationStructures in raytracing.go which creates a
// transient MTLAccelerationStructureCommandEncoder and issues build/refit
// commands for each descriptor.
//
// Reference: Rust wgpu-hal metal/command.rs:1840-1879.
func (e *CommandEncoder) BuildAccelerationStructures(descriptors []hal.BuildAccelerationStructureDescriptor) {
	if e.cmdBuffer == 0 {
		return
	}
	e.buildAccelerationStructures(descriptors)
}

// PlaceAccelerationStructureBarrier is a no-op on Metal.
//
// Metal handles acceleration structure synchronization internally through
// its command buffer scheduling model. Explicit barriers are not required.
//
// Reference: Rust wgpu-hal metal/command.rs:1881-1885 (empty body).
func (e *CommandEncoder) PlaceAccelerationStructureBarrier(_ hal.AccelerationStructureBarrier) {}

// CopyAccelerationStructure copies or compacts an acceleration structure.
//
// Delegates to copyAccelerationStructure in raytracing.go which creates a
// transient MTLAccelerationStructureCommandEncoder and issues copy/compact.
//
// Reference: Rust wgpu-hal metal/command.rs:746-764.
func (e *CommandEncoder) CopyAccelerationStructure(src, dst hal.AccelerationStructure, copyMode gputypes.AccelerationStructureCopyMode) {
	if e.cmdBuffer == 0 {
		return
	}
	e.copyAccelerationStructure(src, dst, copyMode)
}

// ReadAccelerationStructureCompactSize writes the post-compaction size
// of an acceleration structure into a buffer.
//
// Delegates to readAccelerationStructureCompactSize in raytracing.go.
//
// Reference: Rust wgpu-hal metal/command.rs:1887-1898.
func (e *CommandEncoder) ReadAccelerationStructureCompactSize(accelStruct hal.AccelerationStructure, buffer hal.Buffer, offset uint64) {
	if e.cmdBuffer == 0 {
		return
	}
	e.readAccelerationStructureCompactSize(accelStruct, buffer, offset)
}

// BeginRenderPass begins a render pass.
// Returns nil if encoder is not recording (cmdBuffer == 0).
func (e *CommandEncoder) BeginRenderPass(desc *hal.RenderPassDescriptor) hal.RenderPassEncoder {
	if e.cmdBuffer == 0 {
		return nil
	}
	// Keep any temporary Objective-C objects scoped to descriptor construction.
	// rpDesc is created with new, so it remains owned after this pool drains and
	// can safely outlive BeginRenderPass while native encoder creation is deferred.
	pool := NewAutoreleasePool()
	defer pool.Drain()
	rpDesc := MsgSend(ID(GetClass("MTLRenderPassDescriptor")), Sel("new"))
	if rpDesc == 0 {
		return nil
	}
	colorAttachments := MsgSend(rpDesc, Sel("colorAttachments"))
	for i, ca := range desc.ColorAttachments {
		attachment := MsgSend(colorAttachments, Sel("objectAtIndexedSubscript:"), uintptr(i))
		if attachment == 0 {
			continue
		}
		if tv, ok := ca.View.(*TextureView); ok && tv != nil {
			_ = MsgSend(attachment, Sel("setTexture:"), uintptr(tv.raw))
		}
		_ = MsgSend(attachment, Sel("setLoadAction:"), uintptr(loadOpToMTL(ca.LoadOp)))
		if ca.LoadOp == gputypes.LoadOpClear {
			clearColor := MTLClearColor{Red: ca.ClearValue.R, Green: ca.ClearValue.G, Blue: ca.ClearValue.B, Alpha: ca.ClearValue.A}
			msgSendClearColor(attachment, Sel("setClearColor:"), clearColor)
		}
		storeAction := storeOpToMTL(ca.StoreOp)
		if ca.ResolveTarget != nil {
			if rtv, ok := ca.ResolveTarget.(*TextureView); ok && rtv != nil {
				_ = MsgSend(attachment, Sel("setResolveTexture:"), uintptr(rtv.raw))
				// Metal requires MultisampleResolve store action when a resolve
				// texture is set. Without this, Metal silently skips the MSAA
				// resolve and the surface stays uninitialized (purple screen).
				if storeAction == MTLStoreActionStore {
					storeAction = MTLStoreActionStoreAndMultisampleResolve
				} else {
					storeAction = MTLStoreActionMultisampleResolve
				}
			}
		}
		_ = MsgSend(attachment, Sel("setStoreAction:"), uintptr(storeAction))
	}
	if desc.DepthStencilAttachment != nil {
		dsa := desc.DepthStencilAttachment
		tv, _ := dsa.View.(*TextureView)

		// A stencil-only format gets no depth attachment, just as a depth-only
		// one gets no stencil attachment below.
		if tv != nil && tv.texture != nil && tv.texture.format.HasDepth() {
			depthAttachment := MsgSend(rpDesc, Sel("depthAttachment"))
			_ = MsgSend(depthAttachment, Sel("setTexture:"), uintptr(tv.raw))
			depthLoad := loadOpToMTL(dsa.DepthLoadOp)
			if dsa.DepthReadOnly {
				depthLoad = MTLLoadActionLoad
			}
			_ = MsgSend(depthAttachment, Sel("setLoadAction:"), uintptr(depthLoad))
			if dsa.DepthLoadOp == gputypes.LoadOpClear {
				msgSendVoid(depthAttachment, Sel("setClearDepth:"), argFloat64(float64(dsa.DepthClearValue)))
			}
			_ = MsgSend(depthAttachment, Sel("setStoreAction:"), uintptr(storeOpToMTL(dsa.DepthStoreOp)))
		}

		// Stencil attachment — same texture, separate load/store/clear.
		// Metal requires both depth and stencil attachments to be configured
		// independently when using combined depth-stencil formats (e.g.
		// Depth32FloatStencil8). Without this, the stencil load action
		// defaults to MTLLoadActionDontCare, leaving stencil values
		// undefined and causing progressive rendering artifacts on Apple
		// Silicon TBDR GPUs. A depth-only format gets no stencil attachment:
		// the pipeline declares no stencil format for it (CreateRenderPipeline
		// goes by HasStencil too), and Metal rejects a pass whose stencil
		// attachment the pipeline does not match.
		// Reference: Rust wgpu-hal metal/command.rs:705-727.
		if tv != nil && tv.texture != nil && tv.texture.format.HasStencil() {
			stencilAttachment := MsgSend(rpDesc, Sel("stencilAttachment"))
			_ = MsgSend(stencilAttachment, Sel("setTexture:"), uintptr(tv.raw))
			stencilLoad := loadOpToMTL(dsa.StencilLoadOp)
			if dsa.StencilReadOnly {
				stencilLoad = MTLLoadActionLoad
			}
			_ = MsgSend(stencilAttachment, Sel("setLoadAction:"), uintptr(stencilLoad))
			if dsa.StencilLoadOp == gputypes.LoadOpClear {
				_ = MsgSend(stencilAttachment, Sel("setClearStencil:"), uintptr(dsa.StencilClearValue))
			}
			_ = MsgSend(stencilAttachment, Sel("setStoreAction:"), uintptr(storeOpToMTL(dsa.StencilStoreOp)))
		}
	}
	// Keep the descriptor alive but delay creation of the native render encoder
	// until the first draw. Metal requires the ICB translator to run on a compute
	// encoder before the render encoder exists; retaining the descriptor gives
	// that backend-private lowering a narrow seam without journaling draw calls.
	e.passState = renderPassPendingState{}
	pass := &RenderPassEncoder{descriptor: rpDesc, commandEncoder: e, device: e.device, pending: &e.passState}
	e.openPass = pass
	return pass
}

// BeginComputePass begins a compute pass.
// Returns nil if encoder is not recording (cmdBuffer == 0).
func (e *CommandEncoder) BeginComputePass(desc *hal.ComputePassDescriptor) hal.ComputePassEncoder {
	if e.cmdBuffer == 0 {
		return nil
	}
	// Scoped pool: encoder is Retained to survive pool drain.
	pool := NewAutoreleasePool()
	defer pool.Drain()
	encoder := MsgSend(e.cmdBuffer, Sel("computeCommandEncoder"))
	if encoder == 0 {
		return nil
	}
	Retain(encoder)
	if desc != nil && desc.Label != "" {
		nsLabel := NSString(desc.Label)
		_ = MsgSend(encoder, Sel("setLabel:"), uintptr(nsLabel))
		Release(nsLabel)
	}
	pass := &ComputePassEncoder{raw: encoder, device: e.device, commandEncoder: e}
	e.openPass = pass
	return pass
}

// CommandBuffer implements hal.CommandBuffer for Metal.
type CommandBuffer struct {
	raw       ID
	device    *Device
	drawable  ID // Attached drawable for presentation
	icbOwners []*indexedICBOwnership
	external  []externalCopy // for Submit to encode, see externalCopy
	destroy   sync.Once
	// uncommitted is whether the buffer still counts in the device's
	// uncommitted command buffers.
	uncommitted atomic.Bool
}

// settle takes the buffer off the device's uncommitted command buffers, once
// it is committed or released.
func (cb *CommandBuffer) settle() {
	if cb.uncommitted.CompareAndSwap(true, false) {
		cb.device.uncommitted.Add(-1)
	}
}

// Destroy releases the command buffer.
func (cb *CommandBuffer) Destroy() {
	if cb == nil {
		return
	}
	cb.destroy.Do(func() {
		cb.settle()
		if cb.raw != 0 {
			Release(cb.raw)
			cb.raw = 0
		}
		for _, owner := range cb.icbOwners {
			owner.release()
		}
		cb.icbOwners = nil
	})
}

// SetDrawable attaches a drawable for presentation.
// The drawable will be presented when the command buffer is submitted.
func (cb *CommandBuffer) SetDrawable(drawable ID) {
	cb.drawable = drawable
}

// RenderPassEncoder implements hal.RenderPassEncoder for Metal.
type RenderPassEncoder struct {
	raw            ID
	descriptor     ID
	commandEncoder *CommandEncoder
	device         *Device
	pipeline       *RenderPipeline
	currentLayout  *PipelineLayout // set by SetPipeline for SetBindGroup slot offsets
	indexBuffer    *Buffer
	indexFormat    gputypes.IndexFormat
	indexOffset    uint64
	pending        *renderPassPendingState
	// sizesStale is set when the pipeline or a bind group changed since the
	// stages' `_buffer_sizes` were last set.
	sizesStale bool
	// depthRangeStale is set when the pipeline or the viewport changed since
	// the fragment stage's frag_depth range was last set.
	depthRangeStale bool
}

const (
	maxRenderBindGroups     = 4
	maxRenderDynamicOffsets = 16
)

type renderBindGroupState struct {
	group       *BindGroup
	offsets     [maxRenderDynamicOffsets]uint32
	offsetCount uint8
	set         bool
}

type renderVertexBufferState struct {
	buffer *Buffer
	offset uint64
	set    bool
}

type renderPassPendingState struct {
	bindGroups    [maxRenderBindGroups]renderBindGroupState
	vertexBuffers [maxVertexBuffers]renderVertexBufferState
	viewport      MTLViewport
	scissor       MTLScissorRect
	blend         gputypes.Color
	stencil       uint32
	viewportSet   bool
	scissorSet    bool
	blendSet      bool
	stencilSet    bool
}

func (e *RenderPassEncoder) beginNative() bool {
	if e == nil {
		return false
	}
	if e.raw != 0 {
		return true
	}
	if e.descriptor == 0 || e.commandEncoder == nil || e.commandEncoder.cmdBuffer == 0 {
		return false
	}

	pool := NewAutoreleasePool()
	encoder := MsgSend(e.commandEncoder.cmdBuffer, Sel("renderCommandEncoderWithDescriptor:"), uintptr(e.descriptor))
	Release(e.descriptor)
	e.descriptor = 0
	if encoder == 0 {
		e.commandEncoder.failRecording(fmt.Errorf("metal: failed to create render command encoder"))
		pool.Drain()
		return false
	}
	Retain(encoder)
	e.raw = encoder
	pool.Drain()
	e.replayPendingState()
	return true
}

func (e *RenderPassEncoder) replayPendingState() {
	if e.pipeline != nil {
		e.applyPipeline(e.pipeline)
	}
	if e.pending == nil {
		return
	}
	e.applyPendingBindGroups()
	e.applyPendingVertexBuffers()
	if e.pending.viewportSet {
		msgSendVoid(e.raw, Sel("setViewport:"), argStruct(e.pending.viewport, mtlViewportType))
	}
	if e.pending.scissorSet {
		msgSendVoid(e.raw, Sel("setScissorRect:"), argStruct(e.pending.scissor, mtlScissorRectType))
	}
	if e.pending.blendSet {
		e.applyBlendConstant(e.pending.blend)
	}
	if e.pending.stencilSet {
		_ = MsgSend(e.raw, Sel("setStencilReferenceValue:"), uintptr(e.pending.stencil))
	}
}

// applyPendingBindGroups binds the groups set so far, at the slots of the
// current pipeline's layout.
func (e *RenderPassEncoder) applyPendingBindGroups() {
	if e.pending == nil {
		return
	}
	for i := range e.pending.bindGroups {
		state := &e.pending.bindGroups[i]
		if state.set {
			e.applyBindGroup(uint32(i), state.group, state.offsets[:state.offsetCount])
		}
	}
}

// applyPendingVertexBuffers binds the vertex buffers set so far that the
// current pipeline reads.
func (e *RenderPassEncoder) applyPendingVertexBuffers() {
	if e.pending == nil {
		return
	}
	for i := range e.pending.vertexBuffers {
		state := &e.pending.vertexBuffers[i]
		if state.set {
			e.applyVertexBuffer(uint32(i), state.buffer, state.offset)
		}
	}
}

// End finishes the render pass.
func (e *RenderPassEncoder) End() {
	e.beginNative()
	e.close()
	if e.commandEncoder != nil && e.commandEncoder.openPass == e {
		e.commandEncoder.openPass = nil
	}
}

// close ends and releases the native encoder, and releases the descriptor
// of a pass that never made one.
func (e *RenderPassEncoder) close() {
	if e.raw != 0 {
		_ = MsgSend(e.raw, Sel("endEncoding"))
		Release(e.raw)
		e.raw = 0
	}
	if e.descriptor != 0 {
		Release(e.descriptor)
		e.descriptor = 0
	}
}

// SetPipeline sets the render pipeline.
func (e *RenderPassEncoder) SetPipeline(pipeline hal.RenderPipeline) {
	p, ok := pipeline.(*RenderPipeline)
	if !ok || p == nil {
		return
	}
	prev := e.pipeline
	e.pipeline = p
	e.currentLayout = p.layout // store for SetBindGroup slot offset lookup
	e.sizesStale = true
	e.depthRangeStale = true
	if e.raw == 0 {
		return
	}
	e.applyPipeline(p)
	// A group's slots come from the layout: groups set before the first
	// pipeline waited for it, and another layout moves them.
	if prev == nil || prev.layout != p.layout {
		e.applyPendingBindGroups()
	}
	e.applyPendingVertexBuffers()
}

func (e *RenderPassEncoder) applyPipeline(p *RenderPipeline) {
	_ = MsgSend(e.raw, Sel("setRenderPipelineState:"), uintptr(p.raw))
	_ = MsgSend(e.raw, Sel("setCullMode:"), uintptr(p.cullMode))
	_ = MsgSend(e.raw, Sel("setFrontFacingWinding:"), uintptr(p.frontFace))
	_ = MsgSend(e.raw, Sel("setDepthClipMode:"), uintptr(p.depthClipMode))
	if p.depthStencil != 0 {
		_ = MsgSend(e.raw, Sel("setDepthStencilState:"), uintptr(p.depthStencil))
		msgSendVoid(e.raw, Sel("setDepthBias:slopeScale:clamp:"), argFloat32(p.depthBias), argFloat32(p.depthSlopeScale), argFloat32(p.depthClamp))
	}
}

// bindSlotAssignment holds the computed per-type Metal slot index for a bind group entry.
type bindSlotAssignment struct {
	entryIndex int
	slot       uintptr
}

// computeBindSlots calculates per-type sequential Metal slot indices for bind group entries.
//
// Metal uses separate index spaces: [[buffer(N)]], [[texture(M)]], [[sampler(K)]].
// The pipeline layout numbers each resource type independently, so the
// WGSL @binding(N) number (which is unique across all types in a group) is not
// the slot.
func computeBindSlots(entries []gputypes.BindGroupEntry) (bufferSlots, textureSlots, samplerSlots []bindSlotAssignment) {
	var bufferIdx, textureIdx, samplerIdx uintptr
	for i, entry := range entries {
		switch entry.Resource.(type) {
		case gputypes.BufferBinding:
			bufferSlots = append(bufferSlots, bindSlotAssignment{entryIndex: i, slot: bufferIdx})
			bufferIdx++
		case gputypes.TextureViewBinding:
			textureSlots = append(textureSlots, bindSlotAssignment{entryIndex: i, slot: textureIdx})
			textureIdx++
		case gputypes.SamplerBinding:
			samplerSlots = append(samplerSlots, bindSlotAssignment{entryIndex: i, slot: samplerIdx})
			samplerIdx++
		}
	}
	return
}

// SetBindGroup sets a bind group by binding each resource directly on the encoder.
//
// Metal does not use argument buffers for basic resource binding. Instead, resources
// are set individually via setVertexBuffer/setFragmentBuffer, setVertexTexture/
// setFragmentTexture, and setVertexSamplerState/setFragmentSamplerState.
//
// The Metal binding index is the per-type sequential index the pipeline
// layout gives the entry, which its shaders are compiled with.
func (e *RenderPassEncoder) SetBindGroup(index uint32, group hal.BindGroup, offsets []uint32) {
	bg, ok := group.(*BindGroup)
	if !ok || bg == nil {
		return
	}
	e.sizesStale = true
	if e.pending != nil && index < maxRenderBindGroups && len(offsets) <= maxRenderDynamicOffsets {
		state := &e.pending.bindGroups[index]
		state.group = bg
		state.offsetCount = uint8(len(offsets))
		state.set = true
		copy(state.offsets[:], offsets)
		clear(state.offsets[len(offsets):])
		if e.pipeline == nil {
			return // SetPipeline binds it, at its layout's slots
		}
	}
	if e.raw == 0 {
		return
	}
	e.applyBindGroup(index, bg, offsets)
}

func (e *RenderPassEncoder) applyBindGroup(index uint32, bg *BindGroup, offsets []uint32) {
	// Metal uses per-type sequential indices: [[buffer(N)]], [[texture(M)]], [[sampler(K)]].
	// The pipeline layout numbers them across all its bind groups: group 0
	// starts at 0, group 1 where group 0 ended, and so on.
	//
	// Reference: Rust wgpu-hal metal/command.rs:182 (resource_indices.buffers + index).
	off, ok := e.currentLayout.groupSlots(index, bg)
	if !ok {
		return
	}
	bufferSlot, textureSlot, samplerSlot := uintptr(off.Buffers), uintptr(off.Textures), uintptr(off.Samplers)

	var dynamicIdx int
	for _, entry := range bg.entries {
		switch res := entry.Resource.(type) {
		case gputypes.BufferBinding:
			offset := uintptr(res.Offset)
			// Apply dynamic offset if the layout entry has HasDynamicOffset.
			if dynamicIdx < len(offsets) && bg.layout != nil {
				for _, le := range bg.layout.entries {
					if le.Binding == entry.Binding && le.Buffer != nil && le.Buffer.HasDynamicOffset {
						offset += uintptr(offsets[dynamicIdx])
						dynamicIdx++
						break
					}
				}
			}
			_ = MsgSend(e.raw, Sel("setVertexBuffer:offset:atIndex:"), uintptr(res.Buffer), offset, bufferSlot)
			_ = MsgSend(e.raw, Sel("setFragmentBuffer:offset:atIndex:"), uintptr(res.Buffer), offset, bufferSlot)
			bufferSlot++

		case gputypes.TextureViewBinding:
			_ = MsgSend(e.raw, Sel("setVertexTexture:atIndex:"), uintptr(res.TextureView), textureSlot)
			_ = MsgSend(e.raw, Sel("setFragmentTexture:atIndex:"), uintptr(res.TextureView), textureSlot)
			textureSlot++

		case gputypes.SamplerBinding:
			_ = MsgSend(e.raw, Sel("setVertexSamplerState:atIndex:"), uintptr(res.Sampler), samplerSlot)
			_ = MsgSend(e.raw, Sel("setFragmentSamplerState:atIndex:"), uintptr(res.Sampler), samplerSlot)
			samplerSlot++
		}
	}
}

// SetVertexBuffer sets a vertex buffer.
func (e *RenderPassEncoder) SetVertexBuffer(slot uint32, buffer hal.Buffer, offset uint64) {
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil || slot >= maxVertexBuffers {
		return
	}
	if e.pending != nil {
		state := &e.pending.vertexBuffers[slot]
		state.buffer = buf
		state.offset = offset
		state.set = true
	}
	e.sizesStale = true
	if e.raw == 0 {
		return
	}
	e.applyVertexBuffer(slot, buf, offset)
}

func (e *RenderPassEncoder) applyVertexBuffer(slot uint32, buf *Buffer, offset uint64) {
	// The slot of a vertex buffer the pipeline doesn't read may be one of
	// its bind groups' slots.
	if e.pipeline == nil || slot >= e.pipeline.vertexBuffers {
		return
	}
	bufIdx := maxVertexBuffers - 1 - slot
	_ = MsgSend(e.raw, Sel("setVertexBuffer:offset:atIndex:"), uintptr(buf.raw), uintptr(offset), uintptr(bufIdx))
}

// SetIndexBuffer sets the index buffer.
func (e *RenderPassEncoder) SetIndexBuffer(buffer hal.Buffer, format gputypes.IndexFormat, offset uint64) {
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil {
		return
	}
	e.indexBuffer = buf
	e.indexFormat = format
	e.indexOffset = offset
}

// SetViewport sets the viewport.
func (e *RenderPassEncoder) SetViewport(vp gputypes.Viewport) {
	viewport := MTLViewport{OriginX: float64(vp.X), OriginY: float64(vp.Y), Width: float64(vp.Width), Height: float64(vp.Height), ZNear: float64(vp.MinDepth), ZFar: float64(vp.MaxDepth)}
	if e.pending != nil {
		e.pending.viewport = viewport
		e.pending.viewportSet = true
	}
	e.depthRangeStale = true
	if e.raw == 0 {
		return
	}
	msgSendVoid(e.raw, Sel("setViewport:"), argStruct(viewport, mtlViewportType))
}

// SetScissorRect sets the scissor rectangle.
func (e *RenderPassEncoder) SetScissorRect(rect gputypes.ScissorRect) {
	scissor := MTLScissorRect{X: NSUInteger(rect.X), Y: NSUInteger(rect.Y), Width: NSUInteger(rect.Width), Height: NSUInteger(rect.Height)}
	if e.pending != nil {
		e.pending.scissor = scissor
		e.pending.scissorSet = true
	}
	if e.raw == 0 {
		return
	}
	msgSendVoid(e.raw, Sel("setScissorRect:"), argStruct(scissor, mtlScissorRectType))
}

// SetBlendConstant sets the blend constant color.
func (e *RenderPassEncoder) SetBlendConstant(color *gputypes.Color) {
	if color == nil {
		return
	}
	if e.pending != nil {
		e.pending.blend = *color
		e.pending.blendSet = true
	}
	if e.raw == 0 {
		return
	}
	e.applyBlendConstant(*color)
}

func (e *RenderPassEncoder) applyBlendConstant(color gputypes.Color) {
	msgSendVoid(e.raw, Sel("setBlendColorRed:green:blue:alpha:"),
		argFloat32(float32(color.R)),
		argFloat32(float32(color.G)),
		argFloat32(float32(color.B)),
		argFloat32(float32(color.A)),
	)
}

// SetStencilReference sets the stencil reference value.
func (e *RenderPassEncoder) SetStencilReference(ref uint32) {
	if e.pending != nil {
		e.pending.stencil = ref
		e.pending.stencilSet = true
	}
	if e.raw == 0 {
		return
	}
	_ = MsgSend(e.raw, Sel("setStencilReferenceValue:"), uintptr(ref))
}

// beginDraw makes the native encoder and brings the state a draw reads up
// to date.
func (e *RenderPassEncoder) beginDraw() bool {
	if !e.beginNative() {
		return false
	}
	if e.sizesStale && e.pipeline != nil {
		e.sizesStale = false
		e.bindBufferSizes(Sel("setVertexBytes:length:atIndex:"), e.pipeline.vertexSizes, e.pipeline.pulledBuffers)
		e.bindBufferSizes(Sel("setFragmentBytes:length:atIndex:"), e.pipeline.fragmentSizes, nil)
	}
	if e.depthRangeStale && e.pipeline != nil {
		e.depthRangeStale = false
		e.bindFragDepthRange()
	}
	return true
}

// bindFragDepthRange sets the range the fragment stage clamps the frag_depth
// it writes to, when it does: the viewport's depth range, 0 to 1 until the
// pass sets a viewport.
func (e *RenderPassEncoder) bindFragDepthRange() {
	if !e.pipeline.clampsFragDepth {
		return
	}
	depthRange := [2]float32{0, 1}
	if e.pending != nil && e.pending.viewportSet {
		depthRange = [2]float32{float32(e.pending.viewport.ZNear), float32(e.pending.viewport.ZFar)}
	}
	_ = msgSendSetBytes(e.raw, Sel("setFragmentBytes:length:atIndex:"), unsafe.Pointer(&depthRange[0]), 8, NSUInteger(e.pipeline.sizesSlot+1))
}

// bindBufferSizes sets a stage's `_buffer_sizes` argument, the byte sizes of
// the buffers bound at bindings and then of the vertex buffers bound at the
// slots pulled, with the setBytes method sel. See
// ComputePassEncoder.bindBufferSizes.
//
// Reference: Rust wgpu-hal metal/command.rs make_sizes_buffer_update.
func (e *RenderPassEncoder) bindBufferSizes(sel SEL, bindings []ir.ResourceBinding, pulled []uint32) {
	if len(bindings)+len(pulled) == 0 || e.pending == nil {
		return
	}
	sizes := make([]uint32, len(bindings), len(bindings)+len(pulled))
	for i, rb := range bindings {
		if rb.Group >= maxRenderBindGroups || !e.pending.bindGroups[rb.Group].set {
			continue
		}
		for _, entry := range e.pending.bindGroups[rb.Group].group.entries {
			if res, ok := entry.Resource.(gputypes.BufferBinding); ok && entry.Binding == rb.Binding {
				sizes[i] = sizeField(bufferBindingSize(res))
			}
		}
	}
	for _, slot := range pulled {
		var size uint64
		if vb := e.pending.vertexBuffers[slot]; vb.set && vb.buffer.size > vb.offset {
			size = vb.buffer.size - vb.offset
		}
		sizes = append(sizes, sizeField(size))
	}
	_ = msgSendSetBytes(e.raw, sel, unsafe.Pointer(&sizes[0]), NSUInteger(4*len(sizes)), NSUInteger(e.pipeline.sizesSlot))
	runtime.KeepAlive(sizes)
}

// primitive is what the current pipeline's topology assembles.
func (e *RenderPassEncoder) primitive() MTLPrimitiveType {
	if e.pipeline == nil {
		return MTLPrimitiveTypeTriangle
	}
	return e.pipeline.primitive
}

// Draw draws primitives. A draw of nothing is not encoded: Metal's
// validation refuses a vertex or instance count of 0, as Dawn notes.
func (e *RenderPassEncoder) Draw(args gputypes.DrawArgs) {
	if args.VertexCount == 0 || args.InstanceCount == 0 || !e.beginDraw() {
		return
	}
	_ = MsgSend(e.raw, Sel("drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:"),
		uintptr(e.primitive()), uintptr(args.FirstVertex), uintptr(args.VertexCount), uintptr(args.InstanceCount), uintptr(args.FirstInstance))
}

// DrawIndexed draws indexed primitives. A draw of nothing is not encoded,
// as in Draw.
func (e *RenderPassEncoder) DrawIndexed(args gputypes.DrawIndexedArgs) {
	if args.IndexCount == 0 || args.InstanceCount == 0 || e.indexBuffer == nil || !e.beginDraw() {
		return
	}
	indexType := indexFormatToMTL(e.indexFormat)
	indexSize := uint32(2)
	if e.indexFormat == gputypes.IndexFormatUint32 {
		indexSize = 4
	}
	offset := e.indexOffset + uint64(args.FirstIndex)*uint64(indexSize)
	_ = MsgSend(e.raw, Sel("drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:instanceCount:baseVertex:baseInstance:"),
		uintptr(e.primitive()), uintptr(args.IndexCount), uintptr(indexType),
		uintptr(e.indexBuffer.raw), uintptr(offset), uintptr(args.InstanceCount), uintptr(args.BaseVertex), uintptr(args.FirstInstance))
}

// DrawIndirect draws primitives with GPU-generated parameters.
func (e *RenderPassEncoder) DrawIndirect(buffer hal.Buffer, offset uint64, drawCount uint32) {
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil || drawCount == 0 {
		return
	}
	if !indirect.RangeFits(buf.size, offset, 16, drawCount) {
		return
	}
	if _, ok := indirect.RecordOffset(offset, 16, drawCount-1); !ok {
		return
	}
	if !e.beginDraw() {
		return
	}
	for i := uint32(0); i < drawCount; i++ {
		recordOffset, _ := indirect.RecordOffset(offset, 16, i)
		_ = MsgSend(e.raw, Sel("drawPrimitives:indirectBuffer:indirectBufferOffset:"),
			uintptr(e.primitive()), uintptr(buf.raw), uintptr(recordOffset))
	}
}

// DrawIndexedIndirect draws indexed primitives with GPU-generated parameters.
// Metal exposes only the single-record indirect operation, so count is lowered
// to consecutive 20-byte calls.
func (e *RenderPassEncoder) DrawIndexedIndirect(buffer hal.Buffer, offset uint64, drawCount uint32) {
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil || e.indexBuffer == nil || drawCount == 0 {
		return
	}
	if !indirect.RangeFits(buf.size, offset, 20, drawCount) {
		return
	}
	if _, ok := indirect.RecordOffset(offset, 20, drawCount-1); !ok {
		return
	}
	if e.tryFirstIndexedICB(buf, offset, drawCount) {
		return
	}
	if !e.beginDraw() {
		return
	}
	indexType := indexFormatToMTL(e.indexFormat)
	for i := uint32(0); i < drawCount; i++ {
		recordOffset, ok := indirect.RecordOffset(offset, 20, i)
		if !ok {
			return
		}
		_ = MsgSend(e.raw, Sel("drawIndexedPrimitives:indexType:indexBuffer:indexBufferOffset:indirectBuffer:indirectBufferOffset:"),
			uintptr(e.primitive()), uintptr(indexType), uintptr(e.indexBuffer.raw), uintptr(e.indexOffset), uintptr(buf.raw), uintptr(recordOffset))
	}
}

// DrawIndirectCount falls back to maxDrawCount sequential indirect draws on Metal.
func (e *RenderPassEncoder) DrawIndirectCount(
	buffer hal.Buffer, offset uint64,
	countBuffer hal.Buffer, countOffset uint64,
	maxDrawCount uint32,
) {
	hal.RecordIndirectCountMax(e.DrawIndirect, buffer, offset, countBuffer, countOffset, maxDrawCount)
}

// DrawIndexedIndirectCount falls back to maxDrawCount sequential indexed indirect draws.
func (e *RenderPassEncoder) DrawIndexedIndirectCount(
	buffer hal.Buffer, offset uint64,
	countBuffer hal.Buffer, countOffset uint64,
	maxDrawCount uint32,
) {
	hal.RecordIndirectCountMax(e.DrawIndexedIndirect, buffer, offset, countBuffer, countOffset, maxDrawCount)
}

func (e *RenderPassEncoder) tryFirstIndexedICB(buffer *Buffer, offset uint64, count uint32) bool {
	commands, ok := e.prepareIndexedICB(buffer, offset, count)
	if !ok {
		return false
	}
	if !e.beginDraw() {
		return true
	}
	return e.executeIndexedICB(commands, buffer)
}

func indexedIndirectRecordOffset(offset uint64, index uint32) (uint64, bool) {
	return indirect.RecordOffset(offset, 20, index)
}

// ExecuteBundle executes a pre-recorded render bundle.
func (e *RenderPassEncoder) ExecuteBundle(_ hal.RenderBundle) {}

// ComputePassEncoder implements hal.ComputePassEncoder for Metal.
type ComputePassEncoder struct {
	raw            ID
	device         *Device
	commandEncoder *CommandEncoder
	pipeline       *ComputePipeline
	currentLayout  *PipelineLayout // set by SetPipeline for SetBindGroup slot offsets
	// groups are the bind groups set so far, bound again when a pipeline
	// brings another layout, as in RenderPassEncoder.
	groups [maxRenderBindGroups]renderBindGroupState

	// bindingSizes holds the usable byte size of each bound buffer, by group
	// and binding. A dispatch copies these sizes into naga's `_buffer_sizes`
	// argument.
	bindingSizes map[[2]uint32]uint64
}

// End finishes the compute pass.
func (e *ComputePassEncoder) End() {
	e.close()
	if e.commandEncoder != nil && e.commandEncoder.openPass == e {
		e.commandEncoder.openPass = nil
	}
}

func (e *ComputePassEncoder) close() {
	if e.raw != 0 {
		_ = MsgSend(e.raw, Sel("endEncoding"))
		Release(e.raw)
		e.raw = 0
	}
}

// SetPipeline sets the compute pipeline.
func (e *ComputePassEncoder) SetPipeline(pipeline hal.ComputePipeline) {
	p, ok := pipeline.(*ComputePipeline)
	if !ok || p == nil {
		return
	}
	prev := e.pipeline
	e.pipeline = p
	e.currentLayout = p.layout // store for SetBindGroup slot offset lookup
	_ = MsgSend(e.raw, Sel("setComputePipelineState:"), uintptr(p.raw))
	// A group's slots come from the layout: groups set before the first
	// pipeline waited for it, and another layout moves them.
	if prev == nil || prev.layout != p.layout {
		for i := range e.groups {
			if state := &e.groups[i]; state.set {
				e.applyBindGroup(uint32(i), state.group, state.offsets[:state.offsetCount])
			}
		}
	}
}

// SetBindGroup sets a bind group by binding each resource directly on the compute encoder.
//
// See RenderPassEncoder.SetBindGroup for the binding index convention.
func (e *ComputePassEncoder) SetBindGroup(index uint32, group hal.BindGroup, offsets []uint32) {
	bg, ok := group.(*BindGroup)
	if !ok || bg == nil {
		return
	}
	if index < maxRenderBindGroups && len(offsets) <= maxRenderDynamicOffsets {
		state := &e.groups[index]
		state.group = bg
		state.offsetCount = uint8(len(offsets))
		state.set = true
		copy(state.offsets[:], offsets)
		clear(state.offsets[len(offsets):])
		if e.pipeline == nil {
			return // SetPipeline binds it, at its layout's slots
		}
	}
	e.applyBindGroup(index, bg, offsets)
}

func (e *ComputePassEncoder) applyBindGroup(index uint32, bg *BindGroup, offsets []uint32) {
	// Metal uses per-type sequential indices across all bind groups.
	// See RenderPassEncoder.SetBindGroup for detailed explanation.
	off, ok := e.currentLayout.groupSlots(index, bg)
	if !ok {
		return
	}
	bufferSlot, textureSlot, samplerSlot := uintptr(off.Buffers), uintptr(off.Textures), uintptr(off.Samplers)

	var dynamicIdx int
	for _, entry := range bg.entries {
		switch res := entry.Resource.(type) {
		case gputypes.BufferBinding:
			offset := uintptr(res.Offset)
			if dynamicIdx < len(offsets) && bg.layout != nil {
				for _, le := range bg.layout.entries {
					if le.Binding == entry.Binding && le.Buffer != nil && le.Buffer.HasDynamicOffset {
						offset += uintptr(offsets[dynamicIdx])
						dynamicIdx++
						break
					}
				}
			}
			_ = MsgSend(e.raw, Sel("setBuffer:offset:atIndex:"), uintptr(res.Buffer), offset, bufferSlot)
			bufferSlot++

			if e.bindingSizes == nil {
				e.bindingSizes = make(map[[2]uint32]uint64)
			}
			e.bindingSizes[[2]uint32{index, entry.Binding}] = bufferBindingSize(res)

		case gputypes.TextureViewBinding:
			_ = MsgSend(e.raw, Sel("setTexture:atIndex:"), uintptr(res.TextureView), textureSlot)
			textureSlot++

		case gputypes.SamplerBinding:
			_ = MsgSend(e.raw, Sel("setSamplerState:atIndex:"), uintptr(res.Sampler), samplerSlot)
			samplerSlot++
		}
	}
}

// bindBufferSizes binds naga's `_buffer_sizes` argument. Kernels need it when
// they use a runtime-sized array, either through WGSL arrayLength() or through
// naga's bounds checks on storage buffers.
//
// The MSL kernel takes one extra parameter. It holds one uint32 byte size for
// each runtime-sized array. The pipeline compiled that parameter to the buffer
// slot that follows the buffers of the pipeline layout.
//
// This binding is required. If it is missing, the kernel reads a size of zero,
// and naga's arrayLength() formula, 1 + (size - offset - stride) / stride,
// wraps around to a very large number.
//
// Reference: Rust wgpu-hal metal/command.rs:1736-1747.
func (e *ComputePassEncoder) bindBufferSizes() {
	p := e.pipeline
	if p == nil || len(p.sizesBindings) == 0 {
		return
	}
	sizes := make([]uint32, len(p.sizesBindings))
	for i, rb := range p.sizesBindings {
		sizes[i] = sizeField(e.bindingSizes[[2]uint32{rb.Group, rb.Binding}])
	}
	_ = msgSendSetBytes(e.raw, Sel("setBytes:length:atIndex:"),
		unsafe.Pointer(&sizes[0]), NSUInteger(4*len(sizes)), NSUInteger(p.sizesSlot))
	runtime.KeepAlive(sizes)
}

// bufferBindingSize is the byte size of a buffer binding as the shader sees
// it. A zero entry size means the rest of the buffer.
func bufferBindingSize(res gputypes.BufferBinding) uint64 {
	if res.Size != 0 {
		return res.Size
	}
	if l := uint64(MsgSend(ID(res.Buffer), Sel("length"))); l > res.Offset {
		return l - res.Offset
	}
	return 0
}

// sizeField is size as a `_buffer_sizes` field. The fields are 32-bit, so a
// binding above 4 GiB reports the largest value that fits.
func sizeField(size uint64) uint32 {
	return uint32(min(size, math.MaxUint32))
}

// Dispatch dispatches compute workgroups. A dispatch of no workgroups is
// not encoded, as wgpu-hal and Dawn skip it.
func (e *ComputePassEncoder) Dispatch(x, y, z uint32) {
	if e.pipeline == nil || x == 0 || y == 0 || z == 0 {
		return
	}
	e.bindBufferSizes()

	threadgroupsPerGrid := MTLSize{Width: NSUInteger(x), Height: NSUInteger(y), Depth: NSUInteger(z)}
	// Use pipeline's workgroup size instead of hardcoded value
	threadsPerThreadgroup := e.pipeline.workgroupSize

	msgSendVoid(e.raw, Sel("dispatchThreadgroups:threadsPerThreadgroup:"),
		argStruct(threadgroupsPerGrid, mtlSizeType),
		argStruct(threadsPerThreadgroup, mtlSizeType),
	)
	e.insertComputeBarrier()
}

// DispatchIndirect dispatches compute work with GPU-generated parameters.
func (e *ComputePassEncoder) DispatchIndirect(buffer hal.Buffer, offset uint64) {
	if e.pipeline == nil {
		return // No pipeline set
	}
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil {
		return
	}
	e.bindBufferSizes()
	// Use pipeline's workgroup size instead of hardcoded value
	threadsPerThreadgroup := e.pipeline.workgroupSize
	msgSendVoid(e.raw, Sel("dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:"),
		argPointer(uintptr(buf.raw)),
		argUint64(offset),
		argStruct(threadsPerThreadgroup, mtlSizeType),
	)
	e.insertComputeBarrier()
}

// insertComputeBarrier inserts a buffer+texture memory barrier after a dispatch.
// VAL-008: ensures storage buffer/texture writes from one dispatch are visible
// to subsequent dispatches. Uses memoryBarrierWithScope: on the compute encoder.
func (e *ComputePassEncoder) insertComputeBarrier() {
	if e.raw == 0 {
		return
	}
	scope := MTLBarrierScopeBuffers | MTLBarrierScopeTextures
	_ = MsgSend(e.raw, Sel("memoryBarrierWithScope:"), uintptr(scope))
}

// msgSendClearColor sends an Objective-C message with an MTLClearColor argument.
func msgSendClearColor(obj ID, sel SEL, color MTLClearColor) {
	if obj == 0 {
		return
	}
	msgSendVoid(obj, sel, argStruct(color, mtlClearColorType))
}
