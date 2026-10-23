// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// planRegionCopies plans every region of a buffer/texture copy, or records
// the copy as the encoder's error when a region has no plan, so that the
// command buffer fails instead of silently missing the copy.
func (e *CommandEncoder) planRegionCopies(texture *Texture, regions []hal.BufferTextureCopy) [][]bufferTextureCopyPlan {
	regionPlans := make([][]bufferTextureCopyPlan, len(regions))
	for i, region := range regions {
		regionPlans[i] = planBufferTextureCopies(texture, region.TextureBase, region.BufferLayout, region.Size)
		if regionPlans[i] == nil {
			e.fail(fmt.Errorf("dx12: buffer/texture copy of %v aspect %v at %+v size %+v with layout %+v cannot be recorded", texture.format, region.TextureBase.Aspect, region.TextureBase.Origin, region.Size, region.BufferLayout))
		}
	}
	return regionPlans
}

// copyThroughScratch reports a copy of a depth or stencil plane at a buffer
// offset that is not 512-aligned. D3D12 copies a depth/stencil subresource
// whole, with no box inside the buffer footprint on devices without
// programmable sample positions, so such a copy goes through a scratch
// buffer it starts at offset 0 of, as Dawn and Rust's wgpu-hal do.
func copyThroughScratch(texture *Texture, layout hal.ImageDataLayout) bool {
	return texture.format.IsDepthStencil() && layout.Offset%d3d12TexturePlacementAlignment != 0
}

// copyByteSpan is the number of bytes from the layout's offset a copy of
// region reads or writes: every row runs the row pitch but the last row of
// the last image, which runs its blocks only.
func copyByteSpan(texture *Texture, region hal.BufferTextureCopy) (uint64, bool) {
	plane, ok := copyPlane(texture, region.TextureBase.Aspect)
	if !ok {
		return 0, false
	}
	info, rowPitch, blockRows, ok := normalizedCopyLayout(texture, plane, region.BufferLayout, region.Size)
	if !ok {
		return 0, false
	}
	width, height, ok := copyExtentBlocks(info, region.TextureBase.Origin, region.Size)
	if !ok {
		return 0, false
	}
	depth := uint64(max(region.Size.DepthOrArrayLayers, 1))
	return (depth-1)*uint64(blockRows)*uint64(rowPitch) + uint64(height-1)*uint64(rowPitch) + uint64(width)*uint64(info.bytes), true
}

func (e *CommandEncoder) copyBufferToTextureThroughScratch(src *Buffer, texture *Texture, region hal.BufferTextureCopy) {
	span, ok := copyByteSpan(texture, region)
	if !ok {
		e.fail(fmt.Errorf("dx12: buffer/texture copy of %v with layout %+v cannot be recorded", texture.format, region.BufferLayout))
		return
	}
	scratch, err := e.scratchBuffer(span)
	if err != nil {
		e.fail(fmt.Errorf("dx12: buffer/texture copy scratch: %w", err))
		return
	}
	e.transitionCopyBuffers(src, scratch)
	e.cmdList.CopyBufferRegion(scratch.raw, 0, src.raw, region.BufferLayout.Offset, span)
	region.BufferLayout.Offset = 0
	e.copyBufferToTexture(scratch, texture, []hal.BufferTextureCopy{region})
}

func (e *CommandEncoder) copyTextureToBufferThroughScratch(src *Texture, dst *Buffer, region hal.BufferTextureCopy) {
	span, ok := copyByteSpan(src, region)
	if !ok {
		e.fail(fmt.Errorf("dx12: buffer/texture copy of %v with layout %+v cannot be recorded", src.format, region.BufferLayout))
		return
	}
	scratch, err := e.scratchBuffer(span)
	if err != nil {
		e.fail(fmt.Errorf("dx12: buffer/texture copy scratch: %w", err))
		return
	}
	offset := region.BufferLayout.Offset
	region.BufferLayout.Offset = 0
	e.copyTextureToBuffer(src, scratch, []hal.BufferTextureCopy{region})
	e.transitionCopyBuffers(scratch, dst)
	e.cmdList.CopyBufferRegion(dst.raw, offset, scratch.raw, 0, span)
}

// transitionCopyBuffers moves src to COPY_SOURCE and dst to COPY_DEST.
func (e *CommandEncoder) transitionCopyBuffers(src, dst *Buffer) {
	plans := make([]stateBarrierPlan, 0, 2)
	if before, barrier := e.stateTracker.transitionBuffer(src, d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE); barrier {
		plans = append(plans, stateBarrierPlan{resource: src, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE})
	}
	if before, barrier := e.stateTracker.transitionBuffer(dst, d3d12.D3D12_RESOURCE_STATE_COPY_DEST); barrier {
		plans = append(plans, stateBarrierPlan{resource: dst, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_DEST})
	}
	e.emitStateBarrierPlans(plans)
}

func (e *CommandEncoder) copyBufferToTexture(src *Buffer, texture *Texture, regions []hal.BufferTextureCopy) {
	direct := make([]hal.BufferTextureCopy, 0, len(regions))
	for _, region := range regions {
		if copyThroughScratch(texture, region.BufferLayout) {
			e.copyBufferToTextureThroughScratch(src, texture, region)
		} else {
			direct = append(direct, region)
		}
	}
	regions = direct
	barriers := make([]stateBarrierPlan, 0, 1+len(regions))
	if before, barrier := e.stateTracker.transitionBuffer(src, d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE); barrier {
		barriers = append(barriers, stateBarrierPlan{resource: src, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE})
	}
	regionPlans := e.planRegionCopies(texture, regions)
	for _, plans := range regionPlans {
		for _, copyPlan := range plans {
			if before, barrier := e.stateTracker.transitionTexture(texture, copyPlan.subresource, d3d12.D3D12_RESOURCE_STATE_COPY_DEST); barrier {
				barriers = append(barriers, stateBarrierPlan{resource: texture, subresource: copyPlan.subresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_DEST})
			}
		}
	}
	e.emitStateBarrierPlans(barriers)
	for _, plans := range regionPlans {
		for _, plan := range plans {
			srcLoc := placedFootprintLocation(src, texture.format, plan)
			dstLoc := subresourceLocation(texture, plan.subresource)
			box := d3d12.D3D12_BOX{
				Left:   plan.bufferOriginX,
				Top:    plan.bufferOriginY,
				Front:  0,
				Right:  plan.bufferOriginX + plan.copyWidth,
				Bottom: plan.bufferOriginY + plan.copyHeight,
				Back:   plan.footprintDepth,
			}
			e.cmdList.CopyTextureRegion(&dstLoc, plan.textureOriginX, plan.textureOriginY, plan.textureOriginZ, &srcLoc, &box)
		}
	}
}

func (e *CommandEncoder) copyTextureToBuffer(src *Texture, dst *Buffer, regions []hal.BufferTextureCopy) {
	direct := make([]hal.BufferTextureCopy, 0, len(regions))
	for _, region := range regions {
		if copyThroughScratch(src, region.BufferLayout) {
			e.copyTextureToBufferThroughScratch(src, dst, region)
		} else {
			direct = append(direct, region)
		}
	}
	regions = direct
	barriers := make([]stateBarrierPlan, 0, 1+len(regions))
	if before, barrier := e.stateTracker.transitionBuffer(dst, d3d12.D3D12_RESOURCE_STATE_COPY_DEST); barrier {
		barriers = append(barriers, stateBarrierPlan{resource: dst, subresource: d3d12.D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_DEST})
	}
	regionPlans := e.planRegionCopies(src, regions)
	for _, plans := range regionPlans {
		for _, copyPlan := range plans {
			if before, barrier := e.stateTracker.transitionTexture(src, copyPlan.subresource, d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE); barrier {
				barriers = append(barriers, stateBarrierPlan{resource: src, subresource: copyPlan.subresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE})
			}
		}
	}
	e.emitStateBarrierPlans(barriers)
	for _, plans := range regionPlans {
		for _, plan := range plans {
			srcLoc := subresourceLocation(src, plan.subresource)
			dstLoc := placedFootprintLocation(dst, src.format, plan)
			box := d3d12.D3D12_BOX{
				Left:   plan.textureOriginX,
				Top:    plan.textureOriginY,
				Front:  plan.textureOriginZ,
				Right:  plan.textureOriginX + plan.copyWidth,
				Bottom: plan.textureOriginY + plan.copyHeight,
				Back:   plan.textureOriginZ + plan.footprintDepth,
			}
			e.cmdList.CopyTextureRegion(&dstLoc, plan.bufferOriginX, plan.bufferOriginY, 0, &srcLoc, &box)
		}
	}
}

func (e *CommandEncoder) copyTextureToTexture(src, dst *Texture, regions []hal.TextureCopy) {
	plans := make([]stateBarrierPlan, 0, len(regions)*2)
	for _, region := range regions {
		copyPlans := planTextureTextureCopies(src, dst, region)
		if copyPlans == nil {
			e.fail(fmt.Errorf("dx12: texture copy of %v %v to %v %v from %+v to %+v size %+v cannot be recorded",
				src.dimension, src.format, dst.dimension, dst.format, region.SrcBase.Origin, region.DstBase.Origin, region.Size))
		}
		for _, copyPlan := range copyPlans {
			if before, barrier := e.stateTracker.transitionTexture(src, copyPlan.srcSubresource, d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE); barrier {
				plans = append(plans, stateBarrierPlan{resource: src, subresource: copyPlan.srcSubresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE})
			}
			if before, barrier := e.stateTracker.transitionTexture(dst, copyPlan.dstSubresource, d3d12.D3D12_RESOURCE_STATE_COPY_DEST); barrier {
				plans = append(plans, stateBarrierPlan{resource: dst, subresource: copyPlan.dstSubresource, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_DEST})
			}
		}
	}
	e.emitStateBarrierPlans(plans)
	for _, region := range regions {
		for _, plan := range planTextureTextureCopies(src, dst, region) {
			srcLoc := subresourceLocation(src, plan.srcSubresource)
			dstLoc := subresourceLocation(dst, plan.dstSubresource)
			box := d3d12.D3D12_BOX{
				Left:   region.SrcBase.Origin.X,
				Top:    region.SrcBase.Origin.Y,
				Front:  plan.srcFront,
				Right:  region.SrcBase.Origin.X + region.Size.Width,
				Bottom: region.SrcBase.Origin.Y + region.Size.Height,
				Back:   plan.srcBack,
			}
			e.cmdList.CopyTextureRegion(&dstLoc, region.DstBase.Origin.X, region.DstBase.Origin.Y, plan.dstZ, &srcLoc, &box)
		}
	}
}

func placedFootprintLocation(buffer *Buffer, format gputypes.TextureFormat, plan bufferTextureCopyPlan) d3d12.D3D12_TEXTURE_COPY_LOCATION {
	copyFormat, _ := textureCopyFormat(format, plan.plane)
	location := d3d12.D3D12_TEXTURE_COPY_LOCATION{Resource: buffer.raw, Type: d3d12.D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}
	location.SetPlacedFootprint(d3d12.D3D12_PLACED_SUBRESOURCE_FOOTPRINT{
		Offset: plan.bufferOffset,
		Footprint: d3d12.D3D12_SUBRESOURCE_FOOTPRINT{
			Format:   copyFormat,
			Width:    plan.footprintWidth,
			Height:   plan.footprintHeight,
			Depth:    plan.footprintDepth,
			RowPitch: plan.rowPitch,
		},
	})
	return location
}

func subresourceLocation(texture *Texture, subresource uint32) d3d12.D3D12_TEXTURE_COPY_LOCATION {
	location := d3d12.D3D12_TEXTURE_COPY_LOCATION{Resource: texture.raw, Type: d3d12.D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}
	location.SetSubresourceIndex(subresource)
	return location
}
