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

func (e *CommandEncoder) copyBufferToTexture(src *Buffer, texture *Texture, regions []hal.BufferTextureCopy) {
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
		for _, copyPlan := range planTextureTextureCopies(src, dst, region) {
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
