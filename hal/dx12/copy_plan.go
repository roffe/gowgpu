// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"math"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

const d3d12TexturePlacementAlignment = 512

// bufferTextureCopyPlan describes one native D3D12 copy. WebGPU treats a
// 2D-array layer as a separate subresource, while D3D12 treats a 3D slice as
// a depth-one copy into one subresource. The buffer origin fields select the
// logical byte range inside an aligned, possibly enlarged footprint.
type bufferTextureCopyPlan struct {
	subresource     uint32
	plane           uint32
	bufferOffset    uint64
	bufferOriginX   uint32
	bufferOriginY   uint32
	footprintWidth  uint32
	footprintHeight uint32
	footprintDepth  uint32
	textureOriginX  uint32
	textureOriginY  uint32
	textureOriginZ  uint32
	rowPitch        uint32
	copyWidth       uint32
	copyHeight      uint32
}

type textureTextureCopyPlan struct {
	srcSubresource uint32
	dstSubresource uint32
	srcFront       uint32
	srcBack        uint32
	dstZ           uint32
}

type textureBlockInfo struct {
	width  uint32
	height uint32
	bytes  uint32
}

// textureCopyFormat is the format a buffer copy of one physical plane of a
// texture of format goes through and the bytes per block of it: the format
// itself for a colour texture, and for a depth/stencil texture the plane's
// own, as D3D12 copies depth as R16_UNORM or R32_FLOAT and stencil as
// R8_UINT. A plane with no copy, 24-bit depth, has no format.
func textureCopyFormat(format gputypes.TextureFormat, plane uint32) (d3d12.DXGI_FORMAT, uint32) {
	if !format.IsDepthStencil() {
		return textureFormatToD3D12(format), format.BlockCopySize()
	}
	if plane == 1 {
		if format.HasStencil() {
			return d3d12.DXGI_FORMAT_R8_UINT, 1
		}
		return d3d12.DXGI_FORMAT_UNKNOWN, 0
	}
	switch format {
	case gputypes.TextureFormatDepth16Unorm:
		return d3d12.DXGI_FORMAT_R16_UNORM, 2
	case gputypes.TextureFormatDepth32Float, gputypes.TextureFormatDepth32FloatStencil8:
		return d3d12.DXGI_FORMAT_R32_FLOAT, 4
	default:
		return d3d12.DXGI_FORMAT_UNKNOWN, 0
	}
}

func textureBlockInfoForPlane(format gputypes.TextureFormat, plane uint32) (textureBlockInfo, bool) {
	_, bytes := textureCopyFormat(format, plane)
	if bytes == 0 {
		return textureBlockInfo{}, false
	}
	if textureFormatIsBC(format) {
		return textureBlockInfo{width: 4, height: 4, bytes: bytes}, true
	}
	return textureBlockInfo{width: 1, height: 1, bytes: bytes}, true
}

func textureFormatIsBC(format gputypes.TextureFormat) bool {
	switch format {
	case gputypes.TextureFormatBC1RGBAUnorm,
		gputypes.TextureFormatBC1RGBAUnormSrgb,
		gputypes.TextureFormatBC2RGBAUnorm,
		gputypes.TextureFormatBC2RGBAUnormSrgb,
		gputypes.TextureFormatBC3RGBAUnorm,
		gputypes.TextureFormatBC3RGBAUnormSrgb,
		gputypes.TextureFormatBC4RUnorm,
		gputypes.TextureFormatBC4RSnorm,
		gputypes.TextureFormatBC5RGUnorm,
		gputypes.TextureFormatBC5RGSnorm,
		gputypes.TextureFormatBC6HRGBUfloat,
		gputypes.TextureFormatBC6HRGBFloat,
		gputypes.TextureFormatBC7RGBAUnorm,
		gputypes.TextureFormatBC7RGBAUnormSrgb:
		return true
	default:
		return false
	}
}

func textureFormatBlockHeight(format gputypes.TextureFormat) uint32 {
	if info, ok := textureBlockInfoForPlane(format, 0); ok {
		return info.height
	}
	return 1
}

func alignUint32(value, alignment uint32) (uint32, bool) {
	if alignment == 0 {
		return value, true
	}
	if value > math.MaxUint32-(alignment-1) {
		return 0, false
	}
	return (value + alignment - 1) / alignment * alignment, true
}

func copyExtentBlocks(info textureBlockInfo, origin hal.Origin3D, extent hal.Extent3D) (width, height uint32, ok bool) {
	if extent.Width == 0 || extent.Height == 0 {
		return 0, 0, false
	}
	if origin.X > math.MaxUint32-extent.Width || origin.Y > math.MaxUint32-extent.Height {
		return 0, 0, false
	}
	endX := origin.X + extent.Width
	endY := origin.Y + extent.Height
	if endX > math.MaxUint32-(info.width-1) || endY > math.MaxUint32-(info.height-1) {
		return 0, 0, false
	}
	startBlockX := origin.X / info.width
	startBlockY := origin.Y / info.height
	endBlockX := (endX + info.width - 1) / info.width
	endBlockY := (endY + info.height - 1) / info.height
	if endBlockX < startBlockX || endBlockY < startBlockY {
		return 0, 0, false
	}
	return endBlockX - startBlockX, endBlockY - startBlockY, true
}

// placedSlice is one native copy of a placed slice: a footprint at a
// 512-aligned bufferOffset holding the copy's data from bufferOrigin on,
// and the copyWidth texels of the copy's rows from textureOffsetX on that
// it covers.
type placedSlice struct {
	bufferOffset    uint64
	bufferOriginX   uint32
	bufferOriginY   uint32
	footprintWidth  uint32
	footprintHeight uint32
	textureOffsetX  uint32
	copyWidth       uint32
}

// planPlacedBufferSlice places the copyWidth by copyHeight texels whose data
// starts at logicalOffset in footprints at 512-aligned offsets. The data
// starts inside a footprint; when its rows run past the row pitch from there,
// the copy is split in two at the pitch, as Dawn's TextureCopySplitter does:
// the first copy takes the part of each row up to the pitch, the second the
// rest, from a footprint of its own.
func planPlacedBufferSlice(logicalOffset uint64, rowPitch, blockWidth, blockHeight, blockBytes, copyWidth, copyHeight uint32) ([]placedSlice, bool) {
	if rowPitch == 0 || blockWidth == 0 || blockHeight == 0 || blockBytes == 0 || rowPitch%blockBytes != 0 {
		return nil, false
	}
	blocksPerRow := rowPitch / blockBytes
	widthBlocks := copyWidth / blockWidth
	if logicalOffset%d3d12TexturePlacementAlignment != 0 && logicalOffset >= uint64(rowPitch) {
		previousRow := logicalOffset - uint64(rowPitch)
		if previousRow%d3d12TexturePlacementAlignment == 0 && widthBlocks <= blocksPerRow {
			slice, ok := placeSlice(previousRow, 0, 1, widthBlocks, blockWidth, blockHeight, copyHeight)
			if !ok {
				return nil, false
			}
			slice.copyWidth = copyWidth
			return []placedSlice{slice}, true
		}
	}
	x, y, ok := placedOrigin(logicalOffset, blockBytes, blocksPerRow)
	if !ok {
		return nil, false
	}
	bufferOffset := logicalOffset &^ (d3d12TexturePlacementAlignment - 1)
	if x+widthBlocks <= blocksPerRow {
		slice, ok := placeSlice(bufferOffset, x, y, widthBlocks, blockWidth, blockHeight, copyHeight)
		if !ok {
			return nil, false
		}
		slice.copyWidth = copyWidth
		return []placedSlice{slice}, true
	}

	firstBlocks := blocksPerRow - x
	first, ok := placeSlice(bufferOffset, x, y, firstBlocks, blockWidth, blockHeight, copyHeight)
	if !ok {
		return nil, false
	}
	first.copyWidth = firstBlocks * blockWidth

	secondOffset := logicalOffset + uint64(firstBlocks)*uint64(blockBytes)
	secondBlocks := widthBlocks - firstBlocks
	x, y, ok = placedOrigin(secondOffset, blockBytes, blocksPerRow)
	if !ok || x+secondBlocks > blocksPerRow {
		return nil, false
	}
	second, ok := placeSlice(secondOffset&^(d3d12TexturePlacementAlignment-1), x, y, secondBlocks, blockWidth, blockHeight, copyHeight)
	if !ok {
		return nil, false
	}
	second.textureOffsetX = first.copyWidth
	second.copyWidth = secondBlocks * blockWidth
	return []placedSlice{first, second}, true
}

// placedOrigin is the block a copy whose data starts at offset starts at in
// a footprint at offset aligned down to 512 with rows blocksPerRow blocks
// long.
func placedOrigin(offset uint64, blockBytes, blocksPerRow uint32) (x, y uint32, ok bool) {
	delta := offset % d3d12TexturePlacementAlignment
	if delta%uint64(blockBytes) != 0 {
		return 0, 0, false
	}
	blocks := uint32(delta) / blockBytes
	return blocks % blocksPerRow, blocks / blocksPerRow, true
}

// placeSlice is the footprint at bufferOffset holding widthBlocks blocks of
// copyHeight texel rows from block (x, y) on.
func placeSlice(bufferOffset uint64, x, y, widthBlocks, blockWidth, blockHeight, copyHeight uint32) (placedSlice, bool) {
	fpWidth := (uint64(x) + uint64(widthBlocks)) * uint64(blockWidth)
	originY := uint64(y) * uint64(blockHeight)
	fpHeight := originY + uint64(copyHeight)
	if fpWidth > math.MaxUint32 || fpHeight > math.MaxUint32-uint64(blockHeight) {
		return placedSlice{}, false
	}
	alignedHeight, _ := alignUint32(uint32(fpHeight), blockHeight)
	return placedSlice{
		bufferOffset:    bufferOffset,
		bufferOriginX:   x * blockWidth,
		bufferOriginY:   uint32(originY),
		footprintWidth:  uint32(fpWidth),
		footprintHeight: alignedHeight,
	}, true
}

// copyPlane is the one physical plane a buffer copy of aspect of texture
// touches. The planes of a depth/stencil texture have different layouts,
// so a copy addresses one of them.
func copyPlane(texture *Texture, aspect gputypes.TextureAspect) (uint32, bool) {
	planes := textureAspectPlanes(texture, aspect)
	if len(planes) != 1 {
		return 0, false
	}
	return planes[0], true
}

func normalizedCopyLayout(texture *Texture, plane uint32, layout hal.ImageDataLayout, size hal.Extent3D) (textureBlockInfo, uint32, uint32, bool) {
	if texture == nil {
		return textureBlockInfo{}, 0, 0, false
	}
	info, ok := textureBlockInfoForPlane(texture.format, plane)
	if !ok {
		return textureBlockInfo{}, 0, 0, false
	}
	minWidth := (uint64(size.Width) + uint64(info.width) - 1) / uint64(info.width)
	minRowBytes64 := minWidth * uint64(info.bytes)
	if minRowBytes64 == 0 || minRowBytes64 > math.MaxUint32 {
		return textureBlockInfo{}, 0, 0, false
	}
	rowPitch := layout.BytesPerRow
	if rowPitch == 0 {
		if size.Height > info.height {
			return textureBlockInfo{}, 0, 0, false
		}
		aligned, alignedOK := alignUint32(uint32(minRowBytes64), d3d12TexturePitchAlignment)
		if !alignedOK {
			return textureBlockInfo{}, 0, 0, false
		}
		rowPitch = aligned
	} else if uint64(rowPitch) < minRowBytes64 || rowPitch%d3d12TexturePitchAlignment != 0 {
		return textureBlockInfo{}, 0, 0, false
	}
	rowsPerImage := layout.RowsPerImage
	if rowsPerImage == 0 {
		rowsPerImage = size.Height
	}
	if rowsPerImage < size.Height {
		return textureBlockInfo{}, 0, 0, false
	}
	blockRows := (rowsPerImage + info.height - 1) / info.height
	return info, rowPitch, blockRows, true
}

func textureWriteSourceOffset(base uint64, sourceBytesPerRow, blockRows, slice, row uint32) uint64 {
	return base + uint64(slice)*uint64(sourceBytesPerRow)*uint64(blockRows) + uint64(row)*uint64(sourceBytesPerRow)
}

func writeTextureNativeLayout(texture *Texture, aspect gputypes.TextureAspect, layout hal.ImageDataLayout, size hal.Extent3D) (textureBlockInfo, hal.ImageDataLayout, uint32, uint32, bool) {
	if texture == nil || size.Width == 0 {
		return textureBlockInfo{}, hal.ImageDataLayout{}, 0, 0, false
	}
	plane, ok := copyPlane(texture, aspect)
	if !ok {
		return textureBlockInfo{}, hal.ImageDataLayout{}, 0, 0, false
	}
	info, ok := textureBlockInfoForPlane(texture.format, plane)
	if !ok {
		return textureBlockInfo{}, hal.ImageDataLayout{}, 0, 0, false
	}
	logicalRowBytes64 := ((uint64(size.Width) + uint64(info.width) - 1) / uint64(info.width)) * uint64(info.bytes)
	if logicalRowBytes64 == 0 || logicalRowBytes64 > math.MaxUint32 {
		return textureBlockInfo{}, hal.ImageDataLayout{}, 0, 0, false
	}
	sourceBytesPerRow := layout.BytesPerRow
	if sourceBytesPerRow == 0 {
		sourceBytesPerRow = uint32(logicalRowBytes64)
	}
	if uint64(sourceBytesPerRow) < logicalRowBytes64 {
		return textureBlockInfo{}, hal.ImageDataLayout{}, 0, 0, false
	}
	nativeRowPitch, ok := alignUint32(uint32(logicalRowBytes64), d3d12TexturePitchAlignment)
	if !ok {
		return textureBlockInfo{}, hal.ImageDataLayout{}, 0, 0, false
	}
	nativeLayout := layout
	nativeLayout.BytesPerRow = nativeRowPitch
	sourceRowsPerImage := layout.RowsPerImage
	if sourceRowsPerImage == 0 {
		sourceRowsPerImage = size.Height
	}
	if sourceRowsPerImage < size.Height {
		return textureBlockInfo{}, hal.ImageDataLayout{}, 0, 0, false
	}
	sourceBlockRows := (sourceRowsPerImage + info.height - 1) / info.height
	nativeLayout.RowsPerImage = size.Height
	if _, _, _, valid := normalizedCopyLayout(texture, plane, nativeLayout, size); !valid {
		return textureBlockInfo{}, hal.ImageDataLayout{}, 0, 0, false
	}
	return info, nativeLayout, sourceBytesPerRow, sourceBlockRows, true
}

// planBufferTextureCopies converts a WebGPU buffer/texture copy into one plan
// per D3D12 subresource or 3D depth slice, or two where the rows straddle
// the row pitch (see planPlacedBufferSlice). It returns nil when the layout
// cannot be represented without changing logical byte addresses.
func planBufferTextureCopies(texture *Texture, region hal.ImageCopyTexture, layout hal.ImageDataLayout, size hal.Extent3D) []bufferTextureCopyPlan {
	plane, ok := copyPlane(texture, region.Aspect)
	if !ok {
		return nil
	}
	info, rowPitch, blockRows, ok := normalizedCopyLayout(texture, plane, layout, size)
	if !ok {
		return nil
	}
	widthBlocks, heightBlocks, ok := copyExtentBlocks(info, region.Origin, size)
	if !ok || widthBlocks == 0 || heightBlocks == 0 {
		return nil
	}
	if region.MipLevel >= maxTextureMips(texture) {
		return nil
	}
	depth := size.DepthOrArrayLayers
	if depth == 0 {
		depth = 1
	}
	if !validTextureCopyDepth(texture, region.Origin.Z, depth) {
		return nil
	}
	stride := uint64(rowPitch) * uint64(blockRows)
	if stride > math.MaxUint64/uint64(depth) {
		return nil
	}
	plans := make([]bufferTextureCopyPlan, 0, depth)
	for slice := uint32(0); slice < depth; slice++ {
		logicalOffset := layout.Offset + uint64(slice)*stride
		if logicalOffset < layout.Offset {
			return nil
		}
		placed, representable := planPlacedBufferSlice(
			logicalOffset, rowPitch, info.width, info.height, info.bytes,
			widthBlocks*info.width, heightBlocks*info.height)
		if !representable {
			return nil
		}
		subresource := texture.subresourceIndexForPlane(region.MipLevel, 0, plane)
		textureOriginZ := region.Origin.Z + slice
		if texture.dimension != gputypes.TextureDimension3D {
			subresource = texture.subresourceIndexForPlane(region.MipLevel, region.Origin.Z+slice, plane)
			textureOriginZ = 0
		}
		for _, p := range placed {
			plans = append(plans, bufferTextureCopyPlan{
				subresource:     subresource,
				plane:           plane,
				bufferOffset:    p.bufferOffset,
				bufferOriginX:   p.bufferOriginX,
				bufferOriginY:   p.bufferOriginY,
				footprintWidth:  p.footprintWidth,
				footprintHeight: p.footprintHeight,
				footprintDepth:  1,
				textureOriginX:  region.Origin.X + p.textureOffsetX,
				textureOriginY:  region.Origin.Y,
				textureOriginZ:  textureOriginZ,
				rowPitch:        rowPitch,
				copyWidth:       p.copyWidth,
				copyHeight:      heightBlocks * info.height,
			})
		}
	}
	return plans
}

func maxTextureMips(texture *Texture) uint32 {
	if texture.mipLevels == 0 {
		return 1
	}
	return texture.mipLevels
}

func validTextureCopyDepth(texture *Texture, origin, depth uint32) bool {
	if origin > math.MaxUint32-depth {
		return false
	}
	if texture.dimension == gputypes.TextureDimension3D {
		layers := texture.size.DepthOrArrayLayers
		return layers == 0 || origin+depth <= layers
	}
	layers := texture.size.DepthOrArrayLayers
	if layers == 0 {
		layers = 1
	}
	return origin+depth <= layers
}

// planTextureTextureCopies applies D3D12's distinct 3D-volume and 2D-array
// copy models while pairing selected source and destination planes. A copy
// between a 2D array and a 3D texture pairs each layer with a depth slice,
// one copy each, as Dawn does.
func planTextureTextureCopies(src, dst *Texture, region hal.TextureCopy) []textureTextureCopyPlan {
	if src == nil || dst == nil {
		return nil
	}
	depth := region.Size.DepthOrArrayLayers
	if depth == 0 {
		depth = 1
	}
	if region.SrcBase.MipLevel >= maxTextureMips(src) || region.DstBase.MipLevel >= maxTextureMips(dst) {
		return nil
	}
	if !validTextureCopyDepth(src, region.SrcBase.Origin.Z, depth) || !validTextureCopyDepth(dst, region.DstBase.Origin.Z, depth) {
		return nil
	}
	srcPlanes := textureAspectPlanes(src, region.SrcBase.Aspect)
	dstPlanes := textureAspectPlanes(dst, region.DstBase.Aspect)
	if len(srcPlanes) == 0 || len(srcPlanes) != len(dstPlanes) {
		return nil
	}
	if region.Size.Width == 0 || region.Size.Height == 0 {
		return nil
	}
	if src.dimension == gputypes.TextureDimension3D && dst.dimension == gputypes.TextureDimension3D {
		plans := make([]textureTextureCopyPlan, 0, len(srcPlanes))
		for plane := range srcPlanes {
			plans = append(plans, textureTextureCopyPlan{
				srcSubresource: src.subresourceIndexForPlane(region.SrcBase.MipLevel, 0, srcPlanes[plane]),
				dstSubresource: dst.subresourceIndexForPlane(region.DstBase.MipLevel, 0, dstPlanes[plane]),
				srcFront:       region.SrcBase.Origin.Z,
				srcBack:        region.SrcBase.Origin.Z + depth,
				dstZ:           region.DstBase.Origin.Z,
			})
		}
		return plans
	}
	plans := make([]textureTextureCopyPlan, 0, len(srcPlanes)*int(depth))
	for plane := range srcPlanes {
		for layer := uint32(0); layer < depth; layer++ {
			p := textureTextureCopyPlan{srcBack: 1}
			srcZ, dstZ := region.SrcBase.Origin.Z+layer, region.DstBase.Origin.Z+layer
			if src.dimension == gputypes.TextureDimension3D {
				p.srcSubresource = src.subresourceIndexForPlane(region.SrcBase.MipLevel, 0, srcPlanes[plane])
				p.srcFront, p.srcBack = srcZ, srcZ+1
			} else {
				p.srcSubresource = src.subresourceIndexForPlane(region.SrcBase.MipLevel, srcZ, srcPlanes[plane])
			}
			if dst.dimension == gputypes.TextureDimension3D {
				p.dstSubresource = dst.subresourceIndexForPlane(region.DstBase.MipLevel, 0, dstPlanes[plane])
				p.dstZ = dstZ
			} else {
				p.dstSubresource = dst.subresourceIndexForPlane(region.DstBase.MipLevel, dstZ, dstPlanes[plane])
			}
			plans = append(plans, p)
		}
	}
	return plans
}
