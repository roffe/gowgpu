// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin && !(js && wasm)

package metal

import (
	"fmt"
	"math"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

type metalTextureShape struct {
	depth       uint32
	arrayLength uint32
}

// metalTextureDescriptorShape separates WebGPU's combined depth-or-layers
// value into Metal's physical depth and array length properties.
func metalTextureDescriptorShape(dimension gputypes.TextureDimension, depthOrArrayLayers uint32) metalTextureShape {
	if depthOrArrayLayers == 0 {
		depthOrArrayLayers = 1
	}
	if dimension == gputypes.TextureDimension3D {
		return metalTextureShape{depth: depthOrArrayLayers, arrayLength: 1}
	}
	return metalTextureShape{depth: 1, arrayLength: depthOrArrayLayers}
}

type metalTextureCopyRegion struct {
	slice  NSUInteger
	origin MTLOrigin
	size   MTLSize
}

type metalCopyPlan struct {
	operationCount uint32
	splitDepth     bool
}

func planMetalBufferTextureCopy(dimension gputypes.TextureDimension, depthOrArrayLayers uint32) metalCopyPlan {
	return newMetalCopyPlan(dimension != gputypes.TextureDimension3D, depthOrArrayLayers)
}

func planMetalTextureCopy(sourceDimension, destinationDimension gputypes.TextureDimension, depthOrArrayLayers uint32) metalCopyPlan {
	return newMetalCopyPlan(
		sourceDimension != gputypes.TextureDimension3D || destinationDimension != gputypes.TextureDimension3D,
		depthOrArrayLayers,
	)
}

func newMetalCopyPlan(splitDepth bool, depthOrArrayLayers uint32) metalCopyPlan {
	if depthOrArrayLayers == 0 {
		return metalCopyPlan{splitDepth: splitDepth}
	}
	operationCount := uint32(1)
	if splitDepth {
		operationCount = depthOrArrayLayers
	}
	return metalCopyPlan{operationCount: operationCount, splitDepth: splitDepth}
}

func (p metalCopyPlan) textureRegion(dimension gputypes.TextureDimension, origin hal.Origin3D, size hal.Extent3D, operation uint32) (metalTextureCopyRegion, bool) {
	if operation >= p.operationCount {
		return metalTextureCopyRegion{}, false
	}
	region := metalTextureCopyRegion{
		origin: MTLOrigin{X: NSUInteger(origin.X), Y: NSUInteger(origin.Y), Z: NSUInteger(origin.Z)},
		size: MTLSize{
			Width:  NSUInteger(size.Width),
			Height: NSUInteger(size.Height),
			Depth:  NSUInteger(size.DepthOrArrayLayers),
		},
	}
	if dimension != gputypes.TextureDimension3D {
		layer, ok := checkedMetalTextureDataAdd32(origin.Z, operation)
		if !ok {
			return metalTextureCopyRegion{}, false
		}
		region.slice = NSUInteger(layer)
		region.origin.Z = 0
		region.size.Depth = 1
	} else if p.splitDepth {
		depth, ok := checkedMetalTextureDataAdd32(origin.Z, operation)
		if !ok {
			return metalTextureCopyRegion{}, false
		}
		region.origin.Z = NSUInteger(depth)
		region.size.Depth = 1
	}
	return region, true
}

func (p metalCopyPlan) bufferOffset(base, bytesPerImage uint64, operation uint32) (uint64, bool) {
	if operation >= p.operationCount {
		return 0, false
	}
	if !p.splitDepth {
		return base, true
	}
	imageOffset, ok := checkedMetalTextureDataMul(uint64(operation), bytesPerImage)
	if !ok {
		return 0, false
	}
	return checkedMetalTextureDataAdd(base, imageOffset)
}

func metalCopyBytesPerImage(format gputypes.TextureFormat, aspect gputypes.TextureAspect, bytesPerRow, rowsPerImage, imageHeight uint32) (uint64, bool) {
	if rowsPerImage == 0 {
		_, blockHeight, _, ok := metalCopyBlockInfo(format, aspect)
		if !ok {
			return 0, false
		}
		rowsPerImage = uint32((uint64(imageHeight) + uint64(blockHeight) - 1) / uint64(blockHeight))
	}
	return checkedMetalTextureDataMul(uint64(rowsPerImage), uint64(bytesPerRow))
}

type metalTextureDataCopyLayout struct {
	bytesPerRow   uint32
	bytesPerImage uint64
}

type metalCopyStrides struct {
	bytesPerRow   uint64
	bytesPerImage uint64
}

// metalReplaceRegionStrides applies MTLTexture.replaceRegion's dimension rules.
// Metal infers the sole row of a 1D texture and only accepts an image stride for 3D.
func metalReplaceRegionStrides(dimension gputypes.TextureDimension, bytesPerRow, bytesPerImage uint64) metalCopyStrides {
	switch dimension {
	case gputypes.TextureDimension1D:
		return metalCopyStrides{}
	case gputypes.TextureDimension3D:
		return metalCopyStrides{bytesPerRow: bytesPerRow, bytesPerImage: bytesPerImage}
	default:
		return metalCopyStrides{bytesPerRow: bytesPerRow}
	}
}

// metalBufferBlit is one blit between a buffer and a texture.
type metalBufferBlit struct {
	region metalTextureCopyRegion
	offset uint64
	metalCopyStrides
}

// splitMetalBufferBlit returns the blits of region between a texture and a
// buffer of length bytes, at offset with the strides bytesPerRow and
// bytesPerImage, that Metal's validation takes, as Dawn's
// ComputeTextureBufferCopySplit makes them. Metal takes no row stride past
// the widest texture's row, so such a copy goes a row at a time. Its
// validation sizes a copy as whole images of whole rows from the offset,
// so one that ends inside the padding of its last row copies its last
// image and then its last row apart, the last row with a tight stride. An
// image stride only goes with more than one image of a 3D texture.
func splitMetalBufferBlit(region metalTextureCopyRegion, dimension gputypes.TextureDimension, offset, length, bytesPerRow, bytesPerImage uint64, blockWidth, blockHeight, blockSize uint32) []metalBufferBlit {
	blockRows := (uint64(region.size.Height) + uint64(blockHeight) - 1) / uint64(blockHeight)
	rowBytes := (uint64(region.size.Width) + uint64(blockWidth) - 1) / uint64(blockWidth) * uint64(blockSize)
	if bytesPerRow == 0 {
		bytesPerRow = rowBytes
	}
	images := uint64(region.size.Depth)
	imageStride := func(images uint64) uint64 {
		if dimension == gputypes.TextureDimension3D && images > 1 {
			return bytesPerImage
		}
		return 0
	}

	maxRowBytes := uint64(16384) * uint64(blockSize)
	if dimension == gputypes.TextureDimension3D {
		maxRowBytes = 2048 * uint64(blockSize)
	}
	if bytesPerRow > maxRowBytes {
		blits := make([]metalBufferBlit, 0, images*blockRows)
		for image := range images {
			for row := range blockRows {
				r := region
				r.origin.Y += NSUInteger(row * uint64(blockHeight))
				r.origin.Z += NSUInteger(image)
				r.size.Height = NSUInteger(min(uint64(blockHeight), uint64(region.size.Height)-row*uint64(blockHeight)))
				r.size.Depth = 1
				blits = append(blits, metalBufferBlit{r, offset + image*bytesPerImage + row*bytesPerRow, metalCopyStrides{bytesPerRow: rowBytes}})
			}
		}
		return blits
	}

	if offset <= length && length-offset >= images*bytesPerImage && length-offset >= blockRows*bytesPerRow {
		return []metalBufferBlit{{region, offset, metalCopyStrides{bytesPerRow, imageStride(images)}}}
	}
	var blits []metalBufferBlit
	if images > 1 {
		r := region
		r.size.Depth = NSUInteger(images - 1)
		blits = append(blits, metalBufferBlit{r, offset, metalCopyStrides{bytesPerRow, imageStride(images - 1)}})
		offset += (images - 1) * bytesPerImage
		region.origin.Z += r.size.Depth
		region.size.Depth = 1
	}
	if blockRows > 1 {
		r := region
		r.size.Height = NSUInteger((blockRows - 1) * uint64(blockHeight))
		blits = append(blits, metalBufferBlit{r, offset, metalCopyStrides{bytesPerRow: bytesPerRow}})
		offset += (blockRows - 1) * bytesPerRow
		region.origin.Y += r.size.Height
		region.size.Height -= r.size.Height
	}
	return append(blits, metalBufferBlit{region, offset, metalCopyStrides{bytesPerRow: rowBytes}})
}

func validateMetalBufferTextureCopyPlan(
	format gputypes.TextureFormat,
	aspect gputypes.TextureAspect,
	dimension gputypes.TextureDimension,
	layout hal.ImageDataLayout,
	origin hal.Origin3D,
	size hal.Extent3D,
) (metalCopyPlan, uint64, bool) {
	bytesPerImage, ok := metalCopyBytesPerImage(format, aspect, layout.BytesPerRow, layout.RowsPerImage, size.Height)
	if !ok {
		return metalCopyPlan{}, 0, false
	}
	plan := planMetalBufferTextureCopy(dimension, size.DepthOrArrayLayers)
	if plan.operationCount == 0 {
		return metalCopyPlan{}, 0, false
	}
	lastOperation := plan.operationCount - 1
	if _, ok := plan.textureRegion(dimension, origin, size, lastOperation); !ok {
		return metalCopyPlan{}, 0, false
	}
	if _, ok := plan.bufferOffset(layout.Offset, bytesPerImage, lastOperation); !ok {
		return metalCopyPlan{}, 0, false
	}
	return plan, bytesPerImage, true
}

func validateMetalTextureCopyPlan(
	sourceDimension, destinationDimension gputypes.TextureDimension,
	sourceOrigin, destinationOrigin hal.Origin3D,
	size hal.Extent3D,
) (metalCopyPlan, bool) {
	plan := planMetalTextureCopy(sourceDimension, destinationDimension, size.DepthOrArrayLayers)
	if plan.operationCount == 0 {
		return metalCopyPlan{}, false
	}
	lastOperation := plan.operationCount - 1
	if _, ok := plan.textureRegion(sourceDimension, sourceOrigin, size, lastOperation); !ok {
		return metalCopyPlan{}, false
	}
	if _, ok := plan.textureRegion(destinationDimension, destinationOrigin, size, lastOperation); !ok {
		return metalCopyPlan{}, false
	}
	return plan, true
}

// validateMetalTextureDataCopyLayout proves that every byte Metal may read is
// present before either the Shared direct-write or Private staging path starts.
func validateMetalTextureDataCopyLayout(dataLength uint64, format gputypes.TextureFormat, aspect gputypes.TextureAspect, layout hal.ImageDataLayout, size hal.Extent3D) (metalTextureDataCopyLayout, error) {
	if size.Width == 0 || size.Height == 0 || size.DepthOrArrayLayers == 0 {
		return metalTextureDataCopyLayout{}, fmt.Errorf("copy extent must be non-zero")
	}

	rowBytes, blockRows, ok := metalTextureCopyGeometry(format, aspect, size.Width, size.Height)
	if !ok {
		return metalTextureDataCopyLayout{}, fmt.Errorf("texture format %s has no defined copy block size for aspect %d", format, aspect)
	}
	if rowBytes > math.MaxUint32 {
		return metalTextureDataCopyLayout{}, fmt.Errorf("texture row size overflows BytesPerRow")
	}

	bytesPerRow := layout.BytesPerRow
	if bytesPerRow == 0 {
		bytesPerRow = uint32(rowBytes)
	}
	if uint64(bytesPerRow) < rowBytes {
		return metalTextureDataCopyLayout{}, fmt.Errorf("BytesPerRow %d is smaller than the copied row size %d", bytesPerRow, rowBytes)
	}

	rowsPerImage := layout.RowsPerImage
	if rowsPerImage == 0 {
		rowsPerImage = uint32(blockRows)
	}
	if uint64(rowsPerImage) < blockRows {
		return metalTextureDataCopyLayout{}, fmt.Errorf("RowsPerImage %d is smaller than the copied block-row count %d", rowsPerImage, blockRows)
	}

	bytesPerImage, ok := checkedMetalTextureDataMul(uint64(rowsPerImage), uint64(bytesPerRow))
	if !ok {
		return metalTextureDataCopyLayout{}, fmt.Errorf("BytesPerImage overflows uint64")
	}
	lastImageOffset, ok := checkedMetalTextureDataMul(uint64(size.DepthOrArrayLayers-1), bytesPerImage)
	if !ok {
		return metalTextureDataCopyLayout{}, fmt.Errorf("last image offset overflows uint64")
	}
	lastRowOffset, ok := checkedMetalTextureDataMul(blockRows-1, uint64(bytesPerRow))
	if !ok {
		return metalTextureDataCopyLayout{}, fmt.Errorf("last row offset overflows uint64")
	}
	requiredBytes, ok := checkedMetalTextureDataAdd(layout.Offset, lastImageOffset)
	if ok {
		requiredBytes, ok = checkedMetalTextureDataAdd(requiredBytes, lastRowOffset)
	}
	if ok {
		requiredBytes, ok = checkedMetalTextureDataAdd(requiredBytes, rowBytes)
	}
	if !ok {
		return metalTextureDataCopyLayout{}, fmt.Errorf("source data range overflows uint64")
	}
	if requiredBytes > dataLength {
		return metalTextureDataCopyLayout{}, fmt.Errorf("source data is too short: need %d bytes, have %d", requiredBytes, dataLength)
	}

	return metalTextureDataCopyLayout{bytesPerRow: bytesPerRow, bytesPerImage: bytesPerImage}, nil
}

func metalTextureCopyGeometry(format gputypes.TextureFormat, aspect gputypes.TextureAspect, width, height uint32) (rowBytes, blockRows uint64, ok bool) {
	blockWidth, blockHeight, blockSize, ok := metalCopyBlockInfo(format, aspect)
	if !ok {
		return 0, 0, false
	}
	blockColumns := (uint64(width) + uint64(blockWidth) - 1) / uint64(blockWidth)
	blockRows = (uint64(height) + uint64(blockHeight) - 1) / uint64(blockHeight)
	rowBytes, ok = checkedMetalTextureDataMul(blockColumns, uint64(blockSize))
	return rowBytes, blockRows, ok
}

// metalCopyBlockInfo is metalTextureFormatBlockInfo of the aspect a copy
// takes of format: a copy of a combined depth-stencil format takes one, the
// stencil or the depth of Depth32FloatStencil8, which WebGPU lets be copied.
func metalCopyBlockInfo(format gputypes.TextureFormat, aspect gputypes.TextureAspect) (width, height, size uint32, ok bool) {
	if format.HasDepth() && format.HasStencil() {
		switch {
		case aspect == gputypes.TextureAspectStencilOnly:
			return 1, 1, 1, true
		case aspect == gputypes.TextureAspectDepthOnly && format == gputypes.TextureFormatDepth32FloatStencil8:
			return 1, 1, 4, true
		}
		return 0, 0, 0, false
	}
	return metalTextureFormatBlockInfo(format)
}

// metalBlitOption picks the aspect of a combined depth-stencil format, which
// Metal's blits between a buffer and such a texture have to be told.
//
// Reference: Rust wgpu-hal metal/conv.rs get_blit_option.
func metalBlitOption(format gputypes.TextureFormat, aspect gputypes.TextureAspect) MTLBlitOption {
	if !format.HasDepth() || !format.HasStencil() {
		return MTLBlitOptionNone
	}
	if aspect == gputypes.TextureAspectStencilOnly {
		return MTLBlitOptionStencilFromDepthStencil
	}
	return MTLBlitOptionDepthFromDepthStencil
}

func metalTextureFormatBlockInfo(format gputypes.TextureFormat) (width, height, size uint32, ok bool) {
	blockSize := format.BlockCopySize()
	if blockSize == 0 {
		return 0, 0, 0, false
	}

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
		gputypes.TextureFormatBC7RGBAUnormSrgb,
		gputypes.TextureFormatETC2RGB8Unorm,
		gputypes.TextureFormatETC2RGB8UnormSrgb,
		gputypes.TextureFormatETC2RGB8A1Unorm,
		gputypes.TextureFormatETC2RGB8A1UnormSrgb,
		gputypes.TextureFormatETC2RGBA8Unorm,
		gputypes.TextureFormatETC2RGBA8UnormSrgb,
		gputypes.TextureFormatEACR11Unorm,
		gputypes.TextureFormatEACR11Snorm,
		gputypes.TextureFormatEACRG11Unorm,
		gputypes.TextureFormatEACRG11Snorm:
		return 4, 4, blockSize, true
	}

	switch format {
	case gputypes.TextureFormatASTC4x4Unorm, gputypes.TextureFormatASTC4x4UnormSrgb:
		return 4, 4, blockSize, true
	case gputypes.TextureFormatASTC5x4Unorm, gputypes.TextureFormatASTC5x4UnormSrgb:
		return 5, 4, blockSize, true
	case gputypes.TextureFormatASTC5x5Unorm, gputypes.TextureFormatASTC5x5UnormSrgb:
		return 5, 5, blockSize, true
	case gputypes.TextureFormatASTC6x5Unorm, gputypes.TextureFormatASTC6x5UnormSrgb:
		return 6, 5, blockSize, true
	case gputypes.TextureFormatASTC6x6Unorm, gputypes.TextureFormatASTC6x6UnormSrgb:
		return 6, 6, blockSize, true
	case gputypes.TextureFormatASTC8x5Unorm, gputypes.TextureFormatASTC8x5UnormSrgb:
		return 8, 5, blockSize, true
	case gputypes.TextureFormatASTC8x6Unorm, gputypes.TextureFormatASTC8x6UnormSrgb:
		return 8, 6, blockSize, true
	case gputypes.TextureFormatASTC8x8Unorm, gputypes.TextureFormatASTC8x8UnormSrgb:
		return 8, 8, blockSize, true
	case gputypes.TextureFormatASTC10x5Unorm, gputypes.TextureFormatASTC10x5UnormSrgb:
		return 10, 5, blockSize, true
	case gputypes.TextureFormatASTC10x6Unorm, gputypes.TextureFormatASTC10x6UnormSrgb:
		return 10, 6, blockSize, true
	case gputypes.TextureFormatASTC10x8Unorm, gputypes.TextureFormatASTC10x8UnormSrgb:
		return 10, 8, blockSize, true
	case gputypes.TextureFormatASTC10x10Unorm, gputypes.TextureFormatASTC10x10UnormSrgb:
		return 10, 10, blockSize, true
	case gputypes.TextureFormatASTC12x10Unorm, gputypes.TextureFormatASTC12x10UnormSrgb:
		return 12, 10, blockSize, true
	case gputypes.TextureFormatASTC12x12Unorm, gputypes.TextureFormatASTC12x12UnormSrgb:
		return 12, 12, blockSize, true
	default:
		return 1, 1, blockSize, true
	}
}

func checkedMetalTextureDataMul(a, b uint64) (uint64, bool) {
	if a != 0 && b > math.MaxUint64/a {
		return 0, false
	}
	return a * b, true
}

func checkedMetalTextureDataAdd(a, b uint64) (uint64, bool) {
	if b > math.MaxUint64-a {
		return 0, false
	}
	return a + b, true
}

func checkedMetalTextureDataAdd32(a, b uint32) (uint32, bool) {
	if b > math.MaxUint32-a {
		return 0, false
	}
	return a + b, true
}
