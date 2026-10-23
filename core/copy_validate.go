//go:build !(js && wasm)

package core

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// TexelCopyTexture is the texture side of a copy: a mip level of Texture, an
// origin in it and an aspect.
type TexelCopyTexture struct {
	Texture  *Texture
	MipLevel uint32
	Origin   gputypes.Origin3D
	Aspect   gputypes.TextureAspect
}

// validate checks c for a copy of size: a mip level and an aspect the
// texture has, an origin and a size in whole texel blocks, a region inside
// the mip level, rounded up to whole blocks, and whole mip levels of a depth
// or stencil format or a multisampled texture. It returns the size cut to
// the texels the mip level has, where a block sticks out over its edge,
// which is what the backends copy.
//
// Reference: WebGPU "validating GPUTexelCopyTextureInfo" and "validating
// texture copy range"; wgpu-core command/transfer.rs
// validate_texture_copy_range
func (c *TexelCopyTexture) validate(size gputypes.Extent3D) (gputypes.Extent3D, error) {
	t, o := c.Texture, c.Origin
	f := t.Format()
	if c.MipLevel >= t.MipLevelCount() {
		return size, fmt.Errorf("mip level %d of a texture with %d", c.MipLevel, t.MipLevelCount())
	}
	if _, ok := AspectFormat(f, c.Aspect); !ok {
		return size, fmt.Errorf("aspect %v of a %v texture", c.Aspect, f)
	}
	bw, bh := f.BlockDimensions()
	if o.X%bw != 0 || o.Y%bh != 0 || size.Width%bw != 0 || size.Height%bh != 0 {
		return size, fmt.Errorf("origin %d,%d and size %dx%d are not in whole %dx%d blocks of %v", o.X, o.Y, size.Width, size.Height, bw, bh, f)
	}
	texels := t.Size()
	texels.Width, texels.Height = max(texels.Width>>c.MipLevel, 1), max(texels.Height>>c.MipLevel, 1)
	if t.Dimension() == gputypes.TextureDimension3D {
		texels.DepthOrArrayLayers >>= c.MipLevel
	}
	texels.DepthOrArrayLayers = max(texels.DepthOrArrayLayers, 1)
	mip := texels
	mip.Width, mip.Height = (mip.Width+bw-1)/bw*bw, (mip.Height+bh-1)/bh*bh
	if uint64(o.X)+uint64(size.Width) > uint64(mip.Width) || uint64(o.Y)+uint64(size.Height) > uint64(mip.Height) ||
		uint64(o.Z)+uint64(size.DepthOrArrayLayers) > uint64(mip.DepthOrArrayLayers) {
		return size, fmt.Errorf("%dx%dx%d at %d,%d,%d reaches past the %dx%dx%d of mip level %d",
			size.Width, size.Height, size.DepthOrArrayLayers, o.X, o.Y, o.Z, mip.Width, mip.Height, mip.DepthOrArrayLayers, c.MipLevel)
	}
	if (f.IsDepthStencil() || t.SampleCount() > 1) && (o.X != 0 || o.Y != 0 || size.Width != mip.Width || size.Height != mip.Height) {
		return size, fmt.Errorf("a copy of a %v texture with %d samples takes whole mip levels", f, t.SampleCount())
	}
	cut := func(n, origin, edge uint32) uint32 {
		if origin >= edge {
			return 0
		}
		return min(n, edge-origin)
	}
	size.Width, size.Height = cut(size.Width, o.X, texels.Width), cut(size.Height, o.Y, texels.Height)
	return size, nil
}

// ValidateTextureBufferCopy checks a copy of size between the texture side c
// and the bytes of layout in byteSize: c, a single-sampled texture, one
// aspect of a depth or stencil format that buffer copies take that way, and
// the bytes the layout addresses. toTexture is a copy into the texture. A
// copy with a buffer, bufferCopy, also needs its offset in whole texel
// blocks, or a multiple of 4 for depth and stencil; Queue.WriteTexture
// doesn't. It returns the size the backend copies, as validate does.
//
// Reference: WebGPU copyBufferToTexture, copyTextureToBuffer and
// writeTexture; wgpu-core command/transfer.rs validate_texture_buffer_copy
func ValidateTextureBufferCopy(c *TexelCopyTexture, size gputypes.Extent3D, layout hal.ImageDataLayout, byteSize uint64, toTexture, bufferCopy bool) (gputypes.Extent3D, error) {
	copySize, err := c.validate(size)
	if err != nil {
		return size, err
	}
	f := c.Texture.Format()
	af, _ := AspectFormat(f, c.Aspect)
	switch {
	case c.Texture.SampleCount() > 1:
		return size, fmt.Errorf("a buffer copy of a multisampled texture")
	case af.HasDepth() && af.HasStencil():
		return size, fmt.Errorf("a buffer copy of both aspects of %v, it takes one", f)
	case af == gputypes.TextureFormatDepth24Plus:
		return size, fmt.Errorf("a buffer copy of the depth of %v, whose bytes are the backend's", f)
	case toTexture && af == gputypes.TextureFormatDepth32Float:
		return size, fmt.Errorf("a copy into the depth of %v, which only copies out", f)
	}
	blockSize := uint64(af.BlockCopySize())
	what := "data"
	if align := blockSize; bufferCopy {
		if f.IsDepthStencil() {
			align = 4
		}
		if layout.Offset%align != 0 {
			return size, fmt.Errorf("buffer offset %d is not a multiple of %d", layout.Offset, align)
		}
		what = "buffer"
	}
	return copySize, validateLinearTextureData(layout, byteSize, what, af, blockSize, size)
}

// validateLinearTextureData checks the layout of the bytes of a copy of size
// texels of format: bytesPerRow where the copy has more than one row of
// blocks or more than one image, rowsPerImage where it has more than one
// image, each large enough, and the bytes addressed from the offset within
// byteSize.
//
// Reference: WebGPU "validating linear texture data"; wgpu-core
// command/transfer.rs validate_linear_texture_data
func validateLinearTextureData(layout hal.ImageDataLayout, byteSize uint64, what string, format gputypes.TextureFormat, blockSize uint64, size gputypes.Extent3D) error {
	bw, bh := format.BlockDimensions()
	rows, layers := uint64(size.Height/bh), uint64(size.DepthOrArrayLayers)
	rowBytes := uint64(size.Width/bw) * blockSize
	bytesPerRow, rowsPerImage := uint64(layout.BytesPerRow), uint64(layout.RowsPerImage)
	switch {
	case bytesPerRow == 0 && (rows > 1 || layers > 1):
		return fmt.Errorf("bytesPerRow is needed for %d rows of blocks in %d images", rows, layers)
	case rowsPerImage == 0 && layers > 1:
		return fmt.Errorf("rowsPerImage is needed for %d images", layers)
	case bytesPerRow != 0 && bytesPerRow < rowBytes:
		return fmt.Errorf("bytesPerRow %d is less than the %d bytes of a row", bytesPerRow, rowBytes)
	case rowsPerImage != 0 && rowsPerImage < rows:
		return fmt.Errorf("rowsPerImage %d is less than the %d rows of blocks of an image", rowsPerImage, rows)
	}
	var required uint64
	if layers > 0 && rows > 0 {
		required = bytesPerRow*rowsPerImage*(layers-1) + bytesPerRow*(rows-1) + rowBytes
	} else if layers > 0 {
		required = bytesPerRow * rowsPerImage * (layers - 1)
	}
	if layout.Offset > byteSize || required > byteSize-layout.Offset {
		return fmt.Errorf("the copy addresses %d bytes from offset %d, past the %s's %d bytes", required, layout.Offset, what, byteSize)
	}
	return nil
}

// ValidateTextureToTextureCopy checks a copy of size from src to dst: both
// sides, the same sample count, formats that differ at most in being sRGB,
// every aspect of a depth or stencil format, and no subresource both read
// and written. It returns the size the backend copies, what both sides have
// of it, as validate does.
//
// Reference: WebGPU copyTextureToTexture; wgpu-core command/transfer.rs
// copy_texture_to_texture
func ValidateTextureToTextureCopy(src, dst *TexelCopyTexture, size gputypes.Extent3D) (gputypes.Extent3D, error) {
	srcSize, err := src.validate(size)
	if err != nil {
		return size, fmt.Errorf("source: %w", err)
	}
	dstSize, err := dst.validate(size)
	if err != nil {
		return size, fmt.Errorf("destination: %w", err)
	}
	sf, df := src.Texture.Format(), dst.Texture.Format()
	allAspects := func(c *TexelCopyTexture) bool {
		af, _ := AspectFormat(c.Texture.Format(), c.Aspect)
		return af == c.Texture.Format()
	}
	switch {
	case src.Texture.SampleCount() != dst.Texture.SampleCount():
		return size, fmt.Errorf("a copy from %d samples to %d", src.Texture.SampleCount(), dst.Texture.SampleCount())
	case sf != df && !srgbTwins(sf, df):
		return size, fmt.Errorf("a copy from %v to %v, formats that differ in more than being sRGB", sf, df)
	case sf.IsDepthStencil() && (!allAspects(src) || !allAspects(dst)):
		return size, fmt.Errorf("a copy of one aspect of %v, which copies between textures whole", sf)
	case src.Texture == dst.Texture && src.MipLevel == dst.MipLevel &&
		(src.Texture.Dimension() == gputypes.TextureDimension3D ||
			uint64(src.Origin.Z) < uint64(dst.Origin.Z)+uint64(size.DepthOrArrayLayers) &&
				uint64(dst.Origin.Z) < uint64(src.Origin.Z)+uint64(size.DepthOrArrayLayers)):
		return size, fmt.Errorf("a copy within mip level %d that reads and writes the same layers", src.MipLevel)
	}
	return gputypes.Extent3D{
		Width:              min(srcSize.Width, dstSize.Width),
		Height:             min(srcSize.Height, dstSize.Height),
		DepthOrArrayLayers: size.DepthOrArrayLayers,
	}, nil
}
