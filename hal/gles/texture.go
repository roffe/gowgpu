// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"fmt"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// createTexture makes the texture desc describes. The context must be
// current.
func createTexture(glCtx *gl.Context, formats formatCaps, maxMSAA int32, desc *TextureDescriptor) (*Texture, error) {
	glf, ok := formats.glFormat(desc.Format)
	if !ok {
		return nil, fmt.Errorf("gles: texture format %s is not supported on this context", desc.Format)
	}

	sampleCount := desc.SampleCount
	if sampleCount == 0 {
		sampleCount = 1
	}

	// Validate MSAA sample count against driver capability (Rust wgpu-hal parity).
	// Mesa d3d12 GL 4.1 may only support 1x-2x for certain formats.
	if sampleCount > 1 && maxMSAA > 0 && int32(sampleCount) > maxMSAA {
		hal.Logger().Warn("gles: MSAA sample count clamped",
			"requested", sampleCount, "maxMSAA", maxMSAA)
		sampleCount = uint32(maxMSAA)
	}

	// Map dimension to GL target
	target := uint32(gl.TEXTURE_2D)
	switch desc.Dimension {
	case gputypes.TextureDimension1D:
		// GL doesn't have 1D textures in ES, use 2D with height=1
		target = gl.TEXTURE_2D
	case gputypes.TextureDimension2D:
		switch {
		case sampleCount > 1:
			target = gl.TEXTURE_2D_MULTISAMPLE
		case desc.Size.DepthOrArrayLayers > 1:
			target = gl.TEXTURE_2D_ARRAY
		default:
			target = gl.TEXTURE_2D
		}
	case gputypes.TextureDimension3D:
		target = gl.TEXTURE_3D
	}

	// Handle cube maps (only for single-sample textures)
	if sampleCount <= 1 && desc.ViewFormats != nil {
		for _, vf := range desc.ViewFormats {
			if vf == desc.Format {
				// Check if this should be a cube map
				if desc.Size.DepthOrArrayLayers == 6 {
					target = gl.TEXTURE_CUBE_MAP
				}
			}
		}
	}

	id := glCtx.GenTextures(1)
	glCtx.BindTexture(target, id)

	// Allocate texture storage
	switch target {
	case gl.TEXTURE_2D_MULTISAMPLE:
		glCtx.TexImage2DMultisample(target, int32(sampleCount), glf.internal,
			int32(desc.Size.Width), int32(desc.Size.Height), true)
		if glErr := glCtx.GetError(); glErr != 0 {
			glCtx.DeleteTextures(id)
			return nil, fmt.Errorf("gles: TexImage2DMultisample failed: GL error 0x%x (format=0x%x, samples=%d, %dx%d)",
				glErr, glf.internal, sampleCount, desc.Size.Width, desc.Size.Height)
		}

	case gl.TEXTURE_2D:
		for level := uint32(0); level < desc.MipLevelCount; level++ {
			width := max(1, int32(desc.Size.Width>>level))
			height := max(1, int32(desc.Size.Height>>level))
			glCtx.TexImage2D(target, int32(level), int32(glf.internal),
				width, height, 0, glf.format, glf.typ, 0)
			if glErr := glCtx.GetError(); glErr != 0 {
				glCtx.DeleteTextures(id)
				return nil, fmt.Errorf("gles: TexImage2D failed: GL error 0x%x (format=0x%x, level=%d, %dx%d)",
					glErr, glf.internal, level, width, height)
			}
		}

	case gl.TEXTURE_CUBE_MAP:
		for face := uint32(0); face < 6; face++ {
			faceTarget := gl.TEXTURE_CUBE_MAP_POSITIVE_X + face
			for level := uint32(0); level < desc.MipLevelCount; level++ {
				width := max(1, int32(desc.Size.Width>>level))
				height := max(1, int32(desc.Size.Height>>level))
				glCtx.TexImage2D(faceTarget, int32(level), int32(glf.internal),
					width, height, 0, glf.format, glf.typ, 0)
			}
		}
	}

	// Set default texture parameters (multisample textures don't support these).
	if target != gl.TEXTURE_2D_MULTISAMPLE {
		// Set NEAREST filter and MAX_LEVEL on all textures for completeness.
		// Default GL_TEXTURE_MIN_FILTER is GL_NEAREST_MIPMAP_LINEAR and default
		// GL_TEXTURE_MAX_LEVEL is 1000. For non-mipmapped textures (MipLevelCount=1),
		// this makes the texture INCOMPLETE — sampling returns (0,0,0,0).
		// Setting MAX_LEVEL = MipLevelCount-1 tells GL the actual mip chain length.
		glCtx.TexParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
		glCtx.TexParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
		glCtx.TexParameteri(target, gl.TEXTURE_MAX_LEVEL, int32(desc.MipLevelCount-1))
		glCtx.TexParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
		glCtx.TexParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
	}

	glCtx.BindTexture(target, 0)

	hal.Logger().Debug("gles: texture created",
		"label", desc.Label,
		"format", desc.Format,
		"width", desc.Size.Width,
		"height", desc.Size.Height,
	)

	return &Texture{
		id:          id,
		target:      target,
		format:      desc.Format,
		glf:         glf,
		dimension:   desc.Dimension,
		size:        desc.Size,
		mipLevels:   desc.MipLevelCount,
		sampleCount: sampleCount,
		glCtx:       glCtx,
	}, nil
}

// writeTexture uploads data into tex at dst. The context must be current.
func writeTexture(glCtx *gl.Context, tex *Texture, dst *hal.ImageCopyTexture, data []byte, layout *hal.ImageDataLayout, size *hal.Extent3D) {
	f := tex.glf
	glCtx.BindTexture(tex.target, tex.id)

	if tex.target == gl.TEXTURE_2D {
		// The copy lands at dst.Origin, and the source rows may be longer
		// than the copy (a region of a larger image): an atlas fills in one
		// small rectangle at a time.
		if layout != nil && layout.Offset > 0 {
			data = data[layout.Offset:]
		}
		bpp := pixelSize(f.format, f.typ)
		var rowLength int32
		if layout != nil && layout.BytesPerRow > size.Width*bpp {
			rowLength = int32(layout.BytesPerRow / bpp)
			glCtx.PixelStorei(gl.UNPACK_ROW_LENGTH, rowLength)
		}
		// Rows of texels narrower than 4 bytes (R8, RG8) need not start on
		// the default 4-byte GL_UNPACK_ALIGNMENT.
		narrow := bpp < 4
		if narrow {
			glCtx.PixelStorei(gl.UNPACK_ALIGNMENT, 1)
		}
		// Use TexSubImage2D to update existing texture data (Rust wgpu-hal pattern).
		// TexImage2D reallocates storage on every call; TexSubImage2D updates in-place.
		glCtx.TexSubImage2D(tex.target, int32(dst.MipLevel),
			int32(dst.Origin.X), int32(dst.Origin.Y), int32(size.Width), int32(size.Height), f.format, f.typ,
			unsafe.Pointer(&data[0]))
		if rowLength != 0 {
			glCtx.PixelStorei(gl.UNPACK_ROW_LENGTH, 0)
		}
		if narrow {
			glCtx.PixelStorei(gl.UNPACK_ALIGNMENT, 4)
		}
	}

	glCtx.BindTexture(tex.target, 0)

	if hal.DebugEnabled() {
		hal.Logger().Debug("gles: texture written",
			"format", tex.format,
			"width", size.Width,
			"height", size.Height,
		)
	}
}
