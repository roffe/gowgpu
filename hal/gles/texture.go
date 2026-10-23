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
	switch {
	case glf.compressed():
		levels, width, height := int32(desc.MipLevelCount), int32(desc.Size.Width), int32(desc.Size.Height)
		switch target {
		case gl.TEXTURE_2D, gl.TEXTURE_CUBE_MAP:
			glCtx.TexStorage2D(target, levels, glf.internal, width, height)
		case gl.TEXTURE_2D_ARRAY:
			glCtx.TexStorage3D(target, levels, glf.internal, width, height, int32(desc.Size.DepthOrArrayLayers))
		default:
			glCtx.DeleteTextures(id)
			return nil, fmt.Errorf("gles: %s %s textures are not supported", desc.Dimension, desc.Format)
		}
		if glErr := glCtx.GetError(); glErr != 0 {
			glCtx.DeleteTextures(id)
			return nil, fmt.Errorf("gles: TexStorage failed: GL error 0x%x (format=0x%x, %dx%dx%d)",
				glErr, glf.internal, width, height, desc.Size.DepthOrArrayLayers)
		}

	case target == gl.TEXTURE_2D_MULTISAMPLE:
		glCtx.TexImage2DMultisample(target, int32(sampleCount), glf.internal,
			int32(desc.Size.Width), int32(desc.Size.Height), true)
		if glErr := glCtx.GetError(); glErr != 0 {
			glCtx.DeleteTextures(id)
			return nil, fmt.Errorf("gles: TexImage2DMultisample failed: GL error 0x%x (format=0x%x, samples=%d, %dx%d)",
				glErr, glf.internal, sampleCount, desc.Size.Width, desc.Size.Height)
		}

	case target == gl.TEXTURE_2D:
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

	case target == gl.TEXTURE_CUBE_MAP:
		for face := uint32(0); face < 6; face++ {
			faceTarget := gl.TEXTURE_CUBE_MAP_POSITIVE_X + face
			for level := uint32(0); level < desc.MipLevelCount; level++ {
				width := max(1, int32(desc.Size.Width>>level))
				height := max(1, int32(desc.Size.Height>>level))
				glCtx.TexImage2D(faceTarget, int32(level), int32(glf.internal),
					width, height, 0, glf.format, glf.typ, 0)
			}
		}
		if glErr := glCtx.GetError(); glErr != 0 {
			glCtx.DeleteTextures(id)
			return nil, fmt.Errorf("gles: TexImage2D (cube) failed: GL error 0x%x (format=0x%x)", glErr, glf.internal)
		}

	case target == gl.TEXTURE_2D_ARRAY || target == gl.TEXTURE_3D:
		// An array's levels all have its layers; a 3D texture's halve in depth.
		for level := uint32(0); level < desc.MipLevelCount; level++ {
			width := max(1, int32(desc.Size.Width>>level))
			height := max(1, int32(desc.Size.Height>>level))
			depth := int32(desc.Size.DepthOrArrayLayers)
			if target == gl.TEXTURE_3D {
				depth = max(1, depth>>level)
			}
			glCtx.TexImage3D(target, int32(level), int32(glf.internal), width, height, depth, 0, glf.format, glf.typ, 0)
			if glErr := glCtx.GetError(); glErr != 0 {
				glCtx.DeleteTextures(id)
				return nil, fmt.Errorf("gles: TexImage3D failed: GL error 0x%x (format=0x%x, level=%d, %dx%dx%d)",
					glErr, glf.internal, level, width, height, depth)
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
		maxLevel:    int32(desc.MipLevelCount) - 1,
	}, nil
}

// writeTexture uploads data into tex at dst: the layers or slices from
// dst.Origin.Z on of an array, cube or 3D texture. The context must be
// current.
func writeTexture(glCtx *gl.Context, tex *Texture, dst *hal.ImageCopyTexture, data []byte, layout *hal.ImageDataLayout, size *hal.Extent3D) {
	var l hal.ImageDataLayout
	if layout != nil {
		l = *layout
	}
	data = data[l.Offset:]
	depth := max(1, size.DepthOrArrayLayers)
	glCtx.BindTexture(tex.target, tex.id)
	if tex.glf.compressed() {
		writeBlocks(glCtx, tex, dst, data, l, size.Width, size.Height, depth)
	} else {
		writePixels(glCtx, tex, dst, data, l, size.Width, size.Height, depth)
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

// writePixels uploads texels with the texture's transfer pair, the rows and
// images at the layout's pitches: an atlas fills in one small rectangle of a
// bigger image at a time.
func writePixels(glCtx *gl.Context, tex *Texture, dst *hal.ImageCopyTexture, data []byte, l hal.ImageDataLayout, width, height, depth uint32) {
	f := tex.glf
	bpp := pixelSize(f.format, f.typ)
	rows := height
	if l.RowsPerImage > height {
		rows = l.RowsPerImage
	}
	pitch := width * bpp
	if l.BytesPerRow > pitch {
		pitch = l.BytesPerRow
	}
	if pitch != width*bpp {
		glCtx.PixelStorei(gl.UNPACK_ROW_LENGTH, int32(pitch/bpp))
		defer glCtx.PixelStorei(gl.UNPACK_ROW_LENGTH, 0)
	}
	if rows != height && tex.layered() {
		glCtx.PixelStorei(gl.UNPACK_IMAGE_HEIGHT, int32(rows))
		defer glCtx.PixelStorei(gl.UNPACK_IMAGE_HEIGHT, 0)
	}
	// Rows of texels narrower than 4 bytes (R8, RG8) need not start on
	// the default 4-byte GL_UNPACK_ALIGNMENT.
	if bpp < 4 {
		glCtx.PixelStorei(gl.UNPACK_ALIGNMENT, 1)
		defer glCtx.PixelStorei(gl.UNPACK_ALIGNMENT, 4)
	}
	x, y, z, mip := int32(dst.Origin.X), int32(dst.Origin.Y), dst.Origin.Z, int32(dst.MipLevel)
	switch {
	case tex.layered():
		glCtx.TexSubImage3D(tex.target, mip, x, y, int32(z), int32(width), int32(height), int32(depth), f.format, f.typ, unsafe.Pointer(&data[0]))
	case tex.target == gl.TEXTURE_CUBE_MAP:
		for face := range depth {
			glCtx.TexSubImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+z+face, mip, x, y, int32(width), int32(height), f.format, f.typ,
				unsafe.Pointer(&data[face*rows*pitch]))
		}
	default:
		// Use TexSubImage2D to update existing texture data (Rust wgpu-hal pattern).
		// TexImage2D reallocates storage on every call; TexSubImage2D updates in-place.
		glCtx.TexSubImage2D(tex.target, mip, x, y, int32(width), int32(height), f.format, f.typ, unsafe.Pointer(&data[0]))
	}
}

// writeBlocks uploads a compressed region. GL ES has no row length for
// compressed uploads, so rows given at a wider pitch go in packed.
func writeBlocks(glCtx *gl.Context, tex *Texture, dst *hal.ImageCopyTexture, data []byte, l hal.ImageDataLayout, width, height, depth uint32) {
	blocks, rowBytes := compressedRows(tex.format, width, height)
	pitch, imageRows := rowBytes, blocks
	if l.BytesPerRow != 0 {
		pitch = l.BytesPerRow
	}
	if _, bh := tex.format.BlockDimensions(); l.RowsPerImage != 0 {
		imageRows = l.RowsPerImage / bh
	}
	imageSize := rowBytes * blocks
	if pitch != rowBytes || imageRows != blocks {
		packed := make([]byte, imageSize*depth)
		for img := range depth {
			for r := range blocks {
				from := (img*imageRows + r) * pitch
				copy(packed[img*imageSize+r*rowBytes:], data[from:from+rowBytes])
			}
		}
		data = packed
	}
	x, y, z, mip := int32(dst.Origin.X), int32(dst.Origin.Y), dst.Origin.Z, int32(dst.MipLevel)
	switch {
	case tex.layered():
		glCtx.CompressedTexSubImage3D(tex.target, mip, x, y, int32(z), int32(width), int32(height), int32(depth), tex.glf.internal,
			int32(imageSize*depth), unsafe.Pointer(&data[0]))
	case tex.target == gl.TEXTURE_CUBE_MAP:
		for face := range depth {
			glCtx.CompressedTexSubImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X+z+face, mip, x, y, int32(width), int32(height), tex.glf.internal,
				int32(imageSize), unsafe.Pointer(&data[face*imageSize]))
		}
	default:
		glCtx.CompressedTexSubImage2D(tex.target, mip, x, y, int32(width), int32(height), tex.glf.internal, int32(imageSize), unsafe.Pointer(&data[0]))
	}
}

// compressedRows returns how many rows of blocks a width by height region
// of a compressed format has, and how many bytes a row of them takes.
func compressedRows(format gputypes.TextureFormat, width, height uint32) (rows, rowBytes uint32) {
	bw, bh := format.BlockDimensions()
	return (height + bh - 1) / bh, (width + bw - 1) / bw * format.BlockCopySize()
}

// newTextureView makes a view of t from desc, with its mip and layer counts
// resolved. GL has no views of another format (no glTextureView on ES), so
// one is refused, but for the depth or stencil aspect of a combined format.
func newTextureView(t *Texture, desc *TextureViewDescriptor) (*TextureView, error) {
	v := &TextureView{texture: t, mipCount: t.mipLevels, layerCount: t.layers()}
	if desc != nil {
		if !viewFormatAllowed(t.format, desc.Format, desc.Aspect) {
			return nil, fmt.Errorf("gles: a %s view of a %s texture: GL has no views that change the format", desc.Format, t.format)
		}
		v.aspect = desc.Aspect
		v.baseMip, v.baseLayer = desc.BaseMipLevel, desc.BaseArrayLayer
		v.mipCount = desc.MipLevelCount
		if v.mipCount == 0 {
			v.mipCount = t.mipLevels - min(v.baseMip, t.mipLevels)
		}
		v.layerCount = desc.ArrayLayerCount
		if v.layerCount == 0 {
			v.layerCount = t.layers() - min(v.baseLayer, t.layers())
		}
	}
	v.handle = lastViewKey.Add(1)
	views.Store(v.handle, v)
	return v, nil
}

// viewFormatAllowed reports whether a view of a tex texture can be view:
// the texture's own format, or the format of the aspect it views.
func viewFormatAllowed(tex, view gputypes.TextureFormat, aspect gputypes.TextureAspect) bool {
	if view == gputypes.TextureFormatUndefined || view == tex {
		return true
	}
	switch aspect {
	case gputypes.TextureAspectDepthOnly:
		return tex == gputypes.TextureFormatDepth24PlusStencil8 && view == gputypes.TextureFormatDepth24Plus ||
			tex == gputypes.TextureFormatDepth32FloatStencil8 && view == gputypes.TextureFormatDepth32Float
	case gputypes.TextureAspectStencilOnly:
		return tex.HasStencil() && view == gputypes.TextureFormatStencil8
	}
	return false
}

// readsStencil reports whether sampling v reads the stencil of a texture
// stored with both depth and stencil, which DEPTH_STENCIL_TEXTURE_MODE
// picks.
func (v *TextureView) readsStencil() bool {
	return v.texture.glf.format == gl.DEPTH_STENCIL &&
		(v.aspect == gputypes.TextureAspectStencilOnly || v.texture.format == gputypes.TextureFormatStencil8)
}

// viewApplied reports whether t samples with v's mip range and aspect.
func (t *Texture) viewApplied(v *TextureView) bool {
	if t.target == gl.TEXTURE_2D_MULTISAMPLE {
		return true
	}
	return t.baseLevel == int32(v.baseMip) && t.maxLevel == int32(v.baseMip+v.mipCount)-1 && t.stencilMode == v.readsStencil()
}

// applyView sets t, bound on the active unit, to sample with v's mip range
// and aspect, as wgpu-hal does at every bind.
func (t *Texture) applyView(ctx *gl.Context, v *TextureView) {
	if t.target == gl.TEXTURE_2D_MULTISAMPLE {
		return
	}
	if base, top := int32(v.baseMip), int32(v.baseMip+v.mipCount)-1; t.baseLevel != base || t.maxLevel != top {
		ctx.TexParameteri(t.target, gl.TEXTURE_BASE_LEVEL, base)
		ctx.TexParameteri(t.target, gl.TEXTURE_MAX_LEVEL, top)
		t.baseLevel, t.maxLevel = base, top
	}
	if stencil := v.readsStencil(); t.stencilMode != stencil {
		mode := int32(gl.DEPTH_COMPONENT)
		if stencil {
			mode = gl.STENCIL_INDEX
		}
		ctx.TexParameteri(t.target, gl.DEPTH_STENCIL_TEXTURE_MODE, mode)
		t.stencilMode = stencil
	}
}

// bindTexture binds v's texture on unit for the shaders to sample, set to
// the view's mip range and aspect.
func bindTexture(ctx *gl.Context, unit uint32, v *TextureView) {
	t := v.texture
	if t == nil || t.id == 0 {
		return
	}
	sh := &ctx.Shadow
	want := gl.ShadowObject{ID: t.id, Target: t.target, Set: true}
	bound := unit < gl.MaxShadowUnits && sh.Textures[unit] == want
	if bound && t.viewApplied(v) {
		return
	}
	if !sh.ActiveUnitSet || sh.ActiveUnit != unit {
		ctx.ActiveTexture(gl.TEXTURE0 + unit)
		sh.ActiveUnit, sh.ActiveUnitSet = unit, true
	}
	if !bound {
		ctx.BindTexture(t.target, t.id)
		if unit < gl.MaxShadowUnits {
			sh.Textures[unit] = want
		}
	}
	t.applyView(ctx, v)
}
