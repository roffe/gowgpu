// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// createTexture makes the texture desc describes. The context must be
// current.
func createTexture(glCtx *gl.Context, formats formatCaps, maxMSAA int32, desc *TextureDescriptor) (*Texture, error) {
	drainGLErrors(glCtx) // its own calls are checked
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

	if sampleCount > 1 && desc.Usage&gputypes.TextureUsageTextureBinding == 0 {
		return createRenderbuffer(glCtx, glf, sampleCount, desc)
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

	// Immutable storage: what the compressed formats are made with, and
	// what texture views take, so every texture has it where the context
	// has views, as Dawn allocates. ES's unsized BGRA takes none.
	immutable := glf.compressed() || formats.textureView && glf.internal != gl.BGRA

	id := glCtx.GenTextures(1)
	glCtx.BindTexture(target, id)

	// Allocate texture storage
	switch {
	case target == gl.TEXTURE_2D_MULTISAMPLE:
		// A multisampled texture to sample: ES has only immutable storage
		// for one, from 3.1 on, and desktop GL glTexImage2DMultisample.
		switch {
		case formats.texStorageMS:
			glCtx.TexStorage2DMultisample(target, int32(sampleCount), glf.internal,
				int32(desc.Size.Width), int32(desc.Size.Height), true)
		case formats.es:
			glCtx.DeleteTextures(id)
			return nil, fmt.Errorf("gles: multisampled textures to sample need OpenGL ES 3.1")
		default:
			glCtx.TexImage2DMultisample(target, int32(sampleCount), glf.internal,
				int32(desc.Size.Width), int32(desc.Size.Height), true)
		}
		if glErr := glCtx.GetError(); glErr != 0 {
			glCtx.DeleteTextures(id)
			return nil, fmt.Errorf("gles: TexImage2DMultisample failed: GL error 0x%x (format=0x%x, samples=%d, %dx%d)",
				glErr, glf.internal, sampleCount, desc.Size.Width, desc.Size.Height)
		}

	case immutable:
		levels, width, height := int32(desc.MipLevelCount), int32(desc.Size.Width), int32(desc.Size.Height)
		switch {
		case target == gl.TEXTURE_2D || target == gl.TEXTURE_CUBE_MAP:
			glCtx.TexStorage2D(target, levels, glf.internal, width, height)
		case target == gl.TEXTURE_2D_ARRAY || target == gl.TEXTURE_3D && !glf.compressed():
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

	setTextureDefaults(glCtx, target, desc.MipLevelCount)

	t := &Texture{
		id:          id,
		target:      target,
		format:      desc.Format,
		glf:         glf,
		dimension:   desc.Dimension,
		size:        desc.Size,
		mipLevels:   desc.MipLevelCount,
		sampleCount: sampleCount,
		usage:       desc.Usage,
		immutable:   immutable,
		glCtx:       glCtx,
		maxLevel:    int32(desc.MipLevelCount) - 1,
	}
	t.zero(glCtx)
	glCtx.BindTexture(target, 0)

	hal.Logger().Debug("gles: texture created",
		"label", desc.Label,
		"format", desc.Format,
		"width", desc.Size.Width,
		"height", desc.Size.Height,
	)
	return t, nil
}

// setTextureDefaults sets the parameters a texture of levels mip levels,
// bound to target, samples with where no sampler object is bound.
func setTextureDefaults(ctx *gl.Context, target, levels uint32) {
	if target == gl.TEXTURE_2D_MULTISAMPLE {
		return // a multisampled texture has no sampling parameters
	}
	// Set NEAREST filter and MAX_LEVEL on all textures for completeness.
	// Default GL_TEXTURE_MIN_FILTER is GL_NEAREST_MIPMAP_LINEAR and default
	// GL_TEXTURE_MAX_LEVEL is 1000. For non-mipmapped textures (MipLevelCount=1),
	// this makes the texture INCOMPLETE — sampling returns (0,0,0,0).
	// Setting MAX_LEVEL = MipLevelCount-1 tells GL the actual mip chain length.
	ctx.TexParameteri(target, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
	ctx.TexParameteri(target, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
	ctx.TexParameteri(target, gl.TEXTURE_MAX_LEVEL, int32(levels-1))
	ctx.TexParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
	ctx.TexParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
}

// zero sets every texel of t, a new texture bound to its target, to zero
// when it is one the queue's zeroing leaves to the GL backend: a depth,
// stencil or multisampled texture is cleared through a framebuffer, as
// the other backends clear them with a render pass, and a compressed one
// filled with blocks of zeros. GL hands out a texture with whatever its
// memory held.
func (t *Texture) zero(ctx *gl.Context) {
	switch {
	case t.glf.compressed():
		t.zeroBlocks(ctx)
	case t.format.IsDepthStencil() || t.sampleCount > 1:
		t.clearZero(ctx)
	}
}

// clearZero clears every mip level and layer of t to zero through a
// framebuffer: color 0, depth 0, stencil 0.
func (t *Texture) clearZero(ctx *gl.Context) {
	point := attachmentPoint(t.format, 0)
	withFramebuffer(ctx, gl.DRAW_FRAMEBUFFER, func() {
		clearColor := ClearColorBufferCommand{}
		var mask uint32
		if t.format.IsDepthStencil() {
			ctx.DrawBuffers(noDrawBuffers)
			ctx.Disable(gl.SCISSOR_TEST)
			if t.format.HasDepth() {
				ctx.DepthMask(true)
				ctx.ClearDepth(0)
				mask |= gl.DEPTH_BUFFER_BIT
			}
			if t.format.HasStencil() {
				ctx.StencilMaskSeparate(gl.FRONT_AND_BACK, 0xFF)
				ctx.ClearStencil(0)
				mask |= gl.STENCIL_BUFFER_BIT
			}
		} else {
			clearColor.integer, clearColor.signed = t.glf.integer()
		}
		for mip := range t.mipLevels {
			for layer := range t.mipLayers(mip) {
				attach(ctx, gl.DRAW_FRAMEBUFFER, point, fbAttachment{tex: t, mip: mip, layer: layer})
				if mask != 0 {
					ctx.Clear(mask)
				} else {
					clearColor.Execute(ctx)
				}
			}
		}
	})
}

// zeroBlocks fills every mip level and layer of t, a compressed texture
// bound to its target, with blocks of zeros, some rows of blocks at a time.
func (t *Texture) zeroBlocks(ctx *gl.Context) {
	zeros := glZeros[:]
	for mip := range t.mipLevels {
		width, height := t.mipSize(mip)
		blockRows, rowBytes := compressedRows(t.format, width, height)
		if int(rowBytes) > len(zeros) {
			zeros = make([]byte, rowBytes)
		}
		_, bh := t.format.BlockDimensions()
		band := uint32(len(zeros)) / rowBytes // rows of blocks an upload takes
		for layer := range t.mipLayers(mip) {
			for row := uint32(0); row < blockRows; row += band {
				at := hal.ImageCopyTexture{MipLevel: mip, Origin: hal.Origin3D{Y: row * bh, Z: layer}}
				_ = writeBlocks(ctx, t, &at, zeros, hal.ImageDataLayout{}, width, min(band*bh, height-row*bh), 1)
			}
		}
	}
}

// mipLayers is how many layers, cube faces or 3D slices mip level mip of t
// has.
func (t *Texture) mipLayers(mip uint32) uint32 {
	if t.dimension == gputypes.TextureDimension3D {
		return max(1, t.size.DepthOrArrayLayers>>mip)
	}
	return t.layers()
}

// createRenderbuffer makes a multisampled texture that is only rendered to
// as a renderbuffer, which ES 3.0 has multisampled storage for and no
// texture, as wgpu-hal makes its render-only textures.
func createRenderbuffer(glCtx *gl.Context, glf glFormat, samples uint32, desc *TextureDescriptor) (*Texture, error) {
	id := glCtx.GenRenderbuffers(1)
	glCtx.BindRenderbuffer(gl.RENDERBUFFER, id)
	glCtx.RenderbufferStorageMultisample(gl.RENDERBUFFER, int32(samples), glf.internal, int32(desc.Size.Width), int32(desc.Size.Height))
	glCtx.BindRenderbuffer(gl.RENDERBUFFER, 0)
	if glErr := glCtx.GetError(); glErr != 0 {
		glCtx.DeleteRenderbuffers(id)
		return nil, fmt.Errorf("gles: RenderbufferStorageMultisample failed: GL error 0x%x (format=0x%x, samples=%d, %dx%d)",
			glErr, glf.internal, samples, desc.Size.Width, desc.Size.Height)
	}
	t := &Texture{
		id:          id,
		target:      gl.RENDERBUFFER,
		format:      desc.Format,
		glf:         glf,
		dimension:   desc.Dimension,
		size:        desc.Size,
		mipLevels:   1,
		sampleCount: samples,
		glCtx:       glCtx,
	}
	t.zero(glCtx)
	return t, nil
}

// writeTexture uploads data into tex at dst: the layers or slices from
// dst.Origin.Z on of an array, cube or 3D texture. The context must be
// current.
func writeTexture(glCtx *gl.Context, tex *Texture, dst *hal.ImageCopyTexture, data []byte, layout *hal.ImageDataLayout, size *hal.Extent3D) error {
	if size.Width == 0 || size.Height == 0 {
		return nil
	}
	var l hal.ImageDataLayout
	if layout != nil {
		l = *layout
	}
	if l.Offset > uint64(len(data)) {
		return fmt.Errorf("gles: WriteTexture data starts at %d of %d bytes", l.Offset, len(data))
	}
	data = data[l.Offset:]
	depth := max(1, size.DepthOrArrayLayers)
	glCtx.BindTexture(tex.target, tex.id)
	var err error
	switch {
	case tex.glf.compressed():
		err = writeBlocks(glCtx, tex, dst, data, l, size.Width, size.Height, depth)
	case tex.stencilBesideDepth(dst.Aspect):
		err = writeStencilData(glCtx, tex, dst, data, l, size.Width, size.Height, depth)
	default:
		err = writePixels(glCtx, tex, dst, data, l, size.Width, size.Height, depth)
	}
	glCtx.BindTexture(tex.target, 0)
	if err != nil {
		return err
	}

	if hal.DebugEnabled() {
		hal.Logger().Debug("gles: texture written",
			"format", tex.format,
			"width", size.Width,
			"height", size.Height,
		)
	}
	return nil
}

// checkData reports whether data holds images of rows rows of rowBytes, at
// pitch bytes a row and imageRows rows an image.
func checkData(data []byte, rowBytes, rows, images, pitch, imageRows uint32) error {
	need := (uint64(images-1)*uint64(imageRows)+uint64(rows-1))*uint64(pitch) + uint64(rowBytes)
	if uint64(len(data)) < need {
		return fmt.Errorf("gles: WriteTexture needs %d bytes of data, has %d", need, len(data))
	}
	return nil
}

// packRows copies those rows out of data into one packed run.
func packRows(data []byte, rowBytes, rows, images, pitch, imageRows uint32) []byte {
	packed := make([]byte, int(rowBytes)*int(rows)*int(images))
	for img := range int(images) {
		for r := range int(rows) {
			from := (img*int(imageRows) + r) * int(pitch)
			copy(packed[(img*int(rows)+r)*int(rowBytes):], data[from:from+int(rowBytes)])
		}
	}
	return packed
}

// writePixels uploads texels with the texture's transfer pair, the rows and
// images at the layout's pitches: an atlas fills in one small rectangle of a
// bigger image at a time.
func writePixels(glCtx *gl.Context, tex *Texture, dst *hal.ImageCopyTexture, data []byte, l hal.ImageDataLayout, width, height, depth uint32) error {
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
	if err := checkData(data, width*bpp, height, depth, pitch, rows); err != nil {
		return err
	}
	if pitch%bpp != 0 {
		// UNPACK_ROW_LENGTH counts texels, so rows a part texel apart
		// go in packed.
		data, pitch, rows = packRows(data, width*bpp, height, depth, pitch, rows), width*bpp, height
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
	return nil
}

// stencilBesideDepth reports whether a copy of aspect into t moves only
// stencil into storage that holds depth beside it: the stencil aspect of a
// combined format, or a Stencil8 texture made as one.
func (t *Texture) stencilBesideDepth(aspect gputypes.TextureAspect) bool {
	return t.glf.format == gl.DEPTH_STENCIL && (aspect == gputypes.TextureAspectStencilOnly || t.format == gputypes.TextureFormatStencil8)
}

// errStencilWriteES refuses a write of the stencil beside depth on ES, which
// writeStencil needs to read back and ES reads no depth or stencil.
var errStencilWriteES = errors.New("gles: ES can't write the stencil of a texture that also holds depth, as it reads no depth or stencil back")

// writeStencilData writes the stencil bytes of data, at the layout's
// pitches, into tex at dst, whose stencil is stored beside depth.
func writeStencilData(glCtx *gl.Context, tex *Texture, dst *hal.ImageCopyTexture, data []byte, l hal.ImageDataLayout, width, height, depth uint32) error {
	rows, pitch := max(height, l.RowsPerImage), max(width, l.BytesPerRow)
	if err := checkData(data, width, height, depth, pitch, rows); err != nil {
		return err
	}
	return writeStencil(glCtx, tex, dst.MipLevel, dst.Origin, width, height, depth, func(layer, y uint32) []byte {
		at := (int(layer)*int(rows) + int(y)) * int(pitch)
		return data[at : at+int(width)]
	})
}

// writeStencil writes the stencil of a width by height region of tex at mip
// and origin, of layers layers from origin.Z on, whose rows row returns.
// GL writes no stencil alone into storage that holds depth beside it, so
// each layer's depth and stencil are read, the stencil replaced and the
// pair written back; ES, which reads neither, refuses it.
func writeStencil(ctx *gl.Context, tex *Texture, mip uint32, origin hal.Origin3D, width, height, layers uint32, row func(layer, y uint32) []byte) error {
	if ctx.IsGLES() {
		return errStencilWriteES
	}
	f := tex.glf
	texel := pixelSize(f.format, f.typ)
	var stencil uint32 // the byte of a texel the stencil is: the low one of its word
	if f.typ == gl.FLOAT_32_UNSIGNED_INT_24_8_REV {
		stencil = 4
	}
	pixels := make([]byte, width*height*texel)
	withFramebuffer(ctx, gl.READ_FRAMEBUFFER, func() {
		ctx.ReadBuffer(gl.NONE)
		ctx.BindTexture(tex.target, tex.id)
		for z := range layers {
			attach(ctx, gl.READ_FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, fbAttachment{tex: tex, mip: mip, layer: origin.Z + z})
			ctx.ReadPixels(int32(origin.X), int32(origin.Y), int32(width), int32(height), f.format, f.typ, unsafe.Pointer(&pixels[0]))
			for y := range height {
				for x, s := range row(z, y) {
					pixels[(y*width+uint32(x))*texel+stencil] = s
				}
			}
			at := hal.ImageCopyTexture{MipLevel: mip, Origin: hal.Origin3D{X: origin.X, Y: origin.Y, Z: origin.Z + z}}
			_ = writePixels(ctx, tex, &at, pixels, hal.ImageDataLayout{}, width, height, 1)
		}
		ctx.BindTexture(tex.target, 0)
	})
	return nil
}

// writeBlocks uploads a compressed region. GL ES has no row length for
// compressed uploads, so rows given at a wider pitch go in packed.
func writeBlocks(glCtx *gl.Context, tex *Texture, dst *hal.ImageCopyTexture, data []byte, l hal.ImageDataLayout, width, height, depth uint32) error {
	blocks, rowBytes := compressedRows(tex.format, width, height)
	pitch, imageRows := rowBytes, blocks
	if l.BytesPerRow != 0 {
		pitch = l.BytesPerRow
	}
	if l.RowsPerImage != 0 {
		imageRows = l.RowsPerImage
	}
	if err := checkData(data, rowBytes, blocks, depth, pitch, imageRows); err != nil {
		return err
	}
	imageSize := rowBytes * blocks
	if pitch != rowBytes || imageRows != blocks {
		data = packRows(data, rowBytes, blocks, depth, pitch, imageRows)
	}
	width, height = tex.inMip(dst.MipLevel, dst.Origin, width, height)
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
	return nil
}

// inMip returns the part of a width by height region at origin of mip
// level mip of t inside the level. A copy of a compressed format takes
// whole blocks, which stick out past the edge of a level that isn't a whole
// number of them wide or high, and GL takes only the texels in the level,
// as wgpu-hal clamps a copy to the level's size.
func (t *Texture) inMip(mip uint32, origin hal.Origin3D, width, height uint32) (uint32, uint32) {
	w, h := t.mipSize(mip)
	return min(width, w-min(origin.X, w)), min(height, h-min(origin.Y, h))
}

// compressedRows returns how many rows of blocks a width by height region
// of a compressed format has, and how many bytes a row of them takes.
func compressedRows(format gputypes.TextureFormat, width, height uint32) (rows, rowBytes uint32) {
	bw, bh := format.BlockDimensions()
	return (height + bh - 1) / bh, (width + bw - 1) / bw * format.BlockCopySize()
}

// drainGLErrors clears the errors earlier calls left, so that glGetError
// speaks for the calls after it; glGetError returns the oldest. A lost
// context has one for every call, hence the bound.
func drainGLErrors(ctx *gl.Context) {
	for range 8 {
		e := ctx.GetError()
		if e == 0 {
			return
		}
		hal.Logger().Debug("gles: GL error left by an earlier call", "error", e)
	}
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
		v.dimension = desc.Dimension
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
	if v.dimension == gputypes.TextureViewDimensionUndefined {
		// WebGPU's default: the texture's dimension, an array of 2D layers
		// for a 2D texture with more than one.
		switch {
		case t.dimension == gputypes.TextureDimension3D:
			v.dimension = gputypes.TextureViewDimension3D
		case t.dimension == gputypes.TextureDimension1D:
			v.dimension = gputypes.TextureViewDimension1D
		case v.layerCount > 1:
			v.dimension = gputypes.TextureViewDimension2DArray
		default:
			v.dimension = gputypes.TextureViewDimension2D
		}
	}
	v.handle = lastViewKey.Add(1)
	views.Store(v.handle, v)
	return v, nil
}

// viewTarget is the texture target the shaders take a view of dim as, of a
// texture of samples samples: a 1D or 2D view is a 2D texture, as in
// wgpu-hal.
func viewTarget(dim gputypes.TextureViewDimension, samples uint32) uint32 {
	switch dim {
	case gputypes.TextureViewDimension2DArray:
		return gl.TEXTURE_2D_ARRAY
	case gputypes.TextureViewDimensionCube:
		return gl.TEXTURE_CUBE_MAP
	case gputypes.TextureViewDimensionCubeArray:
		return gl.TEXTURE_CUBE_MAP_ARRAY
	case gputypes.TextureViewDimension3D:
		return gl.TEXTURE_3D
	}
	if samples > 1 {
		return gl.TEXTURE_2D_MULTISAMPLE
	}
	return gl.TEXTURE_2D
}

// layered reports whether v binds to an image unit with all its layers or
// slices, an array, cube or 3D view, rather than its one layer.
func (v *TextureView) layered() bool {
	return v.dimension == gputypes.TextureViewDimension2DArray || v.dimension == gputypes.TextureViewDimension3D ||
		v.dimension == gputypes.TextureViewDimensionCube || v.dimension == gputypes.TextureViewDimensionCubeArray
}

// apart reports whether the shaders can't take v through its texture's own
// target: v's dimension takes another one, or v has only some of the
// texture's layers.
func (v *TextureView) apart() bool {
	t := v.texture
	return viewTarget(v.dimension, t.sampleCount) != t.target || v.baseLayer != 0 || v.layerCount != t.layers()
}

// wantsTexture reports whether v is sampled and loaded through a texture of
// its own, a texture view, on a context with them (views): a view apart
// from a texture the shaders bind, made with immutable storage, as Dawn
// makes views on GL.
func (v *TextureView) wantsTexture(views bool) bool {
	t := v.texture
	return views && t.immutable && t.usage&(gputypes.TextureUsageTextureBinding|gputypes.TextureUsageStorageBinding) != 0 && v.apart()
}

// makeTexture makes v's texture view, of its dimension, mip levels, layers
// and aspect. The context must be current.
func (v *TextureView) makeTexture(ctx *gl.Context) {
	t := v.texture
	drainGLErrors(ctx)
	id := ctx.GenTextures(1)
	target := viewTarget(v.dimension, t.sampleCount)
	ctx.TextureView(id, target, t.id, t.glf.internal, v.baseMip, v.mipCount, v.baseLayer, v.layerCount)
	if glErr := ctx.GetError(); glErr != 0 {
		ctx.DeleteTextures(id)
		hal.Logger().Warn("gles: glTextureView failed", "error", fmt.Sprintf("0x%x", glErr), "format", t.format, "dimension", v.dimension)
		return
	}
	ctx.BindTexture(target, id)
	setTextureDefaults(ctx, target, v.mipCount)
	if v.readsStencil() {
		ctx.TexParameteri(target, gl.DEPTH_STENCIL_TEXTURE_MODE, gl.STENCIL_INDEX)
	}
	ctx.BindTexture(target, 0)
	v.id, v.target = id, target
}

// bindable returns why the shaders can't take v as a binding of class, nil
// when they can: a view apart from its texture needs a texture view of its
// own to sample, and to bind more than one layer to an image unit.
func (v *TextureView) bindable(class bindingClass) error {
	if v.id != 0 || v.texture == nil || !v.apart() || class == bindingClassImage && !v.layered() {
		return nil
	}
	return fmt.Errorf("a %s view of layers %d to %d of a texture of %d layers needs a GL texture view: OpenGL 4.3 or ARB, OES or EXT_texture_view",
		v.dimension, v.baseLayer, v.baseLayer+v.layerCount-1, v.texture.layers())
}

// bindImage binds v's mip level to an image unit for shaders to load from
// and store to: all its layers or slices for an array, cube or 3D view, its
// one layer otherwise.
func bindImage(ctx *gl.Context, unit uint32, v *TextureView, access, format uint32) {
	if v.id != 0 {
		ctx.BindImageTexture(unit, v.id, 0, v.layered(), 0, access, format)
		return
	}
	if v.texture == nil || v.texture.id == 0 {
		return
	}
	ctx.BindImageTexture(unit, v.texture.id, int32(v.baseMip), v.layered(), int32(v.baseLayer), access, format)
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

// bindTexture binds v on unit for the shaders to sample: its texture view,
// or its texture set to the view's mip range and aspect.
func bindTexture(ctx *gl.Context, unit uint32, v *TextureView) {
	t := v.texture
	id, target := v.id, v.target
	if id == 0 {
		if t == nil || t.id == 0 {
			return
		}
		id, target = t.id, t.target
	}
	sh := &ctx.Shadow
	want := gl.ShadowObject{ID: id, Target: target, Set: true}
	bound := unit < gl.MaxShadowUnits && sh.Textures[unit] == want
	if bound && (v.id != 0 || t.viewApplied(v)) {
		return
	}
	if !sh.ActiveUnitSet || sh.ActiveUnit != unit {
		ctx.ActiveTexture(gl.TEXTURE0 + unit)
		sh.ActiveUnit, sh.ActiveUnitSet = unit, true
	}
	if !bound {
		ctx.BindTexture(target, id)
		if unit < gl.MaxShadowUnits {
			sh.Textures[unit] = want
		}
	}
	if v.id == 0 {
		t.applyView(ctx, v)
	}
}
