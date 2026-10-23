// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"encoding/binary"
	"math"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// glFormat is how GL stores a texture format and moves its texels.
type glFormat struct {
	internal uint32 // the sized internal format the storage is made with
	format   uint32 // pixel transfer format
	typ      uint32 // pixel transfer type
}

// glFormats are the formats every GL 3.3 and ES 3.0 context makes the same
// way, as wgpu-hal's conv.rs and Dawn's GLFormat.cpp map them.
// formatCaps.glFormat adds the ones that depend on the context.
var glFormats = map[gputypes.TextureFormat]glFormat{
	gputypes.TextureFormatR8Unorm:     {gl.R8, gl.RED, gl.UNSIGNED_BYTE},
	gputypes.TextureFormatR8Snorm:     {gl.R8_SNORM, gl.RED, gl.BYTE},
	gputypes.TextureFormatR8Uint:      {gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE},
	gputypes.TextureFormatR8Sint:      {gl.R8I, gl.RED_INTEGER, gl.BYTE},
	gputypes.TextureFormatR16Unorm:    {gl.R16, gl.RED, gl.UNSIGNED_SHORT},
	gputypes.TextureFormatR16Snorm:    {gl.R16_SNORM, gl.RED, gl.SHORT},
	gputypes.TextureFormatR16Uint:     {gl.R16UI, gl.RED_INTEGER, gl.UNSIGNED_SHORT},
	gputypes.TextureFormatR16Sint:     {gl.R16I, gl.RED_INTEGER, gl.SHORT},
	gputypes.TextureFormatR16Float:    {gl.R16F, gl.RED, gl.HALF_FLOAT},
	gputypes.TextureFormatRG8Unorm:    {gl.RG8, gl.RG, gl.UNSIGNED_BYTE},
	gputypes.TextureFormatRG8Snorm:    {gl.RG8_SNORM, gl.RG, gl.BYTE},
	gputypes.TextureFormatRG8Uint:     {gl.RG8UI, gl.RG_INTEGER, gl.UNSIGNED_BYTE},
	gputypes.TextureFormatRG8Sint:     {gl.RG8I, gl.RG_INTEGER, gl.BYTE},
	gputypes.TextureFormatR32Float:    {gl.R32F, gl.RED, gl.FLOAT},
	gputypes.TextureFormatR32Uint:     {gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT},
	gputypes.TextureFormatR32Sint:     {gl.R32I, gl.RED_INTEGER, gl.INT},
	gputypes.TextureFormatRG16Unorm:   {gl.RG16, gl.RG, gl.UNSIGNED_SHORT},
	gputypes.TextureFormatRG16Snorm:   {gl.RG16_SNORM, gl.RG, gl.SHORT},
	gputypes.TextureFormatRG16Uint:    {gl.RG16UI, gl.RG_INTEGER, gl.UNSIGNED_SHORT},
	gputypes.TextureFormatRG16Sint:    {gl.RG16I, gl.RG_INTEGER, gl.SHORT},
	gputypes.TextureFormatRG16Float:   {gl.RG16F, gl.RG, gl.HALF_FLOAT},
	gputypes.TextureFormatRGBA8Unorm:  {gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE},
	gputypes.TextureFormatRGBA8Snorm:  {gl.RGBA8_SNORM, gl.RGBA, gl.BYTE},
	gputypes.TextureFormatRGBA8Uint:   {gl.RGBA8UI, gl.RGBA_INTEGER, gl.UNSIGNED_BYTE},
	gputypes.TextureFormatRGBA8Sint:   {gl.RGBA8I, gl.RGBA_INTEGER, gl.BYTE},
	gputypes.TextureFormatBGRA8Unorm:  {gl.RGBA8, gl.BGRA, gl.UNSIGNED_BYTE},
	gputypes.TextureFormatRG32Float:   {gl.RG32F, gl.RG, gl.FLOAT},
	gputypes.TextureFormatRG32Uint:    {gl.RG32UI, gl.RG_INTEGER, gl.UNSIGNED_INT},
	gputypes.TextureFormatRG32Sint:    {gl.RG32I, gl.RG_INTEGER, gl.INT},
	gputypes.TextureFormatRGBA16Unorm: {gl.RGBA16, gl.RGBA, gl.UNSIGNED_SHORT},
	gputypes.TextureFormatRGBA16Snorm: {gl.RGBA16_SNORM, gl.RGBA, gl.SHORT},
	gputypes.TextureFormatRGBA16Uint:  {gl.RGBA16UI, gl.RGBA_INTEGER, gl.UNSIGNED_SHORT},
	gputypes.TextureFormatRGBA16Sint:  {gl.RGBA16I, gl.RGBA_INTEGER, gl.SHORT},
	gputypes.TextureFormatRGBA16Float: {gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT},
	gputypes.TextureFormatRGBA32Float: {gl.RGBA32F, gl.RGBA, gl.FLOAT},
	gputypes.TextureFormatRGBA32Uint:  {gl.RGBA32UI, gl.RGBA_INTEGER, gl.UNSIGNED_INT},
	gputypes.TextureFormatRGBA32Sint:  {gl.RGBA32I, gl.RGBA_INTEGER, gl.INT},

	gputypes.TextureFormatRGBA8UnormSrgb: {gl.SRGB8_ALPHA8, gl.RGBA, gl.UNSIGNED_BYTE},
	gputypes.TextureFormatBGRA8UnormSrgb: {gl.SRGB8_ALPHA8, gl.BGRA, gl.UNSIGNED_BYTE},
	gputypes.TextureFormatRGB10A2Uint:    {gl.RGB10_A2UI, gl.RGBA_INTEGER, gl.UNSIGNED_INT_2_10_10_10_REV},
	gputypes.TextureFormatRGB10A2Unorm:   {gl.RGB10_A2, gl.RGBA, gl.UNSIGNED_INT_2_10_10_10_REV},
	gputypes.TextureFormatRG11B10Ufloat:  {gl.R11F_G11F_B10F, gl.RGB, gl.UNSIGNED_INT_10F_11F_11F_REV},
	gputypes.TextureFormatRGB9E5Ufloat:   {gl.RGB9_E5, gl.RGB, gl.UNSIGNED_INT_5_9_9_9_REV},

	gputypes.TextureFormatDepth16Unorm:         {gl.DEPTH_COMPONENT16, gl.DEPTH_COMPONENT, gl.UNSIGNED_SHORT},
	gputypes.TextureFormatDepth24Plus:          {gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT},
	gputypes.TextureFormatDepth24PlusStencil8:  {gl.DEPTH24_STENCIL8, gl.DEPTH_STENCIL, gl.UNSIGNED_INT_24_8},
	gputypes.TextureFormatDepth32Float:         {gl.DEPTH_COMPONENT32F, gl.DEPTH_COMPONENT, gl.FLOAT},
	gputypes.TextureFormatDepth32FloatStencil8: {gl.DEPTH32F_STENCIL8, gl.DEPTH_STENCIL, gl.FLOAT_32_UNSIGNED_INT_24_8_REV},
}

// compressedFormats are the internal formats of the compressed formats,
// which move as blocks with glCompressedTex*Image and no transfer pair.
var compressedFormats = map[gputypes.TextureFormat]uint32{
	gputypes.TextureFormatBC1RGBAUnorm:     gl.COMPRESSED_RGBA_S3TC_DXT1_EXT,
	gputypes.TextureFormatBC1RGBAUnormSrgb: gl.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT,
	gputypes.TextureFormatBC2RGBAUnorm:     gl.COMPRESSED_RGBA_S3TC_DXT3_EXT,
	gputypes.TextureFormatBC2RGBAUnormSrgb: gl.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT,
	gputypes.TextureFormatBC3RGBAUnorm:     gl.COMPRESSED_RGBA_S3TC_DXT5_EXT,
	gputypes.TextureFormatBC3RGBAUnormSrgb: gl.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT,
	gputypes.TextureFormatBC4RUnorm:        gl.COMPRESSED_RED_RGTC1,
	gputypes.TextureFormatBC4RSnorm:        gl.COMPRESSED_SIGNED_RED_RGTC1,
	gputypes.TextureFormatBC5RGUnorm:       gl.COMPRESSED_RG_RGTC2,
	gputypes.TextureFormatBC5RGSnorm:       gl.COMPRESSED_SIGNED_RG_RGTC2,
	gputypes.TextureFormatBC6HRGBUfloat:    gl.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT,
	gputypes.TextureFormatBC6HRGBFloat:     gl.COMPRESSED_RGB_BPTC_SIGNED_FLOAT,
	gputypes.TextureFormatBC7RGBAUnorm:     gl.COMPRESSED_RGBA_BPTC_UNORM,
	gputypes.TextureFormatBC7RGBAUnormSrgb: gl.COMPRESSED_SRGB_ALPHA_BPTC_UNORM,

	gputypes.TextureFormatETC2RGB8Unorm:       gl.COMPRESSED_RGB8_ETC2,
	gputypes.TextureFormatETC2RGB8UnormSrgb:   gl.COMPRESSED_SRGB8_ETC2,
	gputypes.TextureFormatETC2RGB8A1Unorm:     gl.COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2,
	gputypes.TextureFormatETC2RGB8A1UnormSrgb: gl.COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2,
	gputypes.TextureFormatETC2RGBA8Unorm:      gl.COMPRESSED_RGBA8_ETC2_EAC,
	gputypes.TextureFormatETC2RGBA8UnormSrgb:  gl.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC,
	gputypes.TextureFormatEACR11Unorm:         gl.COMPRESSED_R11_EAC,
	gputypes.TextureFormatEACR11Snorm:         gl.COMPRESSED_SIGNED_R11_EAC,
	gputypes.TextureFormatEACRG11Unorm:        gl.COMPRESSED_RG11_EAC,
	gputypes.TextureFormatEACRG11Snorm:        gl.COMPRESSED_SIGNED_RG11_EAC,
}

func init() {
	// The ASTC formats come as a Unorm and a UnormSrgb format a block size,
	// in the order of GL's internal formats.
	for i := range uint32(14) {
		unorm := gputypes.TextureFormatASTC4x4Unorm + gputypes.TextureFormat(2*i)
		compressedFormats[unorm] = gl.COMPRESSED_RGBA_ASTC_4x4 + i
		compressedFormats[unorm+1] = gl.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4 + i
	}
}

// compressed reports whether f moves as compressed blocks.
func (f glFormat) compressed() bool { return f.typ == 0 }

// formatCaps is what a context offers beyond the formats of glFormats.
type formatCaps struct {
	es       bool
	bgra8    bool // ES: EXT_texture_format_BGRA8888 or APPLE_texture_format_BGRA8888
	norm16   bool // ES: EXT_texture_norm16, for the 16-bit unorm and snorm formats
	stencil8 bool // STENCIL_INDEX8 textures: GL 4.4, ES 3.2, ARB_texture_stencil8, OES_texture_stencil8

	// texStorage is glTexStorage2D (GL 4.2, ARB_texture_storage, ES 3.0),
	// which the compressed formats are made with, and texStorageMS
	// glTexStorage2DMultisample (GL 4.3, ARB_texture_storage_multisample,
	// ES 3.1), which multisampled textures to sample are made with on ES.
	texStorage, texStorageMS bool
}

// newFormatCaps reads the formats a context offers from its capabilities.
func newFormatCaps(caps *AdapterCapabilities) formatCaps {
	return formatCaps{
		es:     caps.IsES,
		bgra8:  hasExtension(caps.Extensions, "GL_EXT_texture_format_BGRA8888", "GL_APPLE_texture_format_BGRA8888"),
		norm16: hasExtension(caps.Extensions, "GL_EXT_texture_norm16"),
		stencil8: glVersionAtLeast(caps.GLMajor, caps.GLMinor, caps.IsES, [2]int{3, 2}, [2]int{4, 4}) ||
			hasExtension(caps.Extensions, "GL_ARB_texture_stencil8", "GL_OES_texture_stencil8"),
		texStorage: hasTexStorage(caps.Extensions, caps.GLMajor, caps.GLMinor, caps.IsES),
		texStorageMS: glVersionAtLeast(caps.GLMajor, caps.GLMinor, caps.IsES, [2]int{3, 1}, [2]int{4, 3}) ||
			hasExtension(caps.Extensions, "GL_ARB_texture_storage_multisample"),
	}
}

// hasTexStorage reports whether a context has glTexStorage2D.
func hasTexStorage(exts map[string]bool, glMajor, glMinor int, isES bool) bool {
	return glVersionAtLeast(glMajor, glMinor, isES, [2]int{3, 0}, [2]int{4, 2}) ||
		hasExtension(exts, "GL_ARB_texture_storage")
}

// glFormat returns how GL stores format, false when the context can't.
func (fc formatCaps) glFormat(format gputypes.TextureFormat) (glFormat, bool) {
	if format == gputypes.TextureFormatStencil8 {
		// Without stencil-only textures, the stencil of a combined format,
		// as Dawn does.
		if fc.stencil8 {
			return glFormat{gl.STENCIL_INDEX8, gl.STENCIL_INDEX, gl.UNSIGNED_BYTE}, true
		}
		return glFormat{gl.DEPTH24_STENCIL8, gl.DEPTH_STENCIL, gl.UNSIGNED_INT_24_8}, true
	}
	if internal, ok := compressedFormats[format]; ok {
		return glFormat{internal: internal}, fc.texStorage
	}
	f, ok := glFormats[format]
	if !ok || !fc.es {
		return f, ok
	}
	switch format {
	case gputypes.TextureFormatBGRA8Unorm:
		// ES has no sized BGRA format for glTexImage2D, only the
		// extension's unsized one, and no sRGB BGRA at all.
		return glFormat{gl.BGRA, gl.BGRA, gl.UNSIGNED_BYTE}, fc.bgra8
	case gputypes.TextureFormatBGRA8UnormSrgb:
		return glFormat{}, false
	case gputypes.TextureFormatR16Unorm, gputypes.TextureFormatR16Snorm,
		gputypes.TextureFormatRG16Unorm, gputypes.TextureFormatRG16Snorm,
		gputypes.TextureFormatRGBA16Unorm, gputypes.TextureFormatRGBA16Snorm:
		return f, fc.norm16
	}
	return f, true
}

// swapchainFormat is the renderbuffer format of a surface's swapchain: its
// bytes reach the window through a blit, so BGRA formats are stored as RGBA.
func swapchainFormat(format gputypes.TextureFormat) uint32 {
	if format.IsSrgb() {
		return gl.SRGB8_ALPHA8
	}
	return gl.RGBA8
}

// pixelSize is how many bytes a texel takes in client memory or a pixel
// buffer when it moves with the transfer format and type given.
func pixelSize(format, typ uint32) uint32 {
	switch typ {
	case gl.UNSIGNED_INT_2_10_10_10_REV, gl.UNSIGNED_INT_10F_11F_11F_REV, gl.UNSIGNED_INT_5_9_9_9_REV, gl.UNSIGNED_INT_24_8:
		return 4
	case gl.FLOAT_32_UNSIGNED_INT_24_8_REV:
		return 8
	}
	var n uint32
	switch format {
	case gl.RED, gl.RED_INTEGER, gl.DEPTH_COMPONENT, gl.STENCIL_INDEX:
		n = 1
	case gl.RG, gl.RG_INTEGER:
		n = 2
	case gl.RGB, gl.RGB_INTEGER:
		n = 3
	default:
		n = 4
	}
	switch typ {
	case gl.BYTE, gl.UNSIGNED_BYTE:
		return n
	case gl.SHORT, gl.UNSIGNED_SHORT, gl.HALF_FLOAT:
		return 2 * n
	}
	return 4 * n
}

// esReadback is how an ES context reads a colour format back: glReadPixels
// on ES takes only RGBA with UNSIGNED_BYTE for normalized formats, FLOAT for
// float ones and INT or UNSIGNED_INT for integer ones (UNSIGNED_SHORT for the
// 16-bit norm formats of EXT_texture_norm16), besides an implementation pair
// that varies by driver. The texels read with format and typ are repacked
// into the format's own; repack is nil when the pair is the format's own.
type esReadback struct {
	format, typ uint32
	repack      func(dst, src []byte) // one texel
}

// esReadbackFor returns how ES reads f back, false for a format ES can't
// read with glReadPixels.
func esReadbackFor(f gputypes.TextureFormat) (esReadback, bool) {
	channels := func(order ...int) func(dst, src []byte) {
		return func(dst, src []byte) {
			for i, c := range order {
				dst[i] = src[c]
			}
		}
	}
	wide := func(n, size int) func(dst, src []byte) {
		return func(dst, src []byte) {
			for c := range n {
				copy(dst[c*size:(c+1)*size], src[c*4:c*4+size])
			}
		}
	}
	switch f {
	case gputypes.TextureFormatRGBA8Unorm, gputypes.TextureFormatRGBA8UnormSrgb:
		return esReadback{gl.RGBA, gl.UNSIGNED_BYTE, nil}, true
	case gputypes.TextureFormatR8Unorm:
		return esReadback{gl.RGBA, gl.UNSIGNED_BYTE, channels(0)}, true
	case gputypes.TextureFormatRG8Unorm:
		return esReadback{gl.RGBA, gl.UNSIGNED_BYTE, channels(0, 1)}, true
	case gputypes.TextureFormatBGRA8Unorm, gputypes.TextureFormatBGRA8UnormSrgb:
		return esReadback{gl.RGBA, gl.UNSIGNED_BYTE, channels(2, 1, 0, 3)}, true
	case gputypes.TextureFormatRGB10A2Unorm:
		return esReadback{gl.RGBA, gl.UNSIGNED_INT_2_10_10_10_REV, nil}, true
	case gputypes.TextureFormatR16Unorm:
		return esReadback{gl.RGBA, gl.UNSIGNED_SHORT, channels(0, 1)}, true
	case gputypes.TextureFormatRG16Unorm:
		return esReadback{gl.RGBA, gl.UNSIGNED_SHORT, channels(0, 1, 2, 3)}, true
	case gputypes.TextureFormatRGBA16Unorm:
		return esReadback{gl.RGBA, gl.UNSIGNED_SHORT, nil}, true
	case gputypes.TextureFormatR32Float, gputypes.TextureFormatRG32Float:
		return esReadback{gl.RGBA, gl.FLOAT, wide(int(f.BlockCopySize()/4), 4)}, true
	case gputypes.TextureFormatRGBA32Float:
		return esReadback{gl.RGBA, gl.FLOAT, nil}, true
	case gputypes.TextureFormatR16Float, gputypes.TextureFormatRG16Float, gputypes.TextureFormatRGBA16Float:
		n := int(f.BlockCopySize() / 2)
		return esReadback{gl.RGBA, gl.FLOAT, func(dst, src []byte) {
			for c := range n {
				binary.LittleEndian.PutUint16(dst[c*2:], float16(math.Float32frombits(binary.LittleEndian.Uint32(src[c*4:]))))
			}
		}}, true
	case gputypes.TextureFormatRGBA32Uint, gputypes.TextureFormatRGBA32Sint:
		return esReadback{gl.RGBA_INTEGER, integerReadType(f), nil}, true
	case gputypes.TextureFormatRGB10A2Uint:
		return esReadback{gl.RGBA_INTEGER, gl.UNSIGNED_INT, func(dst, src []byte) {
			r, g, b, a := binary.LittleEndian.Uint32(src), binary.LittleEndian.Uint32(src[4:]), binary.LittleEndian.Uint32(src[8:]), binary.LittleEndian.Uint32(src[12:])
			binary.LittleEndian.PutUint32(dst, r&0x3ff|(g&0x3ff)<<10|(b&0x3ff)<<20|(a&3)<<30)
		}}, true
	}
	if g, ok := glFormats[f]; ok && (g.format == gl.RED_INTEGER || g.format == gl.RG_INTEGER || g.format == gl.RGBA_INTEGER) {
		// Integer formats read as four 32-bit integers a texel, of which
		// the low bytes of the format's channels are its texel.
		size := int(pixelSize(g.format, g.typ) / pixelSize(g.format, gl.UNSIGNED_BYTE))
		return esReadback{gl.RGBA_INTEGER, integerReadType(f), wide(int(pixelSize(g.format, gl.UNSIGNED_BYTE)), size)}, true
	}
	return esReadback{}, false
}

// integerReadType is the type ES reads an integer format back with.
func integerReadType(f gputypes.TextureFormat) uint32 {
	if g := glFormats[f]; g.typ == gl.BYTE || g.typ == gl.SHORT || g.typ == gl.INT {
		return gl.INT
	}
	return gl.UNSIGNED_INT
}

// float16 rounds a float32 to the nearest IEEE half float.
func float16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23&0xff) - 127 + 15
	mant := b & 0x7fffff
	switch {
	case b&0x7fffffff > 0x7f800000: // NaN
		return sign | 0x7e00
	case exp >= 0x1f: // too big, or infinite
		return sign | 0x7c00
	case exp <= 0: // subnormal, or too small
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - exp)
		half := mant >> shift
		if rem := mant & (1<<shift - 1); rem > 1<<(shift-1) || rem == 1<<(shift-1) && half&1 != 0 {
			half++
		}
		return sign | uint16(half)
	}
	half := uint32(exp)<<10 | mant>>13
	if rem := mant & 0x1fff; rem > 0x1000 || rem == 0x1000 && half&1 != 0 {
		half++ // a carry into the exponent is right, up to infinity
	}
	return sign | uint16(half)
}
