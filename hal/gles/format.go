// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
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

// formatCaps is what a context offers beyond the formats of glFormats.
type formatCaps struct {
	es       bool
	bgra8    bool // ES: EXT_texture_format_BGRA8888 or APPLE_texture_format_BGRA8888
	norm16   bool // ES: EXT_texture_norm16, for the 16-bit unorm and snorm formats
	stencil8 bool // STENCIL_INDEX8 textures: GL 4.4, ES 3.2, ARB_texture_stencil8, OES_texture_stencil8
}

// newFormatCaps reads the formats a context offers from its capabilities.
func newFormatCaps(caps *AdapterCapabilities) formatCaps {
	return formatCaps{
		es:     caps.IsES,
		bgra8:  hasExtension(caps.Extensions, "GL_EXT_texture_format_BGRA8888", "GL_APPLE_texture_format_BGRA8888"),
		norm16: hasExtension(caps.Extensions, "GL_EXT_texture_norm16"),
		stencil8: glVersionAtLeast(caps.GLMajor, caps.GLMinor, caps.IsES, [2]int{3, 2}, [2]int{4, 4}) ||
			hasExtension(caps.Extensions, "GL_ARB_texture_stencil8", "GL_OES_texture_stencil8"),
	}
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
