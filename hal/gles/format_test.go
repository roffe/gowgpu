// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

func TestGLFormat(t *testing.T) {
	tests := []struct {
		format gputypes.TextureFormat
		want   glFormat
	}{
		{gputypes.TextureFormatR8Unorm, glFormat{gl.R8, gl.RED, gl.UNSIGNED_BYTE}},
		{gputypes.TextureFormatR8Uint, glFormat{gl.R8UI, gl.RED_INTEGER, gl.UNSIGNED_BYTE}},
		{gputypes.TextureFormatRG8Snorm, glFormat{gl.RG8_SNORM, gl.RG, gl.BYTE}},
		{gputypes.TextureFormatRGBA8UnormSrgb, glFormat{gl.SRGB8_ALPHA8, gl.RGBA, gl.UNSIGNED_BYTE}},
		{gputypes.TextureFormatBGRA8Unorm, glFormat{gl.RGBA8, gl.BGRA, gl.UNSIGNED_BYTE}},
		{gputypes.TextureFormatRGB10A2Unorm, glFormat{gl.RGB10_A2, gl.RGBA, gl.UNSIGNED_INT_2_10_10_10_REV}},
		{gputypes.TextureFormatRG11B10Ufloat, glFormat{gl.R11F_G11F_B10F, gl.RGB, gl.UNSIGNED_INT_10F_11F_11F_REV}},
		{gputypes.TextureFormatRGBA16Sint, glFormat{gl.RGBA16I, gl.RGBA_INTEGER, gl.SHORT}},
		{gputypes.TextureFormatRGBA32Float, glFormat{gl.RGBA32F, gl.RGBA, gl.FLOAT}},
		{gputypes.TextureFormatDepth16Unorm, glFormat{gl.DEPTH_COMPONENT16, gl.DEPTH_COMPONENT, gl.UNSIGNED_SHORT}},
		{gputypes.TextureFormatDepth24PlusStencil8, glFormat{gl.DEPTH24_STENCIL8, gl.DEPTH_STENCIL, gl.UNSIGNED_INT_24_8}},
		{gputypes.TextureFormatDepth32Float, glFormat{gl.DEPTH_COMPONENT32F, gl.DEPTH_COMPONENT, gl.FLOAT}},
		{gputypes.TextureFormatDepth32FloatStencil8, glFormat{gl.DEPTH32F_STENCIL8, gl.DEPTH_STENCIL, gl.FLOAT_32_UNSIGNED_INT_24_8_REV}},
		{gputypes.TextureFormatStencil8, glFormat{gl.DEPTH24_STENCIL8, gl.DEPTH_STENCIL, gl.UNSIGNED_INT_24_8}},
	}
	for _, tt := range tests {
		got, ok := formatCaps{}.glFormat(tt.format)
		if !ok || got != tt.want {
			t.Errorf("%s: got %#x %v, want %#x", tt.format, got, ok, tt.want)
		}
	}
	if _, ok := (formatCaps{}).glFormat(gputypes.TextureFormat(9999)); ok {
		t.Error("an unknown format got a GL format")
	}
	want := glFormat{gl.STENCIL_INDEX8, gl.STENCIL_INDEX, gl.UNSIGNED_BYTE}
	if got, _ := (formatCaps{stencil8: true}).glFormat(gputypes.TextureFormatStencil8); got != want {
		t.Errorf("Stencil8 with stencil-only textures is %#x, want %#x", got, want)
	}
}

// Every texel moved with a format's transfer pair is as big as WebGPU's
// texel: uploads read that much from the caller's data and readbacks write
// that much into a buffer sized from it.
func TestGLFormatTransferSize(t *testing.T) {
	for f := gputypes.TextureFormatR8Unorm; f <= gputypes.TextureFormatDepth32FloatStencil8; f++ {
		size := f.BlockCopySize()
		if size == 0 || f.HasStencil() {
			continue
		}
		g, ok := formatCaps{}.glFormat(f)
		if !ok {
			t.Errorf("%s has no GL format", f)
			continue
		}
		if got := pixelSize(g.format, g.typ); got != size {
			t.Errorf("%s moves %d bytes a texel, want %d", f, got, size)
		}
	}
}

func TestGLFormatES(t *testing.T) {
	es := formatCaps{es: true}
	if _, ok := es.glFormat(gputypes.TextureFormatBGRA8Unorm); ok {
		t.Error("BGRA8Unorm made on ES without EXT_texture_format_BGRA8888")
	}
	if g, ok := (formatCaps{es: true, bgra8: true}).glFormat(gputypes.TextureFormatBGRA8Unorm); !ok || g.internal != gl.BGRA {
		t.Errorf("BGRA8Unorm with EXT_texture_format_BGRA8888 is %#x %v, want the unsized GL_BGRA_EXT", g, ok)
	}
	if _, ok := (formatCaps{es: true, bgra8: true}).glFormat(gputypes.TextureFormatBGRA8UnormSrgb); ok {
		t.Error("BGRA8UnormSrgb made on ES")
	}
	if _, ok := es.glFormat(gputypes.TextureFormatR16Unorm); ok {
		t.Error("R16Unorm made on ES without EXT_texture_norm16")
	}
	if _, ok := (formatCaps{es: true, norm16: true}).glFormat(gputypes.TextureFormatRGBA16Snorm); !ok {
		t.Error("RGBA16Snorm refused on ES with EXT_texture_norm16")
	}
}

func TestGLFormatCompressed(t *testing.T) {
	storage := formatCaps{texStorage: true}
	for format, want := range map[gputypes.TextureFormat]uint32{
		gputypes.TextureFormatBC1RGBAUnorm:       gl.COMPRESSED_RGBA_S3TC_DXT1_EXT,
		gputypes.TextureFormatBC7RGBAUnormSrgb:   gl.COMPRESSED_SRGB_ALPHA_BPTC_UNORM,
		gputypes.TextureFormatETC2RGB8Unorm:      gl.COMPRESSED_RGB8_ETC2,
		gputypes.TextureFormatEACRG11Snorm:       gl.COMPRESSED_SIGNED_RG11_EAC,
		gputypes.TextureFormatASTC4x4Unorm:       0x93B0,
		gputypes.TextureFormatASTC10x8UnormSrgb:  0x93DA,
		gputypes.TextureFormatASTC12x12UnormSrgb: 0x93DD,
	} {
		got, ok := storage.glFormat(format)
		if !ok || got.internal != want || !got.compressed() {
			t.Errorf("%s: got %#x %v, want the compressed %#x", format, got, ok, want)
		}
		if _, ok := (formatCaps{}).glFormat(format); ok {
			t.Errorf("%s made without glTexStorage2D", format)
		}
	}
}
