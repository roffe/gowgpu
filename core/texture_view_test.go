//go:build !(js && wasm)

package core

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func viewTestTexture(format gputypes.TextureFormat, dim gputypes.TextureDimension, w, h, layers, mips, samples uint32) *Texture {
	return NewTexture(nil, nil, format, dim, gputypes.TextureUsageTextureBinding,
		gputypes.Extent3D{Width: w, Height: h, DepthOrArrayLayers: layers}, mips, samples, "view-test")
}

func TestResolveTextureViewDescriptorDefaults(t *testing.T) {
	for _, tc := range []struct {
		name string
		tex  *Texture
		desc *hal.TextureViewDescriptor
		want hal.TextureViewDescriptor
	}{
		{
			name: "2D",
			tex:  viewTestTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 64, 64, 1, 3, 1),
			want: hal.TextureViewDescriptor{Format: gputypes.TextureFormatRGBA8Unorm, Dimension: gputypes.TextureViewDimension2D,
				Aspect: gputypes.TextureAspectAll, MipLevelCount: 3, ArrayLayerCount: 1},
		},
		{
			name: "2D with layers is an array",
			tex:  viewTestTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 64, 64, 4, 1, 1),
			desc: &hal.TextureViewDescriptor{BaseArrayLayer: 1},
			want: hal.TextureViewDescriptor{Format: gputypes.TextureFormatRGBA8Unorm, Dimension: gputypes.TextureViewDimension2DArray,
				Aspect: gputypes.TextureAspectAll, MipLevelCount: 1, BaseArrayLayer: 1, ArrayLayerCount: 3},
		},
		{
			name: "cube",
			tex:  viewTestTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 64, 64, 12, 1, 1),
			desc: &hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimensionCube, BaseArrayLayer: 6},
			want: hal.TextureViewDescriptor{Format: gputypes.TextureFormatRGBA8Unorm, Dimension: gputypes.TextureViewDimensionCube,
				Aspect: gputypes.TextureAspectAll, MipLevelCount: 1, BaseArrayLayer: 6, ArrayLayerCount: 6},
		},
		{
			name: "3D has one layer",
			tex:  viewTestTexture(gputypes.TextureFormatR8Unorm, gputypes.TextureDimension3D, 16, 16, 16, 2, 1),
			desc: &hal.TextureViewDescriptor{BaseMipLevel: 1},
			want: hal.TextureViewDescriptor{Format: gputypes.TextureFormatR8Unorm, Dimension: gputypes.TextureViewDimension3D,
				Aspect: gputypes.TextureAspectAll, BaseMipLevel: 1, MipLevelCount: 1, ArrayLayerCount: 1},
		},
		{
			name: "depth of a combined format keeps the combined format",
			tex:  viewTestTexture(gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureDimension2D, 8, 8, 1, 1, 1),
			desc: &hal.TextureViewDescriptor{Aspect: gputypes.TextureAspectDepthOnly, Format: gputypes.TextureFormatDepth24Plus},
			want: hal.TextureViewDescriptor{Format: gputypes.TextureFormatDepth24PlusStencil8, Dimension: gputypes.TextureViewDimension2D,
				Aspect: gputypes.TextureAspectDepthOnly, MipLevelCount: 1, ArrayLayerCount: 1},
		},
		{
			name: "stencil of a combined format, format left out",
			tex:  viewTestTexture(gputypes.TextureFormatDepth32FloatStencil8, gputypes.TextureDimension2D, 8, 8, 1, 1, 1),
			desc: &hal.TextureViewDescriptor{Aspect: gputypes.TextureAspectStencilOnly},
			want: hal.TextureViewDescriptor{Format: gputypes.TextureFormatDepth32FloatStencil8, Dimension: gputypes.TextureViewDimension2D,
				Aspect: gputypes.TextureAspectStencilOnly, MipLevelCount: 1, ArrayLayerCount: 1},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ResolveTextureViewDescriptor(tc.tex, nil, tc.desc)
			if err != nil {
				t.Fatalf("ResolveTextureViewDescriptor: %v", err)
			}
			if got != tc.want {
				t.Errorf("got %+v, want %+v", got, tc.want)
			}
		})
	}
}

func TestResolveTextureViewDescriptorRejects(t *testing.T) {
	rgba := viewTestTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 64, 64, 1, 1, 1)
	for _, tc := range []struct {
		name        string
		tex         *Texture
		viewFormats []gputypes.TextureFormat
		desc        hal.TextureViewDescriptor
	}{
		{name: "mip past the end", tex: rgba, desc: hal.TextureViewDescriptor{BaseMipLevel: 3}},
		{name: "layer past the end", tex: rgba, desc: hal.TextureViewDescriptor{BaseArrayLayer: 2}},
		{name: "two layers of one", tex: rgba, desc: hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2DArray, ArrayLayerCount: 2}},
		{name: "format not among the view formats", tex: rgba, desc: hal.TextureViewDescriptor{Format: gputypes.TextureFormatRGBA8UnormSrgb}},
		{name: "other format", tex: rgba, viewFormats: []gputypes.TextureFormat{gputypes.TextureFormatRGBA8UnormSrgb},
			desc: hal.TextureViewDescriptor{Format: gputypes.TextureFormatRGBA16Float}},
		{name: "cube of one layer", tex: rgba, desc: hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimensionCube}},
		{name: "cube of a rectangle", tex: viewTestTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 64, 32, 6, 1, 1),
			desc: hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimensionCube}},
		{name: "cube array of 7 layers", tex: viewTestTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 64, 64, 7, 1, 1),
			desc: hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimensionCubeArray}},
		{name: "3D view of a 2D texture", tex: rgba, desc: hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimension3D}},
		{name: "depth of a color format", tex: rgba, desc: hal.TextureViewDescriptor{Aspect: gputypes.TextureAspectDepthOnly}},
		{name: "stencil of a depth format", tex: viewTestTexture(gputypes.TextureFormatDepth32Float, gputypes.TextureDimension2D, 8, 8, 1, 1, 1),
			desc: hal.TextureViewDescriptor{Aspect: gputypes.TextureAspectStencilOnly}},
		{name: "depth aspect in the stencil format", tex: viewTestTexture(gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureDimension2D, 8, 8, 1, 1, 1),
			desc: hal.TextureViewDescriptor{Aspect: gputypes.TextureAspectDepthOnly, Format: gputypes.TextureFormatStencil8}},
		{name: "array view of a multisampled texture", tex: viewTestTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 8, 8, 1, 1, 4),
			desc: hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2DArray}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := ResolveTextureViewDescriptor(tc.tex, tc.viewFormats, &tc.desc); !errors.Is(err, ErrInvalidTextureView) {
				t.Errorf("ResolveTextureViewDescriptor(%+v): %v, want ErrInvalidTextureView", tc.desc, err)
			}
		})
	}
}

func TestResolveTextureViewDescriptorViewFormat(t *testing.T) {
	tex := viewTestTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 64, 64, 1, 1, 1)
	desc := hal.TextureViewDescriptor{Format: gputypes.TextureFormatRGBA8UnormSrgb}
	got, err := ResolveTextureViewDescriptor(tex, []gputypes.TextureFormat{gputypes.TextureFormatRGBA8UnormSrgb}, &desc)
	if err != nil {
		t.Fatalf("ResolveTextureViewDescriptor: %v", err)
	}
	if got.Format != gputypes.TextureFormatRGBA8UnormSrgb {
		t.Errorf("format = %v, want RGBA8UnormSrgb", got.Format)
	}
}

func TestValidateViewFormats(t *testing.T) {
	if err := ValidateViewFormats(gputypes.TextureFormatBGRA8Unorm, []gputypes.TextureFormat{
		gputypes.TextureFormatBGRA8Unorm, gputypes.TextureFormatBGRA8UnormSrgb,
	}); err != nil {
		t.Errorf("BGRA8Unorm with its sRGB twin: %v", err)
	}
	if err := ValidateViewFormats(gputypes.TextureFormatRGBA8UnormSrgb, []gputypes.TextureFormat{gputypes.TextureFormatRGBA8Unorm}); err != nil {
		t.Errorf("RGBA8UnormSrgb with its linear twin: %v", err)
	}
	if err := ValidateViewFormats(gputypes.TextureFormatRGBA8Unorm, []gputypes.TextureFormat{gputypes.TextureFormatBGRA8UnormSrgb}); !errors.Is(err, ErrInvalidTextureView) {
		t.Errorf("RGBA8Unorm with BGRA8UnormSrgb: %v, want ErrInvalidTextureView", err)
	}
}

// A view format besides the texture's own needs the downlevel flag the
// caller names, and the texture's own format needs none.
func TestRequireViewFormats(t *testing.T) {
	without := &Device{}
	with := &Device{downlevel: gputypes.DownlevelCapabilities{Flags: gputypes.DownlevelFlagsSurfaceViewFormats}}
	own := []gputypes.TextureFormat{gputypes.TextureFormatBGRA8Unorm}
	srgb := []gputypes.TextureFormat{gputypes.TextureFormatBGRA8Unorm, gputypes.TextureFormatBGRA8UnormSrgb}
	for _, tc := range []struct {
		name        string
		device      *Device
		viewFormats []gputypes.TextureFormat
		flag        gputypes.DownlevelFlags
		ok          bool
	}{
		{"own format without the flag", without, own, gputypes.DownlevelFlagsSurfaceViewFormats, true},
		{"sRGB twin without the flag", without, srgb, gputypes.DownlevelFlagsSurfaceViewFormats, false},
		{"sRGB twin with the flag", with, srgb, gputypes.DownlevelFlagsSurfaceViewFormats, true},
		{"sRGB twin with another flag", with, srgb, gputypes.DownlevelFlagsViewFormats, false},
		{"another format", with, []gputypes.TextureFormat{gputypes.TextureFormatRGBA8UnormSrgb}, gputypes.DownlevelFlagsSurfaceViewFormats, false},
	} {
		err := tc.device.RequireViewFormats(gputypes.TextureFormatBGRA8Unorm, tc.viewFormats, tc.flag)
		if tc.ok != (err == nil) {
			t.Errorf("%s: %v", tc.name, err)
		}
	}
}
