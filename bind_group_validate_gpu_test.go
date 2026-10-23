//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
)

// CreateBindGroup refuses a resource that doesn't fit its layout entry: a
// buffer at a texture binding, which crashed the driver, a texture without
// TextureBinding, a format of another sample type, both aspects of a
// depth/stencil texture and a sampler that compares where the layout's
// doesn't.
func TestCreateBindGroupChecksTheResources(t *testing.T) {
	device, h := newValidatedDevice(t)

	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: gputypes.ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{
			SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D,
		}},
		{Binding: 1, Visibility: gputypes.ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	depthBGL, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: gputypes.ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{
			SampleType: gputypes.TextureSampleTypeDepth, ViewDimension: gputypes.TextureViewDimension2D,
		}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout(depth): %v", err)
	}
	defer depthBGL.Release()

	view := func(format gputypes.TextureFormat, usage gputypes.TextureUsage, desc *TextureViewDescriptor) *TextureView {
		tex, err := device.CreateTexture(&TextureDescriptor{
			Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
			Dimension: TextureDimension2D, Format: format, Usage: usage,
		})
		if err != nil {
			t.Fatalf("CreateTexture(%v): %v", format, err)
		}
		t.Cleanup(tex.Release)
		v, err := device.CreateTextureView(tex, desc)
		if err != nil {
			t.Fatalf("CreateTextureView(%v): %v", format, err)
		}
		t.Cleanup(v.Release)
		return v
	}
	sampler := func(compare gputypes.CompareFunction) *Sampler {
		s, err := device.CreateSampler(&SamplerDescriptor{
			MagFilter: gputypes.FilterModeLinear, MinFilter: gputypes.FilterModeLinear, LodMaxClamp: 32, Compare: compare,
		})
		if err != nil {
			t.Fatalf("CreateSampler: %v", err)
		}
		t.Cleanup(s.Release)
		return s
	}
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: 256, Usage: BufferUsageUniform})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()

	good := view(TextureFormatRGBA8Unorm, TextureUsageTextureBinding, nil)
	filtering := sampler(gputypes.CompareFunctionUndefined)
	for _, tc := range []struct {
		name    string
		layout  *BindGroupLayout
		entries []BindGroupEntry
		valid   bool
	}{
		{"fitting", bgl, []BindGroupEntry{{Binding: 0, TextureView: good}, {Binding: 1, Sampler: filtering}}, true},
		{"buffer at a texture binding", bgl, []BindGroupEntry{{Binding: 0, Buffer: buf, Size: 256}, {Binding: 1, Sampler: filtering}}, false},
		{"texture without TextureBinding", bgl, []BindGroupEntry{
			{Binding: 0, TextureView: view(TextureFormatRGBA8Unorm, TextureUsageRenderAttachment, nil)}, {Binding: 1, Sampler: filtering},
		}, false},
		{"uint texture at a float binding", bgl, []BindGroupEntry{
			{Binding: 0, TextureView: view(gputypes.TextureFormatR32Uint, TextureUsageTextureBinding, nil)}, {Binding: 1, Sampler: filtering},
		}, false},
		{"comparison sampler at a filtering binding", bgl, []BindGroupEntry{
			{Binding: 0, TextureView: good}, {Binding: 1, Sampler: sampler(gputypes.CompareFunctionLess)},
		}, false},
		{"both aspects of a depth/stencil texture", depthBGL, []BindGroupEntry{
			{Binding: 0, TextureView: view(gputypes.TextureFormatDepth24PlusStencil8, TextureUsageTextureBinding, nil)},
		}, false},
		{"the depth aspect of a depth/stencil texture", depthBGL, []BindGroupEntry{
			{Binding: 0, TextureView: view(gputypes.TextureFormatDepth24PlusStencil8, TextureUsageTextureBinding,
				&TextureViewDescriptor{Aspect: gputypes.TextureAspectDepthOnly})},
		}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: tc.layout, Entries: tc.entries})
			if group != nil {
				group.Release()
			}
			var cbge *core.CreateBindGroupError
			switch {
			case tc.valid && err != nil:
				t.Errorf("CreateBindGroup: %v", err)
			case !tc.valid && !errors.As(err, &cbge):
				t.Errorf("CreateBindGroup: %v, want a CreateBindGroupError", err)
			}
		})
	}
	assertNoValidation(t, h)
}
