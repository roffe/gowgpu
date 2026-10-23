//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
)

// A view gets WebGPU's defaults for what its descriptor leaves out, the
// default view of a texture with layers being a 2D array, a view of one
// aspect of a depth/stencil texture may name that aspect's format, and a
// descriptor WebGPU refuses is an error rather than a view the backend
// makes anyway.
func TestCreateTextureViewResolvesAndChecksTheDescriptor(t *testing.T) {
	device, h := newValidatedDevice(t)

	layered, err := device.CreateTexture(&TextureDescriptor{
		Label: "view-layers", Size: Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 2},
		MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D,
		Format: TextureFormatRGBA8Unorm, Usage: TextureUsageTextureBinding,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer layered.Release()
	view, err := device.CreateTextureView(layered, nil)
	if err != nil {
		t.Fatalf("default view of a texture with 2 layers: %v", err)
	}
	view.Release()

	depth, err := device.CreateTexture(&TextureDescriptor{
		Label: "view-depth", Size: Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1},
		MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D,
		Format: gputypes.TextureFormatDepth24PlusStencil8, Usage: TextureUsageTextureBinding | TextureUsageRenderAttachment,
	})
	if err != nil {
		t.Fatalf("CreateTexture(depth): %v", err)
	}
	defer depth.Release()
	for _, desc := range []TextureViewDescriptor{
		{Aspect: gputypes.TextureAspectDepthOnly, Format: gputypes.TextureFormatDepth24Plus},
		{Aspect: gputypes.TextureAspectStencilOnly, Format: gputypes.TextureFormatStencil8},
	} {
		view, err := device.CreateTextureView(depth, &desc)
		if err != nil {
			t.Fatalf("view of aspect %v: %v", desc.Aspect, err)
		}
		view.Release()
	}

	if device.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsViewFormats) {
		storage, err := device.CreateTexture(&TextureDescriptor{
			Label: "view-srgb-storage", Size: Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1},
			MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D,
			Format: TextureFormatRGBA8Unorm, Usage: TextureUsageTextureBinding | TextureUsageStorageBinding,
			ViewFormats: []gputypes.TextureFormat{gputypes.TextureFormatRGBA8UnormSrgb},
		})
		if err != nil {
			t.Fatalf("CreateTexture(storage): %v", err)
		}
		defer storage.Release()
		srgb, err := device.CreateTextureView(storage, &TextureViewDescriptor{Format: gputypes.TextureFormatRGBA8UnormSrgb})
		if err != nil {
			t.Fatalf("sRGB view of a storage texture: %v", err)
		}
		srgb.Release()
	}

	flat, _ := pixTexture(t, device, "view-flat", TextureUsageTextureBinding)
	defer flat.Release()
	for name, desc := range map[string]TextureViewDescriptor{
		"mip past the end":     {BaseMipLevel: 3},
		"layer past the end":   {BaseArrayLayer: 2},
		"cube of one layer":    {Dimension: gputypes.TextureViewDimensionCube},
		"format not allowed":   {Format: gputypes.TextureFormatRGBA16Float},
		"depth of a color one": {Aspect: gputypes.TextureAspectDepthOnly},
	} {
		if view, err := device.CreateTextureView(flat, &desc); !errors.Is(err, core.ErrInvalidTextureView) {
			if view != nil {
				view.Release()
			}
			t.Errorf("%s: %v, want ErrInvalidTextureView", name, err)
		}
	}
	assertNoValidation(t, h)
}

// A render pass checks its attachments by their views: the default view of
// a texture with two mip levels is no attachment, a view of one of them is.
func TestBeginRenderPassChecksAttachmentViews(t *testing.T) {
	device, h := newValidatedDevice(t)

	tex, err := device.CreateTexture(&TextureDescriptor{
		Label: "attachment-mips", Size: Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1},
		MipLevelCount: 2, SampleCount: 1, Dimension: TextureDimension2D,
		Format: TextureFormatRGBA8Unorm, Usage: TextureUsageRenderAttachment,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	all, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer all.Release()
	one, err := device.CreateTextureView(tex, &TextureViewDescriptor{BaseMipLevel: 1, MipLevelCount: 1})
	if err != nil {
		t.Fatalf("CreateTextureView(mip 1): %v", err)
	}
	defer one.Release()

	for _, tc := range []struct {
		view  *TextureView
		valid bool
	}{{all, false}, {one, true}} {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		pass, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: tc.view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
		}}})
		var rpve *core.RenderPassValidationError
		switch {
		case tc.valid && err != nil:
			t.Errorf("view of one mip level: %v", err)
		case !tc.valid && !errors.As(err, &rpve):
			t.Errorf("view of two mip levels: %v, want a RenderPassValidationError", err)
		}
		if err == nil {
			if err := pass.End(); err != nil {
				t.Fatalf("End: %v", err)
			}
			finishAndSubmit(t, device, enc)
		} else {
			enc.DiscardEncoding()
		}
	}
	assertNoValidation(t, h)
}
