//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestAspectViewOfCombinedDepthStencilTakesTheTextureFormat makes views of
// one aspect of a Depth32FloatStencil8 texture in the aspect's format and
// checks their descriptors are made in the texture's format family.
func TestAspectViewOfCombinedDepthStencilTakesTheTextureFormat(t *testing.T) {
	dev, _ := openTestDevice(t)
	tex := newTestTexture(t, dev, hal.TextureDescriptor{
		Size:      hal.Extent3D{Width: 4, Height: 4},
		Dimension: gputypes.TextureDimension2D,
		Format:    gputypes.TextureFormatDepth32FloatStencil8,
		Usage:     gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding,
	})
	for _, tc := range []struct {
		aspect gputypes.TextureAspect
		format gputypes.TextureFormat
		srv    d3d12.DXGI_FORMAT
	}{
		{gputypes.TextureAspectDepthOnly, gputypes.TextureFormatDepth32Float, d3d12.DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS},
		{gputypes.TextureAspectStencilOnly, gputypes.TextureFormatStencil8, d3d12.DXGI_FORMAT_X32_TYPELESS_G8X24_UINT},
	} {
		raw, err := dev.CreateTextureView(tex, &hal.TextureViewDescriptor{Format: tc.format, Dimension: gputypes.TextureViewDimension2D, Aspect: tc.aspect, MipLevelCount: 1, ArrayLayerCount: 1})
		if err != nil {
			t.Fatalf("CreateTextureView(%v): %v", tc.aspect, err)
		}
		view := raw.(*TextureView)
		if view.format != tex.format {
			t.Errorf("%v view format = %v, want the texture's, %v", tc.aspect, view.format, tex.format)
		}
		if srv, _ := textureFormatToSRV(view.format, view.aspect); srv != tc.srv {
			t.Errorf("%v view SRV format = %v, want %v", tc.aspect, srv, tc.srv)
		}
		dev.DestroyTextureView(raw)
	}
}
