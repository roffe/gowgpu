//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestSRGBViewOfALinearTexture clears an RGBA8Unorm texture through an
// sRGB view, with and without fully typed format casting, and checks the
// texture holds the sRGB encoding of the clear colour.
func TestSRGBViewOfALinearTexture(t *testing.T) {
	dev, queue := openTestDevice(t)
	casting := dev.castingFullyTyped
	t.Cleanup(func() { dev.castingFullyTyped = casting })
	const side, pitch = 4, 256
	for _, tc := range []struct {
		casting bool
		format  d3d12.DXGI_FORMAT
	}{
		{true, d3d12.DXGI_FORMAT_R8G8B8A8_UNORM},
		{false, d3d12.DXGI_FORMAT_R8G8B8A8_TYPELESS},
	} {
		dev.castingFullyTyped = tc.casting
		tex := newTestTexture(t, dev, hal.TextureDescriptor{
			Size:        hal.Extent3D{Width: side, Height: side},
			Dimension:   gputypes.TextureDimension2D,
			Format:      gputypes.TextureFormatRGBA8Unorm,
			Usage:       gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopySrc,
			ViewFormats: []gputypes.TextureFormat{gputypes.TextureFormatRGBA8UnormSrgb},
		})
		if got := tex.raw.GetDesc().Format; got != tc.format {
			t.Errorf("casting %v: resource format = %d, want %d", tc.casting, got, tc.format)
		}
		view, err := dev.CreateTextureView(tex, &hal.TextureViewDescriptor{Format: gputypes.TextureFormatRGBA8UnormSrgb, Dimension: gputypes.TextureViewDimension2D, MipLevelCount: 1, ArrayLayerCount: 1})
		if err != nil {
			t.Fatalf("CreateTextureView: %v", err)
		}
		t.Cleanup(func() { dev.DestroyTextureView(view) })
		out := newTestBuffer(t, dev, pitch*side, gputypes.BufferUsageCopyDst|gputypes.BufferUsageMapRead)
		record(t, dev, queue, func(enc *CommandEncoder) {
			pass := enc.BeginRenderPass(&hal.RenderPassDescriptor{ColorAttachments: []hal.RenderPassColorAttachment{{
				View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
				ClearValue: gputypes.Color{R: 0.5, G: 0.5, B: 0.5, A: 1},
			}}})
			pass.End()
			enc.TransitionTextures([]hal.TextureBarrier{{Texture: tex, Usage: hal.TextureUsageTransition{OldUsage: gputypes.TextureUsageRenderAttachment, NewUsage: gputypes.TextureUsageCopySrc}}})
			enc.CopyTextureToBuffer(tex, out, []hal.BufferTextureCopy{{
				BufferLayout: hal.ImageDataLayout{BytesPerRow: pitch, RowsPerImage: side},
				TextureBase:  hal.ImageCopyTexture{Texture: tex},
				Size:         hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
			}})
		})
		// Linear 0.5 encodes to 188 in sRGB.
		if got := readBuffer(t, dev, out, 4); got[0] < 187 || got[0] > 189 || got[3] != 255 {
			t.Errorf("casting %v: texel = %v, want about [188 188 188 255]", tc.casting, got)
		}
	}
}
