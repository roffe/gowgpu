//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestSurfaceTextureViewsInViewFormats clears a surface texture of
// RGBA8Unorm through a view in its sRGB view format and through one in its
// own format, and checks the back buffer holds the sRGB encoding of the
// clear colour after the first and the colour after the second. A view in
// a format that is not one of the surface's view formats, or of a 3D
// dimension, is refused.
func TestSurfaceTextureViewsInViewFormats(t *testing.T) {
	dev, queue := openTestDevice(t)
	const side, pitch = 4, 256
	tex := newTestTexture(t, dev, hal.TextureDescriptor{
		Size:        hal.Extent3D{Width: side, Height: side},
		Dimension:   gputypes.TextureDimension2D,
		Format:      gputypes.TextureFormatRGBA8Unorm,
		Usage:       gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopySrc,
		ViewFormats: []gputypes.TextureFormat{gputypes.TextureFormatRGBA8UnormSrgb},
	})
	own, err := dev.createViewRTVs(tex.raw, gputypes.TextureFormatUndefined, []gputypes.TextureFormat{gputypes.TextureFormatRGBA8Unorm})
	if err != nil {
		t.Fatalf("createViewRTVs(RGBA8Unorm): %v", err)
	}
	views, err := dev.createViewRTVs(tex.raw, gputypes.TextureFormatRGBA8Unorm, []gputypes.TextureFormat{
		gputypes.TextureFormatRGBA8Unorm, gputypes.TextureFormatRGBA8UnormSrgb, gputypes.TextureFormatRGBA8UnormSrgb,
	})
	if err != nil {
		t.Fatalf("createViewRTVs: %v", err)
	}
	t.Cleanup(func() {
		for _, v := range append(own, views...) {
			dev.rtvHeap.free(v.index)
		}
	})
	if len(views) != 1 || views[0].format != gputypes.TextureFormatRGBA8UnormSrgb {
		t.Fatalf("view RTVs %+v, want one in RGBA8UnormSrgb", views)
	}

	backBuffer := &Texture{
		raw:          tex.raw,
		format:       tex.format,
		dimension:    gputypes.TextureDimension2D,
		size:         hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
		mipLevels:    1,
		samples:      1,
		usage:        gputypes.TextureUsageRenderAttachment,
		device:       dev,
		isExternal:   true,
		currentState: d3d12.D3D12_RESOURCE_STATE_PRESENT,
	}
	backBuffer.stateOwner.setTextureStates([]d3d12.D3D12_RESOURCE_STATES{d3d12.D3D12_RESOURCE_STATE_PRESENT})
	st := &SurfaceTexture{resource: tex.raw, stateOwner: backBuffer, rtvHandle: own[0].handle, viewRTVs: views,
		format: tex.format, width: side, height: side}

	for _, desc := range []hal.TextureViewDescriptor{
		{Format: gputypes.TextureFormatBGRA8UnormSrgb},
		{Dimension: gputypes.TextureViewDimension3D},
	} {
		if view, err := dev.CreateTextureView(st, &desc); err == nil {
			dev.DestroyTextureView(view)
			t.Errorf("CreateTextureView(%+v) succeeded, want it refused", desc)
		}
	}

	for _, tc := range []struct {
		format gputypes.TextureFormat
		want   byte
	}{
		{gputypes.TextureFormatRGBA8UnormSrgb, 188}, // linear 0.5 encoded
		{gputypes.TextureFormatRGBA8Unorm, 128},
	} {
		view, err := dev.CreateTextureView(st, &hal.TextureViewDescriptor{Format: tc.format, Dimension: gputypes.TextureViewDimension2D, MipLevelCount: 1, ArrayLayerCount: 1})
		if err != nil {
			t.Fatalf("CreateTextureView(%v): %v", tc.format, err)
		}
		out := newTestBuffer(t, dev, pitch*side, gputypes.BufferUsageCopyDst|gputypes.BufferUsageMapRead)
		record(t, dev, queue, func(enc *CommandEncoder) {
			pass := enc.BeginRenderPass(&hal.RenderPassDescriptor{ColorAttachments: []hal.RenderPassColorAttachment{{
				View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
				ClearValue: gputypes.Color{R: 0.5, G: 0.5, B: 0.5, A: 1},
			}}})
			pass.End()
			enc.TransitionTextures([]hal.TextureBarrier{{Texture: st, Usage: hal.TextureUsageTransition{OldUsage: gputypes.TextureUsageRenderAttachment, NewUsage: gputypes.TextureUsageCopySrc}}})
			enc.CopyTextureToBuffer(st, out, []hal.BufferTextureCopy{{
				BufferLayout: hal.ImageDataLayout{BytesPerRow: pitch, RowsPerImage: side},
				TextureBase:  hal.ImageCopyTexture{Texture: st},
				Size:         hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
			}})
		})
		dev.DestroyTextureView(view)
		if got := readBuffer(t, dev, out, 4); got[0] < tc.want-1 || got[0] > tc.want+1 || got[3] != 255 {
			t.Errorf("cleared through a %v view: texel = %v, want about [%d %d %d 255]", tc.format, got, tc.want, tc.want, tc.want)
		}
	}
}
