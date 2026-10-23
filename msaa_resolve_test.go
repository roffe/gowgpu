//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestMSAAResolvesEveryAttachment clears two 4-sample attachments, each to
// its own colour, in a pass that resolves both into textures new to it, and
// reads the resolves. GL resolved attachment 0 alone, and the first resolve
// into a new texture blitted the resolve texture onto itself: making its
// framebuffer bound it for reading too.
func TestMSAAResolvesEveryAttachment(t *testing.T) {
	device, _ := newValidatedDevice(t)
	colors := []gputypes.Color{red, green}
	attachments := make([]RenderPassColorAttachment, len(colors))
	resolves := make([]*Texture, len(colors))
	for i, c := range colors {
		msaa, err := device.CreateTexture(&TextureDescriptor{
			Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}, Format: TextureFormatRGBA8Unorm,
			Usage: TextureUsageRenderAttachment, MipLevelCount: 1, SampleCount: 4, Dimension: TextureDimension2D,
		})
		if err != nil {
			t.Fatalf("CreateTexture(msaa): %v", err)
		}
		defer msaa.Release()
		msaaView, err := device.CreateTextureView(msaa, nil)
		if err != nil {
			t.Fatalf("CreateTextureView: %v", err)
		}
		defer msaaView.Release()
		var resolveView *TextureView
		resolves[i], resolveView = pixTexture(t, device, "resolve", TextureUsageRenderAttachment|TextureUsageCopySrc)
		defer releaseTextureView(resolves[i], resolveView)
		attachments[i] = RenderPassColorAttachment{View: msaaView, ResolveTarget: resolveView,
			LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: c}
	}
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: attachments})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	finishAndSubmit(t, device, enc)
	for i, c := range colors {
		pix := readSubresources(t, device, resolves[i], gputypes.TextureAspectAll, []subresource{{width: pixSide, height: pixSide, rowBytes: pixSide * 4, rows: pixSide}})[0]
		expectPixels(t, "resolve of attachment "+string('0'+rune(i)), pix, byte(c.R*255), byte(c.G*255), byte(c.B*255), 255)
	}
}
