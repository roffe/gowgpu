//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// A queue write into one layer of a texture or one aspect of a depth/stencil
// texture moves the whole texture, the state the tracker keeps for it: the
// next pass on another layer starts from the layout it is really in, and
// the stencil write's barriers name both aspects of the image, which the
// validation layer checks.
func TestQueueWriteBarriersTakeTheWholeTexture(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL {
		t.Skip("the GL backend renders into a view's texture, not its layer")
	}

	tex, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 2}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm,
		Usage: TextureUsageRenderAttachment | TextureUsageTextureBinding | TextureUsageCopyDst | TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	layer0, err := device.CreateTextureView(tex, &TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2D, ArrayLayerCount: 1})
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer layer0.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, layer0, gputypes.LoadOpClear, gputypes.Color{G: 1, A: 1}, nil)
	finishAndSubmit(t, device, enc)
	data := make([]byte, pixRow*pixSide)
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex, Origin: Origin3D{Z: 1}}, data,
		&ImageDataLayout{BytesPerRow: pixRow}, &Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	out := readbackBuffer(t, device, "layer 0")
	enc, err = device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, layer0, gputypes.LoadOpLoad, gputypes.Color{}, nil)
	copyOut(enc, tex, out)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "layer 0 after a write into layer 1", readback(t, out), 0, 255, 0, 255)

	depthStencil, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: gputypes.TextureFormatDepth24PlusStencil8,
		Usage: TextureUsageRenderAttachment | TextureUsageTextureBinding | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture(depth/stencil): %v", err)
	}
	defer depthStencil.Release()
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: depthStencil, Aspect: gputypes.TextureAspectStencilOnly},
		make([]byte, pixSide*pixSide), &ImageDataLayout{BytesPerRow: pixSide}, &Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture(stencil): %v", err)
	}
	if _, err := device.Queue().Submit(); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	assertNoValidation(t, h)
}
