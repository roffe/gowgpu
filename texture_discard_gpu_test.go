//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// A texture whose contents a pass discarded reads as zeros, as WebGPU has
// it: a later pass that loads it in the same encoder, a copy in the same
// encoder and a pass that loads it in a later submit all see zeros, not the
// red the discarding pass cleared it to.
func TestDiscardedTextureReadsZeros(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL {
		t.Skip("GL keeps what a store op discards")
	}
	red := gputypes.Color{R: 1, A: 1}
	target := func() (*Texture, *TextureView) {
		tex, view := pixTexture(t, device, "discarded", TextureUsageRenderAttachment|TextureUsageCopySrc)
		t.Cleanup(func() { releaseTextureView(tex, view) })
		return tex, view
	}
	pass := func(enc *CommandEncoder, view *TextureView, load gputypes.LoadOp, store gputypes.StoreOp) {
		p, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{
			{View: view, LoadOp: load, StoreOp: store, ClearValue: red},
		}})
		if err != nil {
			t.Fatalf("BeginRenderPass: %v", err)
		}
		if err := p.End(); err != nil {
			t.Fatalf("End: %v", err)
		}
	}
	encoder := func() *CommandEncoder {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		return enc
	}

	tex, view := target()
	out := readbackBuffer(t, device, "loaded")
	enc := encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	pass(enc, view, gputypes.LoadOpLoad, gputypes.StoreOpStore)
	copyOut(enc, tex, out)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "loaded after a discard in the encoder", readback(t, out), 0, 0, 0, 0)

	tex, view = target()
	out = readbackBuffer(t, device, "copied")
	enc = encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	copyOut(enc, tex, out)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "copied after a discard in the encoder", readback(t, out), 0, 0, 0, 0)

	tex, view = target()
	out = readbackBuffer(t, device, "next submit")
	enc = encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	finishAndSubmit(t, device, enc)
	enc = encoder()
	pass(enc, view, gputypes.LoadOpLoad, gputypes.StoreOpStore)
	copyOut(enc, tex, out)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "loaded in the submit after the discard", readback(t, out), 0, 0, 0, 0)

	tex, view = target()
	out = readbackBuffer(t, device, "cleared again")
	enc = encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	finishAndSubmit(t, device, enc)
	enc = encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpStore)
	copyOut(enc, tex, out)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "cleared and stored after the discard", readback(t, out), 255, 0, 0, 255)
	assertNoValidation(t, h)
}
