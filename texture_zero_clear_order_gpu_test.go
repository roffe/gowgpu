//go:build !(js && wasm)

package wgpu

import (
	"context"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// A pass that clears a new texture spares it the zeroing only when it is
// the texture's first use in what runs: a copy earlier in the same encoder,
// a command buffer submitted before, and a later encoder after the clearing
// one was dropped all read zeros, not the memory a released texture left.
func TestNewTextureClearedLaterIsZeroedFirst(t *testing.T) {
	device, h := newValidatedDevice(t)
	const side = 64
	desc := &TextureDescriptor{
		Size: Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageRenderAttachment | TextureUsageCopySrc,
	}
	ab := gputypes.Color{R: 0xab / 255.0, G: 0xab / 255.0, B: 0xab / 255.0, A: 0xab / 255.0}
	newTarget := func() (*Texture, *TextureView) {
		leaveBehind(t, device, desc, func(old *Texture) {
			view, err := device.CreateTextureView(old, nil)
			if err != nil {
				t.Fatalf("CreateTextureView: %v", err)
			}
			defer view.Release()
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			clearPass(t, enc, view, gputypes.LoadOpClear, ab, nil)
			finishAndSubmit(t, device, enc)
		})
		tex, err := device.CreateTexture(desc)
		if err != nil {
			t.Fatalf("CreateTexture: %v", err)
		}
		t.Cleanup(tex.Release)
		view, err := device.CreateTextureView(tex, nil)
		if err != nil {
			t.Fatalf("CreateTextureView: %v", err)
		}
		t.Cleanup(view.Release)
		return tex, view
	}
	encoder := func() *CommandEncoder {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		return enc
	}
	staging := func() *Buffer {
		buf, err := device.CreateBuffer(&BufferDescriptor{Size: side * side * 4, Usage: BufferUsageCopyDst | BufferUsageMapRead})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		t.Cleanup(buf.Release)
		return buf
	}
	copyOut := func(enc *CommandEncoder, tex *Texture, buf *Buffer) {
		enc.CopyTextureToBuffer(tex, buf, []BufferTextureCopy{{
			BufferLayout: ImageDataLayout{BytesPerRow: side * 4, RowsPerImage: side},
			Size:         Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
		}})
	}
	finish := func(enc *CommandEncoder) *CommandBuffer {
		cb, err := enc.Finish()
		if err != nil {
			t.Fatalf("Finish: %v", err)
		}
		return cb
	}
	submit := func(cbs ...*CommandBuffer) {
		if _, err := device.Queue().Submit(cbs...); err != nil {
			t.Fatalf("Submit: %v", err)
		}
	}
	expectZeros := func(what string, buf *Buffer) {
		t.Helper()
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := buf.Map(ctx, MapModeRead, 0, buf.Size()); err != nil {
			t.Fatalf("Map: %v", err)
		}
		rng, err := buf.MappedRange(0, buf.Size())
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		expectBytes(t, what, rng.Bytes(), 0)
		if err := buf.Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
	}

	tex, view := newTarget()
	buf := staging()
	enc := encoder()
	copyOut(enc, tex, buf)
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{G: 1, A: 1}, nil)
	submit(finish(enc))
	expectZeros("a copy ahead of the clear in its encoder", buf)

	tex, view = newTarget()
	enc = encoder()
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{G: 1, A: 1}, nil)
	enc.DiscardEncoding()
	buf = staging()
	enc = encoder()
	copyOut(enc, tex, buf)
	submit(finish(enc))
	expectZeros("a copy after the clearing encoder was dropped", buf)

	tex, view = newTarget()
	buf = staging()
	reader := encoder()
	copyOut(reader, tex, buf)
	readCB := finish(reader)
	clearer := encoder()
	clearPass(t, clearer, view, gputypes.LoadOpClear, gputypes.Color{G: 1, A: 1}, nil)
	clearCB := finish(clearer)
	submit(readCB, clearCB)
	expectZeros("a command buffer submitted ahead of the clearing one", buf)
	assertNoValidation(t, h)
}
