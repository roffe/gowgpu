//go:build !(js && wasm)

package wgpu_test

import (
	"bytes"
	"context"
	"testing"

	"github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/gputypes"
)

// A new texture reads back as zeros where nothing wrote it, as WebGPU
// promises. The backends hand out whatever the memory held before, so the
// test leaves some 0xab textures behind for the new one to land on.
func TestNewTextureIsZeroed(t *testing.T) {
	inst, adapter, device := newDevice(t)
	defer inst.Release()
	defer adapter.Release()
	defer device.Release()
	q := device.Queue()

	const w, h, row = 64, 64, 3
	desc := &wgpu.TextureDescriptor{
		Size:          wgpu.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRGBA8Unorm,
		Usage:         gputypes.TextureUsageTextureBinding | gputypes.TextureUsageCopySrc | gputypes.TextureUsageCopyDst,
	}
	write := func(tex *wgpu.Texture, y, rows uint32, b byte) {
		t.Helper()
		err := q.WriteTexture(&wgpu.ImageCopyTexture{Texture: tex, Origin: wgpu.Origin3D{Y: y}},
			bytes.Repeat([]byte{b}, w*4*int(rows)),
			&wgpu.ImageDataLayout{BytesPerRow: w * 4, RowsPerImage: rows},
			&wgpu.Extent3D{Width: w, Height: rows, DepthOrArrayLayers: 1})
		if err != nil {
			t.Fatalf("WriteTexture: %v", err)
		}
	}

	for range 4 {
		old, err := device.CreateTexture(desc)
		if err != nil {
			t.Fatalf("CreateTexture: %v", err)
		}
		write(old, 0, h, 0xab)
		if _, err := q.Submit(); err != nil {
			t.Fatalf("Submit: %v", err)
		}
		if err := device.WaitIdle(); err != nil {
			t.Fatalf("WaitIdle: %v", err)
		}
		old.Release()
	}

	tex, err := device.CreateTexture(desc)
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	write(tex, row, 1, 0xff) // a partial write, the rest is never touched

	buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{Size: w * h * 4, Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.TransitionTextures([]wgpu.TextureBarrier{{Texture: tex, Usage: wgpu.TextureUsageTransition{OldUsage: gputypes.TextureUsageTextureBinding, NewUsage: gputypes.TextureUsageCopySrc}}})
	enc.CopyTextureToBuffer(tex, buf, []wgpu.BufferTextureCopy{{
		BufferLayout: wgpu.ImageDataLayout{BytesPerRow: w * 4, RowsPerImage: h},
		TextureBase:  wgpu.ImageCopyTexture{Texture: tex},
		Size:         wgpu.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
	}})
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if _, err := q.Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := buf.Map(context.Background(), wgpu.MapModeRead, 0, w*h*4); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, w*h*4)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	px := rng.Bytes()
	for y := range h {
		want := byte(0)
		if y == row {
			want = 0xff
		}
		for x, b := range px[y*w*4 : (y+1)*w*4] {
			if b != want {
				t.Fatalf("byte %d of row %d is %#x, want %#x", x, y, b, want)
			}
		}
	}
}
