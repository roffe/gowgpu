//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
	"testing"
	"time"
)

// rgbaPattern is w by h RGBA8 texels, each a function of its position and
// seed, so a misplaced row or column shows.
func rgbaPattern(w, h uint32, seed byte) []byte {
	data := make([]byte, 0, w*h*4)
	for y := uint32(0); y < h; y++ {
		for x := uint32(0); x < w; x++ {
			data = append(data, byte(x)+seed, byte(y)+seed, byte(x*y)^seed, 255)
		}
	}
	return data
}

func newRGBATexture(t *testing.T, device *Device, label string, w, h uint32) *Texture {
	t.Helper()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         label,
		Size:          Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageCopyDst | TextureUsageCopySrc | TextureUsageTextureBinding,
	})
	if err != nil {
		t.Fatalf("CreateTexture(%s): %v", label, err)
	}
	t.Cleanup(tex.Release)
	return tex
}

// readRGBA copies w by h RGBA8 texels of tex out and returns them with
// the rows packed.
func readRGBA(t *testing.T, device *Device, tex *Texture, w, h uint32) []byte {
	t.Helper()
	pitch := (w*4 + 255) &^ 255
	size := uint64(pitch) * uint64(h)
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "read-rgba", Size: size, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyTextureToBuffer(tex, buf, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{BytesPerRow: pitch, RowsPerImage: h},
		Size:         Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	out := make([]byte, 0, w*h*4)
	for y := uint32(0); y < h; y++ {
		out = append(out, rng.Bytes()[y*pitch:y*pitch+w*4]...)
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

func expectRGBA(t *testing.T, what string, got, want []byte) {
	t.Helper()
	if bytes.Equal(got, want) {
		return
	}
	for i := 0; i+3 < len(want); i += 4 {
		if !bytes.Equal(got[i:i+4], want[i:i+4]) {
			t.Errorf("%s: texel %d = %v, want %v", what, i/4, got[i:i+4], want[i:i+4])
			return
		}
	}
}

// TestWriteTexturesShareAStagingBatch writes a buffer and textures of
// several widths in one batch, so the later uploads sit at staging offsets
// that are not 512-aligned, and their rows run past the 512-byte row
// pitch from there.
func TestWriteTexturesShareAStagingBatch(t *testing.T) {
	device, _ := newValidatedDevice(t)

	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "batch-buffer", Size: 16, Usage: BufferUsageCopyDst | BufferUsageUniform})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	if err := device.Queue().WriteBuffer(buf, 0, make([]byte, 16)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	sizes := []struct{ w, h uint32 }{{64, 4}, {20, 13}, {128, 16}, {100, 3}, {256, 2}}
	textures := make([]*Texture, len(sizes))
	for i, s := range sizes {
		textures[i] = newRGBATexture(t, device, "batch-texture", s.w, s.h)
		err := device.Queue().WriteTexture(
			&ImageCopyTexture{Texture: textures[i]},
			rgbaPattern(s.w, s.h, byte(i*40)),
			&ImageDataLayout{BytesPerRow: s.w * 4, RowsPerImage: s.h},
			&Extent3D{Width: s.w, Height: s.h, DepthOrArrayLayers: 1},
		)
		if err != nil {
			t.Fatalf("WriteTexture(%dx%d): %v", s.w, s.h, err)
		}
	}
	for i, s := range sizes {
		expectRGBA(t, "texture", readRGBA(t, device, textures[i], s.w, s.h), rgbaPattern(s.w, s.h, byte(i*40)))
	}
}

// TestCopyBufferToTextureAtUnalignedOffset copies into a texture from
// buffer offsets that are not 512-aligned, with rows that fill the whole
// row pitch, and back out to offsets that are not either.
func TestCopyBufferToTextureAtUnalignedOffset(t *testing.T) {
	device, _ := newValidatedDevice(t)
	const w, h, pitch = 64, 4, 256
	for _, offset := range []uint64{4, 256, 260, 508} {
		want := rgbaPattern(w, h, byte(offset))
		src := make([]byte, offset+pitch*h)
		for y := uint64(0); y < h; y++ {
			copy(src[offset+y*pitch:], want[y*w*4:(y+1)*w*4])
		}
		buf, err := device.CreateBuffer(&BufferDescriptor{Label: "unaligned-src", Size: uint64(len(src)), Usage: BufferUsageCopyDst | BufferUsageCopySrc})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		defer buf.Release()
		if err := device.Queue().WriteBuffer(buf, 0, src); err != nil {
			t.Fatalf("WriteBuffer: %v", err)
		}
		tex := newRGBATexture(t, device, "unaligned-dst", w, h)
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		enc.CopyBufferToTexture(buf, tex, []BufferTextureCopy{{
			BufferLayout: ImageDataLayout{Offset: offset, BytesPerRow: pitch, RowsPerImage: h},
			TextureBase:  ImageCopyTexture{Texture: tex},
			Size:         Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
		}})
		finishAndSubmit(t, device, enc)
		expectRGBA(t, "uploaded at an unaligned offset", readRGBA(t, device, tex, w, h), want)

		out, err := device.CreateBuffer(&BufferDescriptor{Label: "unaligned-out", Size: uint64(len(src)), Usage: BufferUsageCopyDst | BufferUsageMapRead})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		defer out.Release()
		enc, err = device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		enc.CopyTextureToBuffer(tex, out, []BufferTextureCopy{{
			TextureBase:  ImageCopyTexture{Texture: tex},
			BufferLayout: ImageDataLayout{Offset: offset, BytesPerRow: pitch, RowsPerImage: h},
			Size:         Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
		}})
		finishAndSubmit(t, device, enc)
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := out.Map(ctx, MapModeRead, 0, uint64(len(src))); err != nil {
			cancel()
			t.Fatalf("Map: %v", err)
		}
		cancel()
		rng, err := out.MappedRange(0, uint64(len(src)))
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		got := make([]byte, 0, w*h*4)
		for y := uint64(0); y < h; y++ {
			got = append(got, rng.Bytes()[offset+y*pitch:offset+y*pitch+w*4]...)
		}
		if err := out.Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
		expectRGBA(t, "read back to an unaligned offset", got, want)
	}
}
