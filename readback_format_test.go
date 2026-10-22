//go:build !(js && wasm)

package wgpu

import (
	"context"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestCopyTextureToBufferKeepsTexelSize reads a 25x16 RG8 texture back, rows
// of 50 bytes, not 4-byte aligned: GL read every texture as RGBA8, 4 bytes a
// texel, so an R8 or RG8 readback came back spread out and cut short.
func TestCopyTextureToBufferKeepsTexelSize(t *testing.T) {
	_, _, device := newTestDevice(t)
	defer device.Release()
	q := device.Queue()
	const W, H, bpp = 25, 16, 2
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: W, Height: H, DepthOrArrayLayers: 1}, Format: gputypes.TextureFormatRG8Unorm,
		Usage: gputypes.TextureUsageCopyDst | gputypes.TextureUsageCopySrc, MipLevelCount: 1, SampleCount: 1, Dimension: gputypes.TextureDimension2D,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer tex.Release()
	data := make([]byte, W*H*bpp)
	for i := range data {
		data[i] = byte(i*7 + 3)
	}
	if err := q.WriteTexture(&ImageCopyTexture{Texture: tex}, data, &ImageDataLayout{BytesPerRow: W * bpp, RowsPerImage: H}, &Extent3D{Width: W, Height: H, DepthOrArrayLayers: 1}); err != nil {
		t.Fatal(err)
	}
	const pitch = 256
	staging, err := device.CreateBuffer(&BufferDescriptor{Size: pitch * H, Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst})
	if err != nil {
		t.Fatal(err)
	}
	defer staging.Release()
	enc, _ := device.CreateCommandEncoder(nil)
	enc.CopyTextureToBuffer(tex, staging, []BufferTextureCopy{{
		BufferLayout: ImageDataLayout{BytesPerRow: pitch, RowsPerImage: H},
		TextureBase:  ImageCopyTexture{Texture: tex},
		Size:         Extent3D{Width: W, Height: H, DepthOrArrayLayers: 1},
	}})
	cb, err := enc.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Submit(cb); err != nil {
		t.Fatal(err)
	}
	if err := staging.Map(context.Background(), MapModeRead, 0, pitch*H); err != nil {
		t.Fatal(err)
	}
	defer staging.Unmap()
	rng, _ := staging.MappedRange(0, pitch*H)
	px := rng.Bytes()
	bad := 0
	for y := range H {
		for x := range W * bpp {
			if px[y*pitch+x] != data[y*W*bpp+x] {
				bad++
			}
		}
	}
	if bad != 0 {
		t.Errorf("%d of %d bytes differ; row 1 got % x want % x", bad, W*H*bpp, px[pitch:pitch+8], data[W*bpp:W*bpp+8])
	}
}
