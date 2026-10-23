//go:build !(js && wasm)

package wgpu

import (
	"context"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// TestCompressedCopies uploads both layers of mip 0 of an 8x8 BC1 texture
// in one write, rowsPerImage 2 rows of blocks, and the 2x2 mip 2 as the one
// block a WebGPU copy of it is, and reads them back. Vulkan took
// rowsPerImage as texel rows, so the second layer came from the first's
// bytes, and copied the block's 4x4 into the 2x2 mip.
func TestCompressedCopies(t *testing.T) {
	device, h := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureTextureCompressionBC))
	if h.backend == gputypes.BackendGL {
		t.Skip("GL reads no compressed texture back")
	}
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         "bc1",
		Size:          Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 2},
		MipLevelCount: 4,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        gputypes.TextureFormatBC1RGBAUnorm,
		Usage:         TextureUsageCopySrc | TextureUsageCopyDst | TextureUsageTextureBinding,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	const blockBytes = 8
	layers := make([]byte, 2*2*2*blockBytes) // two layers of 2x2 blocks
	for i := range layers {
		layers[i] = byte(i + 1)
	}
	q := device.Queue()
	if err := q.WriteTexture(&ImageCopyTexture{Texture: tex}, layers,
		&ImageDataLayout{BytesPerRow: 2 * blockBytes, RowsPerImage: 2}, &Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 2}); err != nil {
		t.Fatalf("WriteTexture(mip 0): %v", err)
	}
	block := []byte{0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8}
	if err := q.WriteTexture(&ImageCopyTexture{Texture: tex, MipLevel: 2}, block,
		&ImageDataLayout{BytesPerRow: blockBytes}, &Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture(mip 2): %v", err)
	}

	const pitch = 256
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "bc1-read", Size: 5 * pitch, Usage: BufferUsageCopyDst | BufferUsageMapRead})
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
		BufferLayout: ImageDataLayout{BytesPerRow: pitch, RowsPerImage: 2},
		Size:         Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 2},
	}, {
		TextureBase:  ImageCopyTexture{Texture: tex, MipLevel: 2},
		BufferLayout: ImageDataLayout{Offset: 4 * pitch, BytesPerRow: pitch},
		Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, MapModeRead, 0, 5*pitch); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, 5*pitch)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	got := rng.Bytes()
	for row := range 4 { // two block rows of each layer
		want := layers[row*2*blockBytes : (row+1)*2*blockBytes]
		if have := got[row*pitch : row*pitch+2*blockBytes]; string(have) != string(want) {
			t.Errorf("layer %d block row %d = % x, want % x", row/2, row%2, have, want)
		}
	}
	if have := got[4*pitch : 4*pitch+blockBytes]; string(have) != string(block) {
		t.Errorf("mip 2 = % x, want % x", have, block)
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	assertNoValidation(t, h)
}
