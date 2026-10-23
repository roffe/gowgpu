//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestFormatRoundTrip writes a texture of each format and reads it back.
// The texels have to come back as they went in, the integer and packed
// formats' too: GL made every format outside a short list as RGBA8 and moved
// it four bytes a texel, reading past the data on the way in and writing
// past its buffer on the way out.
func TestFormatRoundTrip(t *testing.T) {
	_, _, device := newTestDevice(t)
	defer device.Release()
	formats := []gputypes.TextureFormat{
		gputypes.TextureFormatR8Uint, gputypes.TextureFormatR8Sint,
		gputypes.TextureFormatRG8Uint, gputypes.TextureFormatRG8Sint,
		gputypes.TextureFormatR16Uint, gputypes.TextureFormatR16Sint,
		gputypes.TextureFormatRG16Uint, gputypes.TextureFormatRG16Sint,
		gputypes.TextureFormatR32Uint, gputypes.TextureFormatR32Sint, gputypes.TextureFormatR32Float,
		gputypes.TextureFormatRG32Uint, gputypes.TextureFormatRG32Sint, gputypes.TextureFormatRG32Float,
		gputypes.TextureFormatRGBA8Uint, gputypes.TextureFormatRGBA8Sint,
		gputypes.TextureFormatRGBA16Uint, gputypes.TextureFormatRGBA16Sint,
		gputypes.TextureFormatRGBA32Uint, gputypes.TextureFormatRGBA32Sint, gputypes.TextureFormatRGBA32Float,
		gputypes.TextureFormatRGB10A2Unorm, gputypes.TextureFormatRGB10A2Uint,
	}
	const side = 16
	for _, format := range formats {
		t.Run(format.String(), func(t *testing.T) {
			bpp := format.BlockCopySize()
			tex, err := device.CreateTexture(&TextureDescriptor{
				Size:   Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
				Format: format, Usage: gputypes.TextureUsageCopyDst | gputypes.TextureUsageCopySrc | gputypes.TextureUsageRenderAttachment,
				MipLevelCount: 1, SampleCount: 1, Dimension: gputypes.TextureDimension2D,
			})
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			defer tex.Release()
			// The rows go in at a pitch wider than theirs, as an upload of
			// part of a bigger image does.
			const pitch = 256
			data := formatPattern(format, side*side)
			padded := make([]byte, pitch*side)
			for y := range side {
				copy(padded[y*pitch:], data[y*side*int(bpp):(y+1)*side*int(bpp)])
			}
			if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, padded,
				&ImageDataLayout{BytesPerRow: pitch, RowsPerImage: side}, &Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}); err != nil {
				t.Fatalf("WriteTexture: %v", err)
			}
			got := readSubresources(t, device, tex, gputypes.TextureAspectAll, []subresource{{width: side, height: side, rowBytes: side * bpp, rows: side}})[0]
			if !bytes.Equal(got, data) {
				at := 0
				for at < len(got) && got[at] == data[at] {
					at++
				}
				t.Errorf("byte %d differs: got % x, want % x", at, got[at:min(at+8, len(got))], data[at:min(at+8, len(data))])
			}
		})
	}
}

// formatPattern is n texels of format that survive a copy bit for bit:
// ordinary floats for the float formats, any bytes for the others.
func formatPattern(format gputypes.TextureFormat, n int) []byte {
	data := make([]byte, n*int(format.BlockCopySize()))
	switch format {
	case gputypes.TextureFormatR32Float, gputypes.TextureFormatRG32Float, gputypes.TextureFormatRGBA32Float:
		for i := 0; i < len(data); i += 4 {
			binary.LittleEndian.PutUint32(data[i:], math.Float32bits(float32(i)/8-3))
		}
	default:
		for i := range data {
			data[i] = byte(i*37 + 11)
		}
	}
	return data
}

// TestWriteTextureOddPitchAndEmpty writes rows of RGBA8 18 bytes apart,
// which WebGPU allows, and reads every row back: GL stepped the rows by
// whole texels, 16 bytes, so the second row started two bytes early. A
// write of nothing then writes nothing; GL indexed the empty data and
// panicked.
func TestWriteTextureOddPitchAndEmpty(t *testing.T) {
	device, h := newValidatedDevice(t)
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: 4, Height: 2, DepthOrArrayLayers: 1}, Format: gputypes.TextureFormatRGBA8Unorm,
		Usage: TextureUsageCopyDst | TextureUsageCopySrc, MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	data := bytes.Repeat([]byte{0x11}, 18+16)
	copy(data, bytes.Repeat([]byte{0x22}, 16))
	copy(data[18:], bytes.Repeat([]byte{0x22}, 16))
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, data, &ImageDataLayout{BytesPerRow: 18, RowsPerImage: 2},
		&Extent3D{Width: 4, Height: 2, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, nil, &ImageDataLayout{}, &Extent3D{DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture of nothing: %v", err)
	}
	got := readSubresources(t, device, tex, gputypes.TextureAspectAll, []subresource{{width: 4, height: 2, rowBytes: 16, rows: 2}})[0]
	expectBytes(t, "the rows written", got, 0x22)
	assertNoValidation(t, h)
}
