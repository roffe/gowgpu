//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestNorm16Textures uploads and reads back R16Unorm, RG16Snorm and
// RGBA16Unorm textures. Vulkan had no format for the six 16-bit
// normalized formats and made their images VK_FORMAT_UNDEFINED, which the
// validation layer crashed on and NVIDIA gave no memory.
func TestNorm16Textures(t *testing.T) {
	device, h := newValidatedDevice(t)
	for _, tc := range []struct {
		format gputypes.TextureFormat
		texel  int
	}{
		{gputypes.TextureFormatR16Unorm, 2},
		{gputypes.TextureFormatRG16Snorm, 4},
		{gputypes.TextureFormatRGBA16Unorm, 8},
	} {
		tex, err := device.CreateTexture(&TextureDescriptor{
			Label:         tc.format.String(),
			Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
			MipLevelCount: 1,
			SampleCount:   1,
			Dimension:     TextureDimension2D,
			Format:        tc.format,
			Usage:         TextureUsageCopySrc | TextureUsageCopyDst | TextureUsageTextureBinding,
		})
		if err != nil {
			if strings.Contains(err.Error(), "feature") {
				t.Skipf("CreateTexture(%s): %v", tc.format, err)
			}
			t.Fatalf("CreateTexture(%s): %v", tc.format, err)
		}
		defer tex.Release()
		data := make([]byte, pixSide*pixSide*tc.texel)
		for i := 0; i < len(data); i += 2 {
			binary.LittleEndian.PutUint16(data[i:], uint16(i*97+1))
		}
		if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, data,
			&ImageDataLayout{BytesPerRow: uint32(pixSide * tc.texel), RowsPerImage: pixSide}, &Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}); err != nil {
			t.Fatalf("WriteTexture(%s): %v", tc.format, err)
		}
		got := readTextureAspect(t, device, tex, gputypes.TextureAspectAll)
		for y := range pixSide {
			row := pixSide * tc.texel
			if have, want := got[y*pixRow:y*pixRow+row], data[y*row:(y+1)*row]; string(have) != string(want) {
				t.Errorf("%s row %d = % x, want % x", tc.format, y, have, want)
			}
		}
	}
	assertNoValidation(t, h)
}
