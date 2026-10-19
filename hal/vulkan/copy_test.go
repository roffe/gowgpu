//go:build !(js && wasm)

package vulkan

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func TestConvertBufferImageCopyRegionsUsesArrayLayersFor2DTextures(t *testing.T) {
	regions := []hal.BufferTextureCopy{{
		BufferLayout: hal.ImageDataLayout{BytesPerRow: 256, RowsPerImage: 4},
		TextureBase: hal.ImageCopyTexture{
			Origin: hal.Origin3D{X: 2, Y: 3, Z: 1},
			Aspect: gputypes.TextureAspectAll,
		},
		Size: hal.Extent3D{Width: 8, Height: 4, DepthOrArrayLayers: 2},
	}}

	converted := convertBufferImageCopyRegions(nil, regions, gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D)
	got := converted[0]
	if got.ImageSubresource.BaseArrayLayer != 1 || got.ImageSubresource.LayerCount != 2 {
		t.Fatalf("array range = %d+%d, want 1+2", got.ImageSubresource.BaseArrayLayer, got.ImageSubresource.LayerCount)
	}
	if got.ImageOffset.Z != 0 || got.ImageExtent.Depth != 1 {
		t.Fatalf("depth offset+extent = %d+%d, want 0+1", got.ImageOffset.Z, got.ImageExtent.Depth)
	}
}

func TestConvertBufferImageCopyRegionsUsesDepthFor3DTextures(t *testing.T) {
	regions := []hal.BufferTextureCopy{{
		TextureBase: hal.ImageCopyTexture{
			Origin: hal.Origin3D{X: 2, Y: 3, Z: 1},
			Aspect: gputypes.TextureAspectAll,
		},
		Size: hal.Extent3D{Width: 8, Height: 4, DepthOrArrayLayers: 2},
	}}

	converted := convertBufferImageCopyRegions(nil, regions, gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension3D)
	got := converted[0]
	if got.ImageSubresource.BaseArrayLayer != 0 || got.ImageSubresource.LayerCount != 1 {
		t.Fatalf("array range = %d+%d, want 0+1", got.ImageSubresource.BaseArrayLayer, got.ImageSubresource.LayerCount)
	}
	if got.ImageOffset.Z != 1 || got.ImageExtent.Depth != 2 {
		t.Fatalf("depth offset+extent = %d+%d, want 1+2", got.ImageOffset.Z, got.ImageExtent.Depth)
	}
}

// bufferRowLength counts texels: a row of 256 bytes of BC1 is 32 blocks of
// 8 bytes, 128 texels, not 32.
func TestConvertBufferImageCopyRegionsCompressedRowLength(t *testing.T) {
	regions := []hal.BufferTextureCopy{{
		BufferLayout: hal.ImageDataLayout{BytesPerRow: 256},
		TextureBase:  hal.ImageCopyTexture{Aspect: gputypes.TextureAspectAll},
		Size:         hal.Extent3D{Width: 64, Height: 8, DepthOrArrayLayers: 1},
	}}
	for _, tc := range []struct {
		format gputypes.TextureFormat
		want   uint32
	}{
		{gputypes.TextureFormatRGBA8Unorm, 64},
		{gputypes.TextureFormatBC1RGBAUnorm, 128},
		{gputypes.TextureFormatBC7RGBAUnorm, 64},
		{gputypes.TextureFormatASTC10x8Unorm, 160},
	} {
		got := convertBufferImageCopyRegions(nil, regions, tc.format, gputypes.TextureDimension2D)[0].BufferRowLength
		if got != tc.want {
			t.Errorf("%s: bufferRowLength = %d, want %d", tc.format, got, tc.want)
		}
	}
}
