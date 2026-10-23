//go:build !(js && wasm)

package vulkan

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
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

// A copy of a depth or stencil format names the aspect it copies, all of a
// single-aspect format, and its rows are as many texels as bytesPerRow holds
// of that aspect: a stencil texel is one byte, a Depth16Unorm one two.
func TestConvertBufferImageCopyRegionsDepthStencilAspects(t *testing.T) {
	depth, stencil := vk.ImageAspectFlags(vk.ImageAspectDepthBit), vk.ImageAspectFlags(vk.ImageAspectStencilBit)
	for _, tc := range []struct {
		format     gputypes.TextureFormat
		aspect     gputypes.TextureAspect
		wantAspect vk.ImageAspectFlags
		wantRow    uint32
	}{
		{gputypes.TextureFormatDepth32Float, gputypes.TextureAspectUndefined, depth, 64},
		{gputypes.TextureFormatDepth32Float, gputypes.TextureAspectAll, depth, 64},
		{gputypes.TextureFormatDepth16Unorm, gputypes.TextureAspectAll, depth, 128},
		{gputypes.TextureFormatStencil8, gputypes.TextureAspectAll, stencil, 256},
		{gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureAspectStencilOnly, stencil, 256},
		{gputypes.TextureFormatDepth32FloatStencil8, gputypes.TextureAspectStencilOnly, stencil, 256},
		{gputypes.TextureFormatDepth32FloatStencil8, gputypes.TextureAspectDepthOnly, depth, 64},
	} {
		regions := []hal.BufferTextureCopy{{
			BufferLayout: hal.ImageDataLayout{BytesPerRow: 256},
			TextureBase:  hal.ImageCopyTexture{Aspect: tc.aspect},
			Size:         hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		}}
		got := convertBufferImageCopyRegions(nil, regions, tc.format, gputypes.TextureDimension2D)[0]
		if got.ImageSubresource.AspectMask != tc.wantAspect || got.BufferRowLength != tc.wantRow {
			t.Errorf("%s aspect %v: aspect %#x, row %d; want %#x, %d", tc.format, tc.aspect,
				got.ImageSubresource.AspectMask, got.BufferRowLength, tc.wantAspect, tc.wantRow)
		}
	}
}

// A barrier on a combined depth/stencil image covers both aspects whatever
// aspect the barrier's range names: they share their layout.
func TestBarrierAspects(t *testing.T) {
	both := vk.ImageAspectFlags(vk.ImageAspectDepthBit | vk.ImageAspectStencilBit)
	if got := barrierAspects(gputypes.TextureAspectStencilOnly, gputypes.TextureFormatDepth24PlusStencil8); got != both {
		t.Errorf("stencil barrier of Depth24PlusStencil8 = %#x, want %#x", got, both)
	}
	if got := barrierAspects(gputypes.TextureAspectDepthOnly, gputypes.TextureFormatDepth32Float); got != vk.ImageAspectFlags(vk.ImageAspectDepthBit) {
		t.Errorf("depth barrier of Depth32Float = %#x", got)
	}
	if got := barrierAspects(gputypes.TextureAspectAll, gputypes.TextureFormatRGBA8Unorm); got != vk.ImageAspectFlags(vk.ImageAspectColorBit) {
		t.Errorf("colour barrier = %#x", got)
	}
}
