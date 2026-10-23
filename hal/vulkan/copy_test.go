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

	converted := convertBufferImageCopyRegions(nil, regions, copyImage{format: gputypes.TextureFormatRGBA8Unorm, dimension: gputypes.TextureDimension2D, size: Extent3D{Width: 64, Height: 64, Depth: 1}})
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

	converted := convertBufferImageCopyRegions(nil, regions, copyImage{format: gputypes.TextureFormatRGBA8Unorm, dimension: gputypes.TextureDimension3D, size: Extent3D{Width: 64, Height: 64, Depth: 4}})
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
		got := convertBufferImageCopyRegions(nil, regions, copyImage{format: tc.format, dimension: gputypes.TextureDimension2D, size: Extent3D{Width: 256, Height: 256, Depth: 1}})[0].BufferRowLength
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
		got := convertBufferImageCopyRegions(nil, regions, copyImage{format: tc.format, dimension: gputypes.TextureDimension2D, size: Extent3D{Width: 256, Height: 256, Depth: 1}})[0]
		if got.ImageSubresource.AspectMask != tc.wantAspect || got.BufferRowLength != tc.wantRow {
			t.Errorf("%s aspect %v: aspect %#x, row %d; want %#x, %d", tc.format, tc.aspect,
				got.ImageSubresource.AspectMask, got.BufferRowLength, tc.wantAspect, tc.wantRow)
		}
	}
}

// A barrier on a combined depth/stencil image covers both aspects whatever
// aspect the barrier's range names: they share their layout. So does one on
// a Stencil8 texture the device made a combined format.
func TestBarrierAspects(t *testing.T) {
	both := vk.ImageAspectFlags(vk.ImageAspectDepthBit | vk.ImageAspectStencilBit)
	d := &Device{}
	if got := barrierAspects(gputypes.TextureAspectStencilOnly, &Texture{format: gputypes.TextureFormatDepth24PlusStencil8, device: d}); got != both {
		t.Errorf("stencil barrier of Depth24PlusStencil8 = %#x, want %#x", got, both)
	}
	if got := barrierAspects(gputypes.TextureAspectDepthOnly, &Texture{format: gputypes.TextureFormatDepth32Float, device: d}); got != vk.ImageAspectFlags(vk.ImageAspectDepthBit) {
		t.Errorf("depth barrier of Depth32Float = %#x", got)
	}
	if got := barrierAspects(gputypes.TextureAspectAll, &Texture{format: gputypes.TextureFormatRGBA8Unorm, device: d}); got != vk.ImageAspectFlags(vk.ImageAspectColorBit) {
		t.Errorf("colour barrier = %#x", got)
	}
	if got := barrierAspects(gputypes.TextureAspectAll, &Texture{format: gputypes.TextureFormatStencil8, device: &Device{stencil8: vk.FormatD24UnormS8Uint}}); got != both {
		t.Errorf("barrier of a Stencil8 made D24_UNORM_S8_UINT = %#x, want %#x", got, both)
	}
}

// A compressed copy is whole blocks: one of 4x4 into the 2x2 mip 2 of an
// 8x8 BC1 texture covers the mip, and its rowsPerImage counts rows of
// blocks, which bufferImageHeight takes in texels.
func TestConvertBufferImageCopyRegionsCompressedMip(t *testing.T) {
	img := copyImage{format: gputypes.TextureFormatBC1RGBAUnorm, dimension: gputypes.TextureDimension2D, size: Extent3D{Width: 8, Height: 8, Depth: 1}}
	regions := []hal.BufferTextureCopy{{
		BufferLayout: hal.ImageDataLayout{BytesPerRow: 256, RowsPerImage: 2},
		TextureBase:  hal.ImageCopyTexture{MipLevel: 2},
		Size:         hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 2},
	}}
	got := convertBufferImageCopyRegions(nil, regions, img)[0]
	if got.ImageExtent.Width != 2 || got.ImageExtent.Height != 2 {
		t.Errorf("extent %dx%d, want the 2x2 of the mip", got.ImageExtent.Width, got.ImageExtent.Height)
	}
	if got.BufferImageHeight != 8 {
		t.Errorf("bufferImageHeight %d, want 8 texels for 2 rows of blocks", got.BufferImageHeight)
	}
}

// A texture copy is clamped to the mip of each side from its origin.
func TestClampExtent(t *testing.T) {
	img := copyImage{dimension: gputypes.TextureDimension3D, size: Extent3D{Width: 16, Height: 8, Depth: 8}}
	got := img.clampExtent(hal.ImageCopyTexture{MipLevel: 1, Origin: hal.Origin3D{X: 4, Y: 2, Z: 1}}, hal.Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 8})
	if want := (hal.Extent3D{Width: 4, Height: 2, DepthOrArrayLayers: 3}); got != want {
		t.Errorf("clampExtent = %+v, want %+v", got, want)
	}
}
