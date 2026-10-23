//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// mipTexture makes an 8x8 RGBA8 texture with two mip levels, every byte of
// mip 0 0x10 and of mip 1 0x30.
func mipTexture(t *testing.T, device *Device) *Texture {
	t.Helper()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1}, Format: TextureFormatRGBA8Unorm,
		Usage:         TextureUsageTextureBinding | TextureUsageRenderAttachment | TextureUsageCopyDst | TextureUsageCopySrc,
		MipLevelCount: 2, SampleCount: 1, Dimension: TextureDimension2D,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	for mip, fill := range []byte{0x10, 0x30} {
		side := uint32(8 >> mip)
		data := bytes.Repeat([]byte{fill}, int(side*side*4))
		if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex, MipLevel: uint32(mip)}, data,
			&ImageDataLayout{BytesPerRow: side * 4, RowsPerImage: side}, &Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}); err != nil {
			t.Fatalf("WriteTexture(mip %d): %v", mip, err)
		}
	}
	return tex
}

// mipView makes a view of one mip level of tex.
func mipView(t *testing.T, tex *Texture, mip uint32) *TextureView {
	t.Helper()
	view, err := tex.device.CreateTextureView(tex, &TextureViewDescriptor{BaseMipLevel: mip, MipLevelCount: 1})
	if err != nil {
		t.Fatalf("CreateTextureView(mip %d): %v", mip, err)
	}
	return view
}

// TestTextureViewSamplesItsMip loads texels through a view of each mip
// level: GL bound the whole texture, so a view of mip 1 read mip 0.
func TestTextureViewSamplesItsMip(t *testing.T) {
	device, _ := newValidatedDevice(t)
	tex := mipTexture(t, device)
	defer tex.Release()
	for mip, want := range []byte{0x10, 0x30} {
		view := mipView(t, tex, uint32(mip))
		side := uint32(8 >> mip)
		expectPixels(t, "view of mip "+string('0'+rune(mip)), loadPixels(t, device, view, side, side), want, want, want, want)
		view.Release()
	}
}

// TestRenderPassDrawsIntoItsMip clears a view of mip 1 and finds mip 0 as
// it was: GL attached level 0 whatever level the view had.
func TestRenderPassDrawsIntoItsMip(t *testing.T) {
	device, _ := newValidatedDevice(t)
	tex := mipTexture(t, device)
	defer tex.Release()
	mip0, mip1 := mipView(t, tex, 0), mipView(t, tex, 1)
	defer mip0.Release()
	defer mip1.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, mip1, gputypes.LoadOpClear, red, nil)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "the cleared mip 1", loadPixels(t, device, mip1, 4, 4), 255, 0, 0, 255)
	expectPixels(t, "mip 0", loadPixels(t, device, mip0, 8, 8), 0x10, 0x10, 0x10, 0x10)
}

// TestArrayTextureLayers writes both layers of a 2D array texture in one
// WriteTexture, samples each, and clears a view of layer 1: GL made array
// textures without storage, and attached level 0 of the texture whatever
// layer the view had.
func TestArrayTextureLayers(t *testing.T) {
	device, _ := newValidatedDevice(t)
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 2}, Format: TextureFormatRGBA8Unorm,
		Usage:         TextureUsageTextureBinding | TextureUsageRenderAttachment | TextureUsageCopyDst,
		MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	// The layers go in at a row and image pitch wider than theirs.
	const pitch, rows = 32, 6
	data := make([]byte, pitch*rows*2)
	for layer, fill := range []byte{0x10, 0x30} {
		for y := range 4 {
			copy(data[(layer*rows+y)*pitch:], bytes.Repeat([]byte{fill}, 16))
		}
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, data,
		&ImageDataLayout{BytesPerRow: pitch, RowsPerImage: rows}, &Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 2}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	all, err := device.CreateTextureView(tex, &TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2DArray})
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer all.Release()
	expectPixels(t, "layer 0", loadLayerPixels(t, device, all, 0, 4, 4), 0x10, 0x10, 0x10, 0x10)
	expectPixels(t, "layer 1", loadLayerPixels(t, device, all, 1, 4, 4), 0x30, 0x30, 0x30, 0x30)

	layer1, err := device.CreateTextureView(tex, &TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2D, BaseArrayLayer: 1, ArrayLayerCount: 1})
	if err != nil {
		t.Fatalf("CreateTextureView(layer 1): %v", err)
	}
	defer layer1.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, layer1, gputypes.LoadOpClear, red, nil)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "the cleared layer 1", loadLayerPixels(t, device, all, 1, 4, 4), 255, 0, 0, 255)
	expectPixels(t, "layer 0 after the clear", loadLayerPixels(t, device, all, 0, 4, 4), 0x10, 0x10, 0x10, 0x10)
}

// TestCopiesUseTheirMipLevel reads mip 1 back, copies a buffer into mip 1
// and reads both mips: GL's buffer copies wrote and read mip 0 whatever mip
// the copy named.
func TestCopiesUseTheirMipLevel(t *testing.T) {
	device, _ := newValidatedDevice(t)
	tex := mipTexture(t, device)
	defer tex.Release()
	mip1 := subresource{mip: 1, width: 4, height: 4, rowBytes: 16, rows: 4}
	expectBytes(t, "mip 1 read back", readSubresources(t, device, tex, gputypes.TextureAspectAll, []subresource{mip1})[0], 0x30)

	buf, err := device.CreateBuffer(&BufferDescriptor{Size: 256 * 4, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	if err := device.Queue().WriteBuffer(buf, 0, bytes.Repeat([]byte{0x55}, 256*4)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToTexture(buf, tex, []BufferTextureCopy{{
		BufferLayout: ImageDataLayout{BytesPerRow: 256, RowsPerImage: 4},
		TextureBase:  ImageCopyTexture{Texture: tex, MipLevel: 1},
		Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)
	got := readSubresources(t, device, tex, gputypes.TextureAspectAll, []subresource{{width: 8, height: 8, rowBytes: 32, rows: 8}, mip1})
	expectBytes(t, "mip 0 after the copy into mip 1", got[0], 0x10)
	expectBytes(t, "mip 1 after the copy into it", got[1], 0x55)
}
