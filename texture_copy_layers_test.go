//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"strconv"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestCopyTextureToTextureArrayLayer: a copy between array textures goes
// from the layer the source names to the one the destination names, and
// touches no other.
func TestCopyTextureToTextureArrayLayer(t *testing.T) {
	_, _, device := newTestDevice(t)
	defer device.Release()

	const side, layers = 8, 3
	const rowBytes = side * 4
	desc := &TextureDescriptor{
		Size:          Extent3D{Width: side, Height: side, DepthOrArrayLayers: layers},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageCopySrc | TextureUsageCopyDst,
	}
	src, err := device.CreateTexture(desc)
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer src.Release()
	dst, err := device.CreateTexture(desc)
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer dst.Release()

	data := make([]byte, rowBytes*side)
	for i := range data {
		data[i] = byte(i*7 + 3)
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: src, Origin: Origin3D{Z: 1}}, data,
		&ImageDataLayout{BytesPerRow: rowBytes, RowsPerImage: side}, &Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyTextureToTexture(src, dst, []TextureCopy{{
		Source:      ImageCopyTexture{Texture: src, Origin: Origin3D{Z: 1}},
		Destination: ImageCopyTexture{Texture: dst, Origin: Origin3D{Z: 2}},
		Size:        Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)

	subs := make([]subresource, layers)
	for i := range subs {
		subs[i] = subresource{layer: uint32(i), width: side, height: side, rowBytes: rowBytes, rows: side}
	}
	got := readSubresources(t, device, dst, gputypes.TextureAspectAll, subs)
	for layer, b := range got {
		if layer == 2 {
			if !bytes.Equal(b, data) {
				t.Errorf("layer 2 holds %x..., want the source's layer 1 %x...", b[:8], data[:8])
			}
			continue
		}
		expectBytes(t, "layer "+strconv.Itoa(layer), b, 0)
	}
}

// TestCopyTextureToTextureBetween2DArrayAnd3D copies the layers of a 2D
// array into the depth slices of a 3D texture and slices of that back into
// layers, each layer paired with a slice.
func TestCopyTextureToTextureBetween2DArrayAnd3D(t *testing.T) {
	_, _, device := newTestDevice(t)
	defer device.Release()

	const side, layers = 8, 4
	const rowBytes = side * 4
	newTexture := func(dim gputypes.TextureDimension) *Texture {
		tex, err := device.CreateTexture(&TextureDescriptor{
			Size:          Extent3D{Width: side, Height: side, DepthOrArrayLayers: layers},
			MipLevelCount: 1,
			SampleCount:   1,
			Dimension:     dim,
			Format:        TextureFormatRGBA8Unorm,
			Usage:         TextureUsageCopySrc | TextureUsageCopyDst,
		})
		if err != nil {
			t.Fatalf("CreateTexture(%v): %v", dim, err)
		}
		return tex
	}
	array, volume, back := newTexture(TextureDimension2D), newTexture(TextureDimension3D), newTexture(TextureDimension2D)
	defer array.Release()
	defer volume.Release()
	defer back.Release()

	data := make([]byte, rowBytes*side*layers)
	for i := range data {
		data[i] = byte(i*7+3) ^ byte(i/(rowBytes*side)*0x55)
	}
	layer := func(i int) []byte { return data[i*rowBytes*side : (i+1)*rowBytes*side] }
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: array}, data,
		&ImageDataLayout{BytesPerRow: rowBytes, RowsPerImage: side}, &Extent3D{Width: side, Height: side, DepthOrArrayLayers: layers}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyTextureToTexture(array, volume, []TextureCopy{{
		Source:      ImageCopyTexture{Texture: array},
		Destination: ImageCopyTexture{Texture: volume},
		Size:        Extent3D{Width: side, Height: side, DepthOrArrayLayers: layers},
	}})
	enc.CopyTextureToTexture(volume, back, []TextureCopy{{
		Source:      ImageCopyTexture{Texture: volume, Origin: Origin3D{Z: 1}},
		Destination: ImageCopyTexture{Texture: back, Origin: Origin3D{Z: 2}},
		Size:        Extent3D{Width: side, Height: side, DepthOrArrayLayers: 2},
	}})
	finishAndSubmit(t, device, enc)

	subs := make([]subresource, layers)
	for i := range subs {
		subs[i] = subresource{layer: uint32(i), width: side, height: side, rowBytes: rowBytes, rows: side}
	}
	for i, b := range readSubresources(t, device, volume, gputypes.TextureAspectAll, subs) {
		if !bytes.Equal(b, layer(i)) {
			t.Errorf("3D slice %d holds %x..., want layer %d %x...", i, b[:8], i, layer(i)[:8])
		}
	}
	for i, b := range readSubresources(t, device, back, gputypes.TextureAspectAll, subs) {
		switch i {
		case 2, 3:
			if !bytes.Equal(b, layer(i-1)) {
				t.Errorf("layer %d holds %x..., want slice %d %x...", i, b[:8], i-1, layer(i - 1)[:8])
			}
		default:
			expectBytes(t, "layer "+strconv.Itoa(i), b, 0)
		}
	}
}

// TestWriteTextureOfLayersNeedsRowsPerImage: a write of more than one layer
// names the rows of an image, as WebGPU and the encoder's copies require.
func TestWriteTextureOfLayersNeedsRowsPerImage(t *testing.T) {
	_, _, device := newTestDevice(t)
	defer device.Release()

	tex, err := device.CreateTexture(&TextureDescriptor{
		Size:          Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 2},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()

	layout := &ImageDataLayout{BytesPerRow: 16}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, make([]byte, 128), layout,
		&Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 2}); err == nil {
		t.Fatal("WriteTexture of two layers without RowsPerImage succeeded")
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, make([]byte, 64), layout,
		&Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture of one layer without RowsPerImage: %v", err)
	}
}
