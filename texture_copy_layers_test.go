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
	_, adapter, device := newTestDevice(t)
	defer device.Release()
	if adapter.Info().Backend == gputypes.BackendGL {
		t.Skip("array textures are not done on GL")
	}

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
