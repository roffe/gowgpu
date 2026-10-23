//go:build js && wasm

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/internal/browser"
)

func TestBrowserWriteTextureOfLayersNeedsRowsPerImage(t *testing.T) {
	d, _ := newFakeDevice(jsEval("new Promise(() => {})"))
	tex := &Texture{browser: &browser.Texture{}}
	err := d.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, make([]byte, 128), &ImageDataLayout{BytesPerRow: 16},
		&Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 2})
	if err == nil {
		t.Fatal("WriteTexture of two layers without RowsPerImage succeeded")
	}
}
