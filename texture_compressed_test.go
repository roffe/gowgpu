//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// bc1Block is a BC1 block of one RGB565 colour: color0 above color1 picks
// the four-colour mode, and index 0 everywhere picks color0.
func bc1Block(rgb565 uint16) []byte {
	return []byte{byte(rgb565), byte(rgb565 >> 8), 0, 0, 0, 0, 0, 0}
}

// bc1Quadrants are the four 4x4 blocks of an 8x8 BC1 texture, red and green
// on top, blue and white below, each row of blocks at pitch bytes.
func bc1Quadrants(pitch int) []byte {
	data := make([]byte, pitch+16)
	copy(data, append(bc1Block(0xf800), bc1Block(0x07e0)...))
	copy(data[pitch:], append(bc1Block(0x001f), bc1Block(0xffff)...))
	return data
}

// expectQuadrants checks pixels of an 8x8 image for bc1Quadrants' colours.
func expectQuadrants(t *testing.T, what string, pix []byte) {
	t.Helper()
	want := [4][4]byte{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}, {255, 255, 255, 255}}
	for y := range 8 {
		for x := range 8 {
			w := want[y/4*2+x/4]
			if got := pix[(y*8+x)*4 : (y*8+x)*4+4]; [4]byte(got) != w {
				t.Errorf("%s: pixel (%d,%d) = %v, want %v", what, x, y, got, w)
				return
			}
		}
	}
}

// TestCompressedTextureUploads samples a BC1 texture written by WriteTexture,
// with packed rows and with rows at a wider pitch, and by
// CopyBufferToTexture. GL made compressed textures as RGBA8 and uploaded the
// blocks with glTexSubImage2D, reading eight times their size.
func TestCompressedTextureUploads(t *testing.T) {
	device, _ := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureTextureCompressionBC))
	for _, tc := range []struct {
		name  string
		pitch int
		copy  bool
	}{
		{"write_packed", 16, false},
		{"write_padded", 256, false},
		{"copy", 256, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			tex, err := device.CreateTexture(&TextureDescriptor{
				Size: Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1}, Format: gputypes.TextureFormatBC1RGBAUnorm,
				Usage: TextureUsageTextureBinding | TextureUsageCopyDst, MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D,
			})
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			defer tex.Release()
			data := bc1Quadrants(tc.pitch)
			layout := ImageDataLayout{BytesPerRow: uint32(tc.pitch), RowsPerImage: 8}
			size := Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1}
			if tc.copy {
				buf, err := device.CreateBuffer(&BufferDescriptor{Size: uint64(alignUp(uint32(len(data)), 4)), Usage: BufferUsageCopySrc | BufferUsageCopyDst})
				if err != nil {
					t.Fatalf("CreateBuffer: %v", err)
				}
				defer buf.Release()
				if err := device.Queue().WriteBuffer(buf, 0, append(data, make([]byte, int(buf.Size())-len(data))...)); err != nil {
					t.Fatalf("WriteBuffer: %v", err)
				}
				enc, err := device.CreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("CreateCommandEncoder: %v", err)
				}
				enc.CopyBufferToTexture(buf, tex, []BufferTextureCopy{{BufferLayout: layout, TextureBase: ImageCopyTexture{Texture: tex}, Size: size}})
				finishAndSubmit(t, device, enc)
			} else if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, data, &layout, &size); err != nil {
				t.Fatalf("WriteTexture: %v", err)
			}
			view, err := device.CreateTextureView(tex, nil)
			if err != nil {
				t.Fatalf("CreateTextureView: %v", err)
			}
			defer view.Release()
			expectQuadrants(t, tc.name, loadPixels(t, device, view, 8, 8))
		})
	}
}
