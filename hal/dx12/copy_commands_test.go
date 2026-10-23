//go:build windows && !(js && wasm)

package dx12

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// TestDepthStencilCopiesAtUnalignedOffsets copies a depth or stencil plane
// in from a buffer offset that is not 512-aligned and back out to another,
// with rows that fill the row pitch. Both copies go through a scratch
// buffer, since D3D12 takes no box inside the footprint of a depth/stencil
// copy on every device.
func TestDepthStencilCopiesAtUnalignedOffsets(t *testing.T) {
	dev, queue := openTestDevice(t)
	const w, h, pitch = 128, 4, 512
	for _, tc := range []struct {
		name   string
		format gputypes.TextureFormat
		aspect gputypes.TextureAspect
		bytes  uint32
	}{
		{"depth16", gputypes.TextureFormatDepth16Unorm, gputypes.TextureAspectAll, 2},
		{"depth32", gputypes.TextureFormatDepth32Float, gputypes.TextureAspectAll, 4},
		{"stencil8", gputypes.TextureFormatStencil8, gputypes.TextureAspectAll, 1},
		{"stencil of depth24plus-stencil8", gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureAspectStencilOnly, 1},
		{"depth of depth32float-stencil8", gputypes.TextureFormatDepth32FloatStencil8, gputypes.TextureAspectDepthOnly, 4},
		{"stencil of depth32float-stencil8", gputypes.TextureFormatDepth32FloatStencil8, gputypes.TextureAspectStencilOnly, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.bytes == 1 && underWine() {
				t.Skip("vkd3d ignores barriers of the stencil plane alone")
			}
			tex := newTestTexture(t, dev, hal.TextureDescriptor{
				Size:      hal.Extent3D{Width: w, Height: h},
				Dimension: gputypes.TextureDimension2D,
				Format:    tc.format,
				Usage:     gputypes.TextureUsageCopyDst | gputypes.TextureUsageCopySrc | gputypes.TextureUsageRenderAttachment,
			})
			rowBytes := w * tc.bytes
			want := make([]byte, rowBytes*h)
			for i := range want {
				want[i] = byte(i*7 + 3)
			}
			if tc.bytes == 4 {
				for i := 0; i < len(want); i += 4 {
					binary.LittleEndian.PutUint32(want[i:], math.Float32bits(float32(i)/float32(len(want))))
				}
			}
			const in, out = 4, 260
			src := bytes.Repeat([]byte{0xEE}, in+pitch*h)
			for y := uint32(0); y < h; y++ {
				copy(src[in+y*pitch:], want[y*rowBytes:(y+1)*rowBytes])
			}
			srcBuf := newTestBuffer(t, dev, uint64(len(src)), gputypes.BufferUsageCopyDst|gputypes.BufferUsageCopySrc)
			if err := queue.WriteBuffer(srcBuf, 0, src); err != nil {
				t.Fatalf("WriteBuffer: %v", err)
			}
			dstSize := uint64(out + pitch*h)
			dstBuf := newTestBuffer(t, dev, dstSize, gputypes.BufferUsageCopyDst|gputypes.BufferUsageMapRead)
			record(t, dev, queue, func(enc *CommandEncoder) {
				enc.CopyBufferToTexture(srcBuf, tex, []hal.BufferTextureCopy{{
					BufferLayout: hal.ImageDataLayout{Offset: in, BytesPerRow: pitch, RowsPerImage: h},
					TextureBase:  hal.ImageCopyTexture{Texture: tex, Aspect: tc.aspect},
					Size:         hal.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
				}})
				enc.CopyTextureToBuffer(tex, dstBuf, []hal.BufferTextureCopy{{
					BufferLayout: hal.ImageDataLayout{Offset: out, BytesPerRow: pitch, RowsPerImage: h},
					TextureBase:  hal.ImageCopyTexture{Texture: tex, Aspect: tc.aspect},
					Size:         hal.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
				}})
				if len(enc.scratch) != 2 {
					t.Errorf("%d copies went through a scratch buffer, want both", len(enc.scratch))
				}
			})
			got := readBuffer(t, dev, dstBuf, dstSize)
			for y := uint32(0); y < h; y++ {
				row := got[out+y*pitch : out+y*pitch+rowBytes]
				if !bytes.Equal(row, want[y*rowBytes:(y+1)*rowBytes]) {
					t.Fatalf("row %d = % x..., want % x...", y, row[:8], want[y*rowBytes:y*rowBytes+8])
				}
			}
			if !bytes.Equal(got[:out], make([]byte, out)) {
				t.Fatalf("bytes ahead of the copy were written: % x", got[:16])
			}
		})
	}
}
