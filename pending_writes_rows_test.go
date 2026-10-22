//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
	"testing"
	"time"
)

// WebGPU asks a texture write for bytesPerRow times the rows but the last,
// and bytesInLastRow for that one. The padding copy used to drop a last row
// that stopped there.
func TestAlignTextureDataIntoShortLastRow(t *testing.T) {
	const bytesPerRow, aligned, rows, lastRow = 12, 256, 3, 8
	data := make([]byte, bytesPerRow*(rows-1)+lastRow)
	for i := range data {
		data[i] = byte(i + 1)
	}
	var buf []byte
	out := alignTextureDataInto(&buf, data, 0, bytesPerRow, aligned, rows, 1, aligned*rows)
	if len(out) != aligned*rows {
		t.Fatalf("got %d bytes, want %d", len(out), aligned*rows)
	}
	for r := range rows {
		n := bytesPerRow
		if r == rows-1 {
			n = lastRow
		}
		if got, want := out[r*aligned:r*aligned+n], data[r*bytesPerRow:r*bytesPerRow+n]; !bytes.Equal(got, want) {
			t.Errorf("row %d: got %v, want %v", r, got, want)
		}
	}
}

func TestAlignTextureDataFromOffsetWithoutPadding(t *testing.T) {
	const bytesPerRow, rows, offset = 256, 2, 256
	data := make([]byte, offset+bytesPerRow*rows)
	for i := range data {
		data[i] = byte(i)
	}
	var buf []byte
	for name, out := range map[string][]byte{
		"into":   alignTextureDataInto(&buf, data, offset, bytesPerRow, bytesPerRow, rows, 1, bytesPerRow*rows),
		"copy":   copyTextureDataAligned(data, offset, bytesPerRow, bytesPerRow, rows, 1, bytesPerRow*rows),
		"beyond": alignTextureDataInto(&buf, data, uint64(len(data)), bytesPerRow, bytesPerRow, rows, 1, bytesPerRow*rows),
	} {
		want := data[offset:]
		if name == "beyond" {
			want = nil
		}
		if !bytes.Equal(out, want) {
			t.Errorf("%s: got %d bytes starting %v, want %d starting %v", name, len(out), out[:min(4, len(out))], len(want), want[:min(4, len(want))])
		}
	}
}

// A texture write whose staging data is over what one staging buffer holds
// goes up in runs of rows, all of them, and leaves nothing behind for the
// write after it.
func TestWriteTextureOverStagingCap(t *testing.T) {
	_, _, device := newTestDevice(t)
	defer device.Release()
	q := device.Queue()
	if q.pending == nil || q.pending.belt == nil {
		t.Skip("the backend writes textures directly")
	}
	const side = 512
	const row = side * 4
	const size = side * row
	belt := q.pending.belt
	cap := belt.maxStagingBufferSize
	belt.maxStagingBufferSize = size / 4
	defer func() { belt.maxStagingBufferSize = cap }()

	tex, err := device.CreateTexture(&TextureDescriptor{
		Size:          Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageCopyDst | TextureUsageCopySrc,
		MipLevelCount: 1,
		SampleCount:   1,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	pixels := make([]byte, size)
	for i := range pixels {
		pixels[i] = byte(i*7 + i/row)
	}
	if err := q.WriteTexture(&ImageCopyTexture{Texture: tex}, pixels, &ImageDataLayout{BytesPerRow: row, RowsPerImage: side}, &Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	words := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	small, err := device.CreateBuffer(&BufferDescriptor{Size: uint64(len(words)), Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer small.Release()
	if err := q.WriteBuffer(small, 0, words); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	out, err := device.CreateBuffer(&BufferDescriptor{Size: size, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer out.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyTextureToBuffer(tex, out, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{BytesPerRow: row, RowsPerImage: side},
		Size:         Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	for _, b := range []struct {
		name string
		buf  *Buffer
		want []byte
	}{{"texture", out, pixels}, {"buffer written after it", small, words}} {
		n := uint64(len(b.want))
		if err := b.buf.Map(ctx, MapModeRead, 0, n); err != nil {
			t.Fatalf("Map %s: %v", b.name, err)
		}
		rng, err := b.buf.MappedRange(0, n)
		if err != nil {
			t.Fatalf("MappedRange %s: %v", b.name, err)
		}
		got := rng.Bytes()
		if i := bytes.Compare(got, b.want); i != 0 {
			at := 0
			for at < len(got) && at < len(b.want) && got[at] == b.want[at] {
				at++
			}
			t.Errorf("%s: differs from byte %d (row %d) on", b.name, at, at/row)
		}
		if err := b.buf.Unmap(); err != nil {
			t.Fatalf("Unmap %s: %v", b.name, err)
		}
	}
}
