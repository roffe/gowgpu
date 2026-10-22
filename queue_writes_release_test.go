//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestQueueWritesOutliveRelease releases a buffer and a texture between the
// queue writes into them and the Submit. The buffer used to be destroyed
// under the copy, and the released texture made flush drop every write of
// the batch, so the kept buffer never got its data and Submit failed.
func TestQueueWritesOutliveRelease(t *testing.T) {
	_, _, device := newTestDevice(t)
	defer device.Release()
	q := device.Queue()
	newBuf := func(usage gputypes.BufferUsage) *Buffer {
		b, err := device.CreateBuffer(&BufferDescriptor{Size: 256, Usage: usage})
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	gone := newBuf(gputypes.BufferUsageCopyDst | gputypes.BufferUsageCopySrc)
	kept := newBuf(gputypes.BufferUsageCopyDst | gputypes.BufferUsageCopySrc)
	defer kept.Release()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, Format: gputypes.TextureFormatRGBA8Unorm,
		Usage: gputypes.TextureUsageCopyDst, MipLevelCount: 1, SampleCount: 1, Dimension: gputypes.TextureDimension2D,
	})
	if err != nil {
		t.Fatal(err)
	}
	pattern := bytes.Repeat([]byte{0x5a, 0xa5, 0x3c, 0xc3}, 64)
	if err := q.WriteBuffer(gone, 0, pattern); err != nil {
		t.Fatal(err)
	}
	gone.Release()
	if err := q.WriteBuffer(kept, 0, pattern); err != nil {
		t.Fatal(err)
	}
	if err := q.WriteTexture(&ImageCopyTexture{Texture: tex}, make([]byte, 64), &ImageDataLayout{BytesPerRow: 16, RowsPerImage: 4}, &Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}); err != nil {
		t.Fatal(err)
	}
	tex.Release()
	if _, err := q.Submit(); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	staging := newBuf(gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst)
	defer staging.Release()
	enc, _ := device.CreateCommandEncoder(nil)
	enc.CopyBufferToBuffer(kept, 0, staging, 0, 256)
	cb, err := enc.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := q.Submit(cb); err != nil {
		t.Fatal(err)
	}
	if err := staging.Map(context.Background(), MapModeRead, 0, 256); err != nil {
		t.Fatal(err)
	}
	defer staging.Unmap()
	rng, _ := staging.MappedRange(0, 256)
	if got := rng.Bytes(); !bytes.Equal(got, pattern) {
		t.Errorf("kept buffer holds % x..., want % x...", got[:8], pattern[:8])
	}
}
