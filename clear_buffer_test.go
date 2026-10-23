//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
	"encoding/binary"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// clearTestBuffer makes a buffer that goes away with the test.
func clearTestBuffer(t *testing.T, device *Device, usage gputypes.BufferUsage, size uint64) *Buffer {
	t.Helper()
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "clear-test", Size: size, Usage: usage})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(buf.Release)
	return buf
}

// ClearBuffer zeroes the range it is given and nothing else, and size 0
// clears to the end. The range is wider than DX12's zero buffer and GL's
// block of zeros, so both clear it in pieces, and the copy reading it back
// in the same encoder needs the barrier the clear's CopyDst usage gives.
// GOWGPU_TEST_BACKEND picks the backend.
func TestClearBufferZeroesRange(t *testing.T) {
	device, h := newValidatedDevice(t)
	const size, from, n = 320 << 10, 4, 300 << 10

	part := clearTestBuffer(t, device, BufferUsageCopySrc|BufferUsageCopyDst, size)
	all := clearTestBuffer(t, device, BufferUsageCopySrc|BufferUsageCopyDst, size)
	for _, b := range []*Buffer{part, all} {
		if err := device.Queue().WriteBuffer(b, 0, bytes.Repeat([]byte{0xab}, size)); err != nil {
			t.Fatalf("WriteBuffer: %v", err)
		}
	}
	st := clearTestBuffer(t, device, BufferUsageCopyDst|BufferUsageMapRead, 2*size)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.ClearBuffer(part, from, n)
	enc.ClearBuffer(all, 0, 0)
	enc.CopyBufferToBuffer(part, 0, st, 0, size)
	enc.CopyBufferToBuffer(all, 0, st, size, size)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := st.Map(ctx, MapModeRead, 0, 2*size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := st.MappedRange(0, 2*size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	for i, b := range rng.Bytes() {
		want := byte(0)
		if i < from || (i >= from+n && i < size) {
			want = 0xab
		}
		if b != want {
			t.Fatalf("byte %d is %#x, want %#x", i, b, want)
		}
	}
	if err := st.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	assertNoValidation(t, h)
}

// ClearBuffer refuses what WebGPU refuses, and Finish reports it.
func TestClearBufferValidation(t *testing.T) {
	device, _ := newValidatedDevice(t)
	buf := clearTestBuffer(t, device, BufferUsageCopyDst, 256)
	noDst := clearTestBuffer(t, device, BufferUsageCopySrc, 256)

	for _, c := range []struct {
		name         string
		buf          *Buffer
		offset, size uint64
	}{
		{"unaligned offset", buf, 2, 4},
		{"unaligned size", buf, 0, 6},
		{"past the end", buf, 128, 132},
		{"offset past the end", buf, 260, 0},
		{"no CopyDst", noDst, 0, 0},
		{"nil buffer", nil, 0, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			enc.ClearBuffer(c.buf, c.offset, c.size)
			if _, err := enc.Finish(); err == nil {
				t.Fatal("Finish took an invalid ClearBuffer")
			}
		})
	}
}

// A write mapping of part of a buffer uploads that part at Unmap and nothing
// else. GL kept a zero shadow of the whole buffer for a buffer without MapRead
// and uploaded all of it, over what lay outside the mapped range.
// GOWGPU_TEST_BACKEND picks the backend.
func TestMapWriteRangeKeepsTheRest(t *testing.T) {
	device, h := newValidatedDevice(t)
	const size, from, n = 1024, 256, 256
	buf := clearTestBuffer(t, device, BufferUsageMapWrite|BufferUsageCopySrc, size)
	st := clearTestBuffer(t, device, BufferUsageCopyDst|BufferUsageMapRead, size)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	fill := func(offset, length uint64, b byte) {
		t.Helper()
		if err := buf.Map(ctx, MapModeWrite, offset, length); err != nil {
			t.Fatalf("Map(%d, %d): %v", offset, length, err)
		}
		rng, err := buf.MappedRange(offset, length)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		data := rng.Bytes()
		for i := range data {
			data[i] = b
		}
		if err := buf.Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
	}
	fill(0, size, 0xab)
	fill(from, n, 0x11)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToBuffer(buf, 0, st, 0, size)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	if err := st.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := st.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	for i, b := range rng.Bytes() {
		want := byte(0xab)
		if i >= from && i < from+n {
			want = 0x11
		}
		if b != want {
			t.Fatalf("byte %d is %#x, want %#x", i, b, want)
		}
	}
	if err := st.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	assertNoValidation(t, h)
}

// A MapWrite mapping holds the buffer's contents: GL started every mapping
// zeroed and uploaded it whole at Unmap, so a second mapping that wrote 16
// bytes zeroed the other 240.
func TestMapWriteKeepsBufferContents(t *testing.T) {
	device, h := newValidatedDevice(t)
	const size = 256
	src := clearTestBuffer(t, device, BufferUsageMapWrite|BufferUsageCopySrc, size)
	write := func(data []byte) {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := src.Map(ctx, MapModeWrite, 0, size); err != nil {
			t.Fatalf("Map: %v", err)
		}
		rng, err := src.MappedRange(0, size)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		copy(rng.Bytes(), data)
		if err := src.Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
	}
	write(bytes.Repeat([]byte{0xab}, size))
	write(bytes.Repeat([]byte{0xcd}, 16))

	st := clearTestBuffer(t, device, BufferUsageCopyDst|BufferUsageMapRead, size)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToBuffer(src, 0, st, 0, size)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := st.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := st.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	want := append(bytes.Repeat([]byte{0xcd}, 16), bytes.Repeat([]byte{0xab}, size-16)...)
	if got := rng.Bytes(); !bytes.Equal(got, want) {
		t.Errorf("buffer holds % x..., byte 100 is %#x, want 0xab", got[:20], got[100])
	}
	if err := st.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	assertNoValidation(t, h)
}

// TestTimestampWritesResolve writes timestamps at the start and end of a
// render pass and resolves them: GL on Linux offered timestamp queries and
// failed every query set.
func TestTimestampWritesResolve(t *testing.T) {
	device, h := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureTimestampQuery))
	qs, err := device.CreateQuerySet(&QuerySetDescriptor{Label: "timestamps", Type: QueryTypeTimestamp, Count: 2})
	if err != nil {
		t.Fatalf("CreateQuerySet: %v", err)
	}
	defer qs.Release()
	resolved := clearTestBuffer(t, device, BufferUsageQueryResolve|BufferUsageCopySrc, 256)
	st := clearTestBuffer(t, device, BufferUsageCopyDst|BufferUsageMapRead, 256)
	tex, view := pixTexture(t, device, "timestamps", TextureUsageRenderAttachment)
	defer releaseTextureView(tex, view)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	begin, end := uint32(0), uint32(1)
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: red}},
		TimestampWrites:  &RenderPassTimestampWrites{QuerySet: qs, BeginningOfPassWriteIndex: &begin, EndOfPassWriteIndex: &end},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	enc.ResolveQuerySet(qs, 0, 2, resolved, 0)
	enc.CopyBufferToBuffer(resolved, 0, st, 0, 256)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := st.Map(ctx, MapModeRead, 0, 16); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := st.MappedRange(0, 16)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	b := rng.Bytes()
	t0, t1 := binary.LittleEndian.Uint64(b), binary.LittleEndian.Uint64(b[8:])
	if t0 == 0 || t1 < t0 {
		t.Errorf("timestamps %d, %d: want a start and an end at or after it", t0, t1)
	}
	if err := st.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	assertNoValidation(t, h)
}
