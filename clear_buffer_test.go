//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
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
