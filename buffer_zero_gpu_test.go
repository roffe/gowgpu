//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

const zeroTestSize = 4096

// dirtyAndRelease makes a buffer, fills it with fill and releases it once the
// GPU is done, so that the next buffer of its size and kind can get its
// memory. A MapWrite buffer is filled through its mapping at creation, any
// other with WriteBuffer.
func dirtyAndRelease(t *testing.T, device *Device, usage gputypes.BufferUsage, fill byte) {
	t.Helper()
	data := bytes.Repeat([]byte{fill}, zeroTestSize)
	mapWrite := usage&BufferUsageMapWrite != 0
	if !mapWrite {
		usage |= BufferUsageCopyDst
	}
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: zeroTestSize, Usage: usage, MappedAtCreation: mapWrite})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	if mapWrite {
		rng, err := buf.MappedRange(0, zeroTestSize)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		copy(rng.Bytes(), data)
		if err := buf.Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
	} else if err := device.Queue().WriteBuffer(buf, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	if _, err := device.Queue().Submit(); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	buf.Release()
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
}

// mapAll maps all of buf and returns a copy of its bytes.
func mapAll(t *testing.T, buf *Buffer, mode MapMode) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, mode, 0, zeroTestSize); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, zeroTestSize)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	out := bytes.Clone(rng.Bytes())
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

// expectZeros fails on the first byte of got that is not zero.
func expectZeros(t *testing.T, what string, got []byte) {
	t.Helper()
	if i := bytes.IndexFunc(got, func(r rune) bool { return r != 0 }); i >= 0 {
		t.Errorf("%s: byte %d is %#x, want every byte zero", what, i, got[i])
	}
}

// A new buffer holds zeros, as WebGPU hands buffers out, whatever a freed
// buffer of the same size left in the memory: mapped at creation, mapped
// for reading, and read by the GPU.
func TestNewBuffersAreZeroed(t *testing.T) {
	device, h := newValidatedDevice(t)

	t.Run("mapped at creation", func(t *testing.T) {
		dirtyAndRelease(t, device, BufferUsageMapWrite|BufferUsageCopySrc, 0xAB)
		buf, err := device.CreateBuffer(&BufferDescriptor{Size: zeroTestSize, Usage: BufferUsageMapWrite | BufferUsageCopySrc, MappedAtCreation: true})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		defer buf.Release()
		rng, err := buf.MappedRange(0, zeroTestSize)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		expectZeros(t, "mapped at creation", rng.Bytes())
		if err := buf.Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
	})

	t.Run("map read", func(t *testing.T) {
		dirtyAndRelease(t, device, BufferUsageMapRead, 0xCD)
		buf, err := device.CreateBuffer(&BufferDescriptor{Size: zeroTestSize, Usage: BufferUsageMapRead | BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		defer buf.Release()
		expectZeros(t, "mapped for reading", mapAll(t, buf, MapModeRead))
	})

	t.Run("storage", func(t *testing.T) {
		dirtyAndRelease(t, device, BufferUsageStorage|BufferUsageCopySrc, 0xEE)
		buf, err := device.CreateBuffer(&BufferDescriptor{Size: zeroTestSize, Usage: BufferUsageStorage | BufferUsageCopySrc})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		defer buf.Release()
		staging, err := device.CreateBuffer(&BufferDescriptor{Size: zeroTestSize, Usage: BufferUsageMapRead | BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer(staging): %v", err)
		}
		defer staging.Release()
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		enc.CopyBufferToBuffer(buf, 0, staging, 0, zeroTestSize)
		finishAndSubmit(t, device, enc)
		expectZeros(t, "copied out of a new storage buffer", mapAll(t, staging, MapModeRead))
	})

	t.Run("partial write", func(t *testing.T) {
		dirtyAndRelease(t, device, BufferUsageStorage|BufferUsageCopySrc, 0xEE)
		buf, err := device.CreateBuffer(&BufferDescriptor{Size: zeroTestSize, Usage: BufferUsageStorage | BufferUsageCopySrc | BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		defer buf.Release()
		if err := device.Queue().WriteBuffer(buf, 16, []byte{1, 2, 3, 4}); err != nil {
			t.Fatalf("WriteBuffer: %v", err)
		}
		staging, err := device.CreateBuffer(&BufferDescriptor{Size: zeroTestSize, Usage: BufferUsageMapRead | BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer(staging): %v", err)
		}
		defer staging.Release()
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		enc.CopyBufferToBuffer(buf, 0, staging, 0, zeroTestSize)
		finishAndSubmit(t, device, enc)
		got := mapAll(t, staging, MapModeRead)
		if !bytes.Equal(got[16:20], []byte{1, 2, 3, 4}) {
			t.Errorf("written bytes = %v, want [1 2 3 4]", got[16:20])
		}
		expectZeros(t, "around a write into a new buffer", append(got[:16:16], got[20:]...))
	})

	assertNoValidation(t, h)
}
