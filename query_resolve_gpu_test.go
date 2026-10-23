//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
	"testing"
	"time"
)

// TestResolveUnwrittenQueries resolves an occlusion query set nothing has
// written into a QueryResolve|CopySrc buffer filled with 0xff, which WebGPU
// resolves to zeros. Vulkan reset the pool on the host without
// hostQueryReset, copied the queries with VK_QUERY_RESULT_WAIT_BIT although
// none can become available, and made the buffer without TRANSFER_DST.
// Buffers that are only mapped got no usage at all.
func TestResolveUnwrittenQueries(t *testing.T) {
	device, h := newValidatedDevice(t)
	qs, err := device.CreateQuerySet(&QuerySetDescriptor{Label: "occlusion", Type: QueryTypeOcclusion, Count: 4})
	if err != nil {
		t.Skipf("CreateQuerySet(occlusion): %v", err)
	}
	defer qs.Release()
	resolve, err := device.CreateBuffer(&BufferDescriptor{Label: "resolve", Size: 32, Usage: BufferUsageQueryResolve | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer(resolve): %v", err)
	}
	defer resolve.Release()
	if err := device.Queue().WriteBuffer(resolve, 0, bytes.Repeat([]byte{0xff}, 32)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	read, err := device.CreateBuffer(&BufferDescriptor{Label: "read", Size: 32, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer(read): %v", err)
	}
	defer read.Release()
	mapOnly, err := device.CreateBuffer(&BufferDescriptor{Label: "map-read-only", Size: 32, Usage: BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer(map read only): %v", err)
	}
	mapOnly.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.ResolveQuerySet(qs, 0, 4, resolve, 0)
	enc.CopyBufferToBuffer(resolve, 0, read, 0, 32)
	finishAndSubmit(t, device, enc)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := read.Map(ctx, MapModeRead, 0, 32); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := read.MappedRange(0, 32)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	if got := rng.Bytes(); !bytes.Equal(got, make([]byte, 32)) {
		t.Errorf("resolved unwritten queries = % x, want zeros", got)
	}
	if err := read.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	assertNoValidation(t, h)
}
