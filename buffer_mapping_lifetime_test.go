//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"errors"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
)

// A buffer the GC collects while it is mapped keeps its mapping: a slice of
// it holds no reference to the buffer and may still be read.
func TestBufferCollectedWhileMappedKeepsMapping(t *testing.T) {
	inst, adapter, device := newTestDevice(t)
	defer inst.Release()
	defer adapter.Release()
	defer device.Release()

	const size = 1 << 20
	coreBuf, released, data := func() (*core.Buffer, *atomic.Bool, []byte) {
		buf, err := device.CreateBuffer(&BufferDescriptor{
			Label: "collected-while-mapped",
			Size:  size,
			Usage: BufferUsageMapRead | BufferUsageCopyDst,
		})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := buf.Map(ctx, MapModeRead, 0, size); err != nil {
			t.Fatalf("Map: %v", err)
		}
		rng, err := buf.MappedRange(0, size)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		return buf.core, buf.released, rng.Bytes()
	}()

	for i := 0; !released.Load(); i++ {
		if i == 200 {
			t.Fatal("the GC never collected the buffer")
		}
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}
	if coreBuf.IsDestroyed() {
		t.Fatal("the GC destroyed a mapped buffer, whose mapping a slice still points into")
	}
	if got := coreBuf.CurrentMapState(); got != core.BufferMapStateMapped {
		t.Fatalf("map state after the GC = %v, want mapped", got)
	}
	var sum byte
	for _, b := range data {
		sum |= b
	}
	if sum != 0 {
		t.Errorf("the mapping of a new buffer holds %#x, want zeros", sum)
	}
}

// busyWGSL keeps one GPU thread busy for data[1] rounds, long enough for a
// map that waits for the submission to see its short context end first.
const busyWGSL = `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main() {
	var x = data[0];
	for (var i = 0u; i < data[1]; i++) {
		x = x * 1664525u + 1013904223u;
	}
	data[0] = x;
}
`

// submitBusy submits a dispatch that runs for rounds rounds and then a copy
// of its result into dst, and returns the submission's index. It skips the
// test on an adapter without compute shaders.
func submitBusy(t *testing.T, device *Device, rounds uint32, dst *Buffer) uint64 {
	t.Helper()
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "busy", WGSL: busyWGSL})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer layout.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{layout}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "busy", Layout: pl, Module: module, EntryPoint: "main"})
	if err != nil {
		if strings.Contains(err.Error(), "downlevel") {
			t.Skipf("no compute shaders: %v", err)
		}
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipe.Release()
	data, err := device.CreateBuffer(&BufferDescriptor{Label: "busy", Size: 8, Usage: BufferUsageStorage | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer data.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: data}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()
	init := make([]byte, 8)
	binary.LittleEndian.PutUint32(init[4:], rounds)
	if err := device.Queue().WriteBuffer(data, 0, init); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cp, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cp.SetPipeline(pipe)
	cp.SetBindGroup(0, group, nil)
	cp.Dispatch(1, 1, 1)
	if err := cp.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	if dst != nil {
		enc.CopyBufferToBuffer(data, 0, dst, 0, 4)
	}
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	idx, err := device.Queue().Submit(cb)
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	return idx
}

// A Map whose context ends before the GPU finishes leaves the buffer
// unmapped, also once the GPU is done, instead of mapping it behind the
// caller's back.
func TestBufferMapContextEndCancelsTheMap(t *testing.T) {
	inst, adapter, device := newTestDevice(t)
	defer inst.Release()
	defer adapter.Release()
	defer device.Release()

	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "map-ctx", Size: 4, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	submitBusy(t, device, 50_000_000, buf)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	err = buf.Map(ctx, MapModeRead, 0, 4)
	if err == nil {
		t.Skip("the GPU finished before the context ended")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Map = %v, want context.DeadlineExceeded", err)
	}
	device.Poll(PollWait)
	if got := buf.MapState(); got != MapStateUnmapped {
		t.Fatalf("state once the GPU is done = %v, want unmapped", got)
	}
	if err := buf.Map(context.Background(), MapModeRead, 0, 4); err != nil {
		t.Fatalf("Map again: %v", err)
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
}
