//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"sync"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// readU32s copies n words of a storage buffer out and returns them.
func readU32s(t *testing.T, device *Device, buf *Buffer, n int) []uint32 {
	t.Helper()
	size := uint64(n * 4)
	staging, err := device.CreateBuffer(&BufferDescriptor{Label: "read-u32s", Size: size, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer staging.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToBuffer(buf, 0, staging, 0, size)
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := staging.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := staging.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	out := make([]uint32, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(rng.Bytes()[i*4:])
	}
	if err := staging.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

func newStorageBuffer(t *testing.T, device *Device, size uint64) *Buffer {
	t.Helper()
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "storage", Size: size, Usage: BufferUsageStorage | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(buf.Release)
	return buf
}

func storageLayout(t *testing.T, device *Device, kind gputypes.BufferBindingType) *BindGroupLayout {
	t.Helper()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: kind}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	t.Cleanup(layout.Release)
	return layout
}

const groupOneStoreShader = `
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn cs_main() {
	out[0] = 7u;
}
`

// TestOneModuleUnderTwoLayouts uses one shader module in pipelines whose
// layouts have different groups ahead of the one it binds, so the register
// its binding takes differs between them.
func TestOneModuleUnderTwoLayouts(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "group-one-store", WGSL: groupOneStoreShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	uniform := storageLayout(t, device, gputypes.BufferBindingTypeUniform)
	storage := storageLayout(t, device, gputypes.BufferBindingTypeStorage)

	pipeline := func(group0 *BindGroupLayout) *ComputePipeline {
		pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{group0, storage}})
		if err != nil {
			t.Fatalf("CreatePipelineLayout: %v", err)
		}
		t.Cleanup(pl.Release)
		pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "group-one-store", Layout: pl, Module: module, EntryPoint: "cs_main"})
		if err != nil {
			t.Fatalf("CreateComputePipeline: %v", err)
		}
		t.Cleanup(pipe.Release)
		return pipe
	}
	_ = pipeline(uniform)
	second := pipeline(storage)

	group := func(layout *BindGroupLayout, buf *Buffer) *BindGroup {
		g, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: buf, Size: 256}}})
		if err != nil {
			t.Fatalf("CreateBindGroup: %v", err)
		}
		t.Cleanup(g.Release)
		return g
	}
	group0Buf := newStorageBuffer(t, device, 256)
	outBuf := newStorageBuffer(t, device, 256)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	pass.SetPipeline(second)
	pass.SetBindGroup(0, group(storage, group0Buf), nil)
	pass.SetBindGroup(1, group(storage, outBuf), nil)
	pass.Dispatch(1, 1, 1)
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	finishAndSubmit(t, device, enc)
	if got := readU32s(t, device, outBuf, 1)[0]; got != 7 {
		t.Errorf("group 1 buffer = %d, want 7", got)
	}
	if got := readU32s(t, device, group0Buf, 1)[0]; got != 0 {
		t.Errorf("group 0 buffer = %d, want it left at 0", got)
	}
	assertNoValidation(t, h)
}

// TestConcurrentPipelineCreationFromOneModule makes pipelines from one
// shader module on several goroutines at once.
func TestConcurrentPipelineCreationFromOneModule(t *testing.T) {
	device, _ := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "group-one-store", WGSL: groupOneStoreShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	uniform := storageLayout(t, device, gputypes.BufferBindingTypeUniform)
	storage := storageLayout(t, device, gputypes.BufferBindingTypeStorage)
	layouts := make([]*PipelineLayout, 2)
	for i, group0 := range []*BindGroupLayout{uniform, storage} {
		layouts[i], err = device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{group0, storage}})
		if err != nil {
			t.Fatalf("CreatePipelineLayout: %v", err)
		}
		defer layouts[i].Release()
	}

	var wg sync.WaitGroup
	errs := make(chan error, 8*10)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "concurrent", Layout: layouts[(g+i)%2], Module: module, EntryPoint: "cs_main"})
				if err != nil {
					errs <- err
					continue
				}
				pipe.Release()
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("CreateComputePipeline: %v", err)
	}
}
