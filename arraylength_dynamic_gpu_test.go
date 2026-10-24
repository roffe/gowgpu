//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

const dynamicArrayLengthShader = `
struct Data { head: vec4<u32>, items: array<u32> }
@group(0) @binding(0) var<storage, read_write> data: Data;
@compute @workgroup_size(1) fn main() {
	data.head.x = arrayLength(&data.items);
}
`

// TestArrayLengthOfDynamicStorageBinding binds 64 bytes at offset 256 of a
// 1024-byte buffer with a dynamic offset, and has arrayLength() of the
// runtime array after a 16-byte head written at dynamic offsets 0 and 256.
// WebGPU counts the binding: 12 elements at either offset. DX12 counted the
// view to the end of the buffer, 188 elements and still 188 at 256.
func TestArrayLengthOfDynamicStorageBinding(t *testing.T) {
	device, h := newValidatedDevice(t)
	if device.Limits().MaxStorageBuffersPerShaderStage == 0 {
		t.Skip("no storage buffers on this adapter")
	}
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "dynamic-array-length", WGSL: dynamicArrayLengthShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage, HasDynamicOffset: true}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Layout: pl, Module: module, EntryPoint: "main"})
	if err != nil {
		skipWithoutComputeShaders(t, err)
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipe.Release()
	buf := newStorageBuffer(t, device, 1024)
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: buf, Offset: 256, Size: 64}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	pass.SetPipeline(pipe)
	for _, offset := range []uint32{0, 256} {
		pass.SetBindGroup(0, group, []uint32{offset})
		pass.Dispatch(1, 1, 1)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	finishAndSubmit(t, device, enc)
	words := readU32s(t, device, buf, 256)
	for _, at := range []int{256 / 4, 512 / 4} {
		if got := words[at]; got != 12 {
			t.Errorf("arrayLength at byte %d = %d, want 12", at*4, got)
		}
	}
	assertNoValidation(t, h)
}
