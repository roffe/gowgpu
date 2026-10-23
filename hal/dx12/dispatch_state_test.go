//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestDispatchUsesTheGroupsSetLast binds a buffer as storage, replaces the
// group with one binding it as a uniform before the dispatch, and checks
// the dispatch leaves the buffer in the constant buffer state the group in
// use reads it in, not in the UAV state of the group replaced.
func TestDispatchUsesTheGroupsSetLast(t *testing.T) {
	dev, _ := openTestDevice(t)
	buf := newTestBuffer(t, dev, 256, gputypes.BufferUsageStorage|gputypes.BufferUsageUniform)
	layout := func(kind gputypes.BufferBindingType) hal.BindGroupLayout {
		l, err := dev.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
			{Binding: 0, Visibility: gputypes.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: kind}},
		}})
		if err != nil {
			t.Fatalf("CreateBindGroupLayout: %v", err)
		}
		t.Cleanup(func() { dev.DestroyBindGroupLayout(l) })
		return l
	}
	group := func(l hal.BindGroupLayout) hal.BindGroup {
		g, err := dev.CreateBindGroup(&hal.BindGroupDescriptor{Layout: l, Entries: []gputypes.BindGroupEntry{
			{Binding: 0, Resource: gputypes.BufferBinding{Buffer: buf.NativeHandle(), Size: 256}},
		}})
		if err != nil {
			t.Fatalf("CreateBindGroup: %v", err)
		}
		t.Cleanup(func() { dev.DestroyBindGroup(g) })
		return g
	}
	storage, uniform := layout(gputypes.BufferBindingTypeStorage), layout(gputypes.BufferBindingTypeUniform)
	pl, err := dev.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{BindGroupLayouts: []hal.BindGroupLayout{uniform}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	t.Cleanup(func() { dev.DestroyPipelineLayout(pl) })
	module, err := dev.CreateShaderModule(&hal.ShaderModuleDescriptor{Source: hal.ShaderSource{WGSL: `
struct U { v: vec4<u32> }
@group(0) @binding(0) var<uniform> u: U;
@compute @workgroup_size(1) fn main() { _ = u.v; }
`}})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(func() { dev.DestroyShaderModule(module) })
	pipe, err := dev.CreateComputePipeline(&hal.ComputePipelineDescriptor{Layout: pl, Compute: hal.ComputeState{Module: module, EntryPoint: "main"}})
	if err != nil {
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	t.Cleanup(func() { dev.DestroyComputePipeline(pipe) })

	raw, err := dev.CreateCommandEncoder(&hal.CommandEncoderDescriptor{})
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc := raw.(*CommandEncoder)
	t.Cleanup(enc.Destroy)
	if err := enc.BeginEncoding("dispatch-state"); err != nil {
		t.Fatalf("BeginEncoding: %v", err)
	}
	pass := enc.BeginComputePass(&hal.ComputePassDescriptor{})
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, group(storage), nil)
	pass.SetBindGroup(0, group(uniform), nil)
	pass.Dispatch(1, 1, 1)
	pass.End()
	if got := enc.stateTracker.currentState(buf, 0, 0); got != d3d12.D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER {
		t.Errorf("buffer state at the dispatch = %#x, want VERTEX_AND_CONSTANT_BUFFER", got)
	}
	enc.DiscardEncoding()
}
