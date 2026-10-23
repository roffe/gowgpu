//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// A released bind group layout makes no bind group and no pipeline layout:
// its HAL layout may be gone, and DX12 made an empty group from it, which
// hung the GPU when drawn with.
func TestReleasedBindGroupLayoutIsRefused(t *testing.T) {
	device, _ := newValidatedDevice(t)
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: gputypes.ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	sampler, err := device.CreateSampler(nil)
	if err != nil {
		t.Fatalf("CreateSampler: %v", err)
	}
	defer sampler.Release()
	bgl.Release()

	if group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Sampler: sampler}}}); !errors.Is(err, ErrReleased) {
		if group != nil {
			group.Release()
		}
		t.Errorf("CreateBindGroup with a released layout: %v, want ErrReleased", err)
	}
	if layout, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}}); !errors.Is(err, ErrReleased) {
		if layout != nil {
			layout.Release()
		}
		t.Errorf("CreatePipelineLayout with a released layout: %v, want ErrReleased", err)
	}
}

// A pass refuses a released bind group or pipeline instead of recording its
// HAL handle, which the deferred destroy may have zeroed.
func TestPassRefusesReleasedGroupAndPipeline(t *testing.T) {
	device, _ := newValidatedDevice(t)
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	group.Release()
	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: "@compute @workgroup_size(1) fn main() {}"})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	pipeline, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Module: shader, EntryPoint: "main"})
	if err != nil {
		t.Skipf("no compute pipeline: %v", err)
	}
	pipeline.Release()

	for name, record := range map[string]func(*ComputePassEncoder){
		"bind group": func(p *ComputePassEncoder) { p.SetBindGroup(0, group, nil) },
		"pipeline":   func(p *ComputePassEncoder) { p.SetPipeline(pipeline) },
	} {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		pass, err := enc.BeginComputePass(nil)
		if err != nil {
			t.Fatalf("BeginComputePass: %v", err)
		}
		record(pass)
		_ = pass.End()
		if cb, err := enc.Finish(); !errors.Is(err, ErrReleased) {
			if cb != nil {
				cb.Release()
			}
			t.Errorf("released %s: Finish = %v, want ErrReleased", name, err)
		}
	}
}
