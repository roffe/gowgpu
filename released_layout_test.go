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
