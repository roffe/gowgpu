//go:build !(js && wasm)

package vulkan

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// TestIdenticalSamplersShareOne creates two samplers with one descriptor
// and one with another: the first two share a VkSampler, which lives until
// both are destroyed.
func TestIdenticalSamplersShareOne(t *testing.T) {
	d := openTestDevice(t, 0)
	linear := &hal.SamplerDescriptor{Label: "linear", MagFilter: gputypes.FilterModeLinear, MinFilter: gputypes.FilterModeLinear}
	create := func(desc *hal.SamplerDescriptor) *Sampler {
		t.Helper()
		s, err := d.CreateSampler(desc)
		if err != nil {
			t.Fatalf("CreateSampler: %v", err)
		}
		return s.(*Sampler)
	}
	a, b := create(linear), create(linear)
	nearest := create(&hal.SamplerDescriptor{Label: "nearest"})
	defer d.DestroySampler(nearest)
	if a.handle != b.handle {
		t.Errorf("identical samplers got VkSamplers %#x and %#x", a.handle, b.handle)
	}
	if nearest.handle == a.handle {
		t.Errorf("different samplers share VkSampler %#x", a.handle)
	}
	handle := a.handle
	d.DestroySampler(a)
	if shared := d.samplers[b.info]; shared == nil || shared.handle != handle {
		t.Fatalf("destroying one of two identical samplers dropped their VkSampler")
	}
	d.DestroySampler(b)
	if shared := d.samplers[b.info]; shared != nil {
		t.Errorf("VkSampler %#x outlived both samplers", shared.handle)
	}
}
