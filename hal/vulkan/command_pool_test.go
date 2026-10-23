//go:build !(js && wasm)

package vulkan

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

func openTestDevice(t *testing.T) *Device {
	t.Helper()
	if err := vk.Init(); err != nil {
		t.Skipf("Vulkan not available: %v", err)
	}
	instance, err := Backend{}.CreateInstance(&hal.InstanceDescriptor{Backends: gputypes.BackendsVulkan})
	if err != nil {
		t.Skipf("CreateInstance: %v", err)
	}
	t.Cleanup(instance.Destroy)
	adapters := instance.EnumerateAdapters(nil)
	if len(adapters) == 0 {
		t.Skip("no Vulkan adapter")
	}
	open, err := adapters[0].Adapter.Open(0, adapters[0].Capabilities.Limits)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(open.Device.Destroy)
	return open.Device.(*Device)
}

// TestRecycledPoolKeepsCommandBuffers ends a standalone encoder, then
// discards one, then ends another: each encoder that takes the recycled
// pool starts with the command buffers already allocated from it.
func TestRecycledPoolKeepsCommandBuffers(t *testing.T) {
	d := openTestDevice(t)
	for i, discard := range []bool{false, true, false} {
		enc, err := d.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "standalone"})
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		e := enc.(*CommandEncoder)
		if i > 0 && len(e.free) != allocationGranularity {
			t.Fatalf("encoder %d starts with %d command buffers, want the pool's %d", i, len(e.free), allocationGranularity)
		}
		if err := e.BeginEncoding("standalone"); err != nil {
			t.Fatalf("BeginEncoding: %v", err)
		}
		if discard {
			e.DiscardEncoding()
			continue
		}
		cb, err := e.EndEncoding()
		if err != nil {
			t.Fatalf("EndEncoding: %v", err)
		}
		d.FreeCommandBuffer(cb)
	}
}
