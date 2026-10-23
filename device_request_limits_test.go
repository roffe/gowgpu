//go:build !(js && wasm)

package wgpu_test

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
)

// RequestDevice refuses features the adapter lacks and limits better than
// the adapter's, and fills in the limits a descriptor leaves zero with the
// adapter's rather than leaving them 0.
func TestRequestDeviceChecksFeaturesAndLimits(t *testing.T) {
	inst, adapter := newAdapter(t)
	defer inst.Release()
	defer adapter.Release()
	have := adapter.Limits()

	var missing gputypes.Features
	for f := gputypes.Feature(1); f != 0 && missing == 0; f <<= 1 {
		if !adapter.Features().Contains(f) {
			missing = gputypes.Features(f)
		}
	}
	if missing != 0 {
		if device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{RequiredFeatures: missing}); err == nil {
			device.Release()
			t.Errorf("RequestDevice with feature %#x the adapter lacks succeeded", uint64(missing))
		}
	}

	for name, limits := range map[string]gputypes.Limits{
		"larger maximum":    {MaxTextureDimension2D: have.MaxTextureDimension2D * 4},
		"smaller alignment": {MinUniformBufferOffsetAlignment: max(have.MinUniformBufferOffsetAlignment/2, 1)},
	} {
		if name == "smaller alignment" && have.MinUniformBufferOffsetAlignment <= 1 {
			continue
		}
		device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{RequiredLimits: limits})
		if !errors.Is(err, core.ErrLimitExceeded) {
			if device != nil {
				device.Release()
			}
			t.Errorf("%s: %v, want ErrLimitExceeded", name, err)
		}
	}

	device, err := adapter.RequestDevice(&wgpu.DeviceDescriptor{RequiredLimits: gputypes.Limits{MaxBindGroups: 2}})
	if err != nil {
		t.Fatalf("RequestDevice(MaxBindGroups 2): %v", err)
	}
	defer device.Release()
	want := have
	want.MaxBindGroups = 2
	if got := device.Limits(); got != want {
		t.Errorf("limits = %+v, want the adapter's with MaxBindGroups 2", got)
	}
}
