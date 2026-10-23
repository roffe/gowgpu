//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func TestSwapchainTears(t *testing.T) {
	for _, tc := range []struct {
		supported bool
		mode      gputypes.PresentMode
		damage    bool
		want      bool
	}{
		{false, hal.PresentModeImmediate, false, false},
		{true, hal.PresentModeImmediate, false, true},
		{true, hal.PresentModeFifo, false, true},
		{true, hal.PresentModeMailbox, false, true},
		{true, hal.PresentModeFifo, true, false},
		{true, hal.PresentModeImmediate, true, true},
	} {
		config := &hal.SurfaceConfiguration{PresentMode: tc.mode, EnableDamagePresent: tc.damage}
		if got := swapchainTears(tc.supported, config); got != tc.want {
			t.Errorf("supported %v, mode %v, damage %v: tears = %v, want %v", tc.supported, tc.mode, tc.damage, got, tc.want)
		}
	}
}
