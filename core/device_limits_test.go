//go:build !(js && wasm)

package core

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

func TestResolveRequiredLimits(t *testing.T) {
	adapter := gputypes.DefaultLimits()
	adapter.MaxTextureDimension2D = 16384
	adapter.MinUniformBufferOffsetAlignment = 64

	got, err := ResolveRequiredLimits(adapter, gputypes.Limits{})
	if err != nil || got != adapter {
		t.Errorf("no limits asked for: %+v, %v, want the adapter's", got, err)
	}

	got, err = ResolveRequiredLimits(adapter, gputypes.Limits{MaxBindGroups: 2, MinUniformBufferOffsetAlignment: 256})
	if err != nil {
		t.Fatalf("worse limits: %v", err)
	}
	want := adapter
	want.MaxBindGroups, want.MinUniformBufferOffsetAlignment = 2, 256
	if got != want {
		t.Errorf("worse limits: %+v, want %+v", got, want)
	}

	for name, required := range map[string]gputypes.Limits{
		"larger maximum":       {MaxTextureDimension2D: 32768},
		"smaller alignment":    {MinUniformBufferOffsetAlignment: 4},
		"alignment not a pow2": {MinStorageBufferOffsetAlignment: 384},
	} {
		if _, err := ResolveRequiredLimits(adapter, required); !errors.Is(err, ErrLimitExceeded) {
			t.Errorf("%s: %v, want ErrLimitExceeded", name, err)
		}
	}
}
