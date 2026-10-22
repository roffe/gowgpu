//go:build js && wasm

package browser

import (
	"testing"
)

func TestColorTargetStateWriteMaskNoneIsSet(t *testing.T) {
	obj := (&ColorTargetStateJS{Format: "rgba8unorm", WriteMask: 0}).ToJS()
	if v := obj.Get("writeMask"); v.IsUndefined() || v.Int() != 0 {
		t.Fatalf("writeMask = %v, want 0", v)
	}
}

func TestDepthStencilStateZeroMasksAreSet(t *testing.T) {
	obj := (&DepthStencilStateJS{Format: "depth24plus-stencil8"}).ToJS()
	for _, key := range []string{"stencilReadMask", "stencilWriteMask"} {
		if v := obj.Get(key); v.IsUndefined() || v.Int() != 0 {
			t.Errorf("%s = %v, want 0", key, v)
		}
	}
}
