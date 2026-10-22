//go:build js && wasm

package browser

import (
	"syscall/js"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
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

func TestBuildImageCopyTextureAspect(t *testing.T) {
	tests := []struct {
		aspect gputypes.TextureAspect
		want   string
	}{
		{gputypes.TextureAspectUndefined, ""},
		{gputypes.TextureAspectAll, ""},
		{gputypes.TextureAspectDepthOnly, "depth-only"},
		{gputypes.TextureAspectStencilOnly, "stencil-only"},
	}
	for _, tc := range tests {
		v := BuildImageCopyTexture(js.Undefined(), 0, 0, 0, 0, tc.aspect).Get("aspect")
		got := ""
		if !v.IsUndefined() {
			got = v.String()
		}
		if got != tc.want {
			t.Errorf("aspect %v: got %q, want %q", tc.aspect, got, tc.want)
		}
	}
}
