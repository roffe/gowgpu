//go:build js && wasm

package browser

import (
	"slices"
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

func TestBuildRenderPassDescriptorDepthStencilOpsFollowTheFormat(t *testing.T) {
	keys := []string{"depthLoadOp", "depthStoreOp", "depthClearValue", "stencilLoadOp", "stencilStoreOp", "stencilClearValue"}
	tests := []struct {
		name            string
		format          gputypes.TextureFormat
		depthReadOnly   bool
		stencilReadOnly bool
		want            []string
	}{
		{"depth only", gputypes.TextureFormatDepth24Plus, false, false, keys[:3]},
		{"stencil only", gputypes.TextureFormatStencil8, false, false, keys[3:]},
		{"depth and stencil", gputypes.TextureFormatDepth24PlusStencil8, false, false, keys},
		{"read-only depth", gputypes.TextureFormatDepth32Float, true, false, nil},
		{"read-only stencil", gputypes.TextureFormatDepth32FloatStencil8, false, true, keys[:3]},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			desc := BuildRenderPassDescriptor("", nil, &RenderPassDepthStencilAttachmentJS{
				View:            js.Undefined(),
				Format:          tc.format,
				DepthLoadOp:     "clear",
				DepthStoreOp:    "store",
				DepthClearValue: 1,
				DepthReadOnly:   tc.depthReadOnly,
				StencilLoadOp:   "clear",
				StencilStoreOp:  "store",
				StencilReadOnly: tc.stencilReadOnly,
			})
			ds := desc.Get("depthStencilAttachment")
			var got []string
			for _, key := range keys {
				if !ds.Get(key).IsUndefined() {
					got = append(got, key)
				}
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("members %v, want %v", got, tc.want)
			}
		})
	}
}
