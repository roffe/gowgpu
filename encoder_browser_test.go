//go:build js && wasm

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/internal/browser"
)

func TestBrowserDepthOnlyAttachmentSendsNoStencilOps(t *testing.T) {
	view := &TextureView{
		browser: &browser.TextureView{},
		texture: &Texture{format: gputypes.TextureFormatDepth24Plus},
	}
	desc := buildRenderPassDescriptorJS(&RenderPassDescriptor{
		DepthStencilAttachment: &RenderPassDepthStencilAttachment{
			View:            view,
			DepthLoadOp:     gputypes.LoadOpClear,
			DepthStoreOp:    gputypes.StoreOpStore,
			DepthClearValue: 1,
		},
	})
	ds := desc.Get("depthStencilAttachment")
	if ds.Get("depthLoadOp").String() != "clear" || ds.Get("depthStoreOp").String() != "store" {
		t.Fatalf("depth ops = %v/%v, want clear/store", ds.Get("depthLoadOp"), ds.Get("depthStoreOp"))
	}
	for _, key := range []string{"stencilLoadOp", "stencilStoreOp"} {
		if v := ds.Get(key); !v.IsUndefined() {
			t.Errorf("%s = %v, want it left out for a depth-only format", key, v)
		}
	}
}
