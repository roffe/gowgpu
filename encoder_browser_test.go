//go:build js && wasm

package wgpu

import (
	"math"
	"strings"
	"syscall/js"
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

func TestBrowserSparseColorSlotsAreNull(t *testing.T) {
	view := &TextureView{browser: &browser.TextureView{}, texture: &Texture{format: gputypes.TextureFormatRGBA8Unorm}}
	desc := buildRenderPassDescriptorJS(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{}, {View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
	})
	if attachments := desc.Get("colorAttachments"); !attachments.Index(0).IsNull() || attachments.Index(1).IsNull() {
		t.Fatalf("colorAttachments = [%v, %v], want [null, attachment]", attachments.Index(0), attachments.Index(1))
	}

	pipeline := convertRenderPipelineDescriptor(&RenderPipelineDescriptor{
		Fragment: &FragmentState{Targets: []gputypes.ColorTargetState{{}, {Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if targets := pipeline.Get("fragment").Get("targets"); !targets.Index(0).IsNull() || targets.Index(1).Get("format").String() != "rgba8unorm" {
		t.Fatalf("targets = [%v, %v], want [null, rgba8unorm target]", targets.Index(0), targets.Index(1))
	}
}

func fakeEncoder(t *testing.T, pass string) *CommandEncoder {
	t.Helper()
	enc := js.Global().Get("Function").New(`return {
		beginRenderPass() { return ` + pass + ` },
		finish() { return {} },
	}`).Invoke()
	return &CommandEncoder{browser: browser.NewCommandEncoder(enc)}
}

func TestBrowserBeginRenderPassExceptionFailsTheEncoder(t *testing.T) {
	enc := fakeEncoder(t, `(() => { throw new TypeError("missing view") })()`)
	if _, err := enc.BeginRenderPass(&RenderPassDescriptor{}); err == nil {
		t.Fatal("BeginRenderPass: nil error for a throwing beginRenderPass")
	}
	if _, err := enc.Finish(); err == nil {
		t.Fatal("Finish: nil error after a failed BeginRenderPass")
	}
}

func TestBrowserPassExceptionFailsFinish(t *testing.T) {
	enc := fakeEncoder(t, `{
		setViewport() { throw new TypeError("non-finite viewport") },
		setBlendConstant() {},
		end() {},
	}`)
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	pass.SetViewport(gputypes.Viewport{Width: float32(math.NaN()), Height: 1, MaxDepth: 1})
	pass.SetBlendConstant(&gputypes.Color{})
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	if _, err := enc.Finish(); err == nil || !strings.Contains(err.Error(), "SetViewport") {
		t.Fatalf("Finish = %v, want the SetViewport exception", err)
	}
}
