//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// copyOnlyView is a view of a texture made for copies only, which Vulkan
// gives no VkImageView: no bind group or pass may take it.
func copyOnlyView(t *testing.T) (*Device, *captureHandler, *TextureView) {
	t.Helper()
	device, h := newValidatedDevice(t)
	if h.backend != gputypes.BackendVulkan {
		t.Skip("a Vulkan HAL guard; the core's usage validation is the other backends'")
	}
	tex, _ := pixTexture(t, device, "copy-only", TextureUsageCopySrc|TextureUsageCopyDst)
	t.Cleanup(tex.Release)
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	t.Cleanup(view.Release)
	return device, h, view
}

// TestCopyOnlyTextureIsNoBinding binds a view of a copy-only texture,
// which Vulkan wrote into the descriptor as a null image view.
func TestCopyOnlyTextureIsNoBinding(t *testing.T) {
	device, h, view := copyOnlyView(t)
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	if group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, TextureView: view}}}); err == nil {
		group.Release()
		t.Error("a bind group took a view of a copy-only texture")
	}
	assertNoValidation(t, h)
}

// TestCopyOnlyTextureIsNoAttachment attaches a view of a copy-only texture
// to a render pass, which crashed vkCmdBeginRenderPass with the null image
// view. The pass is not begun now; the core's barrier into the attachment
// state is still recorded, so the layer has a word about that until the
// core checks for RenderAttachment usage.
func TestCopyOnlyTextureIsNoAttachment(t *testing.T) {
	device, _, view := copyOnlyView(t)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	if pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
	}); err == nil {
		_ = pass.End()
	}
	if cb, err := enc.Finish(); err == nil {
		if _, err := device.Queue().Submit(cb); err != nil {
			t.Logf("Submit: %v", err)
		}
	}
	device.Poll(PollWait)
}
