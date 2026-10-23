//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

func newBundleRig(t *testing.T, device *Device, h *captureHandler) (*dynamicRig, *RenderPipeline, *RenderBundle) {
	t.Helper()
	rig := newDynamicRig(t, device)
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "bundle",
		Layout:    rig.layout,
		Vertex:    VertexState{Module: rig.module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: rig.module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	t.Cleanup(pipe.Release)

	be, err := device.CreateRenderBundleEncoder(&RenderBundleEncoderDescriptor{
		ColorFormats: []gputypes.TextureFormat{gputypes.TextureFormatRGBA8Unorm},
		SampleCount:  1,
	})
	if err != nil {
		t.Fatalf("CreateRenderBundleEncoder: %v", err)
	}
	if h.backend == gputypes.BackendGL {
		be.SetPipeline(pipe) // GL binds a group against the pipeline set at the time, see the README's known issues
	}
	// Before the pipeline, at the green of the uniform buffer. The bundle
	// keeps its own offsets: changing the slice afterwards changes nothing.
	offsets := []uint32{256, 0}
	be.SetBindGroup(0, rig.group, offsets)
	offsets[0] = 0
	be.SetPipeline(pipe)
	be.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	bundle, err := be.Finish(&RenderBundleDescriptor{Label: "green"})
	if err != nil {
		t.Fatalf("RenderBundleEncoder.Finish: %v", err)
	}
	t.Cleanup(bundle.Release)
	return rig, pipe, bundle
}

// TestRenderBundleDraws executes a bundle that draws a full-screen triangle
// in green on the left half of the target, the viewport the pass set, and
// draws red on the right half directly after it in the same pass. Vulkan
// recorded bundles into a secondary command buffer begun without a render
// pass, which NVIDIA crashed on in vkBeginCommandBuffer.
func TestRenderBundleDraws(t *testing.T) {
	device, h := newValidatedDevice(t)
	rig, pipe, bundle := newBundleRig(t, device, h)
	tex, view := pixTexture(t, device, "bundle", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "bundle-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	half := float32(pixSide / 2)
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetViewport(gputypes.Viewport{Width: half, Height: pixSide, MaxDepth: 1})
		p.ExecuteBundles(bundle)
		p.SetPipeline(pipe)
		p.SetBindGroup(0, rig.group, []uint32{0, 0})
		p.SetViewport(gputypes.Viewport{X: half, Width: half, Height: pixSide, MaxDepth: 1})
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	pix := readback(t, st)
	var left, right []byte
	for y := range pixSide {
		row := pix[y*pixSide*4 : (y+1)*pixSide*4]
		left = append(left, row[:pixSide*2]...)
		right = append(right, row[pixSide*2:]...)
	}
	expectPixels(t, "bundle draw", left, 0, 255, 0, 255)
	expectPixels(t, "draw after the bundle", right, 255, 0, 0, 255)
	assertNoValidation(t, h)
}

// TestExecuteBundlesResetsBindings draws after a bundle without setting a
// pipeline again: WebGPU leaves the pass without one after executeBundles,
// so the draw is a validation error.
func TestExecuteBundlesResetsBindings(t *testing.T) {
	device, h := newValidatedDevice(t)
	rig, pipe, bundle := newBundleRig(t, device, h)
	tex, view := pixTexture(t, device, "bundle-reset", TextureUsageRenderAttachment)
	defer releaseTextureView(tex, view)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, rig.group, []uint32{0, 0})
	pass.ExecuteBundles(bundle)
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	if err := pass.End(); !errors.Is(err, ErrDrawMissingPipeline) {
		t.Fatalf("RenderPass.End() error = %v, want ErrDrawMissingPipeline", err)
	}
}
