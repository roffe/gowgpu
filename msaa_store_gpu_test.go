//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// msaaLoadShader writes sample 0 of a multisampled texture to the target.
const msaaLoadShader = `
@group(0) @binding(0) var ms: texture_multisampled_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	return textureLoad(ms, vec2<i32>(pos.xy), 0);
}
`

// TestMultisampledAttachmentWithResolveIsStored clears a multisampled
// texture to green in a pass that resolves it and stores it, resolves it
// again from a pass that loads it, and samples it. Vulkan stored a
// multisampled attachment with a resolve target DONT_CARE whatever the pass
// said, and left it in COLOR_ATTACHMENT_OPTIMAL where its sampled usage
// keeps it in GENERAL.
func TestMultisampledAttachmentWithResolveIsStored(t *testing.T) {
	device, h := newValidatedDevice(t)
	ms, err := device.CreateTexture(&TextureDescriptor{
		Label:         "msaa",
		Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   4,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageRenderAttachment | TextureUsageTextureBinding,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer ms.Release()
	msView, err := device.CreateTextureView(ms, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer msView.Release()
	first, firstView := pixTexture(t, device, "msaa-first", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(first, firstView)
	second, secondView := pixTexture(t, device, "msaa-second", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(second, secondView)
	sampled, sampledView := pixTexture(t, device, "msaa-sampled", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(sampled, sampledView)

	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "msaa-load", WGSL: msaaLoadShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{
			SampleType: gputypes.TextureSampleTypeUnfilterableFloat, ViewDimension: gputypes.TextureViewDimension2D, Multisampled: true}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "msaa-load",
		Layout:    pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, TextureView: msView}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	resolvePass := func(load gputypes.LoadOp, resolve *TextureView) {
		pass, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: msView, ResolveTarget: resolve, LoadOp: load, StoreOp: gputypes.StoreOpStore,
			ClearValue: gputypes.Color{G: 1, A: 1},
		}}})
		if err != nil {
			t.Fatalf("BeginRenderPass: %v", err)
		}
		if err := pass.End(); err != nil {
			t.Fatalf("RenderPass.End: %v", err)
		}
	}
	resolvePass(gputypes.LoadOpClear, firstView)
	resolvePass(gputypes.LoadOpLoad, secondView)
	clearPass(t, enc, sampledView, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetBindGroup(0, group, nil)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	firstSt := readbackBuffer(t, device, "msaa-first-st")
	secondSt := readbackBuffer(t, device, "msaa-second-st")
	sampledSt := readbackBuffer(t, device, "msaa-sampled-st")
	copyOut(enc, first, firstSt)
	copyOut(enc, second, secondSt)
	copyOut(enc, sampled, sampledSt)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	expectPixels(t, "resolve of the clear", readback(t, firstSt), 0, 255, 0, 255)
	expectPixels(t, "resolve of the loaded attachment", readback(t, secondSt), 0, 255, 0, 255)
	expectPixels(t, "sample 0 of the attachment", readback(t, sampledSt), 0, 255, 0, 255)
	assertNoValidation(t, h)
}
