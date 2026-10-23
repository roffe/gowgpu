//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestEmptyViewportAndListStripFormat draws with a viewport of no width or
// height, which WebGPU allows, and makes a triangle list pipeline with a
// strip index format. Vulkan passed the zero width on
// (VUID-VkViewport-width-01770) and turned primitive restart on for the
// list (VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252).
func TestEmptyViewportAndListStripFormat(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "grey", WGSL: greyShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	pipeline := func(strip *gputypes.IndexFormat) (*RenderPipeline, error) {
		return device.CreateRenderPipeline(&RenderPipelineDescriptor{
			Label:     "grey",
			Layout:    pl,
			Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
			Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList, StripIndexFormat: strip},
			Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
				Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
		})
	}
	strip := gputypes.IndexFormatUint16
	if listStrip, err := pipeline(&strip); err == nil {
		listStrip.Release()
	}
	pipe, err := pipeline(nil)
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	tex, view := pixTexture(t, device, "empty-viewport", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "empty-viewport-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetViewport(gputypes.Viewport{X: 1, Y: 1, Width: 0, Height: 0, MaxDepth: 1})
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "draw into an empty viewport", readback(t, st), 0, 0, 0, 0)
	assertNoValidation(t, h)
}

// TestUnsetDepthStencilStateDraws draws with a Depth24PlusStencil8 pipeline
// whose depth compare and stencil faces are left unset, depth writes off
// and stencil masks full, over depth cleared to 1. WebGPU compares Always
// what is unset and keeps the stencil, so the draw lands. Vulkan compared
// depth and both stencil faces Never and drew nothing.
func TestUnsetDepthStencilStateDraws(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "grey", WGSL: greyShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "unset-depth-stencil",
		Layout:    pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		DepthStencil: &DepthStencilState{Format: gputypes.TextureFormatDepth24PlusStencil8,
			StencilReadMask: 0xff, StencilWriteMask: 0xff},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Skipf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	_, depthView := depthStencilTexture(t, device, gputypes.TextureFormatDepth24PlusStencil8, TextureUsageRenderAttachment)
	tex, view := pixTexture(t, device, "unset-depth-stencil", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "unset-depth-stencil-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
		DepthStencilAttachment: &RenderPassDepthStencilAttachment{View: depthView,
			DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 1,
			StencilLoadOp: gputypes.LoadOpClear, StencilStoreOp: gputypes.StoreOpStore},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	pass.SetPipeline(pipe)
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	if err := pass.End(); err != nil {
		t.Fatalf("RenderPass.End: %v", err)
	}
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "draw with depth compare and stencil faces unset", readback(t, st), 64, 64, 64, 255)
	assertNoValidation(t, h)
}
