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
