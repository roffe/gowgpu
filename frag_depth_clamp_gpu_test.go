//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

const fragDepthClampShader = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
struct Out {
	@location(0) color: vec4<f32>,
	@builtin(frag_depth) depth: f32,
}
@fragment fn fs_main() -> Out {
	return Out(vec4<f32>(1.0), 0.75);
}
`

// TestFragDepthClampsToViewport draws with a viewport depth range of 0 to
// 0.5 and a fragment shader that writes depth 0.75. WebGPU clamps a
// written depth to the viewport's depth range, so 0.5 is stored.
func TestFragDepthClampsToViewport(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL {
		t.Skip("GL reads no depth back")
	}
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "frag-depth", WGSL: fragDepthClampShader})
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
		Label:        "frag-depth",
		Layout:       pl,
		Vertex:       VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive:    gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		DepthStencil: &DepthStencilState{Format: TextureFormatDepth32Float, DepthWriteEnabled: true, DepthCompare: gputypes.CompareFunctionAlways},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	tex, view := depthStencilTexture(t, device, gputypes.TextureFormatDepth32Float, TextureUsageRenderAttachment|TextureUsageCopySrc)
	color, colorView := pixTexture(t, device, "frag-depth-color", TextureUsageRenderAttachment)
	defer releaseTextureView(color, colorView)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{View: colorView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
		DepthStencilAttachment: &RenderPassDepthStencilAttachment{View: view,
			DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 1},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	pass.SetPipeline(pipe)
	pass.SetViewport(gputypes.Viewport{Width: pixSide, Height: pixSide, MinDepth: 0, MaxDepth: 0.5})
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	if err := pass.End(); err != nil {
		t.Fatalf("RenderPass.End: %v", err)
	}
	finishAndSubmit(t, device, enc)

	got := readTextureAspect(t, device, tex, gputypes.TextureAspectAll)
	for y := range pixSide {
		for x := range pixSide {
			if d := math.Float32frombits(binary.LittleEndian.Uint32(got[y*pixRow+x*4:])); d != 0.5 {
				t.Fatalf("depth at %d,%d = %v, want 0.5", x, y, d)
			}
		}
	}
	assertNoValidation(t, h)
}
