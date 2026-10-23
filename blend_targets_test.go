//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// whiteToTwoTargetsShader draws a full-screen triangle in white to two
// colour targets.
const whiteToTwoTargetsShader = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
struct Out { @location(0) a: vec4<f32>, @location(1) b: vec4<f32> }
@fragment fn fs_main() -> Out { return Out(vec4<f32>(1.0), vec4<f32>(1.0)); }
`

// TestColorTargetsMaskedApart draws white into two targets cleared to
// transparent black, the second masked to green: GL applied the first
// target's mask, and blending, to every target although it offered
// independent blending.
func TestColorTargetsMaskedApart(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "two-targets", WGSL: whiteToTwoTargetsShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "two-targets",
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main", Targets: []gputypes.ColorTargetState{
			{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll},
			{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskGreen},
		}},
	})
	if err != nil {
		if strings.Contains(err.Error(), "draw_buffers_indexed") {
			t.Skipf("no per-target write masks on this GL context: %v", err)
		}
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	a, aView := pixTexture(t, device, "a", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(a, aView)
	b, bView := pixTexture(t, device, "b", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(b, bView)
	aSt, bSt := readbackBuffer(t, device, "a-st"), readbackBuffer(t, device, "b-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{
		{View: aView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore},
		{View: bView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore},
	}})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	pass.SetPipeline(pipe)
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	copyOut(enc, a, aSt)
	copyOut(enc, b, bSt)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "the target with every channel", readback(t, aSt), 255, 255, 255, 255)
	expectPixels(t, "the target masked to green", readback(t, bSt), 0, 255, 0, 0)
	assertNoValidation(t, h)
}
