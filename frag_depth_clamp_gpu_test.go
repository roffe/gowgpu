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
@fragment fn fs_low() -> @builtin(frag_depth) f32 {
	return 0.125;
}
`

// TestFragDepthClampsToViewport draws the left half with a viewport depth
// range of 0 to 0.5 and a fragment shader that writes depth 0.75, then the
// right half with a range of 0.625 to 1 and one that writes 0.125, setting
// the viewport after the pipeline the first time and before it the second.
// WebGPU clamps a written depth to the viewport's depth range, so 0.5 and
// 0.625 are stored.
func TestFragDepthClampsToViewport(t *testing.T) {
	device, h := newValidatedDevice(t)
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
	pipeline := func(entry string) *RenderPipeline {
		pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
			Label:        "frag-depth-" + entry,
			Layout:       pl,
			Vertex:       VertexState{Module: module, EntryPoint: "vs_main"},
			Primitive:    gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
			DepthStencil: &DepthStencilState{Format: TextureFormatDepth32Float, DepthWriteEnabled: true, DepthCompare: gputypes.CompareFunctionAlways},
			Fragment: &FragmentState{Module: module, EntryPoint: entry,
				Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
		})
		if err != nil {
			t.Fatalf("CreateRenderPipeline(%s): %v", entry, err)
		}
		t.Cleanup(pipe.Release)
		return pipe
	}
	high, low := pipeline("fs_main"), pipeline("fs_low")
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
	const half = pixSide / 2
	pass.SetPipeline(high)
	pass.SetViewport(gputypes.Viewport{Width: half, Height: pixSide, MinDepth: 0, MaxDepth: 0.5})
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	pass.SetViewport(gputypes.Viewport{X: half, Width: half, Height: pixSide, MinDepth: 0.625, MaxDepth: 1})
	pass.SetPipeline(low)
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	if err := pass.End(); err != nil {
		t.Fatalf("RenderPass.End: %v", err)
	}
	finishAndSubmit(t, device, enc)

	got := readTextureAspect(t, device, tex, gputypes.TextureAspectAll)
	for y := range pixSide {
		for x := range pixSide {
			want := float32(0.5)
			if x >= half {
				want = 0.625
			}
			if d := math.Float32frombits(binary.LittleEndian.Uint32(got[y*pixRow+x*4:])); d != want {
				t.Fatalf("depth at %d,%d = %v, want %v", x, y, d, want)
			}
		}
	}
	assertNoValidation(t, h)
}
