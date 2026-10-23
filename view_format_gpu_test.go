//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// greyShader draws a full-screen triangle in linear grey 0.25.
const greyShader = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.25, 0.25, 0.25, 1.0); }
`

// TestRenderThroughSrgbView clears an RGBA8Unorm texture through an
// RGBA8UnormSrgb view to linear 0.5 and draws linear 0.25 on its right
// half with a pipeline of the view's format: the texture keeps them sRGB
// encoded, 188 and 137. Vulkan made the render pass of the texture's
// format, which the framebuffer's sRGB view and the pipeline did not match.
func TestRenderThroughSrgbView(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL {
		t.Skip("GL ignores the views of a texture, see the README's known issues")
	}
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         "unorm",
		Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageRenderAttachment | TextureUsageCopySrc,
		ViewFormats:   []gputypes.TextureFormat{gputypes.TextureFormatRGBA8UnormSrgb},
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	view, err := device.CreateTextureView(tex, &TextureViewDescriptor{Format: gputypes.TextureFormatRGBA8UnormSrgb})
	if err != nil {
		t.Fatalf("CreateTextureView(sRGB): %v", err)
	}
	defer view.Release()
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
		Label:     "grey",
		Layout:    pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8UnormSrgb, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()

	st := readbackBuffer(t, device, "srgb-view-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	half := float32(pixSide / 2)
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{R: 0.5, G: 0.5, B: 0.5, A: 1}, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
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
	expectPixels(t, "clear through the sRGB view", left, 188, 188, 188, 255)
	expectPixels(t, "draw through the sRGB view", right, 137, 137, 137, 255)
	assertNoValidation(t, h)
}
