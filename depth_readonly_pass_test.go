//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestDepthWriteThenReadOnlyPass: a depth texture written by one pass and
// attached read-only (DepthReadOnly && StencilReadOnly, LoadOp Load) by the
// next. The tracker transitions DepthStencilWrite -> DepthStencilRead between
// them, and the barrier has to name the attachment layout: Vulkan's render
// pass keeps a read-only depth attachment in DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
// and a barrier into SHADER_READ_ONLY_OPTIMAL failed the draw under the
// validation layer. In one encoder the second pass is split off with its
// transit CB; across two submits the barrier is the submit's.
func TestDepthWriteThenReadOnlyPass(t *testing.T) {
	for _, tc := range []struct {
		name        string
		sameEncoder bool
	}{
		{"same_encoder", true},
		{"two_submits", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			device, h := newValidatedDevice(t)

			tex, err := device.CreateTexture(&TextureDescriptor{
				Label:         "depth",
				Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
				MipLevelCount: 1,
				SampleCount:   1,
				Dimension:     TextureDimension2D,
				Format:        TextureFormatDepth32Float,
				Usage:         TextureUsageRenderAttachment | TextureUsageTextureBinding,
			})
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			defer tex.Release()
			view, err := device.CreateTextureView(tex, nil)
			if err != nil {
				t.Fatalf("CreateTextureView: %v", err)
			}
			defer view.Release()

			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
				DepthStencilAttachment: &RenderPassDepthStencilAttachment{
					View: view, DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 1,
				},
			})
			if err != nil {
				t.Fatalf("BeginRenderPass(write): %v", err)
			}
			if err := pass.End(); err != nil {
				t.Fatalf("End(write): %v", err)
			}
			if !tc.sameEncoder {
				finishAndSubmit(t, device, enc)
				enc, err = device.CreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("CreateCommandEncoder: %v", err)
				}
			}
			pass, err = enc.BeginRenderPass(&RenderPassDescriptor{
				DepthStencilAttachment: &RenderPassDepthStencilAttachment{
					View: view, DepthLoadOp: gputypes.LoadOpLoad, DepthStoreOp: gputypes.StoreOpStore,
					DepthReadOnly: true, StencilReadOnly: true,
				},
			})
			if err != nil {
				t.Fatalf("BeginRenderPass(read-only): %v", err)
			}
			if err := pass.End(); err != nil {
				t.Fatalf("End(read-only): %v", err)
			}
			finishAndSubmit(t, device, enc)
			if err := device.WaitIdle(); err != nil {
				t.Fatalf("WaitIdle: %v", err)
			}
			assertNoValidation(t, h)
		})
	}
}

// depthSampleShader draws a full-screen triangle at z 0.5 and writes the
// depth it samples at each pixel to the colour target.
const depthSampleShader = `
@group(0) @binding(0) var t: texture_depth_2d;
@group(0) @binding(1) var s: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
	var o: VOut;
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	o.pos = vec4<f32>(x, y, 0.5, 1.0);
	o.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
	return o;
}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
	let d = textureSample(t, s, in.uv);
	return vec4<f32>(d, d, d, 1.0);
}
`

// depthLoadShader is depthSampleShader fetching the texel under the pixel
// with textureLoad, no sampler.
const depthLoadShader = `
@group(0) @binding(0) var t: texture_depth_2d;
struct VOut { @builtin(position) pos: vec4<f32> }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
	var o: VOut;
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	o.pos = vec4<f32>(x, y, 0.5, 1.0);
	return o;
}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
	let d = textureLoad(t, vec2<i32>(in.pos.xy), 0);
	return vec4<f32>(d, d, d, 1.0);
}
`

// depthRig is a pipeline that draws a sampled depth texture over the whole
// target: with the depth texture attached read-only (depth test Always, no
// write) when attached is set, in a colour-only pass otherwise.
type depthRig struct {
	layout   *BindGroupLayout
	pipeline *RenderPipeline
	sampler  *Sampler
}

// skipDepthSamplingOnGL skips a test whose shader samples or loads a
// texture_depth_2d: naga's GLSL writer declares one as sampler2DShadow, which
// texture() without a reference and texelFetch don't take, so the pipeline
// doesn't compile on GL. GL has no image layouts for these tests to check.
func skipDepthSamplingOnGL(t *testing.T, h *captureHandler) {
	t.Helper()
	if h.backend == gputypes.BackendGL {
		t.Skip("naga declares texture_depth_2d as sampler2DShadow in GLSL, which this shader can't sample or load")
	}
}

func newDepthRig(t *testing.T, device *Device, attached bool) *depthRig {
	t.Helper()
	entries := []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeDepth, ViewDimension: gputypes.TextureViewDimension2D}},
	}
	shader := depthLoadShader
	if attached {
		entries = append(entries, gputypes.BindGroupLayoutEntry{Binding: 1, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeNonFiltering}})
		shader = depthSampleShader
	}
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: entries})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	t.Cleanup(layout.Release)
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{layout}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	t.Cleanup(pl.Release)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "depth-sample", WGSL: shader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(module.Release)
	desc := &RenderPipelineDescriptor{
		Label:     "depth-sample",
		Layout:    pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	}
	if attached {
		desc.DepthStencil = &DepthStencilState{Format: TextureFormatDepth32Float, DepthWriteEnabled: false, DepthCompare: gputypes.CompareFunctionAlways}
	}
	pipe, err := device.CreateRenderPipeline(desc)
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	t.Cleanup(pipe.Release)
	rig := &depthRig{layout: layout, pipeline: pipe}
	if attached {
		rig.sampler, err = device.CreateSampler(&SamplerDescriptor{MagFilter: gputypes.FilterModeNearest, MinFilter: gputypes.FilterModeNearest, LodMaxClamp: 32})
		if err != nil {
			t.Fatalf("CreateSampler: %v", err)
		}
		t.Cleanup(rig.sampler.Release)
	}
	return rig
}

func (r *depthRig) group(t *testing.T, device *Device, view *TextureView) *BindGroup {
	t.Helper()
	entries := []BindGroupEntry{{Binding: 0, TextureView: view}}
	if r.sampler != nil {
		entries = append(entries, BindGroupEntry{Binding: 1, Sampler: r.sampler})
	}
	g, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: r.layout, Entries: entries})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(g.Release)
	return g
}

// depthTexture makes a pixSide square Depth32Float texture that can be
// attached and sampled, with its view.
func depthTexture(t *testing.T, device *Device) (*Texture, *TextureView) {
	t.Helper()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         "depth",
		Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatDepth32Float,
		Usage:         TextureUsageRenderAttachment | TextureUsageTextureBinding,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	t.Cleanup(tex.Release)
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	t.Cleanup(view.Release)
	return tex, view
}

// clearDepth records a pass that clears view to depth.
func clearDepth(t *testing.T, enc *CommandEncoder, view *TextureView, depth float32) {
	t.Helper()
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		DepthStencilAttachment: &RenderPassDepthStencilAttachment{
			View: view, DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: depth,
		},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass(depth clear): %v", err)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End(depth clear): %v", err)
	}
}

// expectDepthPixels fails on the first pixel that is not the depth value,
// grey in all three channels, give or take the unorm rounding.
func expectDepthPixels(t *testing.T, what string, pix []byte, depth float32) {
	t.Helper()
	want := int(depth*255 + 0.5)
	for i := 0; i+3 < len(pix); i += 4 {
		for c := 0; c < 3; c++ {
			if d := int(pix[i+c]) - want; d < -1 || d > 1 {
				t.Errorf("%s: pixel %d = %v, want about [%d %d %d 255]", what, i/4, pix[i:i+4], want, want, want)
				return
			}
		}
		if pix[i+3] != 255 {
			t.Errorf("%s: pixel %d = %v, want alpha 255", what, i/4, pix[i:i+4])
			return
		}
	}
}

// TestDepthReadOnlyAttachedAndSampled: a depth texture cleared to 0.25 by one
// pass is attached read-only AND sampled by the fragment shader of the next,
// which draws the sampled depth to a colour target. Both uses of the pass
// put the texture in DEPTH_STENCIL_READ_ONLY_OPTIMAL on Vulkan: the barrier
// ahead of the pass (DepthStencilWrite -> DepthStencilRead|Resource), the
// render pass attachment layout and the sampled image descriptor have to
// agree, or the layer flags the draw and the pixels miss. In one encoder the
// second pass is split off with its transit CB; across two submits the
// barrier is the submit's.
func TestDepthReadOnlyAttachedAndSampled(t *testing.T) {
	for _, tc := range []struct {
		name        string
		sameEncoder bool
	}{
		{"same_encoder", true},
		{"two_submits", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			device, h := newValidatedDevice(t)
			skipDepthSamplingOnGL(t, h)
			rig := newDepthRig(t, device, true)
			_, depthView := depthTexture(t, device)
			dst, dstView := pixTexture(t, device, "depth-sampled-dst", TextureUsageRenderAttachment|TextureUsageCopySrc)
			defer releaseTextureView(dst, dstView)
			group := rig.group(t, device, depthView)
			st := readbackBuffer(t, device, "depth-sampled-st")

			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			clearDepth(t, enc, depthView, 0.25)
			if !tc.sameEncoder {
				finishAndSubmit(t, device, enc)
				enc, err = device.CreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("CreateCommandEncoder: %v", err)
				}
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
				ColorAttachments: []RenderPassColorAttachment{{View: dstView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: black}},
				DepthStencilAttachment: &RenderPassDepthStencilAttachment{
					View: depthView, DepthLoadOp: gputypes.LoadOpLoad, DepthStoreOp: gputypes.StoreOpStore,
					DepthReadOnly: true, StencilReadOnly: true,
				},
			})
			if err != nil {
				t.Fatalf("BeginRenderPass(read-only, sampled): %v", err)
			}
			pass.SetPipeline(rig.pipeline)
			pass.SetBindGroup(0, group, nil)
			pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
			if err := pass.End(); err != nil {
				t.Fatalf("End(read-only, sampled): %v", err)
			}
			copyOut(enc, dst, st)
			finishAndSubmit(t, device, enc)

			expectDepthPixels(t, "depth sampled by the pass it is attached read-only to", readback(t, st), 0.25)
			assertNoValidation(t, h)
		})
	}
}

// TestDepthSampledInLaterPass: a depth texture cleared to 0.25 by one pass
// is fetched (textureLoad) by a later colour-only pass. The barrier between
// them is DepthStencilWrite -> Resource, which on Vulkan has to land in
// DEPTH_STENCIL_READ_ONLY_OPTIMAL, the layout the sampled image descriptor
// of a depth view names.
func TestDepthSampledInLaterPass(t *testing.T) {
	for _, tc := range []struct {
		name        string
		sameEncoder bool
	}{
		{"same_encoder", true},
		{"two_submits", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			device, h := newValidatedDevice(t)
			skipDepthSamplingOnGL(t, h)
			rig := newDepthRig(t, device, false)
			_, depthView := depthTexture(t, device)
			dst, dstView := pixTexture(t, device, "depth-later-dst", TextureUsageRenderAttachment|TextureUsageCopySrc)
			defer releaseTextureView(dst, dstView)
			group := rig.group(t, device, depthView)
			st := readbackBuffer(t, device, "depth-later-st")

			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			clearDepth(t, enc, depthView, 0.25)
			if !tc.sameEncoder {
				finishAndSubmit(t, device, enc)
				enc, err = device.CreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("CreateCommandEncoder: %v", err)
				}
			}
			clearPass(t, enc, dstView, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) {
				pass.SetPipeline(rig.pipeline)
				pass.SetBindGroup(0, group, nil)
				pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
			})
			copyOut(enc, dst, st)
			finishAndSubmit(t, device, enc)

			expectDepthPixels(t, "depth fetched by a later pass", readback(t, st), 0.25)
			assertNoValidation(t, h)
		})
	}
}
