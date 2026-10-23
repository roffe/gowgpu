//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// fullScreenGreenShader draws a full-screen triangle at depth 0.5 in green.
const fullScreenGreenShader = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	return vec4<f32>(x, y, 0.5, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
	return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
`

// newGreenPipeline makes a pipeline drawing fullScreenGreenShader into an
// RGBA8 target, with depthStencil.
func newGreenPipeline(t *testing.T, device *Device, depthStencil *DepthStencilState) *RenderPipeline {
	t.Helper()
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "green", WGSL: fullScreenGreenShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(module.Release)
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:        "green",
		Vertex:       VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive:    gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		DepthStencil: depthStencil,
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	t.Cleanup(pipe.Release)
	return pipe
}

// TestDepthPipelineWithZeroStencilState draws with a depth test and the
// stencil fields left at their zero values, which leave the stencil test
// off, on depth formats with and without a stencil plane.
func TestDepthPipelineWithZeroStencilState(t *testing.T) {
	for _, format := range []gputypes.TextureFormat{
		gputypes.TextureFormatDepth24Plus,
		gputypes.TextureFormatDepth24PlusStencil8,
		gputypes.TextureFormatDepth32Float,
	} {
		t.Run(format.String(), func(t *testing.T) {
			device, h := newValidatedDevice(t)
			color, colorView := pixTexture(t, device, "zero-stencil-color", TextureUsageRenderAttachment|TextureUsageCopySrc)
			defer releaseTextureView(color, colorView)
			depth, err := device.CreateTexture(&TextureDescriptor{
				Label:         "zero-stencil-depth",
				Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
				MipLevelCount: 1,
				SampleCount:   1,
				Dimension:     TextureDimension2D,
				Format:        format,
				Usage:         TextureUsageRenderAttachment,
			})
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			defer depth.Release()
			depthView, err := device.CreateTextureView(depth, nil)
			if err != nil {
				t.Fatalf("CreateTextureView: %v", err)
			}
			defer depthView.Release()

			pipe := newGreenPipeline(t, device, &DepthStencilState{Format: format, DepthWriteEnabled: true, DepthCompare: gputypes.CompareFunctionLess})
			staging := readbackBuffer(t, device, "zero-stencil-staging")
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			ds := &RenderPassDepthStencilAttachment{View: depthView, DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 1}
			if format.HasStencil() {
				ds.StencilLoadOp, ds.StencilStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
				ColorAttachments:       []RenderPassColorAttachment{{View: colorView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: black}},
				DepthStencilAttachment: ds,
			})
			if err != nil {
				t.Fatalf("BeginRenderPass: %v", err)
			}
			pass.SetPipeline(pipe)
			pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
			if err := pass.End(); err != nil {
				t.Fatalf("End: %v", err)
			}
			copyOut(enc, color, staging)
			finishAndSubmit(t, device, enc)
			expectPixels(t, "depth-tested draw", readback(t, staging), 0, 255, 0, 255)
			assertNoValidation(t, h)
		})
	}
}

// TestStencilReferenceStartsAtZeroEachPass sets a stencil reference in one
// pass and draws with a stencil test Equal against a stencil cleared to 0
// in the next pass of the same encoder, which starts at reference 0.
func TestStencilReferenceStartsAtZeroEachPass(t *testing.T) {
	device, h := newValidatedDevice(t)
	color, colorView := pixTexture(t, device, "stencil-ref-color", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(color, colorView)
	stencil, err := device.CreateTexture(&TextureDescriptor{
		Label:         "stencil-ref",
		Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        gputypes.TextureFormatDepth24PlusStencil8,
		Usage:         TextureUsageRenderAttachment,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer stencil.Release()
	stencilView, err := device.CreateTextureView(stencil, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer stencilView.Release()

	equal := StencilFaceState{Compare: gputypes.CompareFunctionEqual, FailOp: gputypes.StencilOperationKeep, DepthFailOp: gputypes.StencilOperationKeep, PassOp: gputypes.StencilOperationKeep}
	pipe := newGreenPipeline(t, device, &DepthStencilState{
		Format:           gputypes.TextureFormatDepth24PlusStencil8,
		DepthCompare:     gputypes.CompareFunctionAlways,
		StencilFront:     equal,
		StencilBack:      equal,
		StencilReadMask:  0xFF,
		StencilWriteMask: 0xFF,
	})
	staging := readbackBuffer(t, device, "stencil-ref-staging")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	for i := range 2 {
		pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
			ColorAttachments: []RenderPassColorAttachment{{View: colorView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: black}},
			DepthStencilAttachment: &RenderPassDepthStencilAttachment{
				View:        stencilView,
				DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 1,
				StencilLoadOp: gputypes.LoadOpClear, StencilStoreOp: gputypes.StoreOpStore,
			},
		})
		if err != nil {
			t.Fatalf("BeginRenderPass: %v", err)
		}
		pass.SetPipeline(pipe)
		if i == 0 {
			pass.SetStencilReference(1)
		}
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		if err := pass.End(); err != nil {
			t.Fatalf("End: %v", err)
		}
	}
	copyOut(enc, color, staging)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "draw with the reference a new pass starts with", readback(t, staging), 0, 255, 0, 255)
	assertNoValidation(t, h)
}

const sampleLevelOneShader = `
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.5, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
	return textureSampleLevel(t, s, vec2<f32>(0.5, 0.5), 1.0);
}
`

// TestSamplerZeroLodMaxClampDoesNotClamp samples level 1 of a red-over-green
// texture through a sampler with LodMaxClamp left at 0, WebGPU's default
// of 32 in Go, which reaches level 1.
func TestSamplerZeroLodMaxClampDoesNotClamp(t *testing.T) {
	device, h := newValidatedDevice(t)
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         "two-levels",
		Size:          Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		MipLevelCount: 2,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageTextureBinding | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	for level, c := range [][4]byte{{255, 0, 0, 255}, {0, 255, 0, 255}} {
		side := uint32(4 >> level)
		data := make([]byte, 0, side*side*4)
		for range side * side {
			data = append(data, c[:]...)
		}
		if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex, MipLevel: uint32(level)}, data,
			&ImageDataLayout{BytesPerRow: side * 4, RowsPerImage: side}, &Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}); err != nil {
			t.Fatalf("WriteTexture level %d: %v", level, err)
		}
	}
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer view.Release()
	sampler, err := device.CreateSampler(&SamplerDescriptor{})
	if err != nil {
		t.Fatalf("CreateSampler: %v", err)
	}
	defer sampler.Release()

	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 1, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer layout.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{layout}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "sample-level-one", WGSL: sampleLevelOneShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "sample-level-one",
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
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, TextureView: view}, {Binding: 1, Sampler: sampler}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	target, targetView := pixTexture(t, device, "sample-level-one", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(target, targetView)
	staging := readbackBuffer(t, device, "sample-level-one")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, targetView, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) {
		pass.SetPipeline(pipe)
		pass.SetBindGroup(0, group, nil)
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, target, staging)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "level 1", readback(t, staging), 0, 255, 0, 255)
	assertNoValidation(t, h)
}
