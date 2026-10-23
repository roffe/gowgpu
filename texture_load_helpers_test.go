//go:build !(js && wasm)

package wgpu

import (
	"context"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// texelLoadShader writes the texel of t under each pixel, at the view's first
// level.
const texelLoadShader = `
@group(0) @binding(0) var t: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	return vec4<f32>(x, y, 0.5, 1.0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	return textureLoad(t, vec2<i32>(pos.xy), 0);
}
`

// loadPixels draws view into a new width by height RGBA8 target with
// textureLoad, a pixel a texel, and returns the target's pixels, rows packed.
func loadPixels(t *testing.T, device *Device, view *TextureView, width, height uint32) []byte {
	t.Helper()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
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
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "load", WGSL: texelLoadShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label: "load", Layout: pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, TextureView: view}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	target, err := device.CreateTexture(&TextureDescriptor{
		Label: "load-target", Size: Extent3D{Width: width, Height: height, DepthOrArrayLayers: 1},
		MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm,
		Usage: TextureUsageRenderAttachment | TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer target.Release()
	targetView, err := device.CreateTextureView(target, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer targetView.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{View: targetView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, group, nil)
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	pitch := alignUp(width*4, copyBytesPerRowAlignment)
	size := uint64(pitch * height)
	staging, err := device.CreateBuffer(&BufferDescriptor{Label: "load-staging", Size: size, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer staging.Release()
	enc.CopyTextureToBuffer(target, staging, []BufferTextureCopy{{
		BufferLayout: ImageDataLayout{BytesPerRow: pitch, RowsPerImage: height},
		TextureBase:  ImageCopyTexture{Texture: target},
		Size:         Extent3D{Width: width, Height: height, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := staging.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	defer staging.Unmap()
	rng, err := staging.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	data := rng.Bytes()
	out := make([]byte, 0, width*height*4)
	for y := range height {
		out = append(out, data[y*pitch:y*pitch+width*4]...)
	}
	return out
}
