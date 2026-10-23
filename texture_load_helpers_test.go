//go:build !(js && wasm)

package wgpu

import (
	"context"
	"strconv"
	"strings"
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

// layerLoadShader is texelLoadShader for a layer of a texture_2d_array.
const layerLoadShader = `
@group(0) @binding(0) var t: texture_2d_array<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	return vec4<f32>(x, y, 0.5, 1.0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	return textureLoad(t, vec2<i32>(pos.xy), LAYER, 0);
}
`

// loadPixels draws view into a new width by height RGBA8 target with
// textureLoad, a pixel a texel, and returns the target's pixels, rows packed.
func loadPixels(t *testing.T, device *Device, view *TextureView, width, height uint32) []byte {
	t.Helper()
	return loadPixelsWith(t, device, view, gputypes.TextureViewDimension2D, texelLoadShader, width, height)
}

// loadLayerPixels is loadPixels for a layer of a 2D array view.
func loadLayerPixels(t *testing.T, device *Device, view *TextureView, layer int, width, height uint32) []byte {
	t.Helper()
	shader := strings.Replace(layerLoadShader, "LAYER", strconv.Itoa(layer), 1)
	return loadPixelsWith(t, device, view, gputypes.TextureViewDimension2DArray, shader, width, height)
}

func loadPixelsWith(t *testing.T, device *Device, view *TextureView, dimension gputypes.TextureViewDimension, shader string, width, height uint32) []byte {
	t.Helper()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: dimension}},
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
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "load", WGSL: shader})
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

// sampleAndLoadShader samples ta with s and loads tb, and shows tb: ta only
// stays in the shader.
const sampleAndLoadShader = `
@group(0) @binding(0) var ta: texture_2d<f32>;
@group(0) @binding(1) var tb: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	return vec4<f32>(x, y, 0.5, 1.0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	let a = textureSample(ta, s, pos.xy / 4.0);
	let b = textureLoad(tb, vec2<i32>(pos.xy), 0);
	return vec4<f32>(b.rgb + a.rgb * 0.0, 1.0);
}
`

// TestTextureLoadedWithoutSampler loads a texture that has no sampler next
// to one sampled with a sampler. Below GLSL 4.20, where the texture units are
// set after linking, GL set only the sampled textures', so the loaded one
// read texture unit 0, the sampled texture.
func TestTextureLoadedWithoutSampler(t *testing.T) {
	device, _ := newValidatedDevice(t)
	tex := func(c gputypes.Color) (*Texture, *TextureView) {
		t.Helper()
		tx, view := pixTexture(t, device, "texture", TextureUsageRenderAttachment|TextureUsageTextureBinding)
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		clearPass(t, enc, view, gputypes.LoadOpClear, c, nil)
		finishAndSubmit(t, device, enc)
		return tx, view
	}
	ta, taView := tex(red)
	defer releaseTextureView(ta, taView)
	tb, tbView := tex(green)
	defer releaseTextureView(tb, tbView)
	sampler, err := device.CreateSampler(&SamplerDescriptor{MagFilter: gputypes.FilterModeNearest, MinFilter: gputypes.FilterModeNearest, LodMaxClamp: 32})
	if err != nil {
		t.Fatalf("CreateSampler: %v", err)
	}
	defer sampler.Release()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 1, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 2, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
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
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "sample-and-load", WGSL: sampleAndLoadShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label: "sample-and-load", Layout: pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{
		{Binding: 0, TextureView: taView}, {Binding: 1, TextureView: tbView}, {Binding: 2, Sampler: sampler},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()
	target, view := pixTexture(t, device, "target", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(target, view)
	st := readbackBuffer(t, device, "target-staging")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetBindGroup(0, group, nil)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, target, st)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "the loaded texture", readback(t, st), 0, 255, 0, 255)
}

// clampSampleShader samples tb past its right edge with s, ta only staying
// in the shader.
const clampSampleShader = `
@group(0) @binding(0) var ta: texture_2d<f32>;
@group(0) @binding(1) var tb: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.5, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
	let a = textureSample(ta, s, vec2<f32>(0.5, 0.5));
	let b = textureSample(tb, s, vec2<f32>(1.25, 0.5));
	return vec4<f32>(b.rgb + a.rgb * 0.0, 1.0);
}
`

// TestUnusedSamplerBindsNowhere samples a texture past its edge with a
// clamping sampler, next to a repeating sampler of the same bind group the
// shader doesn't use. GL bound the unused sampler to the texture unit
// numbered like its sampler slot, which was the sampled texture's, and the
// texture repeated.
func TestUnusedSamplerBindsNowhere(t *testing.T) {
	device, _ := newValidatedDevice(t)
	newTex := func(texels []byte) (*Texture, *TextureView) {
		t.Helper()
		tex, err := device.CreateTexture(&TextureDescriptor{
			Size: Extent3D{Width: 2, Height: 1, DepthOrArrayLayers: 1}, Format: TextureFormatRGBA8Unorm,
			Usage: TextureUsageTextureBinding | TextureUsageCopyDst, MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D,
		})
		if err != nil {
			t.Fatalf("CreateTexture: %v", err)
		}
		if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, texels,
			&ImageDataLayout{BytesPerRow: 8, RowsPerImage: 1}, &Extent3D{Width: 2, Height: 1, DepthOrArrayLayers: 1}); err != nil {
			t.Fatalf("WriteTexture: %v", err)
		}
		view, err := device.CreateTextureView(tex, nil)
		if err != nil {
			t.Fatalf("CreateTextureView: %v", err)
		}
		return tex, view
	}
	ta, taView := newTex([]byte{255, 0, 0, 255, 255, 0, 0, 255})
	defer releaseTextureView(ta, taView)
	tb, tbView := newTex([]byte{0, 255, 0, 255, 0, 0, 255, 255}) // green, then blue
	defer releaseTextureView(tb, tbView)
	newSampler := func(mode gputypes.AddressMode) *Sampler {
		t.Helper()
		s, err := device.CreateSampler(&SamplerDescriptor{AddressModeU: mode, AddressModeV: mode, AddressModeW: mode,
			MagFilter: gputypes.FilterModeNearest, MinFilter: gputypes.FilterModeNearest, LodMaxClamp: 32})
		if err != nil {
			t.Fatalf("CreateSampler: %v", err)
		}
		return s
	}
	clamp, repeat := newSampler(gputypes.AddressModeClampToEdge), newSampler(gputypes.AddressModeRepeat)
	defer clamp.Release()
	defer repeat.Release()
	tex := &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}
	smp := &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: tex},
		{Binding: 1, Visibility: ShaderStageFragment, Texture: tex},
		{Binding: 2, Visibility: ShaderStageFragment, Sampler: smp},
		{Binding: 3, Visibility: ShaderStageFragment, Sampler: smp},
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
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "clamp-sample", WGSL: clampSampleShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label: "clamp-sample", Layout: pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{
		{Binding: 0, TextureView: taView}, {Binding: 1, TextureView: tbView}, {Binding: 2, Sampler: clamp}, {Binding: 3, Sampler: repeat},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()
	target, view := pixTexture(t, device, "target", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(target, view)
	st := readbackBuffer(t, device, "target-staging")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetBindGroup(0, group, nil)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, target, st)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "past the edge with the clamping sampler", readback(t, st), 0, 0, 255, 255)
}
