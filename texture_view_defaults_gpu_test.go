//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// viewDefaultsShader draws layer 1 of an array view on the left half of
// the target and a 2D view on the right half.
const viewDefaultsShader = `
@group(0) @binding(0) var arr: texture_2d_array<f32>;
@group(0) @binding(1) var one: texture_2d<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	let p = vec2<i32>(pos.xy);
	if (pos.x < 2.0) {
		return textureLoad(arr, p, 1, 0);
	}
	return textureLoad(one, p, 0);
}
`

// TestTextureViewDefaults samples a two-layer texture, layer 0 red and
// layer 1 green, through its default view as a texture_2d_array, and
// through a 2D view of layer 1 that leaves the layer count unset. WebGPU
// makes the first an array of both layers and the second one layer;
// Vulkan made a 2D view of every layer for the first, which sampled
// layer 0, and of all remaining layers for the second. GL sampled the
// array texture through the 2D view, which read nothing.
func TestTextureViewDefaults(t *testing.T) {
	device, h := newValidatedDevice(t)
	skipWithoutGLTextureViews(t, device, h)
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         "two-layers",
		Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 2},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageTextureBinding | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	for layer, c := range [][4]byte{{255, 0, 0, 255}, {0, 255, 0, 255}} {
		data := make([]byte, pixSide*pixSide*4)
		for i := 0; i < len(data); i += 4 {
			copy(data[i:], c[:])
		}
		if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex, Origin: Origin3D{Z: uint32(layer)}}, data,
			&ImageDataLayout{BytesPerRow: pixSide * 4, RowsPerImage: pixSide}, &Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}); err != nil {
			t.Fatalf("WriteTexture(layer %d): %v", layer, err)
		}
	}
	arrayView, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView(default): %v", err)
	}
	defer arrayView.Release()
	layerView, err := device.CreateTextureView(tex, &TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2D, BaseArrayLayer: 1})
	if err != nil {
		t.Fatalf("CreateTextureView(layer 1): %v", err)
	}
	defer layerView.Release()

	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "view-defaults", WGSL: viewDefaultsShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2DArray}},
		{Binding: 1, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "view-defaults",
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
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{
		{Binding: 0, TextureView: arrayView},
		{Binding: 1, TextureView: layerView},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	dst, dstView := pixTexture(t, device, "view-defaults-dst", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(dst, dstView)
	st := readbackBuffer(t, device, "view-defaults-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, dstView, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetBindGroup(0, group, nil)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, dst, st)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	pix := readback(t, st)
	var left, right []byte
	for y := range pixSide {
		row := pix[y*pixSide*4 : (y+1)*pixSide*4]
		left = append(left, row[:pixSide*2]...)
		right = append(right, row[pixSide*2:]...)
	}
	expectPixels(t, "layer 1 of the default view", left, 0, 255, 0, 255)
	expectPixels(t, "2D view of layer 1", right, 0, 255, 0, 255)
	assertNoValidation(t, h)
}

// skipWithoutGLTextureViews skips on a GL context without texture views,
// which the GL backend samples a view of some of a texture's layers or mip
// levels through: without them it refuses such a view to sample, and an
// image unit that stores to one level of a texture while a view of another
// is sampled stores nothing.
func skipWithoutGLTextureViews(t *testing.T, device *Device, h *captureHandler) {
	t.Helper()
	if h.backend != gputypes.BackendGL {
		return
	}
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label: "view-probe", Size: Extent3D{Width: 1, Height: 1, DepthOrArrayLayers: 2}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageTextureBinding,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	view, err := device.CreateTextureView(tex, &TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2D, BaseArrayLayer: 1, ArrayLayerCount: 1})
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer view.Release()
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, TextureView: view}}})
	if err != nil && strings.Contains(err.Error(), "needs a GL texture view") {
		t.Skipf("no texture views on this GL context: %v", err)
	}
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	group.Release()
}
