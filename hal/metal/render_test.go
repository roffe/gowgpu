//go:build darwin && !(js && wasm)

package metal

import (
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

const renderTestSide, renderTestPitch = 4, 256

// renderTestPipeline makes a pipeline drawing vs_main and fs_main of wgsl
// into an RGBA8 target, with layout, or an empty one when it is nil, and
// vertex buffers.
func renderTestPipeline(t *testing.T, device *Device, wgsl string, layout *PipelineLayout, primitive gputypes.PrimitiveState, buffers ...gputypes.VertexBufferLayout) *RenderPipeline {
	t.Helper()
	if layout == nil {
		raw, err := device.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{})
		if err != nil {
			t.Fatalf("CreatePipelineLayout failed: %v", err)
		}
		layout = raw.(*PipelineLayout)
	}
	module, err := device.CreateShaderModule(&hal.ShaderModuleDescriptor{Source: hal.ShaderSource{WGSL: wgsl}})
	if err != nil {
		t.Fatalf("CreateShaderModule failed: %v", err)
	}
	defer module.Destroy()
	raw, err := device.CreateRenderPipeline(&hal.RenderPipelineDescriptor{
		Layout:    layout,
		Vertex:    hal.VertexState{Module: module, EntryPoint: "vs_main", Buffers: buffers},
		Primitive: primitive,
		Fragment: &hal.FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline failed: %v", err)
	}
	pipeline := raw.(*RenderPipeline)
	t.Cleanup(pipeline.Destroy)
	return pipeline
}

// renderTestDraw records record's draws in a pass on a 4x4 RGBA8 target
// cleared to black, and returns the target's pixels, rows packed.
func renderTestDraw(t *testing.T, device *Device, queue *Queue, record func(*RenderPassEncoder)) []byte {
	t.Helper()
	raw, err := device.CreateTexture(&hal.TextureDescriptor{
		Size:          hal.Extent3D{Width: renderTestSide, Height: renderTestSide, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRGBA8Unorm,
		Usage:         gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture failed: %v", err)
	}
	target := raw.(*Texture)
	defer target.Destroy()
	view, err := device.CreateTextureView(target, nil)
	if err != nil {
		t.Fatalf("CreateTextureView failed: %v", err)
	}
	defer view.Destroy()
	readback := createMetalTextureCopyTestReadback(t, device, renderTestPitch*renderTestSide)
	defer readback.Destroy()

	encoder := &CommandEncoder{device: device}
	if err := encoder.BeginEncoding("render test"); err != nil {
		t.Fatalf("BeginEncoding failed: %v", err)
	}
	pass := encoder.BeginRenderPass(&hal.RenderPassDescriptor{ColorAttachments: []hal.RenderPassColorAttachment{{
		View:       view,
		LoadOp:     gputypes.LoadOpClear,
		StoreOp:    gputypes.StoreOpStore,
		ClearValue: gputypes.Color{A: 1},
	}}})
	record(pass.(*RenderPassEncoder))
	pass.End()
	encoder.CopyTextureToBuffer(target, readback, []hal.BufferTextureCopy{{
		BufferLayout: hal.ImageDataLayout{BytesPerRow: renderTestPitch, RowsPerImage: renderTestSide},
		TextureBase:  hal.ImageCopyTexture{Texture: target},
		Size:         hal.Extent3D{Width: renderTestSide, Height: renderTestSide, DepthOrArrayLayers: 1},
	}})
	submitMetalTextureCopyTestEncoder(t, device, queue, encoder)

	data := unsafe.Slice((*byte)(readback.Contents()), renderTestPitch*renderTestSide)
	pixels := make([]byte, 0, renderTestSide*renderTestSide*4)
	for y := range renderTestSide {
		pixels = append(pixels, data[y*renderTestPitch:y*renderTestPitch+renderTestSide*4]...)
	}
	return pixels
}

// TestPointListDrawsOnePixel draws one point into the top left pixel: a
// WebGPU point is one pixel, and Metal leaves the size of a point the vertex
// shader doesn't write undefined.
func TestPointListDrawsOnePixel(t *testing.T) {
	device, queue := newMetalTextureCopyTestDevice(t)
	pipeline := renderTestPipeline(t, device, `
@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(-0.75, 0.75, 0.0, 1.0); }
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
`, nil, gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyPointList})
	pixels := renderTestDraw(t, device, queue, func(pass *RenderPassEncoder) {
		pass.SetPipeline(pipeline)
		pass.Draw(gputypes.DrawArgs{VertexCount: 1, InstanceCount: 1})
	})
	for i := 0; i < len(pixels); i += 4 {
		want := []byte{0, 0, 0, 255}
		if i == 0 {
			want = []byte{0, 255, 0, 255}
		}
		if got := pixels[i : i+4]; string(got) != string(want) {
			t.Errorf("pixel %d = %v, want %v", i/4, got, want)
		}
	}
}

// TestVertexPulling draws one point per pixel of the top row, its color from
// an attribute at offset 2, which Metal's vertex fetch doesn't take, in a
// vertex buffer bound at an offset that leaves room for two vertices: the
// points past them read zero, as WebGPU's robust vertex access allows.
func TestVertexPulling(t *testing.T) {
	device, queue := newMetalTextureCopyTestDevice(t)
	pipeline := renderTestPipeline(t, device, `
struct Out { @builtin(position) pos: vec4<f32>, @location(0) color: vec2<f32> }
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @location(0) color: vec2<f32>) -> Out {
	return Out(vec4<f32>(f32(vi) * 0.5 - 0.75, 0.75, 0.0, 1.0), color);
}
@fragment fn fs_main(@location(0) color: vec2<f32>) -> @location(0) vec4<f32> { return vec4<f32>(color, 1.0, 1.0); }
`, nil, gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyPointList}, gputypes.VertexBufferLayout{
		ArrayStride: 4,
		StepMode:    gputypes.VertexStepModeVertex,
		Attributes:  []gputypes.VertexAttribute{{Format: gputypes.VertexFormatUnorm8x2, Offset: 2, ShaderLocation: 0}},
	})
	vertices := []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0xff, 0, 0, 0, 0, 0xff}
	buffer := createMetalTextureCopyTestUpload(t, device, 0, vertices)
	defer buffer.Destroy()
	pixels := renderTestDraw(t, device, queue, func(pass *RenderPassEncoder) {
		pass.SetPipeline(pipeline)
		pass.SetVertexBuffer(0, buffer, 12)
		pass.Draw(gputypes.DrawArgs{VertexCount: 4, InstanceCount: 1})
	})
	want := [][]byte{{255, 0, 255, 255}, {0, 255, 255, 255}, {0, 0, 255, 255}, {0, 0, 255, 255}}
	for i := 0; i < len(pixels); i += 4 {
		w := []byte{0, 0, 0, 255}
		if i/4 < len(want) {
			w = want[i/4]
		}
		if got := pixels[i : i+4]; string(got) != string(w) {
			t.Errorf("pixel %d = %v, want %v", i/4, got, w)
		}
	}
}

// TestCompileOptionsPreserveInvariance checks the option @invariant needs,
// and that a shader using it builds and draws.
func TestCompileOptionsPreserveInvariance(t *testing.T) {
	device, queue := newMetalTextureCopyTestDevice(t)
	options := newCompileOptions()
	defer Release(options)
	if !MsgSendBool(options, Sel("preserveInvariance")) {
		t.Error("compile options do not preserve invariance")
	}
	pipeline := renderTestPipeline(t, device, `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @invariant @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
`, nil, gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList})
	pixels := renderTestDraw(t, device, queue, func(pass *RenderPassEncoder) {
		pass.SetPipeline(pipeline)
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	for i := 0; i < len(pixels); i += 4 {
		if got := pixels[i : i+4]; string(got) != string([]byte{0, 255, 0, 255}) {
			t.Fatalf("pixel %d = %v, want green", i/4, got)
		}
	}
}

// TestVertexBufferSlotsKeepOffBindGroups binds a vertex buffer at a slot
// the pipeline doesn't read, whose Metal buffer index is that of the 24th
// bind group buffer, which the fragment shader reads; and checks that a
// pipeline whose vertex buffers would reach its bind group buffers fails.
func TestVertexBufferSlotsKeepOffBindGroups(t *testing.T) {
	device, queue := newMetalTextureCopyTestDevice(t)
	const buffers = 24
	entries := make([]gputypes.BindGroupLayoutEntry, buffers)
	for i := range entries {
		entries[i] = gputypes.BindGroupLayoutEntry{Binding: uint32(i), Visibility: gputypes.ShaderStageFragment,
			Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}}
	}
	bgl, err := device.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{Entries: entries})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout failed: %v", err)
	}
	rawLayout, err := device.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{BindGroupLayouts: []hal.BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout failed: %v", err)
	}
	layout := rawLayout.(*PipelineLayout)

	color := createMetalTextureCopyTestUpload(t, device, 0, unsafe.Slice((*byte)(unsafe.Pointer(&[4]float32{0, 1, 0, 1})), 16))
	defer color.Destroy()
	junk := createMetalTextureCopyTestUpload(t, device, 0, make([]byte, 256))
	defer junk.Destroy()
	groupEntries := make([]gputypes.BindGroupEntry, buffers)
	for i := range groupEntries {
		groupEntries[i] = gputypes.BindGroupEntry{Binding: uint32(i),
			Resource: gputypes.BufferBinding{Buffer: uintptr(color.raw), Size: 16}}
	}
	group, err := device.CreateBindGroup(&hal.BindGroupDescriptor{Layout: bgl, Entries: groupEntries})
	if err != nil {
		t.Fatalf("CreateBindGroup failed: %v", err)
	}

	pipeline := renderTestPipeline(t, device, `
@group(0) @binding(23) var<uniform> color: vec4<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return color; }
`, layout, gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList})
	pixels := renderTestDraw(t, device, queue, func(pass *RenderPassEncoder) {
		pass.SetVertexBuffer(maxVertexBuffers-1-buffers+1, junk, 0)
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, group, nil)
		pass.SetVertexBuffer(maxVertexBuffers-1-buffers+1, junk, 0)
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	for i := 0; i < len(pixels); i += 4 {
		if got := pixels[i : i+4]; string(got) != string([]byte{0, 255, 0, 255}) {
			t.Fatalf("pixel %d = %v, want the uniform's green", i/4, got)
		}
	}

	module, err := device.CreateShaderModule(&hal.ShaderModuleDescriptor{Source: hal.ShaderSource{WGSL: `
@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }
`}})
	if err != nil {
		t.Fatalf("CreateShaderModule failed: %v", err)
	}
	defer module.Destroy()
	vertexBuffers := make([]gputypes.VertexBufferLayout, maxVertexBuffers-buffers)
	for i := range vertexBuffers {
		vertexBuffers[i] = gputypes.VertexBufferLayout{ArrayStride: 16, StepMode: gputypes.VertexStepModeVertex}
	}
	if p, err := device.CreateRenderPipeline(&hal.RenderPipelineDescriptor{
		Layout:    layout,
		Vertex:    hal.VertexState{Module: module, EntryPoint: "vs_main", Buffers: vertexBuffers},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
	}); err == nil {
		p.Destroy()
		t.Fatalf("a pipeline with %d vertex buffers and %d bind group buffers was made", len(vertexBuffers), buffers)
	}
}

// TestEmptyWorkIsNotEncoded draws and copies nothing, which WebGPU allows
// and Metal's validation refuses: none of it may reach Metal or fail the
// command buffer, and a real draw after it still lands.
func TestEmptyWorkIsNotEncoded(t *testing.T) {
	device, queue := newMetalTextureCopyTestDevice(t)
	pipeline := renderTestPipeline(t, device, `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
`, nil, gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList})
	pixels := renderTestDraw(t, device, queue, func(pass *RenderPassEncoder) {
		pass.SetPipeline(pipeline)
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 0})
		pass.Draw(gputypes.DrawArgs{VertexCount: 0, InstanceCount: 1})
		if pass.raw != 0 {
			t.Error("a draw of nothing made the render command encoder")
		}
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	for i := 0; i < len(pixels); i += 4 {
		if got := pixels[i : i+4]; string(got) != string([]byte{0, 255, 0, 255}) {
			t.Fatalf("pixel %d = %v, want green", i/4, got)
		}
	}

	texture := createMetalTextureCopyTestTexture(t, device, gputypes.TextureDimension2D, 4, 4, 1)
	defer texture.Destroy()
	buffer := createMetalTextureCopyTestReadback(t, device, 1024)
	defer buffer.Destroy()
	encoder := &CommandEncoder{device: device}
	if err := encoder.BeginEncoding("empty copies"); err != nil {
		t.Fatalf("BeginEncoding failed: %v", err)
	}
	encoder.CopyBufferToBuffer(buffer, buffer, []hal.BufferCopy{{SrcOffset: 0, DstOffset: 256, Size: 0}})
	for _, size := range []hal.Extent3D{{Width: 0, Height: 4, DepthOrArrayLayers: 1}, {Width: 4, Height: 0, DepthOrArrayLayers: 1}, {Width: 4, Height: 4}} {
		region := []hal.BufferTextureCopy{{BufferLayout: hal.ImageDataLayout{BytesPerRow: 256}, TextureBase: hal.ImageCopyTexture{Texture: texture}, Size: size}}
		encoder.CopyBufferToTexture(buffer, texture, region)
		encoder.CopyTextureToBuffer(texture, buffer, region)
		encoder.CopyTextureToTexture(texture, texture, []hal.TextureCopy{{SrcBase: hal.ImageCopyTexture{Texture: texture}, DstBase: hal.ImageCopyTexture{Texture: texture}, Size: size}})
		if err := queue.WriteTexture(&region[0].TextureBase, nil, &region[0].BufferLayout, &size); err != nil {
			t.Errorf("WriteTexture of %+v: %v", size, err)
		}
	}
	submitMetalTextureCopyTestEncoder(t, device, queue, encoder)
}

// TestOpenPassesEndWithTheEncoder leaves passes open, with and without a
// native encoder made, and discards or finishes the command encoder: every
// pass's encoder is ended and released with it, and a finished one
// submits.
func TestOpenPassesEndWithTheEncoder(t *testing.T) {
	device, queue := newMetalTextureCopyTestDevice(t)
	pipeline := renderTestPipeline(t, device, `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
`, nil, gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList})
	target, err := device.CreateTexture(&hal.TextureDescriptor{
		Size:          hal.Extent3D{Width: renderTestSide, Height: renderTestSide, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatRGBA8Unorm,
		Usage:         gputypes.TextureUsageRenderAttachment,
	})
	if err != nil {
		t.Fatalf("CreateTexture failed: %v", err)
	}
	defer target.Destroy()
	view, err := device.CreateTextureView(target, nil)
	if err != nil {
		t.Fatalf("CreateTextureView failed: %v", err)
	}
	defer view.Destroy()
	begin := func() *CommandEncoder {
		t.Helper()
		encoder := &CommandEncoder{device: device}
		if err := encoder.BeginEncoding("open pass"); err != nil {
			t.Fatalf("BeginEncoding failed: %v", err)
		}
		return encoder
	}
	renderPass := func(encoder *CommandEncoder, draw bool) *RenderPassEncoder {
		t.Helper()
		pass := encoder.BeginRenderPass(&hal.RenderPassDescriptor{ColorAttachments: []hal.RenderPassColorAttachment{{
			View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
		}}}).(*RenderPassEncoder)
		if draw {
			pass.SetPipeline(pipeline)
			pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		}
		return pass
	}

	for _, draw := range []bool{false, true} {
		encoder := begin()
		pass := renderPass(encoder, draw)
		encoder.DiscardEncoding()
		if pass.raw != 0 || pass.descriptor != 0 {
			t.Errorf("discarded render pass (drawn in: %v) kept encoder %#x, descriptor %#x", draw, pass.raw, pass.descriptor)
		}
	}
	encoder := begin()
	compute := encoder.BeginComputePass(nil).(*ComputePassEncoder)
	encoder.DiscardEncoding()
	if compute.raw != 0 {
		t.Errorf("discarded compute pass kept encoder %#x", compute.raw)
	}

	encoder = begin()
	pass := renderPass(encoder, true)
	submitMetalTextureCopyTestEncoder(t, device, queue, encoder)
	if pass.raw != 0 {
		t.Errorf("finished render pass kept encoder %#x", pass.raw)
	}
}
