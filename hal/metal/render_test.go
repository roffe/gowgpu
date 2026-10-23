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
// into an RGBA8 target, with layout, or an empty one when it is nil.
func renderTestPipeline(t *testing.T, device *Device, wgsl string, layout *PipelineLayout, primitive gputypes.PrimitiveState) *RenderPipeline {
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
		Vertex:    hal.VertexState{Module: module, EntryPoint: "vs_main"},
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
