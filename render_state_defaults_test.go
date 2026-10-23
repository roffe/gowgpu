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
