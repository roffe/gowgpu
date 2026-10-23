//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// greenAtHalfDepthShader draws a full-screen triangle at depth 0.5 in green.
const greenAtHalfDepthShader = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	return vec4<f32>(x, y, 0.5, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
	return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
`

// dsDraw is a depth/stencil attachment to clear and the pipeline state to
// draw greenAtHalfDepthShader against it with.
type dsDraw struct {
	format     gputypes.TextureFormat
	state      DepthStencilState
	depthClear float32
	stencil    uint32 // the stencil clear value
	ref        uint32 // the stencil reference
}

// drawGreenAgainst clears a pixTexture colour target red, clears a new
// pixSide square texture of d.format to d's values, draws the green
// triangle with d.state and returns the colour target's pixels: green where
// the depth and stencil tests passed.
func drawGreenAgainst(t *testing.T, device *Device, d dsDraw) []byte {
	t.Helper()
	ds, err := device.CreateTexture(&TextureDescriptor{
		Label: "depth-stencil", Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
		MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D, Format: d.format, Usage: TextureUsageRenderAttachment,
	})
	if err != nil {
		t.Fatalf("CreateTexture(%s): %v", d.format, err)
	}
	defer ds.Release()
	dsView, err := device.CreateTextureView(ds, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer dsView.Release()
	color, colorView := pixTexture(t, device, "color", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(color, colorView)

	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "green", WGSL: greenAtHalfDepthShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	state := d.state
	state.Format = d.format
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:        "green",
		Vertex:       VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive:    gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		DepthStencil: &state,
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()

	att := &RenderPassDepthStencilAttachment{View: dsView}
	if d.format.HasDepth() {
		att.DepthLoadOp, att.DepthStoreOp, att.DepthClearValue = gputypes.LoadOpClear, gputypes.StoreOpStore, d.depthClear
	}
	if d.format.HasStencil() {
		att.StencilLoadOp, att.StencilStoreOp, att.StencilClearValue = gputypes.LoadOpClear, gputypes.StoreOpStore, d.stencil
	}
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments:       []RenderPassColorAttachment{{View: colorView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: red}},
		DepthStencilAttachment: att,
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	pass.SetPipeline(pipe)
	pass.SetStencilReference(d.ref)
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	staging := readbackBuffer(t, device, "staging")
	enc.CopyTextureToBuffer(color, staging, []BufferTextureCopy{{
		BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
		TextureBase:  ImageCopyTexture{Texture: color},
		Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)
	return readback(t, staging)
}

// lessThanCleared tests the triangle's depth against the cleared one.
var lessThanCleared = DepthStencilState{DepthCompare: gputypes.CompareFunctionLess,
	StencilFront: StencilFaceState{Compare: gputypes.CompareFunctionAlways},
	StencilBack:  StencilFaceState{Compare: gputypes.CompareFunctionAlways}}

// expectDepthTested draws against format's attachment cleared behind and in
// front of the triangle.
func expectDepthTested(t *testing.T, device *Device, format gputypes.TextureFormat) {
	t.Helper()
	expectPixels(t, "cleared behind the triangle", drawGreenAgainst(t, device, dsDraw{format: format, state: lessThanCleared, depthClear: 0.75}), 0, 255, 0, 255)
	expectPixels(t, "cleared in front of the triangle", drawGreenAgainst(t, device, dsDraw{format: format, state: lessThanCleared, depthClear: 0.25}), 255, 0, 0, 255)
}

// TestDepthFormatsTestDepth draws against every depth format's attachment.
// GL made Depth32Float as the unorm GL_DEPTH_COMPONENT32, which ES has no
// such format for.
func TestDepthFormatsTestDepth(t *testing.T) {
	device, _ := newValidatedDevice(t)
	for _, format := range []gputypes.TextureFormat{
		gputypes.TextureFormatDepth16Unorm, gputypes.TextureFormatDepth24Plus,
		gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureFormatDepth32Float,
	} {
		t.Run(format.String(), func(t *testing.T) { expectDepthTested(t, device, format) })
	}
}

// TestDepth32FloatStencil8TestsDepth: GL made Depth32FloatStencil8 with a
// transfer type its internal format doesn't take, so the texture had no
// storage on any GL.
func TestDepth32FloatStencil8TestsDepth(t *testing.T) {
	device, _ := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureDepth32FloatStencil8))
	expectDepthTested(t, device, gputypes.TextureFormatDepth32FloatStencil8)
}

// skipStencil8OnVulkan skips a test drawing against a Stencil8 attachment
// on Vulkan, where NVIDIA's driver loses the device now and then doing so
// with gowgpu's S8_UINT attachment.
func skipStencil8OnVulkan(t *testing.T, h *captureHandler) {
	t.Helper()
	if h.backend == gputypes.BackendVulkan {
		t.Skip("drawing against an S8_UINT attachment loses the Vulkan device on NVIDIA now and then")
	}
}

// TestStencil8Attachment draws against a Stencil8 attachment, which GL made
// as RGBA8 and attached to the stencil attachment point, an incomplete
// framebuffer that drew nothing.
func TestStencil8Attachment(t *testing.T) {
	device, h := newValidatedDevice(t)
	skipStencil8OnVulkan(t, h)
	equal := DepthStencilState{DepthCompare: gputypes.CompareFunctionAlways,
		StencilFront:    StencilFaceState{Compare: gputypes.CompareFunctionEqual},
		StencilBack:     StencilFaceState{Compare: gputypes.CompareFunctionEqual},
		StencilReadMask: 0xff, StencilWriteMask: 0xff}
	expectPixels(t, "stencil equal to the reference", drawGreenAgainst(t, device, dsDraw{format: gputypes.TextureFormatStencil8, state: equal}), 0, 255, 0, 255)
	expectPixels(t, "stencil not equal to the reference", drawGreenAgainst(t, device, dsDraw{format: gputypes.TextureFormatStencil8, state: equal, ref: 1}), 255, 0, 0, 255)
}

// TestStencilClearValue clears stencil to 1 and tests against references 1
// and 0: GL cleared every stencil to 0, glClearStencil never set.
func TestStencilClearValue(t *testing.T) {
	device, h := newValidatedDevice(t)
	equal := DepthStencilState{DepthCompare: gputypes.CompareFunctionAlways,
		StencilFront:    StencilFaceState{Compare: gputypes.CompareFunctionEqual},
		StencilBack:     StencilFaceState{Compare: gputypes.CompareFunctionEqual},
		StencilReadMask: 0xff, StencilWriteMask: 0xff}
	for _, format := range []gputypes.TextureFormat{gputypes.TextureFormatStencil8, gputypes.TextureFormatDepth24PlusStencil8} {
		t.Run(format.String(), func(t *testing.T) {
			if format == gputypes.TextureFormatStencil8 {
				skipStencil8OnVulkan(t, h)
			}
			expectPixels(t, "reference 1", drawGreenAgainst(t, device, dsDraw{format: format, state: equal, depthClear: 1, stencil: 1, ref: 1}), 0, 255, 0, 255)
			expectPixels(t, "reference 0", drawGreenAgainst(t, device, dsDraw{format: format, state: equal, depthClear: 1, stencil: 1}), 255, 0, 0, 255)
		})
	}
}

// beyondFarShader draws a full-screen triangle at depth 1.5, past the far
// plane, in green.
const beyondFarShader = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 1.5, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
`

// TestUnclippedDepth draws a triangle past the far plane with depth
// clipping and without: GL clipped it either way, GL_DEPTH_CLAMP never set
// although depth-clip-control was offered.
func TestUnclippedDepth(t *testing.T) {
	device, _ := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureDepthClipControl))
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "beyond-far", WGSL: beyondFarShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	for _, unclipped := range []bool{true, false} {
		pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
			Label:     "beyond-far",
			Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
			Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList, UnclippedDepth: unclipped},
			Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
				Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
		})
		if err != nil {
			t.Fatalf("CreateRenderPipeline: %v", err)
		}
		tex, view := pixTexture(t, device, "beyond-far", TextureUsageRenderAttachment|TextureUsageCopySrc)
		st := readbackBuffer(t, device, "beyond-far-st")
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		clearPass(t, enc, view, gputypes.LoadOpClear, red, func(p *RenderPassEncoder) {
			p.SetPipeline(pipe)
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		})
		copyOut(enc, tex, st)
		finishAndSubmit(t, device, enc)
		if unclipped {
			expectPixels(t, "unclipped depth", readback(t, st), 0, 255, 0, 255)
		} else {
			expectPixels(t, "clipped depth", readback(t, st), 255, 0, 0, 255)
		}
		releaseTextureView(tex, view)
		pipe.Release()
	}
}

// TestDepthAndStencilCopyToBuffer clears a depth or stencil attachment and
// copies the aspect to a buffer, which GL advertises on desktop contexts.
func TestDepthAndStencilCopyToBuffer(t *testing.T) {
	device, h := newValidatedDevice(t)
	for _, tc := range []struct {
		format gputypes.TextureFormat
		aspect gputypes.TextureAspect
		want   func([]byte) bool
	}{
		{gputypes.TextureFormatDepth32Float, gputypes.TextureAspectDepthOnly, func(b []byte) bool {
			return binary.LittleEndian.Uint32(b) == math.Float32bits(0.5)
		}},
		{gputypes.TextureFormatDepth16Unorm, gputypes.TextureAspectDepthOnly, func(b []byte) bool {
			v := binary.LittleEndian.Uint16(b)
			return v == 32767 || v == 32768
		}},
		{gputypes.TextureFormatStencil8, gputypes.TextureAspectStencilOnly, func(b []byte) bool { return b[0] == 7 }},
		{gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureAspectStencilOnly, func(b []byte) bool { return b[0] == 7 }},
	} {
		t.Run(tc.format.String(), func(t *testing.T) {
			tex, err := device.CreateTexture(&TextureDescriptor{
				Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}, Format: tc.format,
				Usage: TextureUsageRenderAttachment | TextureUsageCopySrc, MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D,
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
			ds := &RenderPassDepthStencilAttachment{View: view}
			if tc.format.HasDepth() {
				ds.DepthLoadOp, ds.DepthStoreOp, ds.DepthClearValue = gputypes.LoadOpClear, gputypes.StoreOpStore, 0.5
			}
			if tc.format.HasStencil() {
				ds.StencilLoadOp, ds.StencilStoreOp, ds.StencilClearValue = gputypes.LoadOpClear, gputypes.StoreOpStore, 7
			}
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{DepthStencilAttachment: ds})
			if err != nil {
				t.Fatalf("BeginRenderPass: %v", err)
			}
			if err := pass.End(); err != nil {
				t.Fatalf("End: %v", err)
			}
			st := readbackBuffer(t, device, "depth-copy")
			enc.CopyTextureToBuffer(tex, st, []BufferTextureCopy{{
				BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
				TextureBase:  ImageCopyTexture{Texture: tex, Aspect: tc.aspect},
				Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
			}})
			finishAndSubmit(t, device, enc)
			if got := readback(t, st); !tc.want(got) {
				t.Errorf("copied % x", got[:8])
			}
			assertNoValidation(t, h)
		})
	}
}
