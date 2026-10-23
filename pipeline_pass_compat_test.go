//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

const solidShader = `
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[i], 0.5, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
`

// SetPipeline refuses a pipeline made for other attachments than the pass
// has, another color format, sample count or depth format, and one that
// writes depth into a read-only depth attachment.
func TestSetPipelineChecksThePass(t *testing.T) {
	device, h := newValidatedDevice(t)

	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: solidShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	pipeline := func(format gputypes.TextureFormat, samples uint32, depth *DepthStencilState) *RenderPipeline {
		p, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
			Vertex: VertexState{Module: shader, EntryPoint: "vs_main"},
			Fragment: &FragmentState{Module: shader, EntryPoint: "fs_main", Targets: []gputypes.ColorTargetState{
				{Format: format, WriteMask: gputypes.ColorWriteMaskAll},
			}},
			Primitive:    gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
			DepthStencil: depth,
			Multisample:  gputypes.MultisampleState{Count: samples, Mask: 0xFFFFFFFF},
		})
		if err != nil {
			t.Fatalf("CreateRenderPipeline: %v", err)
		}
		t.Cleanup(p.Release)
		return p
	}
	readDepth := &DepthStencilState{Format: gputypes.TextureFormatDepth32Float, DepthCompare: gputypes.CompareFunctionLessEqual}
	writeDepth := &DepthStencilState{Format: gputypes.TextureFormatDepth32Float, DepthCompare: gputypes.CompareFunctionLessEqual, DepthWriteEnabled: true}

	color, colorView := pixTexture(t, device, "compat-color", TextureUsageRenderAttachment)
	defer releaseTextureView(color, colorView)
	depthTex, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: gputypes.TextureFormatDepth32Float, Usage: TextureUsageRenderAttachment,
	})
	if err != nil {
		t.Fatalf("CreateTexture(depth): %v", err)
	}
	defer depthTex.Release()
	depthView, err := device.CreateTextureView(depthTex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView(depth): %v", err)
	}
	defer depthView.Release()

	for _, tc := range []struct {
		name     string
		pipeline *RenderPipeline
		depth    *RenderPassDepthStencilAttachment
		wantErr  string
	}{
		{name: "matching", pipeline: pipeline(TextureFormatRGBA8Unorm, 1, nil)},
		{name: "other color format", pipeline: pipeline(gputypes.TextureFormatRGBA16Float, 1, nil), wantErr: "format"},
		{name: "other sample count", pipeline: pipeline(TextureFormatRGBA8Unorm, 4, nil), wantErr: "sample count"},
		{name: "depth the pass doesn't have", pipeline: pipeline(TextureFormatRGBA8Unorm, 1, readDepth), wantErr: "depth/stencil format"},
		{name: "reading a read-only depth", pipeline: pipeline(TextureFormatRGBA8Unorm, 1, readDepth),
			depth: &RenderPassDepthStencilAttachment{View: depthView, DepthReadOnly: true}},
		{name: "writing a read-only depth", pipeline: pipeline(TextureFormatRGBA8Unorm, 1, writeDepth),
			depth: &RenderPassDepthStencilAttachment{View: depthView, DepthReadOnly: true}, wantErr: "read-only"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
				ColorAttachments:       []RenderPassColorAttachment{{View: colorView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
				DepthStencilAttachment: tc.depth,
			})
			if err != nil {
				t.Fatalf("BeginRenderPass: %v", err)
			}
			pass.SetPipeline(tc.pipeline)
			pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
			_ = pass.End()
			cb, err := enc.Finish()
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("Finish: %v", err)
			case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
				t.Errorf("Finish: %v, want an error about %q", err, tc.wantErr)
			}
			if err == nil {
				if _, err := device.Queue().Submit(cb); err != nil {
					t.Fatalf("Submit: %v", err)
				}
			}
		})
	}
	assertNoValidation(t, h)
}
