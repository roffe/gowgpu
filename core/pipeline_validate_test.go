//go:build !(js && wasm)

package core

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func TestValidateRenderPipelineVertexLimitsAndTargets(t *testing.T) {
	limits := gputypes.DefaultLimits()
	base := func() *hal.RenderPipelineDescriptor {
		return &hal.RenderPipelineDescriptor{
			Vertex: hal.VertexState{Module: mockShaderModule{}, EntryPoint: "vs", Buffers: []gputypes.VertexBufferLayout{{
				ArrayStride: 16,
				Attributes: []gputypes.VertexAttribute{
					{Format: gputypes.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
					{Format: gputypes.VertexFormatFloat32x2, Offset: 8, ShaderLocation: 1},
				},
			}}},
			Fragment: &hal.FragmentState{Module: mockShaderModule{}, EntryPoint: "fs", Targets: []gputypes.ColorTargetState{
				{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll},
			}},
			Multisample: gputypes.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		}
	}
	if err := ValidateRenderPipelineDescriptor(base(), limits, 0); err != nil {
		t.Fatalf("valid pipeline: %v", err)
	}
	for _, tc := range []struct {
		name   string
		change func(*hal.RenderPipelineDescriptor)
		want   CreateRenderPipelineErrorKind
	}{
		{"too many vertex buffers", func(d *hal.RenderPipelineDescriptor) {
			for range limits.MaxVertexBuffers {
				d.Vertex.Buffers = append(d.Vertex.Buffers, gputypes.VertexBufferLayout{ArrayStride: 4})
			}
		}, CreateRenderPipelineErrorTooManyVertexBuffers},
		{"location too large", func(d *hal.RenderPipelineDescriptor) {
			d.Vertex.Buffers[0].Attributes[1].ShaderLocation = limits.MaxVertexAttributes
		}, CreateRenderPipelineErrorVertexLocationTooLarge},
		{"location clash", func(d *hal.RenderPipelineDescriptor) {
			d.Vertex.Buffers[0].Attributes[1].ShaderLocation = 0
		}, CreateRenderPipelineErrorVertexLocationClash},
		{"misaligned attribute", func(d *hal.RenderPipelineDescriptor) {
			d.Vertex.Buffers[0].Attributes[1].Offset = 6
		}, CreateRenderPipelineErrorVertexAttributeMisaligned},
		{"too many vertex attributes", func(d *hal.RenderPipelineDescriptor) {
			var attrs []gputypes.VertexAttribute
			for i := range limits.MaxVertexAttributes + 1 {
				attrs = append(attrs, gputypes.VertexAttribute{Format: gputypes.VertexFormatFloat32, ShaderLocation: i})
			}
			d.Vertex.Buffers = []gputypes.VertexBufferLayout{{ArrayStride: 4, Attributes: attrs}}
		}, CreateRenderPipelineErrorTooManyVertexAttributes},
		{"unrenderable target", func(d *hal.RenderPipelineDescriptor) {
			d.Fragment.Targets[0].Format = gputypes.TextureFormatRGBA8Snorm
		}, CreateRenderPipelineErrorColorTargetNotRenderable},
		{"blending an integer target", func(d *hal.RenderPipelineDescriptor) {
			d.Fragment.Targets[0].Format = gputypes.TextureFormatRGBA32Uint
			d.Fragment.Targets[0].Blend = &gputypes.BlendState{}
		}, CreateRenderPipelineErrorColorTargetNotBlendable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			desc := base()
			tc.change(desc)
			var crpe *CreateRenderPipelineError
			if err := ValidateRenderPipelineDescriptor(desc, limits, 0); !errors.As(err, &crpe) || crpe.Kind != tc.want {
				t.Errorf("got %v, want kind %d", err, tc.want)
			}
		})
	}
}
