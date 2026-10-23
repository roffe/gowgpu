//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

const vertexInstanceShader = `
@vertex
fn vs_main(@location(0) pos: vec2<f32>, @location(1) shift: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos + shift, 0.5, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
`

// Vertex and index buffers are checked where they are set, for usage, slot,
// offset alignment and bounds, and at draws, for a buffer at every slot the
// pipeline reads and for the vertices, instances and indices the draw reads.
func TestVertexAndIndexBuffersAreChecked(t *testing.T) {
	device, h := newValidatedDevice(t)

	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: vertexInstanceShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	pipeline, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Vertex: VertexState{Module: shader, EntryPoint: "vs_main", Buffers: []gputypes.VertexBufferLayout{
			{ArrayStride: 8, StepMode: gputypes.VertexStepModeVertex, Attributes: []gputypes.VertexAttribute{
				{Format: gputypes.VertexFormatFloat32x2, ShaderLocation: 0},
			}},
			{ArrayStride: 8, StepMode: gputypes.VertexStepModeInstance, Attributes: []gputypes.VertexAttribute{
				{Format: gputypes.VertexFormatFloat32x2, ShaderLocation: 1},
			}},
		}},
		Fragment: &FragmentState{Module: shader, EntryPoint: "fs_main", Targets: []gputypes.ColorTargetState{
			{Format: TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll},
		}},
		Primitive:   gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Multisample: gputypes.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipeline.Release()

	buffer := func(size uint64, usage gputypes.BufferUsage) *Buffer {
		b, err := device.CreateBuffer(&BufferDescriptor{Size: size, Usage: usage})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		t.Cleanup(b.Release)
		return b
	}
	vertices := buffer(24, BufferUsageVertex)  // 3 vertices
	instances := buffer(8, BufferUsageVertex)  // 1 instance
	indices := buffer(12, BufferUsageIndex)    // 6 Uint16 indices
	copyOnly := buffer(24, BufferUsageCopyDst) // no Vertex or Index usage
	color, colorView := pixTexture(t, device, "vertex-index-color", TextureUsageRenderAttachment)
	defer releaseTextureView(color, colorView)

	bound := func(p *RenderPassEncoder) {
		p.SetVertexBuffer(0, vertices, 0)
		p.SetVertexBuffer(1, instances, 0)
		p.SetIndexBuffer(indices, gputypes.IndexFormatUint16, 0)
	}
	slotLimit := device.Limits().MaxVertexBuffers
	for _, tc := range []struct {
		name    string
		record  func(p *RenderPassEncoder)
		wantErr string
	}{
		{"draw", func(p *RenderPassEncoder) {
			bound(p)
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		}, ""},
		{"indexed draw", func(p *RenderPassEncoder) {
			bound(p)
			p.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 6, InstanceCount: 1})
		}, ""},
		{"slot the pipeline reads without buffer", func(p *RenderPassEncoder) {
			p.SetVertexBuffer(1, instances, 0)
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		}, "slot 0"},
		{"vertex buffer without Vertex usage", func(p *RenderPassEncoder) { p.SetVertexBuffer(0, copyOnly, 0) }, "BufferUsageVertex"},
		{"vertex buffer offset past the end", func(p *RenderPassEncoder) { p.SetVertexBuffer(0, vertices, 28) }, "past the end"},
		{"unaligned vertex buffer offset", func(p *RenderPassEncoder) { p.SetVertexBuffer(0, vertices, 2) }, "multiple of 4"},
		{"vertex buffer slot over the limit", func(p *RenderPassEncoder) { p.SetVertexBuffer(slotLimit, vertices, 0) }, "limit"},
		{"more vertices than the buffer holds", func(p *RenderPassEncoder) {
			bound(p)
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1, FirstVertex: 1})
		}, "vertex 4"},
		{"vertices past the offset", func(p *RenderPassEncoder) {
			bound(p)
			p.SetVertexBuffer(0, vertices, 8)
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		}, "holds 2"},
		{"more instances than the buffer holds", func(p *RenderPassEncoder) {
			bound(p)
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 2})
		}, "instance 2"},
		{"more indices than the buffer holds", func(p *RenderPassEncoder) {
			bound(p)
			p.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 6, InstanceCount: 1, FirstIndex: 1})
		}, "index 7"},
		{"indices past the offset", func(p *RenderPassEncoder) {
			bound(p)
			p.SetIndexBuffer(indices, gputypes.IndexFormatUint16, 4)
			p.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 5, InstanceCount: 1})
		}, "holds 4"},
		{"indexed draw of more instances than the buffer holds", func(p *RenderPassEncoder) {
			bound(p)
			p.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 3, InstanceCount: 2})
		}, "instance 2"},
		{"index buffer without Index usage", func(p *RenderPassEncoder) {
			p.SetIndexBuffer(vertices, gputypes.IndexFormatUint16, 0)
		}, "BufferUsageIndex"},
		{"index offset not a multiple of the index size", func(p *RenderPassEncoder) {
			p.SetIndexBuffer(indices, gputypes.IndexFormatUint32, 2)
		}, "multiple"},
		{"index offset past the end", func(p *RenderPassEncoder) {
			p.SetIndexBuffer(indices, gputypes.IndexFormatUint16, 14)
		}, "past the end"},
		{"undefined index format", func(p *RenderPassEncoder) {
			p.SetIndexBuffer(indices, gputypes.IndexFormatUndefined, 0)
		}, "index format"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
				ColorAttachments: []RenderPassColorAttachment{{View: colorView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
			})
			if err != nil {
				t.Fatalf("BeginRenderPass: %v", err)
			}
			pass.SetPipeline(pipeline)
			tc.record(pass)
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
