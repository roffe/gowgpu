//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// vertexStorageShader draws a triangle from its vertex buffer, in the colour
// the vertex shader reads from the same buffer as read-only storage.
const vertexStorageShader = `
@group(0) @binding(0) var<storage, read> data: array<vec4<f32>>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(pos, 0.0, 1.0);
    o.color = data[3];
    return o;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return in.color;
}
`

// A buffer a copy fills and a render pass then reads as its vertex buffer
// and as read-only storage gets one barrier into both reads: on DX12 the
// read-only storage state is an SRV one, which the vertex buffer state goes
// with, where the UAV state does not.
func TestVertexAndReadOnlyStorageOfOneBuffer(t *testing.T) {
	device, h := newValidatedDevice(t)
	if !device.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsVertexStorage) {
		t.Skip("no storage buffers in vertex shaders on this adapter")
	}
	queue := device.Queue()

	// Three vec2 positions, then the colour at vec4 index 3.
	data := make([]byte, 64)
	for i, f := range []float32{-1, -1, 3, -1, -1, 3} {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(f))
	}
	for i, f := range []float32{0, 1, 0, 1} {
		binary.LittleEndian.PutUint32(data[48+i*4:], math.Float32bits(f))
	}
	src, err := device.CreateBuffer(&BufferDescriptor{Label: "vs-src", Size: 64, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer src.Release()
	if err := queue.WriteBuffer(src, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	both, err := device.CreateBuffer(&BufferDescriptor{Label: "vs-both", Size: 64, Usage: BufferUsageVertex | BufferUsageStorage | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer both.Release()

	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "vertex-storage", WGSL: vertexStorageShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: gputypes.ShaderStageVertex, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	layout, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer layout.Release()
	pipeline, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Layout: layout,
		Vertex: VertexState{Module: shader, EntryPoint: "vs_main", Buffers: []gputypes.VertexBufferLayout{{
			ArrayStride: 8, StepMode: gputypes.VertexStepModeVertex,
			Attributes: []gputypes.VertexAttribute{{Format: gputypes.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0}},
		}}},
		Fragment: &FragmentState{Module: shader, EntryPoint: "fs_main", Targets: []gputypes.ColorTargetState{
			{Format: TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll},
		}},
		Primitive:   gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Multisample: gputypes.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
	})
	if err != nil {
		t.Skipf("no render pipeline that reads storage in the vertex stage: %v", err)
	}
	defer pipeline.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: both}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	target, view := pixTexture(t, device, "vs-target", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(target, view)
	staging := readbackBuffer(t, device, "vs-staging")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToBuffer(src, 0, both, 0, 64)
	clearPass(t, enc, view, gputypes.LoadOpClear, black, func(p *RenderPassEncoder) {
		p.SetPipeline(pipeline)
		p.SetBindGroup(0, group, nil)
		p.SetVertexBuffer(0, both, 0)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, target, staging)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "drawn from one buffer read two ways", readback(t, staging), 0, 255, 0, 255)
	assertNoValidation(t, h)
}
