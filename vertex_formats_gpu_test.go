//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// integerAttribShader draws a full-screen triangle green when every integer
// attribute holds what the vertex buffer has, and red with a bit per wrong
// attribute in the red channel when one doesn't.
const integerAttribShader = `
struct VSOut {
	@builtin(position) pos: vec4<f32>,
	@location(0) @interpolate(flat) bad: u32,
}
@vertex fn vs_main(@builtin(vertex_index) vi: u32,
	@location(0) u32x4: vec4<u32>, @location(1) s32x2: vec2<i32>,
	@location(2) u8x4: vec4<u32>, @location(3) s16x2: vec2<i32>, @location(4) s8x4: vec4<i32>) -> VSOut {
	var out: VSOut;
	out.pos = vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
	var bad = 0u;
	if any(u32x4 != vec4<u32>(4000000000u, 7u, 65536u, 1u)) { bad |= 1u; }
	if any(s32x2 != vec2<i32>(-5, 123456789)) { bad |= 2u; }
	if any(u8x4 != vec4<u32>(1u, 2u, 250u, 255u)) { bad |= 4u; }
	if any(s16x2 != vec2<i32>(-300, 32767)) { bad |= 8u; }
	if any(s8x4 != vec4<i32>(-128, -1, 0, 127)) { bad |= 16u; }
	out.bad = bad;
	return out;
}
@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	if in.bad != 0u {
		return vec4<f32>(f32(in.bad) / 255.0, 0.0, 0.0, 1.0);
	}
	return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
`

// TestIntegerVertexAttributes draws with uint and sint attributes of 8, 16
// and 32 bits, values a float can't hold among them, which the vertex
// shader reads as integers. GL handed every attribute to the shader as
// floats.
func TestIntegerVertexAttributes(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "integer attributes", WGSL: integerAttribShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	const stride = 36
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:  "integer attributes",
		Layout: pl,
		Vertex: VertexState{Module: module, EntryPoint: "vs_main", Buffers: []gputypes.VertexBufferLayout{{
			ArrayStride: stride, StepMode: gputypes.VertexStepModeVertex,
			Attributes: []gputypes.VertexAttribute{
				{Format: gputypes.VertexFormatUint32x4, Offset: 0, ShaderLocation: 0},
				{Format: gputypes.VertexFormatSint32x2, Offset: 16, ShaderLocation: 1},
				{Format: gputypes.VertexFormatUint8x4, Offset: 24, ShaderLocation: 2},
				{Format: gputypes.VertexFormatSint16x2, Offset: 28, ShaderLocation: 3},
				{Format: gputypes.VertexFormatSint8x4, Offset: 32, ShaderLocation: 4},
			},
		}}},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()

	vertex := make([]byte, stride)
	le := binary.LittleEndian
	for i, v := range []uint32{4000000000, 7, 65536, 1} {
		le.PutUint32(vertex[i*4:], v)
	}
	for i, v := range []int32{-5, 123456789} {
		le.PutUint32(vertex[16+i*4:], uint32(v))
	}
	copy(vertex[24:], []byte{1, 2, 250, 255})
	for i, v := range []int16{-300, 32767} {
		le.PutUint16(vertex[28+i*2:], uint16(v))
	}
	for i, v := range []int8{-128, -1, 0, 127} {
		vertex[32+i] = byte(v)
	}
	data := make([]byte, 0, 3*stride)
	for range 3 {
		data = append(data, vertex...)
	}
	vb, err := device.CreateBuffer(&BufferDescriptor{Label: "integer attributes", Size: uint64(len(data)), Usage: BufferUsageVertex | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer vb.Release()
	if err := device.Queue().WriteBuffer(vb, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	tex, view := pixTexture(t, device, "integer attributes", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "integer attributes")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetVertexBuffer(0, vb, 0)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "integer attributes", readback(t, st), 0, 255, 0, 255)
	assertNoValidation(t, h)
}
