//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"strings"
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
)

// dynamicShader draws a full-screen triangle, or stores from a compute
// shader, in the colour of its uniform.
const dynamicShader = `
@group(0) @binding(0) var<uniform> color: vec4<f32>;
@group(0) @binding(1) var<storage, read_write> out: vec4<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return color; }
@compute @workgroup_size(1) fn cs_main() { out = color; }
`

// dynamicRig is a bind group with dynamic offsets on both its buffers, for
// dynamicShader: binding 0 a uniform buffer with red at 0 and green at 256,
// binding 1 a 512-byte storage buffer.
type dynamicRig struct {
	module     *ShaderModule
	layout     *PipelineLayout
	group      *BindGroup
	out        *Buffer
	red, green []byte
}

func newDynamicRig(t *testing.T, device *Device) *dynamicRig {
	t.Helper()
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "dynamic-offsets", WGSL: dynamicShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(module.Release)
	// Out of binding order on purpose: the offsets go by binding number.
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 1, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage, HasDynamicOffset: true}},
		{Binding: 0, Visibility: ShaderStageFragment | ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform, HasDynamicOffset: true}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	t.Cleanup(bgl.Release)
	layout, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	t.Cleanup(layout.Release)
	colors, err := device.CreateBuffer(&BufferDescriptor{Label: "colors", Size: 512, Usage: BufferUsageUniform | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(colors.Release)
	out, err := device.CreateBuffer(&BufferDescriptor{Label: "out", Size: 512, Usage: BufferUsageStorage | BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(out.Release)
	data := make([]float32, 128)
	copy(data[0:], []float32{1, 0, 0, 1})
	copy(data[64:], []float32{0, 1, 0, 1})
	raw := unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), 512)
	if err := device.Queue().WriteBuffer(colors, 0, raw); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{
		{Binding: 0, Buffer: colors, Size: 16},
		{Binding: 1, Buffer: out, Size: 16},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(group.Release)
	return &dynamicRig{module: module, layout: layout, group: group, out: out, red: raw[0:16], green: raw[256:272]}
}

// TestDynamicOffsetsDraw draws with one bind group at two dynamic offsets
// into its uniform buffer, each draw on one half of the target. DX12
// ignored dynamic offsets, so both halves were red.
func TestDynamicOffsetsDraw(t *testing.T) {
	device, h := newValidatedDevice(t)
	rig := newDynamicRig(t, device)
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "dynamic-offsets",
		Layout:    rig.layout,
		Vertex:    VertexState{Module: rig.module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: rig.module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	tex, view := pixTexture(t, device, "dynamic-offsets", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "dynamic-offsets-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	half := float32(pixSide / 2)
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		if h.backend == gputypes.BackendGL {
			p.SetPipeline(pipe) // GL binds a group against the pipeline set at the time, see the README's known issues
		}
		// Before the pipeline, so SetPipeline binds the group at these offsets.
		p.SetBindGroup(0, rig.group, []uint32{256, 0})
		p.SetPipeline(pipe)
		p.SetViewport(gputypes.Viewport{Width: half, Height: pixSide, MaxDepth: 1})
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		p.SetBindGroup(0, rig.group, []uint32{0, 0})
		p.SetViewport(gputypes.Viewport{X: half, Width: half, Height: pixSide, MaxDepth: 1})
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	pix := readback(t, st)
	var left, right []byte
	for y := range pixSide {
		row := pix[y*pixSide*4 : (y+1)*pixSide*4]
		left = append(left, row[:pixSide*2]...)
		right = append(right, row[pixSide*2:]...)
	}
	expectPixels(t, "draw at uniform offset 256", left, 0, 255, 0, 255)
	expectPixels(t, "draw at uniform offset 0", right, 255, 0, 0, 255)
	assertNoValidation(t, h)
}

// TestDynamicOffsetsDispatch stores the uniform into the storage buffer
// twice with one bind group, the offsets crossed: green into out[0], then
// red into out[256]. DX12 ignored dynamic offsets, so both stored red at 0.
func TestDynamicOffsetsDispatch(t *testing.T) {
	device, h := newValidatedDevice(t)
	rig := newDynamicRig(t, device)
	cpipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "dynamic-offsets", Layout: rig.layout, Module: rig.module, EntryPoint: "cs_main"})
	if err != nil {
		if strings.Contains(err.Error(), "ComputeShaders") {
			t.Skipf("no compute shaders on this adapter: %v", err)
		}
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer cpipe.Release()
	st := readbackBuffer(t, device, "dynamic-offsets-out")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cp, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cp.SetPipeline(cpipe)
	cp.SetBindGroup(0, rig.group, []uint32{256, 0})
	cp.Dispatch(1, 1, 1)
	cp.SetBindGroup(0, rig.group, []uint32{0, 256})
	cp.Dispatch(1, 1, 1)
	if err := cp.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	enc.CopyBufferToBuffer(rig.out, 0, st, 0, 512)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	// readback keeps 16 bytes of every 256-byte row: out[0:16], then out[256:272].
	stored := readback(t, st)
	if !bytes.Equal(stored[:16], rig.green) {
		t.Errorf("dispatch at offsets 256, 0 left % x at out[0], want the green at 256", stored[:16])
	}
	if !bytes.Equal(stored[16:32], rig.red) {
		t.Errorf("dispatch at offsets 0, 256 left % x at out[256], want the red at 0", stored[16:32])
	}
	assertNoValidation(t, h)
}
