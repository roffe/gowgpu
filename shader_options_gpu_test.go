//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// runU32Compute runs one invocation of cs_main of wgsl with constants over
// a storage buffer of n u32s holding in, and returns them after.
func runU32Compute(t *testing.T, device *Device, wgsl string, constants map[string]float64, in []uint32) []uint32 {
	t.Helper()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer layout.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{layout}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "u32-compute", WGSL: wgsl})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "u32-compute", Layout: pl, Module: module, EntryPoint: "cs_main", Constants: constants})
	if err != nil {
		skipWithoutComputeShaders(t, err)
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipe.Release()
	size := uint64(len(in) * 4)
	data, err := device.CreateBuffer(&BufferDescriptor{Label: "u32-data", Size: size, Usage: BufferUsageStorage | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer data.Release()
	raw := make([]byte, size)
	for i, v := range in {
		binary.LittleEndian.PutUint32(raw[i*4:], v)
	}
	if err := device.Queue().WriteBuffer(data, 0, raw); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	read, err := device.CreateBuffer(&BufferDescriptor{Label: "u32-read", Size: size, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer read.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: data, Size: size}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cp, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cp.SetPipeline(pipe)
	cp.SetBindGroup(0, group, nil)
	cp.Dispatch(1, 1, 1)
	if err := cp.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	enc.CopyBufferToBuffer(data, 0, read, 0, size)
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := read.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := read.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	out := make([]uint32, len(in))
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(rng.Bytes()[i*4:])
	}
	if err := read.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

// oobShader fills a function-local array with 99 and reads it at an index
// from the buffer.
const oobShader = `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1) fn cs_main() {
	var a: array<u32, 4>;
	for (var i = 0u; i < 4u; i++) {
		a[i] = 99u;
	}
	data[1] = a[data[0]];
}
`

// TestOutOfBoundsIndexStaysInBounds reads a four-element local array at
// index 1000, which WGSL makes an element of the array or zero. Vulkan
// compiled WGSL without bounds checks, and NVIDIA read 4000.
func TestOutOfBoundsIndexStaysInBounds(t *testing.T) {
	device, h := newValidatedDevice(t)
	if got := runU32Compute(t, device, oobShader, nil, []uint32{1000, 1})[1]; got != 99 && got != 0 {
		t.Errorf("a[1000] of a four-element array of 99 read %d, want 99 or 0", got)
	}
	assertNoValidation(t, h)
}

// overrideShader stores its overrides.
const overrideShader = `
override answer: u32 = 1u;
override required: u32;
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1) fn cs_main() {
	data[0] = answer;
	data[1] = required;
}
`

// TestComputePipelineConstants sets one override with a default and one
// without through the pipeline's constants. Vulkan compiled every module
// with the defaults, so the first stayed 1 and the module failed for the
// second, and GL compiled it without them, which naga refuses.
func TestComputePipelineConstants(t *testing.T) {
	device, h := newValidatedDevice(t)
	got := runU32Compute(t, device, overrideShader, map[string]float64{"answer": 42, "required": 7}, []uint32{0, 0})
	if got[0] != 42 || got[1] != 7 {
		t.Errorf("overrides read %v, want [42 7]", got)
	}
	assertNoValidation(t, h)
}

// pointShader draws a point at the pixel in the bottom-left corner of a
// 4x4 target.
const pointShader = `
@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(-0.75, -0.75, 0.0, 1.0); }
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
`

// TestPointListDrawsOnePixel draws one point, which WebGPU makes one
// pixel. Vulkan never wrote PointSize, which a point list pipeline has to
// (VUID-VkGraphicsPipelineCreateInfo-topology-08773).
func TestPointListDrawsOnePixel(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "point", WGSL: pointShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "point",
		Layout:    pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyPointList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	tex, view := pixTexture(t, device, "point", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "point-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.Draw(gputypes.DrawArgs{VertexCount: 1, InstanceCount: 1})
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	pix := readback(t, st)
	lit := 0
	for i := 0; i < len(pix); i += 4 {
		if pix[i+1] == 255 {
			lit++
		}
	}
	if lit != 1 || pix[(3*pixSide)*4+1] != 255 {
		t.Errorf("a point lit %d pixels, the bottom-left one %v; want just that one", lit, pix[(3*pixSide)*4:(3*pixSide)*4+4])
	}
	assertNoValidation(t, h)
}
