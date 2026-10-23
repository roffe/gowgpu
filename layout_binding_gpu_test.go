//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
)

// Pipelines whose layouts hold more than their shaders use. WebGPU lets a
// layout list bindings a shader never declares; the shader's resources must
// still be the ones bound at their group and binding.

const fullscreenVS = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
`

// layoutRig makes bind group layouts, pipeline layouts and groups for one
// test, released when it ends.
type layoutRig struct {
	t      *testing.T
	device *Device
}

func (r layoutRig) bgl(entries ...gputypes.BindGroupLayoutEntry) *BindGroupLayout {
	r.t.Helper()
	l, err := r.device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: entries})
	if err != nil {
		r.t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	r.t.Cleanup(l.Release)
	return l
}

func (r layoutRig) layout(bgls ...*BindGroupLayout) *PipelineLayout {
	r.t.Helper()
	l, err := r.device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: bgls})
	if err != nil {
		r.t.Fatalf("CreatePipelineLayout: %v", err)
	}
	r.t.Cleanup(l.Release)
	return l
}

func (r layoutRig) group(l *BindGroupLayout, entries ...BindGroupEntry) *BindGroup {
	r.t.Helper()
	g, err := r.device.CreateBindGroup(&BindGroupDescriptor{Layout: l, Entries: entries})
	if err != nil {
		r.t.Fatalf("CreateBindGroup: %v", err)
	}
	r.t.Cleanup(g.Release)
	return g
}

func (r layoutRig) module(wgsl string) *ShaderModule {
	r.t.Helper()
	m, err := r.device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: wgsl})
	if err != nil {
		r.t.Fatalf("CreateShaderModule: %v", err)
	}
	r.t.Cleanup(m.Release)
	return m
}

// buffer makes a 256-byte buffer holding v at its start.
func (r layoutRig) buffer(usage gputypes.BufferUsage, v [4]float32) *Buffer {
	r.t.Helper()
	b, err := r.device.CreateBuffer(&BufferDescriptor{Size: 256, Usage: usage | BufferUsageCopyDst})
	if err != nil {
		r.t.Fatalf("CreateBuffer: %v", err)
	}
	r.t.Cleanup(b.Release)
	if err := r.device.Queue().WriteBuffer(b, 0, unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), 16)); err != nil {
		r.t.Fatalf("WriteBuffer: %v", err)
	}
	return b
}

// renderPipeline makes a pipeline drawing fullscreenVS and fs_main of fs
// into pixTexture's format.
func (r layoutRig) renderPipeline(layout *PipelineLayout, fs string) *RenderPipeline {
	r.t.Helper()
	m := r.module(fullscreenVS + fs)
	p, err := r.device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Layout:    layout,
		Vertex:    VertexState{Module: m, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: m, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		r.t.Fatalf("CreateRenderPipeline: %v", err)
	}
	r.t.Cleanup(p.Release)
	return p
}

func (r layoutRig) computePipeline(layout *PipelineLayout, wgsl string) *ComputePipeline {
	r.t.Helper()
	p, err := r.device.CreateComputePipeline(&ComputePipelineDescriptor{Layout: layout, Module: r.module(wgsl), EntryPoint: "cs_main"})
	if err != nil {
		if strings.Contains(err.Error(), "ComputeShaders") {
			r.t.Skipf("no compute shaders on this adapter: %v", err)
		}
		r.t.Fatalf("CreateComputePipeline: %v", err)
	}
	r.t.Cleanup(p.Release)
	return p
}

// draw renders record's draws into a pixTexture and returns its pixels.
func (r layoutRig) draw(record func(*RenderPassEncoder)) []byte {
	r.t.Helper()
	tex, view := pixTexture(r.t, r.device, "layout-target", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(r.t, r.device, "layout-st")
	enc, err := r.device.CreateCommandEncoder(nil)
	if err != nil {
		r.t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(r.t, enc, view, gputypes.LoadOpClear, black, record)
	copyOut(enc, tex, st)
	finishAndSubmit(r.t, r.device, enc)
	return readback(r.t, st)
}

// dispatch runs record's dispatches and returns the first vec4 of each of
// bufs.
func (r layoutRig) dispatch(record func(*ComputePassEncoder), bufs ...*Buffer) [][]float32 {
	r.t.Helper()
	enc, err := r.device.CreateCommandEncoder(nil)
	if err != nil {
		r.t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		r.t.Fatalf("BeginComputePass: %v", err)
	}
	record(pass)
	if err := pass.End(); err != nil {
		r.t.Fatalf("ComputePass.End: %v", err)
	}
	sts := make([]*Buffer, len(bufs))
	for i, b := range bufs {
		sts[i] = readbackBuffer(r.t, r.device, "layout-dispatch-st")
		enc.CopyBufferToBuffer(b, 0, sts[i], 0, 16)
	}
	finishAndSubmit(r.t, r.device, enc)
	out := make([][]float32, len(bufs))
	for i, st := range sts {
		out[i] = readFloats(r.t, st, 4)
	}
	return out
}

func uniformEntry(binding uint32, vis gputypes.ShaderStages) gputypes.BindGroupLayoutEntry {
	return gputypes.BindGroupLayoutEntry{Binding: binding, Visibility: vis, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}}
}

func storageEntry(binding uint32) gputypes.BindGroupLayoutEntry {
	return gputypes.BindGroupLayoutEntry{Binding: binding, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}}
}

// TestShaderSkipsLayoutEntries draws and dispatches with shaders that use
// group 1 of a layout whose group 0 they never declare.
func TestShaderSkipsLayoutEntries(t *testing.T) {
	t.Run("render", func(t *testing.T) {
		device, h := newValidatedDevice(t)
		r := layoutRig{t, device}
		bgl0 := r.bgl(uniformEntry(0, ShaderStageVertex|ShaderStageFragment))
		bgl1 := r.bgl(uniformEntry(0, ShaderStageFragment))
		pipe := r.renderPipeline(r.layout(bgl0, bgl1), `
@group(1) @binding(0) var<uniform> material: vec4<f32>;
@fragment fn fs_main() -> @location(0) vec4<f32> { return material; }
`)
		camera := r.group(bgl0, BindGroupEntry{Binding: 0, Buffer: r.buffer(BufferUsageUniform, [4]float32{1, 0, 0, 1}), Size: 16})
		material := r.group(bgl1, BindGroupEntry{Binding: 0, Buffer: r.buffer(BufferUsageUniform, [4]float32{0, 1, 0, 1}), Size: 16})
		pix := r.draw(func(p *RenderPassEncoder) {
			p.SetPipeline(pipe)
			p.SetBindGroup(0, camera, nil)
			p.SetBindGroup(1, material, nil)
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		})
		expectPixels(t, "material colour", pix, 0, 255, 0, 255)
		assertNoValidation(t, h)
	})
	t.Run("compute", func(t *testing.T) {
		device, h := newValidatedDevice(t)
		r := layoutRig{t, device}
		bgl0, bgl1 := r.bgl(storageEntry(0)), r.bgl(storageEntry(0))
		pipe := r.computePipeline(r.layout(bgl0, bgl1), `
@group(1) @binding(0) var<storage, read_write> out: vec4<f32>;
@compute @workgroup_size(1) fn cs_main() { out = vec4<f32>(1.0, 2.0, 3.0, 4.0); }
`)
		decoy := r.buffer(BufferUsageStorage|BufferUsageCopySrc, [4]float32{})
		out := r.buffer(BufferUsageStorage|BufferUsageCopySrc, [4]float32{})
		g0 := r.group(bgl0, BindGroupEntry{Binding: 0, Buffer: decoy, Size: 16})
		g1 := r.group(bgl1, BindGroupEntry{Binding: 0, Buffer: out, Size: 16})
		got := r.dispatch(func(p *ComputePassEncoder) {
			p.SetPipeline(pipe)
			p.SetBindGroup(0, g0, nil)
			p.SetBindGroup(1, g1, nil)
			p.Dispatch(1, 1, 1)
		}, out, decoy)
		expectFloats4(t, "group 1 buffer", got[0], [4]float32{1, 2, 3, 4})
		expectFloats4(t, "group 0 buffer", got[1], [4]float32{})
		assertNoValidation(t, h)
	})
}

// TestArrayLengthInRenderStages reads arrayLength of a storage array bound
// to the vertex and fragment stages. Metal compiled naga's `_buffer_sizes`
// argument into render stages but never bound it.
func TestArrayLengthInRenderStages(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL {
		t.Skip("desktop GL declares storage buffers as uniform blocks (README known issues)")
	}
	r := layoutRig{t, device}
	bgl := r.bgl(gputypes.BindGroupLayoutEntry{Binding: 0, Visibility: ShaderStageVertex | ShaderStageFragment,
		Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}})
	m := r.module(`
@group(0) @binding(0) var<storage, read> data: array<vec4<f32>>;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) @interpolate(flat) n: u32 };
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
	var out: VOut;
	out.pos = vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
	out.n = arrayLength(&data);
	return out;
}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
	return vec4<f32>(f32(in.n) / 255.0, f32(arrayLength(&data)) / 255.0, data[3].z, 1.0);
}
`)
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Layout:    r.layout(bgl),
		Vertex:    VertexState{Module: m, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: m, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	data := r.buffer(BufferUsageStorage, [4]float32{})
	if err := device.Queue().WriteBuffer(data, 48, unsafe.Slice((*byte)(unsafe.Pointer(&[4]float32{0, 0, 1, 0})), 16)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	group := r.group(bgl, BindGroupEntry{Binding: 0, Buffer: data, Size: 64})
	pix := r.draw(func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetBindGroup(0, group, nil)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	expectPixels(t, "arrayLength in the vertex and fragment stage", pix, 4, 4, 255, 255)
	assertNoValidation(t, h)
}
