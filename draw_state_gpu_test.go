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
	if device.Limits().MaxStorageBuffersPerShaderStage == 0 {
		t.Skip("no storage buffers on this adapter")
	}
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
	// Out of binding order here too: GL took the offsets in entry order.
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{
		{Binding: 1, Buffer: out, Size: 16},
		{Binding: 0, Buffer: colors, Size: 16},
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

// strideShader draws its vertex buffer's positions in green.
const strideShader = `
@vertex fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(pos, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
`

// TestVertexStrideFollowsPipeline draws from a vertex buffer that has every
// position twice, so a 16-byte stride reads a full-screen triangle and an
// 8-byte or 0 stride a flat one that draws nothing. DX12 put the stride of
// the pipeline set at SetVertexBuffer in the buffer view: 0 when the buffer
// came before the pipeline, the old pipeline's when the pipeline changed.
func TestVertexStrideFollowsPipeline(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "stride", WGSL: strideShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	pipeline := func(stride uint64) *RenderPipeline {
		pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
			Label:  "stride",
			Layout: pl,
			Vertex: VertexState{Module: module, EntryPoint: "vs_main", Buffers: []gputypes.VertexBufferLayout{{
				ArrayStride: stride, StepMode: gputypes.VertexStepModeVertex,
				Attributes: []gputypes.VertexAttribute{{Format: gputypes.VertexFormatFloat32x2, ShaderLocation: 0}},
			}}},
			Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
			Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
				Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
		})
		if err != nil {
			t.Fatalf("CreateRenderPipeline(stride %d): %v", stride, err)
		}
		t.Cleanup(pipe.Release)
		return pipe
	}
	wide, narrow := pipeline(16), pipeline(8)
	pos := []float32{-1, -1, -1, -1, 3, -1, 3, -1, -1, 3, -1, 3}
	vb, err := device.CreateBuffer(&BufferDescriptor{Label: "positions", Size: uint64(len(pos) * 4), Usage: BufferUsageVertex | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer vb.Release()
	if err := device.Queue().WriteBuffer(vb, 0, unsafe.Slice((*byte)(unsafe.Pointer(&pos[0])), len(pos)*4)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	first, firstView := pixTexture(t, device, "stride-first", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(first, firstView)
	changed, changedView := pixTexture(t, device, "stride-changed", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(changed, changedView)
	firstSt := readbackBuffer(t, device, "stride-first-st")
	changedSt := readbackBuffer(t, device, "stride-changed-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	draw := gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1}
	clearPass(t, enc, firstView, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetVertexBuffer(0, vb, 0)
		p.SetPipeline(wide)
		p.Draw(draw)
	})
	copyOut(enc, first, firstSt)
	clearPass(t, enc, changedView, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetPipeline(narrow)
		p.SetVertexBuffer(0, vb, 0)
		p.Draw(draw) // flat
		p.SetPipeline(wide)
		p.Draw(draw)
	})
	copyOut(enc, changed, changedSt)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	expectPixels(t, "vertex buffer set before the pipeline", readback(t, firstSt), 0, 255, 0, 255)
	expectPixels(t, "vertex buffer kept across a stride change", readback(t, changedSt), 0, 255, 0, 255)
	assertNoValidation(t, h)
}

// instanceShader draws a full-screen triangle in its per-instance colour.
const instanceShader = `
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> }
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @location(0) color: vec4<f32>) -> VSOut {
	var out: VSOut;
	out.pos = vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
	out.color = color;
	return out;
}
@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> { return in.color; }
`

// TestDrawFirstInstance draws one instance starting at the second entry of
// a per-instance colour buffer, red then green. GL drew from instance 0
// whatever firstInstance said.
func TestDrawFirstInstance(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "first-instance", WGSL: instanceShader})
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
		Label:  "first-instance",
		Layout: pl,
		Vertex: VertexState{Module: module, EntryPoint: "vs_main", Buffers: []gputypes.VertexBufferLayout{{
			ArrayStride: 16, StepMode: gputypes.VertexStepModeInstance,
			Attributes: []gputypes.VertexAttribute{{Format: gputypes.VertexFormatFloat32x4, ShaderLocation: 0}},
		}}},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	colors := []float32{1, 0, 0, 1, 0, 1, 0, 1}
	vb, err := device.CreateBuffer(&BufferDescriptor{Label: "instance-colors", Size: uint64(len(colors) * 4), Usage: BufferUsageVertex | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer vb.Release()
	if err := device.Queue().WriteBuffer(vb, 0, unsafe.Slice((*byte)(unsafe.Pointer(&colors[0])), len(colors)*4)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	tex, view := pixTexture(t, device, "first-instance", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "first-instance-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetVertexBuffer(0, vb, 0)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1, FirstInstance: 1})
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)
	expectPixels(t, "draw from instance 1", readback(t, st), 0, 255, 0, 255)
	assertNoValidation(t, h)
}

// layoutChangeShader draws a full-screen triangle in the colour of group 1,
// reading group 0 too.
const layoutChangeShader = `
@group(0) @binding(0) var<uniform> zero: vec4<f32>;
@group(1) @binding(0) var<uniform> color: vec4<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return color + zero; }
`

// TestBindGroupsAcrossLayouts sets three bind groups before a pipeline whose
// layout has two, draws the left half, then sets a pipeline whose group 0
// layout is another one and only group 0 again, and draws the right half
// with the group 1 set before. Vulkan bound the third group with the first
// layout, which has no set for it (NVIDIA crashed in vkCmdBindDescriptorSets),
// and kept group 1 bound with the first layout, which the second does not
// match below it.
func TestBindGroupsAcrossLayouts(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "layout-change", WGSL: layoutChangeShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	uniform := func(visibility gputypes.ShaderStages) *BindGroupLayout {
		bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
			{Binding: 0, Visibility: visibility, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
		}})
		if err != nil {
			t.Fatalf("CreateBindGroupLayout: %v", err)
		}
		t.Cleanup(bgl.Release)
		return bgl
	}
	zeroA, zeroB, colorLayout := uniform(ShaderStageFragment), uniform(ShaderStageVertex|ShaderStageFragment), uniform(ShaderStageFragment)
	pipeline := func(zeroLayout *BindGroupLayout) *RenderPipeline {
		pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{zeroLayout, colorLayout}})
		if err != nil {
			t.Fatalf("CreatePipelineLayout: %v", err)
		}
		t.Cleanup(pl.Release)
		pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
			Label:     "layout-change",
			Layout:    pl,
			Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
			Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
			Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
				Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
		})
		if err != nil {
			t.Fatalf("CreateRenderPipeline: %v", err)
		}
		t.Cleanup(pipe.Release)
		return pipe
	}
	pipeA, pipeB := pipeline(zeroA), pipeline(zeroB)

	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "uniforms", Size: 512, Usage: BufferUsageUniform | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	green := []float32{0, 1, 0, 1}
	if err := device.Queue().WriteBuffer(buf, 256, unsafe.Slice((*byte)(unsafe.Pointer(&green[0])), 16)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	group := func(layout *BindGroupLayout, offset uint64) *BindGroup {
		g, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: buf, Offset: offset, Size: 16}}})
		if err != nil {
			t.Fatalf("CreateBindGroup: %v", err)
		}
		t.Cleanup(g.Release)
		return g
	}
	groupA, groupB, colorGroup, extra := group(zeroA, 0), group(zeroB, 0), group(colorLayout, 256), group(colorLayout, 0)

	tex, view := pixTexture(t, device, "layout-change", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "layout-change-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	half := float32(pixSide / 2)
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetBindGroup(0, groupA, nil)
		p.SetBindGroup(1, colorGroup, nil)
		p.SetBindGroup(2, extra, nil)
		p.SetPipeline(pipeA)
		p.SetViewport(gputypes.Viewport{Width: half, Height: pixSide, MaxDepth: 1})
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		p.SetPipeline(pipeB)
		p.SetBindGroup(0, groupB, nil)
		p.SetViewport(gputypes.Viewport{X: half, Width: half, Height: pixSide, MaxDepth: 1})
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	expectPixels(t, "draws with group 1 across a layout change", readback(t, st), 0, 255, 0, 255)
	assertNoValidation(t, h)
}

// instanceIndexShader draws a full-screen triangle green when its
// instance_index is 1, red otherwise.
const instanceIndexShader = `
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) @interpolate(flat) instance: u32 }
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
	var out: VSOut;
	out.pos = vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
	out.instance = ii;
	return out;
}
@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
	return select(vec4<f32>(1.0, 0.0, 0.0, 1.0), vec4<f32>(0.0, 1.0, 0.0, 1.0), in.instance == 1u);
}
`

// TestInstanceIndexStartsAtFirstInstance draws instance 1 alone, with Draw
// and with DrawIndexed: instance_index has to be 1. GL never set naga's
// naga_vs_first_instance uniform, so it was gl_InstanceID alone, which
// starts at 0 even in a base instance draw.
func TestInstanceIndexStartsAtFirstInstance(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "instance-index", WGSL: instanceIndexShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "instance-index",
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	indices := []uint16{0, 1, 2, 0}
	ib, err := device.CreateBuffer(&BufferDescriptor{Label: "indices", Size: 8, Usage: BufferUsageIndex | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer ib.Release()
	if err := device.Queue().WriteBuffer(ib, 0, unsafe.Slice((*byte)(unsafe.Pointer(&indices[0])), 8)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	for _, indexed := range []bool{false, true} {
		tex, view := pixTexture(t, device, "instance-index", TextureUsageRenderAttachment|TextureUsageCopySrc)
		st := readbackBuffer(t, device, "instance-index-st")
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
			p.SetPipeline(pipe)
			if indexed {
				p.SetIndexBuffer(ib, gputypes.IndexFormatUint16, 0)
				p.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 3, InstanceCount: 1, FirstInstance: 1})
			} else {
				p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1, FirstInstance: 1})
			}
		})
		copyOut(enc, tex, st)
		finishAndSubmit(t, device, enc)
		what := "Draw from instance 1"
		if indexed {
			what = "DrawIndexed from instance 1"
		}
		expectPixels(t, what, readback(t, st), 0, 255, 0, 255)
		releaseTextureView(tex, view)
	}
	assertNoValidation(t, h)
}

// stripRestartShader places strip vertices by vertex_index: a thin
// triangle down each side, and every vertex past 5, the restart index as a
// vertex, at the top centre.
const stripRestartShader = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	var pos = array<vec2<f32>, 6>(
		vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(-0.5, -1.0),
		vec2<f32>(0.5, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, -1.0));
	var p = vec2<f32>(0.0, 1.0);
	if vi < 6u {
		p = pos[vi];
	}
	return vec4<f32>(p, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
`

// TestStripRestartsAtTheLargestIndex draws a triangle strip of two
// triangles split by 0xFFFF: the middle of the target stays clear. GL never
// enabled primitive restart, so the strip went on through a vertex 65535
// and filled the middle.
func TestStripRestartsAtTheLargestIndex(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "strip-restart", WGSL: stripRestartShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	stripFormat := gputypes.IndexFormatUint16
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:  "strip-restart",
		Vertex: VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleStrip,
			StripIndexFormat: &stripFormat},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	indices := []uint16{0, 1, 2, 0xffff, 3, 4, 5, 0}
	ib, err := device.CreateBuffer(&BufferDescriptor{Label: "strip-indices", Size: 16, Usage: BufferUsageIndex | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer ib.Release()
	if err := device.Queue().WriteBuffer(ib, 0, unsafe.Slice((*byte)(unsafe.Pointer(&indices[0])), 16)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	tex, view := pixTexture(t, device, "strip-restart", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "strip-restart-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, red, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetIndexBuffer(ib, gputypes.IndexFormatUint16, 0)
		p.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 7, InstanceCount: 1})
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	pix := readback(t, st)
	at := func(x, y int) []byte { return pix[(y*pixSide+x)*4 : (y*pixSide+x)*4+4] }
	expectPixels(t, "the left triangle", at(0, 3), 0, 255, 0, 255)
	expectPixels(t, "the middle, between the triangles", append(at(1, 2), at(2, 2)...), 255, 0, 0, 255)
	assertNoValidation(t, h)
}

const indirectShader = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	var pos = array<vec2<f32>, 9>(
		vec2<f32>(-1.0, -1.0), vec2<f32>(-1.0, 1.0), vec2<f32>(-0.5, -1.0),
		vec2<f32>(0.5, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, -1.0),
		vec2<f32>(-0.5, -1.0), vec2<f32>(0.0, 1.0), vec2<f32>(0.5, -1.0));
	var p = vec2<f32>(0.0, 1.0);
	if vi < 9u {
		p = pos[vi];
	}
	return vec4<f32>(p, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }
`

// TestIndirectDraws draws a triangle at each end and one in the middle of
// the bottom row from indirect records: DrawIndirect from the fourth
// vertex, DrawIndexedIndirect through an index buffer bound at its start
// and through one bound past it. GL recorded no indirect draw at all.
func TestIndirectDraws(t *testing.T) {
	device, h := newValidatedDevice(t)
	skipRootConstantIndirectOnVkd3d(t, h)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "indirect", WGSL: indirectShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "indirect",
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	newBuffer := func(label string, usage gputypes.BufferUsage, data []uint32) *Buffer {
		buf, err := device.CreateBuffer(&BufferDescriptor{Label: label, Size: uint64(4 * len(data)), Usage: usage | BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		if err := device.Queue().WriteBuffer(buf, 0, unsafe.Slice((*byte)(unsafe.Pointer(&data[0])), 4*len(data))); err != nil {
			t.Fatalf("WriteBuffer: %v", err)
		}
		return buf
	}
	args := newBuffer("indirect-args", BufferUsageIndirect, []uint32{
		3, 1, 3, 0, // the right triangle
		3, 1, 0, 0, 0, // indices 0-2 of the index buffer bound
	})
	defer args.Release()
	middle := newBuffer("indirect-middle", BufferUsageIndex, []uint32{6 | 7<<16, 8})
	defer middle.Release()
	left := newBuffer("indirect-left", BufferUsageIndex, []uint32{5 | 5<<16, 0 | 1<<16, 2})
	defer left.Release()

	tex, view := pixTexture(t, device, "indirect", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "indirect-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, red, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.DrawIndirect(args, 0)
		p.SetIndexBuffer(middle, gputypes.IndexFormatUint16, 0)
		p.DrawIndexedIndirect(args, 16)
		p.SetIndexBuffer(left, gputypes.IndexFormatUint16, 4)
		p.DrawIndexedIndirect(args, 16)
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	pix := readback(t, st)
	at := func(x, y int) []byte { return pix[(y*pixSide+x)*4 : (y*pixSide+x)*4+4] }
	expectPixels(t, "the left triangle, indices bound past their buffer's start", at(0, 3), 0, 255, 0, 255)
	expectPixels(t, "the middle triangle, indexed", append(at(1, 3), at(2, 3)...), 0, 255, 0, 255)
	expectPixels(t, "the right triangle", at(3, 3), 0, 255, 0, 255)
	expectPixels(t, "the top row", at(1, 0), 255, 0, 0, 255)
	assertNoValidation(t, h)
}

// baseVertexShader places each vertex from its buffer and turns it green
// when vertex_index is the index the buffer gives it, blue otherwise.
const baseVertexShader = `
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> }
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @location(0) v: vec3<f32>) -> VSOut {
	var out: VSOut;
	out.pos = vec4<f32>(v.xy, 0.0, 1.0);
	out.color = select(vec4<f32>(0.0, 0.0, 1.0, 1.0), vec4<f32>(0.0, 1.0, 0.0, 1.0), u32(v.z) == vi);
	return out;
}
@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> { return in.color; }
`

// TestDrawIndexedFromBaseVertex draws one triangle of a vertex buffer of
// three a pass, through indices 0-2 from base vertices 3 and 6 and through
// indices 3-5 from base vertex -3: with DrawIndexed, from an indirect
// record, and from one with the index buffer bound past its start. Each
// vertex has to come from its index plus the base vertex, and vertex_index
// has to be that too. GL ignored the base vertex of the direct draws and
// drew the triangle the indices alone name.
func TestDrawIndexedFromBaseVertex(t *testing.T) {
	device, h := newValidatedDevice(t)
	if !device.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsBaseVertex) {
		t.Skip("the adapter has no base vertex draws")
	}
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "base-vertex", WGSL: baseVertexShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label: "base-vertex",
		Vertex: VertexState{Module: module, EntryPoint: "vs_main", Buffers: []gputypes.VertexBufferLayout{{
			ArrayStride: 12, StepMode: gputypes.VertexStepModeVertex,
			Attributes: []gputypes.VertexAttribute{{Format: gputypes.VertexFormatFloat32x3, ShaderLocation: 0}},
		}}},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	newBuffer := func(label string, usage gputypes.BufferUsage, data []byte) *Buffer {
		buf, err := device.CreateBuffer(&BufferDescriptor{Label: label, Size: uint64(len(data)), Usage: usage | BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		if err := device.Queue().WriteBuffer(buf, 0, data); err != nil {
			t.Fatalf("WriteBuffer: %v", err)
		}
		return buf
	}
	vertices := []float32{
		-1, -1, 0, -1, 1, 1, -0.5, -1, 2, // left
		0.5, -1, 3, 1, 1, 4, 1, -1, 5, // right
		-0.5, -1, 6, 0, 1, 7, 0.5, -1, 8, // middle
	}
	vb := newBuffer("base-vertex-vertices", BufferUsageVertex, unsafe.Slice((*byte)(unsafe.Pointer(&vertices[0])), 4*len(vertices)))
	defer vb.Release()
	indices := []uint16{0, 1, 2, 0, 3, 4, 5, 0}
	ib := newBuffer("base-vertex-indices", BufferUsageIndex, unsafe.Slice((*byte)(unsafe.Pointer(&indices[0])), 2*len(indices)))
	defer ib.Release()
	minus3 := int32(-3)
	records := []uint32{
		3, 1, 0, 6, 0, // the middle triangle
		3, 1, 0, uint32(minus3), 0, // the left one, through indices 3-5
	}
	args := newBuffer("base-vertex-args", BufferUsageIndirect, unsafe.Slice((*byte)(unsafe.Pointer(&records[0])), 4*len(records)))
	defer args.Release()

	// at is the bottom pixel of the left, middle and right triangles.
	at := func(pix []byte, triangle int) []byte {
		x := []int{0, 1, 3}[triangle]
		return pix[(3*pixSide+x)*4 : (3*pixSide+x)*4+4]
	}
	const left, middle, right = 0, 1, 2
	tex, view := pixTexture(t, device, "base-vertex", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	for _, tc := range []struct {
		name        string
		indexOffset uint64
		draw        func(*RenderPassEncoder)
		want, wrong int
	}{
		{"DrawIndexed", 0, func(p *RenderPassEncoder) {
			p.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 3, InstanceCount: 1, BaseVertex: 3})
		}, right, left},
		{"DrawIndexedIndirect", 0, func(p *RenderPassEncoder) { p.DrawIndexedIndirect(args, 0) }, middle, left},
		{"DrawIndexedIndirect past the index buffer's start", 8, func(p *RenderPassEncoder) { p.DrawIndexedIndirect(args, 20) }, left, right},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if strings.Contains(tc.name, "Indirect") {
				skipRootConstantIndirectOnVkd3d(t, h)
			}
			st := readbackBuffer(t, device, "base-vertex-st")
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			clearPass(t, enc, view, gputypes.LoadOpClear, red, func(p *RenderPassEncoder) {
				p.SetPipeline(pipe)
				p.SetVertexBuffer(0, vb, 0)
				p.SetIndexBuffer(ib, gputypes.IndexFormatUint16, tc.indexOffset)
				tc.draw(p)
			})
			copyOut(enc, tex, st)
			finishAndSubmit(t, device, enc)
			device.Poll(PollWait)
			pix := readback(t, st)
			expectPixels(t, "the triangle the base vertex picks", at(pix, tc.want), 0, 255, 0, 255)
			expectPixels(t, "the triangle the indices alone pick", at(pix, tc.wrong), 255, 0, 0, 255)
		})
	}
	assertNoValidation(t, h)
}
