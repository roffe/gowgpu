//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	naga "github.com/roffe/naga"
	"github.com/roffe/naga/ir"
)

// expectEntries fails the test when got are not the layout entries want.
func expectEntries(t *testing.T, what string, got, want []gputypes.BindGroupLayoutEntry) {
	t.Helper()
	if len(got) == 0 && len(want) == 0 {
		return
	}
	if !reflect.DeepEqual(got, want) {
		g, _ := json.Marshal(got)
		w, _ := json.Marshal(want)
		t.Errorf("%s:\n got %s\nwant %s", what, g, w)
	}
}

// parseModule is src as CreateShaderModule keeps it, without a device.
func parseModule(t *testing.T, src string) *ShaderModule {
	t.Helper()
	ast, err := naga.Parse(src)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	mod, err := naga.Lower(ast)
	if err != nil {
		t.Fatalf("Lower: %v", err)
	}
	return &ShaderModule{irModule: mod}
}

// TestAutoLayoutEntries checks what an automatic layout gets out of the
// naga IR, no GPU needed.
func TestAutoLayoutEntries(t *testing.T) {
	const vf = ShaderStageVertex | ShaderStageFragment
	uniform := func(b uint32, vis gputypes.ShaderStages, size uint64) gputypes.BindGroupLayoutEntry {
		return gputypes.BindGroupLayoutEntry{Binding: b, Visibility: vis, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform, MinBindingSize: size}}
	}
	texture := func(b uint32, vis gputypes.ShaderStages, st gputypes.TextureSampleType, dim gputypes.TextureViewDimension) gputypes.BindGroupLayoutEntry {
		return gputypes.BindGroupLayoutEntry{Binding: b, Visibility: vis, Texture: &gputypes.TextureBindingLayout{SampleType: st, ViewDimension: dim}}
	}
	sampler := func(b uint32, vis gputypes.ShaderStages, typ gputypes.SamplerBindingType) gputypes.BindGroupLayoutEntry {
		return gputypes.BindGroupLayoutEntry{Binding: b, Visibility: vis, Sampler: &gputypes.SamplerBindingLayout{Type: typ}}
	}

	// One module for both stages: u in both, lights only through a call
	// from the vertex stage, t only sampled inside a function it is passed
	// to, lut only loaded, unused not at all, group 2 empty.
	scene := parseModule(t, `
struct U { m: mat4x4<f32>, tint: vec4<f32> }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage> lights: array<vec4<f32>>;
@group(1) @binding(0) var t: texture_2d<f32>;
@group(1) @binding(1) var s: sampler;
@group(1) @binding(2) var lut: texture_2d_array<f32>;
@group(1) @binding(3) var unused: texture_2d<f32>;
@group(3) @binding(0) var shadow: texture_depth_2d;
@group(3) @binding(1) var shadowSampler: sampler_comparison;
fn light(i: u32) -> vec4<f32> { return lights[i]; }
fn tex(tt: texture_2d<f32>, ss: sampler, uv: vec2<f32>) -> vec4<f32> { return textureSample(tt, ss, uv); }
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> { return u.m * light(i); }
@fragment fn fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
	let shade = textureSampleCompare(shadow, shadowSampler, p.xy, p.z);
	return tex(t, s, p.xy) * textureLoad(lut, vec2<i32>(p.xy), 0, 0) * u.tint * shade;
}`)
	// Two modules that declare the same bindings each their own way: the
	// vertex stage only loads heights, the fragment stage samples it and
	// has the larger uniform.
	vsOnly := parseModule(t, `
struct A { a: vec4<f32> }
@group(0) @binding(0) var<uniform> u: A;
@group(0) @binding(1) var heights: texture_2d<f32>;
@vertex fn vs() -> @builtin(position) vec4<f32> { return u.a * textureLoad(heights, vec2<i32>(0), 0); }`)
	fsOnly := parseModule(t, `
struct B { a: vec4<f32>, b: vec4<f32> }
@group(0) @binding(0) var<uniform> u: B;
@group(0) @binding(1) var heights: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
@fragment fn fs() -> @location(0) vec4<f32> { return u.b * textureSample(heights, s, vec2<f32>(0.5)); }`)
	conflicting := parseModule(t, `
@group(0) @binding(0) var<storage> u: array<f32>;
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(u[0]); }`)
	compute := parseModule(t, `
@group(0) @binding(0) var img: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> counts: array<atomic<u32>>;
@compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
	textureStore(img, vec2<i32>(id.xy), vec4<f32>(1.0));
	atomicAdd(&counts[0], 1u);
}`)

	for _, tc := range []struct {
		name   string
		stages []autoStage
		want   [][]gputypes.BindGroupLayoutEntry
		err    string
	}{
		{"one module", []autoStage{{scene, "vs", ir.StageVertex, ShaderStageVertex}, {scene, "fs", ir.StageFragment, ShaderStageFragment}}, [][]gputypes.BindGroupLayoutEntry{
			{uniform(0, vf, 80), {Binding: 1, Visibility: ShaderStageVertex, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage, MinBindingSize: 16}}},
			{
				texture(0, ShaderStageFragment, gputypes.TextureSampleTypeFloat, gputypes.TextureViewDimension2D),
				sampler(1, ShaderStageFragment, gputypes.SamplerBindingTypeFiltering),
				texture(2, ShaderStageFragment, gputypes.TextureSampleTypeUnfilterableFloat, gputypes.TextureViewDimension2DArray),
			},
			nil,
			{texture(0, ShaderStageFragment, gputypes.TextureSampleTypeDepth, gputypes.TextureViewDimension2D), sampler(1, ShaderStageFragment, gputypes.SamplerBindingTypeComparison)},
		}, ""},
		{"merged across modules", []autoStage{{vsOnly, "vs", ir.StageVertex, ShaderStageVertex}, {fsOnly, "fs", ir.StageFragment, ShaderStageFragment}}, [][]gputypes.BindGroupLayoutEntry{
			{uniform(0, vf, 32), texture(1, vf, gputypes.TextureSampleTypeFloat, gputypes.TextureViewDimension2D), sampler(2, ShaderStageFragment, gputypes.SamplerBindingTypeFiltering)},
		}, ""},
		{"storage texture and buffer", []autoStage{{compute, "main", ir.StageCompute, ShaderStageCompute}}, [][]gputypes.BindGroupLayoutEntry{{
			{Binding: 0, Visibility: ShaderStageCompute, StorageTexture: &gputypes.StorageTextureBindingLayout{
				Access: gputypes.StorageTextureAccessWriteOnly, Format: gputypes.TextureFormatRGBA8Unorm, ViewDimension: gputypes.TextureViewDimension2D}},
			{Binding: 1, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage, MinBindingSize: 4}},
		}}, ""},
		{"conflict", []autoStage{{vsOnly, "vs", ir.StageVertex, ShaderStageVertex}, {conflicting, "fs", ir.StageFragment, ShaderStageFragment}}, nil, "@group(0) @binding(0)"},
		{"wrong stage", []autoStage{{scene, "fs", ir.StageVertex, ShaderStageVertex}}, nil, `no entry point "fs"`},
		{"SPIR-V", []autoStage{{&ShaderModule{}, "main", ir.StageCompute, ShaderStageCompute}}, nil, "WGSL"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			groups, err := autoLayoutEntries(tc.stages)
			if tc.err != "" {
				if err == nil || !strings.Contains(err.Error(), tc.err) {
					t.Fatalf("err = %v, want one about %s", err, tc.err)
				}
				return
			}
			if err != nil {
				t.Fatalf("autoLayoutEntries: %v", err)
			}
			if len(groups) != len(tc.want) {
				t.Fatalf("%d groups, want %d", len(groups), len(tc.want))
			}
			for i := range groups {
				expectEntries(t, fmt.Sprint("group ", i), groups[i], tc.want[i])
			}
		})
	}
}

// autoRenderShader draws its texture times a uniform tint. Both stages read
// the tint and group 1 is skipped, so the derived layout has a group two
// stages see and an empty one.
const autoRenderShader = `
struct Tint { color: vec4<f32> }
@group(0) @binding(0) var<uniform> tint: Tint;
@group(2) @binding(0) var t: texture_2d<f32>;
@group(2) @binding(1) var s: sampler;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	return vec4<f32>(x, y, 0.0, tint.color.a);
}
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	return textureSample(t, s, pos.xy / 4.0) * tint.color;
}
`

// TestAutoLayoutRenderPipeline: a render pipeline created without a Layout
// gets the one its shader implies, GetBindGroupLayout hands out its groups,
// and a draw through bind groups made from them samples a yellow texture
// times a cyan tint into green.
func TestAutoLayoutRenderPipeline(t *testing.T) {
	device, h := newValidatedDevice(t)

	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "auto-render", WGSL: autoRenderShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "auto-render",
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()

	want := [][]gputypes.BindGroupLayoutEntry{
		{{Binding: 0, Visibility: ShaderStageVertex | ShaderStageFragment, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform, MinBindingSize: 16}}},
		nil,
		{
			{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
			{Binding: 1, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
		},
	}
	for i := range want {
		layout := pipe.GetBindGroupLayout(uint32(i))
		if layout == nil {
			t.Fatalf("GetBindGroupLayout(%d) = nil", i)
		}
		expectEntries(t, fmt.Sprint("group ", i), layout.entries, want[i])
	}
	if pipe.GetBindGroupLayout(3) != nil {
		t.Error("GetBindGroupLayout(3) is not nil, past the last group")
	}

	tint, err := device.CreateBuffer(&BufferDescriptor{Label: "auto-tint", Size: 16, Usage: BufferUsageUniform | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer tint.Release()
	cyan := make([]byte, 16)
	for i, v := range []float32{0, 1, 1, 1} {
		binary.LittleEndian.PutUint32(cyan[i*4:], math.Float32bits(v))
	}
	if err := device.Queue().WriteBuffer(tint, 0, cyan); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	src, srcView := pixTexture(t, device, "auto-src", TextureUsageRenderAttachment|TextureUsageTextureBinding)
	defer releaseTextureView(src, srcView)
	dst, dstView := pixTexture(t, device, "auto-dst", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(dst, dstView)
	smp, err := device.CreateSampler(&SamplerDescriptor{MagFilter: gputypes.FilterModeNearest, MinFilter: gputypes.FilterModeNearest, LodMaxClamp: 32})
	if err != nil {
		t.Fatalf("CreateSampler: %v", err)
	}
	defer smp.Release()
	st := readbackBuffer(t, device, "auto-st")

	groups := make([]*BindGroup, 0, 3)
	for i, entries := range [][]BindGroupEntry{
		{{Binding: 0, Buffer: tint}},
		nil,
		{{Binding: 0, TextureView: srcView}, {Binding: 1, Sampler: smp}},
	} {
		g, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: pipe.GetBindGroupLayout(uint32(i)), Entries: entries})
		if err != nil {
			t.Fatalf("CreateBindGroup(%d): %v", i, err)
		}
		t.Cleanup(g.Release)
		groups = append(groups, g)
	}

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, srcView, gputypes.LoadOpClear, gputypes.Color{R: 1, G: 1, B: 0, A: 1}, nil)
	clearPass(t, enc, dstView, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) {
		pass.SetPipeline(pipe)
		for i, g := range groups {
			pass.SetBindGroup(uint32(i), g, nil)
		}
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, dst, st)
	finishAndSubmit(t, device, enc)

	expectPixels(t, "yellow sampled times a cyan tint", readback(t, st), 0, 255, 0, 255)
	assertNoValidation(t, h)
}

const autoComputeShader = `
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@compute @workgroup_size(64) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
	if id.x < arrayLength(&dst) {
		dst[id.x] = src[id.x] * 2.0 + 1.0;
	}
}
`

// TestAutoLayoutComputePipeline: the same for a compute pipeline, whose
// dispatch through the derived group maps 64 floats to 2x+1. The derived
// layout is the pipeline's: Release on it does nothing, and the pipeline's
// last reference releases it.
func TestAutoLayoutComputePipeline(t *testing.T) {
	device, h := newValidatedDevice(t)

	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "auto-compute", WGSL: autoComputeShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "auto-compute", Module: module, EntryPoint: "cs_main"})
	if err != nil {
		if strings.Contains(err.Error(), "ComputeShaders") {
			t.Skipf("no compute shaders on this adapter: %v", err)
		}
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipe.Release()
	layout := pipe.GetBindGroupLayout(0)
	if layout == nil {
		t.Fatal("GetBindGroupLayout(0) = nil")
	}
	expectEntries(t, "group 0", layout.entries, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage, MinBindingSize: 4}},
		{Binding: 1, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage, MinBindingSize: 4}},
	})
	if pipe.GetBindGroupLayout(1) != nil {
		t.Error("GetBindGroupLayout(1) is not nil, past the last group")
	}

	const n = 64
	buffer := func(label string, usage gputypes.BufferUsage) *Buffer {
		b, err := device.CreateBuffer(&BufferDescriptor{Label: label, Size: n * 4, Usage: usage})
		if err != nil {
			t.Fatalf("CreateBuffer(%s): %v", label, err)
		}
		t.Cleanup(b.Release)
		return b
	}
	src := buffer("auto-src", BufferUsageStorage|BufferUsageCopyDst)
	dst := buffer("auto-dst", BufferUsageStorage|BufferUsageCopySrc)
	st := buffer("auto-st", BufferUsageMapRead|BufferUsageCopyDst)
	data := make([]byte, n*4)
	for i := range n {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i)))
	}
	if err := device.Queue().WriteBuffer(src, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	// The group is made from the layout and runs after this.
	layout.Release()
	if layout.released {
		t.Error("Release released a layout the pipeline owns")
	}
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: src}, {Binding: 1, Buffer: dst}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, group, nil)
	pass.Dispatch(1, 1, 1)
	if err := pass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	enc.CopyBufferToBuffer(dst, 0, st, 0, n*4)
	finishAndSubmit(t, device, enc)

	for i, v := range readFloats(t, st, n) {
		if want := float32(i)*2 + 1; v != want {
			t.Fatalf("dst[%d] = %v, want %v", i, v, want)
		}
	}

	// A pipeline that never ran drops its last reference at Release.
	spare, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "auto-spare", Module: module, EntryPoint: "cs_main"})
	if err != nil {
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	spareLayout := spare.GetBindGroupLayout(0)
	spare.Release()
	if !spareLayout.released {
		t.Error("the pipeline's release left its derived layout alive")
	}
	assertNoValidation(t, h)
}
