// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package mslmap_test

import (
	"reflect"
	"regexp"
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/metal/mslmap"
	"github.com/roffe/naga"
	"github.com/roffe/naga/ir"
	"github.com/roffe/naga/msl"
)

// wgslRuntimeArray calls arrayLength() on a runtime-sized storage array and
// also reads a uniform buffer. The generated MSL therefore has both the sizes
// argument and a normal buffer binding.
const wgslRuntimeArray = `
struct Params {
	scale: u32,
}

@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	if (gid.x < arrayLength(&out)) {
		out[gid.x] = arrayLength(&out) * params.scale;
	}
}
`

// runtimeArrayLayout is the layout wgslRuntimeArray is written for.
var runtimeArrayLayout = [][]gputypes.BindGroupLayoutEntry{{
	{Binding: 0, Visibility: gputypes.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
	{Binding: 1, Visibility: gputypes.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
}}

// wgslMaterialOnly declares only the second group of a layout whose first
// group holds a uniform buffer too, and its fragment stage only binding 1 of
// group 0.
const wgslMaterialOnly = `
struct M { color: vec4<f32> }
@group(1) @binding(0) var<uniform> material: M;
@group(0) @binding(1) var<uniform> tint: M;
@group(0) @binding(2) var tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
	return material.color + vec4<f32>(f32(i));
}
@fragment fn fs() -> @location(0) vec4<f32> {
	return tint.color * textureSample(tex, samp, vec2<f32>(0.5));
}
`

// materialLayout has a camera buffer at group 0 binding 0 that neither stage
// of wgslMaterialOnly declares.
var materialLayout = [][]gputypes.BindGroupLayoutEntry{
	{
		{Binding: 3, Visibility: gputypes.ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
		{Binding: 0, Visibility: gputypes.ShaderStageVertex, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
		{Binding: 2, Visibility: gputypes.ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 1, Visibility: gputypes.ShaderStageFragment, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
	},
	{
		{Binding: 0, Visibility: gputypes.ShaderStageVertex, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
	},
}

// slotRe finds each MSL parameter name and its slot, for example
// "out [[buffer(0)]]".
var slotRe = regexp.MustCompile(`(\w+)\s*\[\[(buffer|texture|sampler)\((\d+)\)\]\]`)

func lower(t *testing.T, source string) *ir.Module {
	t.Helper()
	ast, err := naga.Parse(source)
	if err != nil {
		t.Fatalf("naga.Parse: %v", err)
	}
	module, err := naga.LowerWithSource(ast, source)
	if err != nil {
		t.Fatalf("naga.LowerWithSource: %v", err)
	}
	return module
}

func resources(t *testing.T, groups [][]gputypes.BindGroupLayoutEntry) map[ir.ResourceBinding]msl.BindTarget {
	t.Helper()
	res, err := mslmap.Resources(groups)
	if err != nil {
		t.Fatalf("mslmap.Resources: %v", err)
	}
	return res
}

// compileEntryPoint writes the MSL of one entry point against the layout.
func compileEntryPoint(t *testing.T, module *ir.Module, stage ir.ShaderStage, name string, groups [][]gputypes.BindGroupLayoutEntry, sizesSlot int) string {
	t.Helper()
	opts, err := mslmap.Options(module, stage, name, resources(t, groups), sizesSlot)
	if err != nil {
		t.Fatalf("mslmap.Options(%s): %v", name, err)
	}
	out, _, err := msl.CompileWithPipeline(module, opts, msl.PipelineOptions{EntryPoint: &msl.EntryPointSelector{Stage: stage, Name: name}})
	if err != nil {
		t.Fatalf("msl.CompileWithPipeline(%s): %v", name, err)
	}
	return out
}

// slots returns the "kind(N)" slot of each parameter.
func slots(mslSource string) map[string]string {
	out := make(map[string]string)
	for _, m := range slotRe.FindAllStringSubmatch(mslSource, -1) {
		out[m[1]] = m[2] + "(" + m[3] + ")"
	}
	return out
}

// TestDefaultOptionsLeaveSizesBufferUnbound records why this package exists.
// With msl.DefaultOptions, naga reports that the sizes buffer is required,
// but it writes the argument without a [[buffer(N)]] attribute.
func TestDefaultOptionsLeaveSizesBufferUnbound(t *testing.T) {
	t.Parallel()

	module := lower(t, wgslRuntimeArray)
	source, info, err := msl.Compile(module, msl.DefaultOptions())
	if err != nil {
		t.Fatalf("msl.Compile: %v", err)
	}
	if !info.RequiresSizesBuffer {
		t.Fatal("RequiresSizesBuffer = false; want true for a runtime-sized array")
	}
	if !strings.Contains(source, "_buffer_sizes") {
		t.Fatal("generated MSL has no _buffer_sizes argument")
	}
	if slot, ok := slots(source)["_buffer_sizes"]; ok {
		t.Fatalf("_buffer_sizes has [[%s]] under DefaultOptions; want no attribute", slot)
	}
}

// TestOptionsBindSizesBuffer checks that the sizes argument goes at the slot
// the caller asks for, and the layout's resources at theirs.
func TestOptionsBindSizesBuffer(t *testing.T) {
	t.Parallel()

	module := lower(t, wgslRuntimeArray)
	source := compileEntryPoint(t, module, ir.StageCompute, "main", runtimeArrayLayout, 2)
	want := map[string]string{"out": "buffer(0)", "params": "buffer(1)", "_buffer_sizes": "buffer(2)"}
	got := slots(source)
	for name, slot := range want {
		if got[name] != slot {
			t.Errorf("%s at %q, want %s\nMSL:\n%s", name, got[name], slot, source)
		}
	}

	bindings, err := mslmap.SizesBindings(module, ir.StageCompute, "main")
	if err != nil {
		t.Fatalf("mslmap.SizesBindings: %v", err)
	}
	if len(bindings) != 1 || bindings[0] != (ir.ResourceBinding{Group: 0, Binding: 0}) {
		t.Errorf("SizesBindings = %v; want exactly the storage array binding (0,0)", bindings)
	}
}

// TestOptionsNumberByLayout checks that the slots come from the layout, not
// from the globals the module declares: a layout entry the shader does not
// use still takes its slot, and each stage gets the same numbering.
func TestOptionsNumberByLayout(t *testing.T) {
	t.Parallel()

	module := lower(t, wgslMaterialOnly)
	vs := slots(compileEntryPoint(t, module, ir.StageVertex, "vs", materialLayout, 3))
	if vs["material"] != "buffer(2)" {
		t.Errorf("vertex: material at %q, want buffer(2) after group 0's two buffers", vs["material"])
	}
	fs := slots(compileEntryPoint(t, module, ir.StageFragment, "fs", materialLayout, 3))
	for name, want := range map[string]string{"tint": "buffer(1)", "tex": "texture(0)", "samp": "sampler(0)"} {
		if fs[name] != want {
			t.Errorf("fragment: %s at %q, want %s", name, fs[name], want)
		}
	}
	if _, ok := fs["material"]; ok {
		t.Error("fragment entry point takes material, which only the vertex stage uses")
	}
}

// TestOptionsRejectBindingsTheLayoutLacks checks that a used global missing
// from the layout, or of another kind there, fails instead of falling back to
// naga's own numbering.
func TestOptionsRejectBindingsTheLayoutLacks(t *testing.T) {
	t.Parallel()

	module := lower(t, wgslMaterialOnly)
	noMaterial := materialLayout[:1]
	if _, err := mslmap.Options(module, ir.StageVertex, "vs", resources(t, noMaterial), 3); err == nil {
		t.Error("vertex stage using @group(1) @binding(0) compiled against a layout without group 1")
	}
	if _, err := mslmap.Options(module, ir.StageFragment, "fs", resources(t, noMaterial), 3); err != nil {
		t.Errorf("fragment stage, which does not use group 1: %v", err)
	}

	swapped := [][]gputypes.BindGroupLayoutEntry{
		{
			{Binding: 1, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
			{Binding: 2, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
			{Binding: 3, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat}},
		},
	}
	if _, err := mslmap.Options(module, ir.StageFragment, "fs", resources(t, swapped), 3); err == nil {
		t.Error("a texture compiled against a layout with a sampler at its binding")
	}
	if _, err := mslmap.Options(module, ir.StageFragment, "nope", resources(t, materialLayout), 3); err == nil {
		t.Error("an entry point the module lacks compiled")
	}
}

// TestSizesBindingsOnlyForEntryPointsThatUseThem checks that an entry point
// with no runtime-sized array gets no sizes, though its module has one.
func TestSizesBindingsOnlyForEntryPointsThatUseThem(t *testing.T) {
	t.Parallel()

	module := lower(t, `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read_write> fixed: array<u32, 4>;
fn helper() { out[0] = arrayLength(&out); }
@compute @workgroup_size(1) fn sized() { helper(); }
@compute @workgroup_size(1) fn unsized() { fixed[0] = 1u; }
`)
	sized, err := mslmap.SizesBindings(module, ir.StageCompute, "sized")
	if err != nil || len(sized) != 1 {
		t.Errorf("SizesBindings(sized) = %v, %v; want the runtime array, through the call", sized, err)
	}
	unsized, err := mslmap.SizesBindings(module, ir.StageCompute, "unsized")
	if err != nil || unsized != nil {
		t.Errorf("SizesBindings(unsized) = %v, %v; want none", unsized, err)
	}
}

// TestOptionsRejectsOutOfRangeSlot checks the guard for a slot that does not
// fit in a uint8.
func TestOptionsRejectsOutOfRangeSlot(t *testing.T) {
	t.Parallel()

	module := lower(t, wgslRuntimeArray)
	if _, err := mslmap.Options(module, ir.StageCompute, "main", resources(t, runtimeArrayLayout), 256); err == nil {
		t.Error("mslmap.Options with sizes slot 256 = nil error; want an out-of-range error")
	}
}

// TestVertexBuffersPullVertices maps a pipeline's vertex buffers for naga's
// vertex pulling, and checks the MSL takes them at the slots the encoder
// binds them at, and that `_buffer_sizes` holds the module's runtime array
// sizes and then the buffers' sizes, in the order the encoder writes them.
func TestVertexBuffersPullVertices(t *testing.T) {
	t.Parallel()

	buffers := []gputypes.VertexBufferLayout{
		{ArrayStride: 8, StepMode: gputypes.VertexStepModeVertex, Attributes: []gputypes.VertexAttribute{
			{Format: gputypes.VertexFormatFloat32x2, ShaderLocation: 0},
		}},
		{StepMode: gputypes.VertexStepModeVertexBufferNotUsed},
		{StepMode: gputypes.VertexStepModeInstance, Attributes: []gputypes.VertexAttribute{
			{Format: gputypes.VertexFormatUnorm8x4, ShaderLocation: 1},
			{Format: gputypes.VertexFormatFloat32x2, Offset: 4, ShaderLocation: 2},
		}},
		{ArrayStride: 4, StepMode: gputypes.VertexStepModeInstance, Attributes: []gputypes.VertexAttribute{
			{Format: gputypes.VertexFormatUnorm8x2, Offset: 2, ShaderLocation: 3},
		}},
	}
	mappings, err := mslmap.VertexBuffers(buffers, 30)
	if err != nil {
		t.Fatalf("mslmap.VertexBuffers: %v", err)
	}
	want := []msl.VertexBufferMapping{
		{ID: 30, Stride: 8, StepMode: msl.VertexStepModeByVertex, Attributes: []msl.AttributeMapping{
			{ShaderLocation: 0, Format: msl.VertexFormatFloat32x2},
		}},
		{ID: 28, Stride: 12, StepMode: msl.VertexStepModeConstant, Attributes: []msl.AttributeMapping{
			{ShaderLocation: 1, Format: msl.VertexFormatUnorm8x4},
			{ShaderLocation: 2, Offset: 4, Format: msl.VertexFormatFloat32x2},
		}},
		{ID: 27, Stride: 4, StepMode: msl.VertexStepModeByInstance, Attributes: []msl.AttributeMapping{
			{ShaderLocation: 3, Offset: 2, Format: msl.VertexFormatUnorm8x2},
		}},
	}
	if !reflect.DeepEqual(mappings, want) {
		t.Fatalf("mappings = %+v\nwant %+v", mappings, want)
	}

	module := lower(t, `
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<uniform> tint: vec4<f32>;
@vertex fn vs(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>, @location(2) offset: vec2<f32>, @location(3) uv: vec2<f32>) -> @builtin(position) vec4<f32> {
	return vec4<f32>(pos + offset + uv, 0.0, 1.0) * color * tint;
}
@compute @workgroup_size(1) fn cs() { _ = arrayLength(&data); }
`)
	layout := [][]gputypes.BindGroupLayoutEntry{{
		{Binding: 0, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
		{Binding: 1, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
	}}
	opts, err := mslmap.Options(module, ir.StageVertex, "vs", resources(t, layout), 2)
	if err != nil {
		t.Fatalf("mslmap.Options: %v", err)
	}
	opts.VertexPullingTransform = true
	opts.VertexBufferMappings = mappings
	source, _, err := msl.CompileWithPipeline(module, opts, msl.PipelineOptions{EntryPoint: &msl.EntryPointSelector{Stage: ir.StageVertex, Name: "vs"}})
	if err != nil {
		t.Fatalf("msl.CompileWithPipeline: %v", err)
	}
	got := slots(source)
	for name, slot := range map[string]string{"tint": "buffer(1)", "vb_30_in": "buffer(30)", "vb_28_in": "buffer(28)", "vb_27_in": "buffer(27)", "_buffer_sizes": "buffer(2)"} {
		if got[name] != slot {
			t.Errorf("%s at %q, want %s", name, got[name], slot)
		}
	}
	if strings.Contains(source, "stage_in") {
		t.Error("the vertex stage still takes its attributes as stage_in")
	}
	sizes := regexp.MustCompile(`struct _mslBufferSizes \{\s*([^}]*)\}`).FindStringSubmatch(source)
	runtime, err := mslmap.RuntimeArrayBindings(module)
	if err != nil {
		t.Fatalf("mslmap.RuntimeArrayBindings: %v", err)
	}
	if sizes == nil || len(runtime) != 1 || strings.Join(strings.Fields(sizes[1]), " ") != "uint size0; uint buffer_size30; uint buffer_size28; uint buffer_size27;" {
		t.Errorf("_mslBufferSizes holds %q, want the size of data, then of vertex buffers 0, 2 and 3", sizes)
	}
	for _, name := range []string{"pos", "color", "offset", "uv"} {
		if !strings.Contains(source, " "+name+" = {};") {
			t.Errorf("attribute %s is not pulled into a local of its name, which the body reads", name)
		}
	}
	if t.Failed() {
		t.Logf("MSL:\n%s", source)
	}

	for f := range gputypes.VertexFormatUnorm1010102 + 1 {
		_, err := mslmap.VertexBuffers([]gputypes.VertexBufferLayout{{ArrayStride: 16, Attributes: []gputypes.VertexAttribute{{Format: f}}}}, 30)
		if (err == nil) != (f.Size() > 0) {
			t.Errorf("vertex format %v: err = %v", f, err)
		}
	}
}
