//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// GPU tests of WGSL semantics the shader translator has to keep on every
// backend: zero values, remainders, private arrays, uniform matrix layout,
// storage buffers and texture-sampler pairs in fragment shaders.

// semanticsStorage makes a storage buffer holding words.
func semanticsStorage(t *testing.T, device *Device, label string, words []uint32) *Buffer {
	t.Helper()
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: label, Size: uint64(len(words) * 4),
		Usage: BufferUsageStorage | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer(%s): %v", label, err)
	}
	t.Cleanup(buf.Release)
	data := make([]byte, len(words)*4)
	for i, w := range words {
		binary.LittleEndian.PutUint32(data[i*4:], w)
	}
	if err := device.Queue().WriteBuffer(buf, 0, data); err != nil {
		t.Fatalf("WriteBuffer(%s): %v", label, err)
	}
	return buf
}

// semanticsDispatch runs cs_main of wgsl once with bufs bound at bindings 0,
// 1, ... as read-write storage buffers, and returns their words afterwards.
func semanticsDispatch(t *testing.T, device *Device, label, wgsl string, bufs ...*Buffer) [][]uint32 {
	t.Helper()
	types := make([]gputypes.BufferBindingType, len(bufs))
	for i := range types {
		types[i] = gputypes.BufferBindingTypeStorage
	}
	return semanticsDispatchTypes(t, device, label, wgsl, types, bufs...)
}

// semanticsDispatchTypes is semanticsDispatch with buffer i bound as a
// buffer of types[i].
func semanticsDispatchTypes(t *testing.T, device *Device, label, wgsl string, types []gputypes.BufferBindingType, bufs ...*Buffer) [][]uint32 {
	t.Helper()
	entries := make([]gputypes.BindGroupLayoutEntry, len(bufs))
	group := make([]BindGroupEntry, len(bufs))
	for i, b := range bufs {
		entries[i] = gputypes.BindGroupLayoutEntry{Binding: uint32(i), Visibility: ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: types[i]}}
		group[i] = BindGroupEntry{Binding: uint32(i), Buffer: b, Size: b.Size()}
	}
	pipe, layout := computePipeline(t, device, label, wgsl, entries)
	bg, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: group})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(bg.Release)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	pass.SetPipeline(pipe)
	pass.SetBindGroup(0, bg, nil)
	pass.Dispatch(1, 1, 1)
	if err := pass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	staging := make([]*Buffer, len(bufs))
	for i, b := range bufs {
		staging[i], err = device.CreateBuffer(&BufferDescriptor{Label: label + "-staging", Size: b.Size(), Usage: BufferUsageMapRead | BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer(staging): %v", err)
		}
		t.Cleanup(staging[i].Release)
		enc.CopyBufferToBuffer(b, 0, staging[i], 0, b.Size())
	}
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	out := make([][]uint32, len(staging))
	for i, s := range staging {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := s.Map(ctx, MapModeRead, 0, s.Size()); err != nil {
			cancel()
			t.Fatalf("Map: %v", err)
		}
		cancel()
		rng, err := s.MappedRange(0, s.Size())
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		data := rng.Bytes()
		out[i] = make([]uint32, len(data)/4)
		for j := range out[i] {
			out[i][j] = binary.LittleEndian.Uint32(data[j*4:])
		}
		if err := s.Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
	}
	return out
}

// TestShaderZeroValues: a function or private variable declared without an
// initializer starts at zero, a var declared in a loop body at zero on every
// iteration, and a private variable at its initializer. The variables are
// written only on a branch that doesn't run, where a driver is free to fold
// an undefined start into the value written there.
func TestShaderZeroValues(t *testing.T) {
	device, h := newValidatedDevice(t)
	d := semanticsStorage(t, device, "zero-values", []uint32{0, 3, 1, 1, 1, 1, 1, 1, 1})
	got := semanticsDispatch(t, device, "zero-values", `
@group(0) @binding(0) var<storage, read_write> d: array<u32>;
var<private> p: u32;
var<private> pa: array<u32, 4>;
var<private> five: u32 = 5u;

@compute @workgroup_size(1)
fn cs_main() {
    var x: u32;
    var v: vec4<u32>;
    var arr: array<u32, 16>;
    if d[0] == 12345u {
        x = 77u;
        v = vec4(55u);
        p = 99u;
        arr[d[1]] = 44u;
        pa[d[1]] = 33u;
        five = 66u;
    }
    var acc = 0u;
    for (var i = 0u; i < 4u; i++) {
        var y: u32;
        y += 1u;
        acc += y;
    }
    d[2] = x;
    d[3] = v.y;
    d[4] = p;
    d[5] = arr[3];
    d[6] = pa[3];
    d[7] = acc;
    d[8] = five;
}
`, d)[0]
	for i, name := range []string{"var x", "var v.y", "var<private> p", "var arr[3]", "var<private> pa[3]", "sum of a loop var", "var<private> five"} {
		want := []uint32{0, 0, 0, 0, 0, 4, 5}[i]
		if got[2+i] != want {
			t.Errorf("%s = %d, want %d", name, got[2+i], want)
		}
	}
	assertNoValidation(t, h)
}

// TestShaderRemainder: WGSL's % truncates, e1 - e2 * trunc(e1 / e2), so the
// result has the sign of e1, for floats and integers, negative operands
// included.
func TestShaderRemainder(t *testing.T) {
	device, h := newValidatedDevice(t)
	f := semanticsStorage(t, device, "remainder-f", []uint32{math.Float32bits(-1.5), math.Float32bits(1), 0, 0, 0})
	i := semanticsStorage(t, device, "remainder-i", []uint32{uint32(0xFFFFFFF9), 3, 0, 0, 0, 0}) // -7, 3
	got := semanticsDispatch(t, device, "remainder", `
@group(0) @binding(0) var<storage, read_write> f: array<f32>;
@group(0) @binding(1) var<storage, read_write> i: array<i32>;

@compute @workgroup_size(1)
fn cs_main() {
    let a = f[0];
    let b = f[1];
    f[2] = a % b;
    f[3] = (vec2(a, -a) % vec2(b)).x;
    f[4] = -a % -b;
    let x = i[0];
    let y = i[1];
    i[2] = x % y;
    i[3] = -x % -y;
    i[4] = (vec2(x) % vec2(y)).y;
    i[5] = i32(u32(-x) % u32(y));
}
`, f, i)
	for k, want := range []float32{-0.5, -0.5, 0.5} {
		if g := math.Float32frombits(got[0][2+k]); g != want {
			t.Errorf("float remainder %d = %v, want %v", k, g, want)
		}
	}
	for k, want := range []int32{-1, 1, -1, 1} {
		if g := int32(got[1][2+k]); g != want {
			t.Errorf("integer remainder %d = %d, want %d", k, g, want)
		}
	}
	assertNoValidation(t, h)
}

// TestShaderPrivateArray: module-scope private arrays, initialized and
// zeroed, indexed at run time.
func TestShaderPrivateArray(t *testing.T) {
	device, h := newValidatedDevice(t)
	d := semanticsStorage(t, device, "private-array", []uint32{1, 0})
	got := semanticsDispatch(t, device, "private-array", `
@group(0) @binding(0) var<storage, read_write> d: array<u32>;
var<private> lut: array<u32, 4> = array<u32, 4>(10u, 20u, 30u, 40u);
var<private> scratch: array<u32, 4>;

@compute @workgroup_size(1)
fn cs_main() {
    let k = d[0];
    scratch[k] = lut[k];
    d[1] = lut[k + 1u] + scratch[k] + scratch[(k + 1u) & 3u];
}
`, d)[0]
	if got[1] != 50 {
		t.Errorf("lut[2] + scratch[1] + scratch[2] = %d, want 30 + 20 + 0", got[1])
	}
	assertNoValidation(t, h)
}

const semanticsFullscreenVS = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
`

// semanticsDraw draws a full-screen triangle with vs_main and fs_main of
// wgsl into a pixTexture, with group 0 laid out by layout and bound to
// entries, and returns the pixels, or the error creating the pipeline.
func semanticsDraw(t *testing.T, device *Device, label, wgsl string, layout []gputypes.BindGroupLayoutEntry, entries []BindGroupEntry) ([]byte, error) {
	t.Helper()
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: layout})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	t.Cleanup(bgl.Release)
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	t.Cleanup(pl.Release)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: label, WGSL: semanticsFullscreenVS + wgsl})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(module.Release)
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     label,
		Layout:    pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		return nil, err
	}
	t.Cleanup(pipe.Release)
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: entries})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(group.Release)

	tex, view := pixTexture(t, device, label, TextureUsageRenderAttachment|TextureUsageCopySrc)
	t.Cleanup(func() { releaseTextureView(tex, view) })
	st := readbackBuffer(t, device, label+"-st")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, black, func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetBindGroup(0, group, nil)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)
	return readback(t, st), nil
}

// semanticsTexture makes a texture of width x 1 RGBA8 texels for sampling.
func semanticsTexture(t *testing.T, device *Device, label string, texels ...[4]byte) *TextureView {
	t.Helper()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         label,
		Size:          Extent3D{Width: uint32(len(texels)), Height: 1, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageTextureBinding | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture(%s): %v", label, err)
	}
	data := make([]byte, 0, len(texels)*4)
	for _, c := range texels {
		data = append(data, c[:]...)
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, data,
		&ImageDataLayout{BytesPerRow: uint32(len(data))},
		&Extent3D{Width: uint32(len(texels)), Height: 1, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture(%s): %v", label, err)
	}
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView(%s): %v", label, err)
	}
	t.Cleanup(func() { releaseTextureView(tex, view) })
	return view
}

// semanticsSampler makes a nearest sampler with the address mode.
func semanticsSampler(t *testing.T, device *Device, mode gputypes.AddressMode) *Sampler {
	t.Helper()
	s, err := device.CreateSampler(&SamplerDescriptor{AddressModeU: mode, AddressModeV: mode, AddressModeW: mode,
		MagFilter: gputypes.FilterModeNearest, MinFilter: gputypes.FilterModeNearest, LodMaxClamp: 32})
	if err != nil {
		t.Fatalf("CreateSampler: %v", err)
	}
	t.Cleanup(s.Release)
	return s
}

var (
	semanticsGreen = [4]byte{0, 255, 0, 255}
	semanticsBlue  = [4]byte{0, 0, 255, 255}
	semanticsRed   = [4]byte{255, 0, 0, 255}
)

func semanticsUniform(t *testing.T, device *Device, label string, floats ...float32) *Buffer {
	t.Helper()
	data := make([]byte, len(floats)*4)
	for i, f := range floats {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(f))
	}
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: label, Size: uint64(len(data)), Usage: BufferUsageUniform | BufferUsageStorage | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer(%s): %v", label, err)
	}
	t.Cleanup(buf.Release)
	if err := device.Queue().WriteBuffer(buf, 0, data); err != nil {
		t.Fatalf("WriteBuffer(%s): %v", label, err)
	}
	return buf
}

// TestUniformMatCx2Layout: WGSL puts the columns of a mat2x2<f32> 8 bytes
// apart in a uniform buffer and the vec2 after it at 16, where std140 puts
// matrix columns 16 bytes apart. The column is read by a constant and by a
// computed index.
func TestUniformMatCx2Layout(t *testing.T) {
	device, h := newValidatedDevice(t)
	// m = [(0.25, 0.5), (0.75, 1.0)], v = (0.125, 0.875) at offset 16.
	u := semanticsUniform(t, device, "matcx2", 0.25, 0.5, 0.75, 1.0, 0.125, 0.875, 0, 0)
	pix, err := semanticsDraw(t, device, "matcx2", `
struct U {
    m: mat2x2<f32>,
    v: vec2<f32>,
}
@group(0) @binding(0) var<uniform> u: U;
@fragment fn fs_main() -> @location(0) vec4<f32> {
    let i = i32(u.v.x * 8.0);
    return vec4<f32>(u.m[1].x, u.m[i].y, u.v);
}
`, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
	}, []BindGroupEntry{{Binding: 0, Buffer: u, Size: 32}})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	expectPixels(t, "m[1], v", pix, 191, 255, 32, 223)
	assertNoValidation(t, h)
}

// TestFragmentStorageBufferBesideUniform reads a read-only storage buffer
// (green) in a fragment shader that also has a uniform buffer (red), each
// the first of its kind. A GL context whose GLSL has no storage buffers
// refuses the pipeline.
func TestFragmentStorageBufferBesideUniform(t *testing.T) {
	device, h := newValidatedDevice(t)
	red := semanticsUniform(t, device, "storage-red", 1, 0, 0, 1)
	green := semanticsUniform(t, device, "storage-green", 0, 1, 0, 1)
	pix, err := semanticsDraw(t, device, "fragment-storage", `
struct C { c: vec4<f32> }
@group(0) @binding(0) var<uniform> u: C;
@group(0) @binding(1) var<storage, read> s: array<C>;
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return s[0].c + u.c * 0.0;
}
`, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
		{Binding: 1, Visibility: ShaderStageFragment, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
	}, []BindGroupEntry{{Binding: 0, Buffer: red, Size: 16}, {Binding: 1, Buffer: green, Size: 16}})
	if err != nil {
		if h.backend == gputypes.BackendGL && strings.Contains(err.Error(), "std430") {
			t.Skipf("this GL context has no storage buffers in its GLSL version: %v", err)
		}
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	expectPixels(t, "storage read beside a uniform", pix, 0, 255, 0, 255)
	assertNoValidation(t, h)
}

// TestSampleInHelperFunction samples texture 1 (green, blue) at u = 1.25
// through a helper that takes the texture and a repeat sampler: repeat wraps
// to the green texel, where a clamped lookup gives blue and texture 0 is red.
func TestSampleInHelperFunction(t *testing.T) {
	device, h := newValidatedDevice(t)
	t0 := semanticsTexture(t, device, "helper-t0", semanticsRed, semanticsRed)
	t1 := semanticsTexture(t, device, "helper-t1", semanticsGreen, semanticsBlue)
	repeat := semanticsSampler(t, device, gputypes.AddressModeRepeat)
	pix, err := semanticsDraw(t, device, "sample-in-helper", `
@group(0) @binding(0) var t0: texture_2d<f32>;
@group(0) @binding(1) var t1: texture_2d<f32>;
@group(0) @binding(2) var s: sampler;
fn fetch(t: texture_2d<f32>, smp: sampler, uv: vec2<f32>) -> vec4<f32> {
    return textureSample(t, smp, uv);
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return fetch(t1, s, vec2<f32>(1.25, 0.5)) + textureSample(t0, s, vec2<f32>(0.5)) * 0.0;
}
`, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 1, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 2, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
	}, []BindGroupEntry{{Binding: 0, TextureView: t0}, {Binding: 1, TextureView: t1}, {Binding: 2, Sampler: repeat}})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	expectPixels(t, "texture 1 through a helper, repeat sampler", pix, 0, 255, 0, 255)
	assertNoValidation(t, h)
}

// TestTextureWithTwoSamplers samples one texture (green, blue) at u = 1.25
// with a repeat sampler (green) and a clamp sampler (blue). A GL texture unit
// holds one sampler, so GL refuses the pipeline, as wgpu does; elsewhere
// both apply.
func TestTextureWithTwoSamplers(t *testing.T) {
	device, h := newValidatedDevice(t)
	tex := semanticsTexture(t, device, "two-samplers", semanticsGreen, semanticsBlue)
	repeat := semanticsSampler(t, device, gputypes.AddressModeRepeat)
	clamp := semanticsSampler(t, device, gputypes.AddressModeClampToEdge)
	pix, err := semanticsDraw(t, device, "two-samplers", `
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var repeat_s: sampler;
@group(0) @binding(2) var clamp_s: sampler;
@fragment fn fs_main() -> @location(0) vec4<f32> {
    let uv = vec2<f32>(1.25, 0.5);
    return vec4<f32>(textureSample(t, repeat_s, uv).g, textureSample(t, clamp_s, uv).b, 0.0, 1.0);
}
`, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 1, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
		{Binding: 2, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
	}, []BindGroupEntry{{Binding: 0, TextureView: tex}, {Binding: 1, Sampler: repeat}, {Binding: 2, Sampler: clamp}})
	if h.backend == gputypes.BackendGL {
		if err == nil || !strings.Contains(err.Error(), "two samplers") {
			t.Fatalf("CreateRenderPipeline on GL: err = %v, want the two samplers error", err)
		}
		return
	}
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	expectPixels(t, "repeat green, clamp blue", pix, 255, 255, 0, 255)
	assertNoValidation(t, h)
}

// TestDepthTexturesSampledLoadedAndCompared samples a depth texture cleared
// to 0.25 with a sampler that doesn't compare, in a helper function, and
// loads it, and samples one cleared to 0.75 with a less comparison against
// 0.5, in one fragment shader. GLSL declares the first a sampler2D and the
// second a sampler2DShadow.
func TestDepthTexturesSampledLoadedAndCompared(t *testing.T) {
	device, h := newValidatedDevice(t)
	_, plainView := depthTexture(t, device)
	_, shadowView := depthTexture(t, device)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearDepth(t, enc, plainView, 0.25)
	clearDepth(t, enc, shadowView, 0.75)
	finishAndSubmit(t, device, enc)
	nearest := &SamplerDescriptor{MagFilter: gputypes.FilterModeNearest, MinFilter: gputypes.FilterModeNearest, LodMaxClamp: 32}
	s, err := device.CreateSampler(nearest)
	if err != nil {
		t.Fatalf("CreateSampler: %v", err)
	}
	t.Cleanup(s.Release)
	compare := *nearest
	compare.Compare = gputypes.CompareFunctionLess
	sc, err := device.CreateSampler(&compare)
	if err != nil {
		t.Fatalf("CreateSampler(compare): %v", err)
	}
	t.Cleanup(sc.Release)

	depthEntry := func(binding uint32) gputypes.BindGroupLayoutEntry {
		return gputypes.BindGroupLayoutEntry{Binding: binding, Visibility: ShaderStageFragment,
			Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeDepth, ViewDimension: gputypes.TextureViewDimension2D}}
	}
	pix, err := semanticsDraw(t, device, "depth-kinds", `
@group(0) @binding(0) var plain: texture_depth_2d;
@group(0) @binding(1) var shadow: texture_depth_2d;
@group(0) @binding(2) var s: sampler;
@group(0) @binding(3) var sc: sampler_comparison;
fn sampled(t: texture_depth_2d, uv: vec2<f32>) -> f32 {
    return textureSample(t, s, uv);
}
@fragment fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = p.xy / 4.0;
    return vec4<f32>(sampled(plain, uv), textureLoad(plain, vec2<i32>(p.xy), 0), textureSampleCompare(shadow, sc, uv, 0.5), 1.0);
}
`, []gputypes.BindGroupLayoutEntry{
		depthEntry(0),
		depthEntry(1),
		{Binding: 2, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeNonFiltering}},
		{Binding: 3, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeComparison}},
	}, []BindGroupEntry{{Binding: 0, TextureView: plainView}, {Binding: 1, TextureView: shadowView}, {Binding: 2, Sampler: s}, {Binding: 3, Sampler: sc}})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	expectPixels(t, "sampled and loaded 0.25, compared 1", pix, 64, 64, 255, 255)
	assertNoValidation(t, h)
}

// TestUniformLayoutReads reads a uniform struct whose members WGSL packs
// with two-row matrices 8 bytes a column, padding after them and a u32 and
// an i32 array among the floats, by constant and run-time indices into
// vectors and matrix columns, as a whole, and by run-time indices into a
// matrix and vectors loaded from it. Each 4-byte slot k of the buffer
// holds k+1, as a float or an integer as its member has it, so each value
// read is the slot it was read from, plus 1.
func TestUniformLayoutReads(t *testing.T) {
	device, h := newValidatedDevice(t)
	const wgsl = `
struct U {
    a: vec2<f32>,
    m2: mat2x2<f32>,
    b: f32,
    m3: mat3x2<f32>,
    m4: mat4x2<f32>,
    m33: mat3x3<f32>,
    c: vec3<f32>,
    d: u32,
    arr: array<vec4<i32>, 3>,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<storage, read_write> idx: array<u32>;
@compute @workgroup_size(1) fn cs_main() {
    let i1 = idx[1];
    let i2 = idx[2];
    let i3 = idx[3];
    out[0] = u.m2[1].x;
    out[1] = u.m2[i1].y;
    out[2] = u.m2[idx[0]][i1];
    out[3] = u.m3[2].y;
    out[4] = u.m3[i2].x;
    out[5] = u.m4[i3][i1];
    out[6] = u.m33[i2].z;
    out[7] = u.m33[1][i2];
    out[8] = u.c[i2];
    out[9] = f32(u.d);
    out[10] = f32(u.arr[i2][i3]);
    out[11] = u.b;
    out[12] = u.a[i1];
    let whole = u;
    out[13] = whole.m2[1].y;
    out[14] = whole.c.z;
    out[15] = f32(whole.d);
    out[16] = f32(whole.arr[2].z);
    let m = u.m4;
    out[17] = m[i2].x;
    let c = u.c;
    out[18] = c[i2];
    let v = vec3<f32>(u.b, u.a.x, u.a.y);
    out[19] = v[i2];
}
`
	// The byte offset of each value cs_main writes, in the order it writes
	// them: m2 at 8, b at 24, m3 at 32, m4 at 56, m33 at 96, c at 144, d at
	// 156 and arr at 160.
	offsets := []uint32{16, 20, 12, 52, 48, 84, 136, 120, 152, 156, 204, 24, 4, 20, 152, 156, 200, 72, 152, 4}
	words := make([]uint32, 208/4)
	for k := range words {
		switch {
		case k == 156/4:
			words[k] = uint32(k + 1)
		case k >= 160/4:
			words[k] = uint32(k + 1)
		default:
			words[k] = math.Float32bits(float32(k + 1))
		}
	}
	data := make([]byte, len(words)*4)
	for k, w := range words {
		binary.LittleEndian.PutUint32(data[k*4:], w)
	}
	u, err := device.CreateBuffer(&BufferDescriptor{Label: "uniform-layout", Size: uint64(len(data)),
		Usage: BufferUsageUniform | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(u.Release)
	if err := device.Queue().WriteBuffer(u, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	got := semanticsDispatchTypes(t, device, "uniform-layout", wgsl,
		[]gputypes.BufferBindingType{gputypes.BufferBindingTypeUniform, gputypes.BufferBindingTypeStorage, gputypes.BufferBindingTypeStorage},
		u, semanticsStorage(t, device, "uniform-layout-out", make([]uint32, len(offsets))), semanticsStorage(t, device, "uniform-layout-idx", []uint32{0, 1, 2, 3}))
	for i, off := range offsets {
		if v, want := math.Float32frombits(got[1][i]), float32(off/4+1); v != want {
			t.Errorf("out[%d] = %v, want %v, the value at byte %d", i, v, want, off)
		}
	}
	assertNoValidation(t, h)
}
