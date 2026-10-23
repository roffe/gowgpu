//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"strings"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// Bind groups may be set in any order with the pipelines, as WebGPU allows:
// one at a slot past the current pipeline's layout, and one whose layout
// only the next pipeline takes. Each reaches the backend once a pipeline's
// layout takes it, and every dispatch sees its groups.
func TestBindGroupsSetAheadOfTheirPipeline(t *testing.T) {
	device, h := newValidatedDevice(t)

	storage := func(binding uint32) gputypes.BindGroupLayoutEntry {
		return gputypes.BindGroupLayoutEntry{Binding: binding, Visibility: gputypes.ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}}
	}
	layout := func(entries ...gputypes.BindGroupLayoutEntry) *BindGroupLayout {
		l, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: entries})
		if err != nil {
			t.Fatalf("CreateBindGroupLayout: %v", err)
		}
		t.Cleanup(l.Release)
		return l
	}
	one, two := layout(storage(0)), layout(storage(0), storage(1))
	pipeline := func(src string, layouts ...*BindGroupLayout) *ComputePipeline {
		shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: src})
		if err != nil {
			t.Fatalf("CreateShaderModule: %v", err)
		}
		t.Cleanup(shader.Release)
		pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: layouts})
		if err != nil {
			t.Fatalf("CreatePipelineLayout: %v", err)
		}
		t.Cleanup(pl.Release)
		p, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Layout: pl, Module: shader, EntryPoint: "main"})
		if err != nil {
			if strings.Contains(err.Error(), "ComputeShaders") {
				t.Skipf("no compute shaders: %v", err)
			}
			t.Fatalf("CreateComputePipeline: %v", err)
		}
		t.Cleanup(p.Release)
		return p
	}
	first := pipeline(`
@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@compute @workgroup_size(1) fn main() { a[0] = 1u; }`, one)
	second := pipeline(`
@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(1) @binding(0) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(1) fn main() { a[1] = 2u; b[0] = 3u; }`, one, one)
	third := pipeline(`
@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> c: array<u32>;
@compute @workgroup_size(1) fn main() { c[0] = 4u; }`, two)

	buffer := func() *Buffer {
		b, err := device.CreateBuffer(&BufferDescriptor{Size: 16, Usage: BufferUsageStorage | BufferUsageCopySrc})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		t.Cleanup(b.Release)
		return b
	}
	a, b, c := buffer(), buffer(), buffer()
	group := func(l *BindGroupLayout, bufs ...*Buffer) *BindGroup {
		entries := make([]BindGroupEntry, len(bufs))
		for i, buf := range bufs {
			entries[i] = BindGroupEntry{Binding: uint32(i), Buffer: buf, Size: 16}
		}
		g, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: l, Entries: entries})
		if err != nil {
			t.Fatalf("CreateBindGroup: %v", err)
		}
		t.Cleanup(g.Release)
		return g
	}

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	pass.SetPipeline(first)
	pass.SetBindGroup(0, group(one, a), nil)
	pass.SetBindGroup(1, group(one, b), nil) // past first's layout
	pass.Dispatch(1, 1, 1)
	pass.SetPipeline(second)
	pass.Dispatch(1, 1, 1)
	pass.SetBindGroup(0, group(two, a, c), nil) // a layout only third takes
	pass.SetPipeline(third)
	pass.Dispatch(1, 1, 1)
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	read, err := device.CreateBuffer(&BufferDescriptor{Size: 48, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer read.Release()
	for i, src := range []*Buffer{a, b, c} {
		enc.CopyBufferToBuffer(src, 0, read, uint64(i)*16, 16)
	}
	finishAndSubmit(t, device, enc)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := read.Map(ctx, MapModeRead, 0, 48); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := read.MappedRange(0, 48)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	data := rng.Bytes()
	for _, w := range []struct {
		what string
		at   int
		want uint32
	}{{"a[0]", 0, 1}, {"a[1]", 4, 2}, {"b[0]", 16, 3}, {"c[0]", 32, 4}} {
		if got := binary.LittleEndian.Uint32(data[w.at:]); got != w.want {
			t.Errorf("%s = %d, want %d", w.what, got, w.want)
		}
	}
	if err := read.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	assertNoValidation(t, h)
}

// Two bind group layouts with the same entries listed in another order are
// the same layout: a group of one binds where the pipeline has the other,
// and each binding gets its own buffer.
func TestBindGroupLayoutEntryOrderDoesNotMatter(t *testing.T) {
	device, h := newValidatedDevice(t)

	storage := func(binding uint32) gputypes.BindGroupLayoutEntry {
		return gputypes.BindGroupLayoutEntry{Binding: binding, Visibility: gputypes.ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}}
	}
	layout := func(entries ...gputypes.BindGroupLayoutEntry) *BindGroupLayout {
		l, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: entries})
		if err != nil {
			t.Fatalf("CreateBindGroupLayout: %v", err)
		}
		t.Cleanup(l.Release)
		return l
	}
	forward, backward := layout(storage(0), storage(1)), layout(storage(1), storage(0))
	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: `
@group(0) @binding(0) var<storage, read_write> a: array<u32>;
@group(0) @binding(1) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(1) fn main() { a[0] = 5u; b[0] = 6u; }`})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{backward}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	pipeline, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Layout: pl, Module: shader, EntryPoint: "main"})
	if err != nil {
		if strings.Contains(err.Error(), "ComputeShaders") {
			t.Skipf("no compute shaders: %v", err)
		}
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipeline.Release()
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: 512, Usage: BufferUsageStorage | BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: forward, Entries: []BindGroupEntry{
		{Binding: 0, Buffer: buf, Offset: 0, Size: 16},
		{Binding: 1, Buffer: buf, Offset: 256, Size: 16},
	}})
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
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, group, nil)
	pass.Dispatch(1, 1, 1)
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	read, err := device.CreateBuffer(&BufferDescriptor{Size: 512, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer read.Release()
	enc.CopyBufferToBuffer(buf, 0, read, 0, 512)
	finishAndSubmit(t, device, enc)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := read.Map(ctx, MapModeRead, 0, 512); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := read.MappedRange(0, 512)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	data := rng.Bytes()
	if a, b := binary.LittleEndian.Uint32(data), binary.LittleEndian.Uint32(data[256:]); a != 5 || b != 6 {
		t.Errorf("a[0], b[0] = %d, %d, want 5, 6", a, b)
	}
	if err := read.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	assertNoValidation(t, h)
}
