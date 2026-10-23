// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestDXILStorageScalarWidths copies structs of f16, and of f32, u32, u64
// and f64 fields, through a storage buffer with naga's DXIL, and changes some
// of their fields: each field at its WGSL offset, the padding between them
// left alone. naga loaded an f16 as the bits of a 32-bit word and wrote
// invalid DXIL for a 64-bit field in a struct of mixed types.
func TestDXILStorageScalarWidths(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU integration test in short mode")
	}
	t.Run("f16", func(t *testing.T) {
		// S is a: f16 at 0, b: vec2<f16> at 4, c: f32 at 8, d: vec3<f16>
		// at 16, m: mat2x2<f16> at 24 (columns 4 bytes apart), 32 bytes.
		in := make([]byte, 64)
		for i := range in {
			in[i] = 0xab
		}
		put16 := func(off int, f float32) { binary.LittleEndian.PutUint16(in[off:], f16Bits(f)) }
		put16(0, 1.5)
		put16(4, 2)
		put16(6, 3)
		binary.LittleEndian.PutUint32(in[8:], math.Float32bits(4))
		put16(16, 5)
		put16(18, 6)
		put16(20, 7)
		put16(24, 8)
		put16(26, 9)
		put16(28, 10)
		put16(30, 11)
		got := runDXILStorage(t, `
enable f16;
struct S { a: f16, b: vec2<f16>, c: f32, d: vec3<f16>, m: mat2x2<f16> }
@group(0) @binding(0) var<storage, read_write> s: array<S>;
@compute @workgroup_size(1) fn main() {
	s[1] = s[0];
	s[1].a = s[0].a * 2.0h;
	s[1].b = s[0].b + vec2<f16>(1.0h);
}
`, in)
		want := append([]byte(nil), in...)
		copy(want[32:], in[:32])
		binary.LittleEndian.PutUint16(want[32:], f16Bits(3))
		binary.LittleEndian.PutUint16(want[36:], f16Bits(3))
		binary.LittleEndian.PutUint16(want[38:], f16Bits(4))
		expectBytes(t, got, want)
	})
	t.Run("64-bit", func(t *testing.T) {
		// T is a: f32 at 0, b: u32 at 4, c: u64 at 8, d: f64 at 16, 24 bytes.
		in := make([]byte, 48)
		binary.LittleEndian.PutUint32(in[0:], math.Float32bits(1.5))
		binary.LittleEndian.PutUint32(in[4:], 7)
		binary.LittleEndian.PutUint64(in[8:], 1<<40+3)
		binary.LittleEndian.PutUint64(in[16:], math.Float64bits(2.25))
		got := runDXILStorage(t, `
struct T { a: f32, b: u32, c: u64, d: f64 }
@group(0) @binding(0) var<storage, read_write> s: array<T>;
@compute @workgroup_size(1) fn main() {
	s[1] = s[0];
	s[1].c = s[0].c + 1lu;
	s[1].d = s[0].d * 2.0lf;
}
`, in)
		want := append([]byte(nil), in...)
		copy(want[24:], in[:24])
		binary.LittleEndian.PutUint64(want[32:], 1<<40+4)
		binary.LittleEndian.PutUint64(want[40:], math.Float64bits(4.5))
		expectBytes(t, got, want)
	})
}

// f16Bits is the binary16 encoding of f, a normal number binary16 holds
// exactly.
func f16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	exp := int(b>>23&0xff) - 127 + 15
	return uint16(b>>16&0x8000) | uint16(exp)<<10 | uint16(b>>13&0x3ff)
}

func expectBytes(t *testing.T, got, want []byte) {
	t.Helper()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("byte %d = %#x, want %#x (got % x)", i, got[i], want[i], got)
			return
		}
	}
}

// runDXILStorage runs the one-invocation compute entry point main of wgsl,
// compiled with naga's DXIL, on the first DX12 adapter, with a read-write
// storage buffer of data at binding 0, and returns the buffer. It skips the
// test on a device without the shader ops the WGSL's types need.
func runDXILStorage(t *testing.T, wgsl string, data []byte) []byte {
	t.Helper()
	t.Setenv("GOGPU_DX12_DXIL", "1")
	raw, err := Backend{}.CreateInstance(&hal.InstanceDescriptor{Backends: gputypes.BackendsDX12})
	if err != nil {
		t.Skipf("DX12 unavailable: %v", err)
	}
	t.Cleanup(raw.Destroy)
	adapters := raw.EnumerateAdapters(nil)
	if len(adapters) == 0 {
		t.Skip("no DX12 adapters")
	}
	open, err := adapters[0].Adapter.Open(0, adapters[0].Capabilities.Limits)
	if err != nil {
		t.Skipf("Open: %v", err)
	}
	dev, queue := open.Device, open.Queue
	t.Cleanup(dev.Destroy)
	if missing := missingShaderOps(dev.(*Device).raw, wgsl); missing != "" {
		t.Skipf("the device has no %s", missing)
	}

	size := uint64(len(data))
	newBuf := func(label string, usage gputypes.BufferUsage) hal.Buffer {
		b, err := dev.CreateBuffer(&hal.BufferDescriptor{Label: label, Size: size, Usage: usage})
		if err != nil {
			t.Fatalf("CreateBuffer(%s): %v", label, err)
		}
		t.Cleanup(func() { dev.DestroyBuffer(b) })
		return b
	}
	buf := newBuf("data", gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc|gputypes.BufferUsageCopyDst)
	staging := newBuf("staging", gputypes.BufferUsageCopyDst|gputypes.BufferUsageMapRead)
	if err := queue.WriteBuffer(buf, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	module, err := dev.CreateShaderModule(&hal.ShaderModuleDescriptor{Label: "widths", Source: hal.ShaderSource{WGSL: wgsl}})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(func() { dev.DestroyShaderModule(module) })
	bgl, err := dev.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{Label: "widths", Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: gputypes.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	t.Cleanup(func() { dev.DestroyBindGroupLayout(bgl) })
	bg, err := dev.CreateBindGroup(&hal.BindGroupDescriptor{Label: "widths", Layout: bgl, Entries: []gputypes.BindGroupEntry{
		{Binding: 0, Resource: gputypes.BufferBinding{Buffer: buf.NativeHandle(), Size: size}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(func() { dev.DestroyBindGroup(bg) })
	pl, err := dev.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{Label: "widths", BindGroupLayouts: []hal.BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	t.Cleanup(func() { dev.DestroyPipelineLayout(pl) })
	pipeline, err := dev.CreateComputePipeline(&hal.ComputePipelineDescriptor{Label: "widths", Layout: pl,
		Compute: hal.ComputeState{Module: module, EntryPoint: "main"}})
	if err != nil {
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	t.Cleanup(func() { dev.DestroyComputePipeline(pipeline) })

	enc, err := dev.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "widths"})
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	if err := enc.BeginEncoding("widths"); err != nil {
		t.Fatalf("BeginEncoding: %v", err)
	}
	pass := enc.BeginComputePass(&hal.ComputePassDescriptor{Label: "widths"})
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bg, nil)
	pass.Dispatch(1, 1, 1)
	pass.End()
	enc.CopyBufferToBuffer(buf, staging, []hal.BufferCopy{{Size: size}})
	cmd, err := enc.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding: %v", err)
	}
	if _, err := queue.Submit([]hal.CommandBuffer{cmd}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := dev.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	m, err := dev.MapBuffer(staging, 0, size)
	if err != nil {
		t.Fatalf("MapBuffer: %v", err)
	}
	got := append([]byte(nil), unsafe.Slice((*byte)(m.Ptr), size)...)
	if err := dev.UnmapBuffer(staging); err != nil {
		t.Fatalf("UnmapBuffer: %v", err)
	}
	return got
}

// missingShaderOps names the shader ops wgsl needs that raw doesn't have:
// native 16-bit ones for f16, and 64-bit integer and double ones for u64
// and f64. A query that fails leaves its options zero, none supported.
func missingShaderOps(raw *d3d12.ID3D12Device, wgsl string) string {
	var options d3d12.D3D12_FEATURE_DATA_D3D12_OPTIONS
	var options1 [6]int32 // D3D12_FEATURE_DATA_D3D12_OPTIONS1, Int64ShaderOps last
	var options4 [3]int32 // D3D12_FEATURE_DATA_D3D12_OPTIONS4, Native16BitShaderOpsSupported last
	_ = raw.CheckFeatureSupport(d3d12.D3D12_FEATURE_D3D12_OPTIONS, unsafe.Pointer(&options), uint32(unsafe.Sizeof(options)))
	_ = raw.CheckFeatureSupport(d3d12.D3D12_FEATURE_D3D12_OPTIONS1, unsafe.Pointer(&options1), uint32(unsafe.Sizeof(options1)))
	_ = raw.CheckFeatureSupport(d3d12.D3D12_FEATURE_D3D12_OPTIONS4, unsafe.Pointer(&options4), uint32(unsafe.Sizeof(options4)))
	switch {
	case strings.Contains(wgsl, "f16") && options4[2] == 0:
		return "native 16-bit shader ops"
	case strings.Contains(wgsl, "u64") && options1[5] == 0:
		return "64-bit integer shader ops"
	case strings.Contains(wgsl, "f64") && options.DoublePrecisionFloatShaderOps == 0:
		return "double shader ops"
	}
	return ""
}
