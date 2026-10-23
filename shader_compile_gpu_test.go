//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// readU32s copies n words of a storage buffer out and returns them.
func readU32s(t *testing.T, device *Device, buf *Buffer, n int) []uint32 {
	t.Helper()
	size := uint64(n * 4)
	staging, err := device.CreateBuffer(&BufferDescriptor{Label: "read-u32s", Size: size, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer staging.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToBuffer(buf, 0, staging, 0, size)
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := staging.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := staging.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	out := make([]uint32, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(rng.Bytes()[i*4:])
	}
	if err := staging.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

// skipWithoutComputeShaders skips the test when a compute pipeline failed
// for want of compute shaders, as on desktop GL 3.3.
func skipWithoutComputeShaders(t *testing.T, err error) {
	t.Helper()
	if strings.Contains(err.Error(), "ComputeShaders") {
		t.Skipf("no compute shaders on this adapter: %v", err)
	}
}

func newStorageBuffer(t *testing.T, device *Device, size uint64) *Buffer {
	t.Helper()
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "storage", Size: size, Usage: BufferUsageStorage | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(buf.Release)
	return buf
}

func storageLayout(t *testing.T, device *Device, kind gputypes.BufferBindingType) *BindGroupLayout {
	t.Helper()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: kind}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	t.Cleanup(layout.Release)
	return layout
}

const groupOneStoreShader = `
@group(1) @binding(0) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn cs_main() {
	out[0] = 7u;
}
`

// TestOneModuleUnderTwoLayouts uses one shader module in pipelines whose
// layouts have different groups ahead of the one it binds, so the register
// its binding takes differs between them.
func TestOneModuleUnderTwoLayouts(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "group-one-store", WGSL: groupOneStoreShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	uniform := storageLayout(t, device, gputypes.BufferBindingTypeUniform)
	storage := storageLayout(t, device, gputypes.BufferBindingTypeStorage)

	pipeline := func(group0 *BindGroupLayout) *ComputePipeline {
		pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{group0, storage}})
		if err != nil {
			t.Fatalf("CreatePipelineLayout: %v", err)
		}
		t.Cleanup(pl.Release)
		pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "group-one-store", Layout: pl, Module: module, EntryPoint: "cs_main"})
		if err != nil {
			skipWithoutComputeShaders(t, err)
			t.Fatalf("CreateComputePipeline: %v", err)
		}
		t.Cleanup(pipe.Release)
		return pipe
	}
	_ = pipeline(uniform)
	second := pipeline(storage)

	group := func(layout *BindGroupLayout, buf *Buffer) *BindGroup {
		g, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: buf, Size: 256}}})
		if err != nil {
			t.Fatalf("CreateBindGroup: %v", err)
		}
		t.Cleanup(g.Release)
		return g
	}
	group0Buf := newStorageBuffer(t, device, 256)
	outBuf := newStorageBuffer(t, device, 256)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	pass.SetPipeline(second)
	pass.SetBindGroup(0, group(storage, group0Buf), nil)
	pass.SetBindGroup(1, group(storage, outBuf), nil)
	pass.Dispatch(1, 1, 1)
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	finishAndSubmit(t, device, enc)
	if got := readU32s(t, device, outBuf, 1)[0]; got != 7 {
		t.Errorf("group 1 buffer = %d, want 7", got)
	}
	if got := readU32s(t, device, group0Buf, 1)[0]; got != 0 {
		t.Errorf("group 0 buffer = %d, want it left at 0", got)
	}
	assertNoValidation(t, h)
}

// TestConcurrentPipelineCreationFromOneModule makes pipelines from one
// shader module on several goroutines at once.
func TestConcurrentPipelineCreationFromOneModule(t *testing.T) {
	device, _ := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "group-one-store", WGSL: groupOneStoreShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	uniform := storageLayout(t, device, gputypes.BufferBindingTypeUniform)
	storage := storageLayout(t, device, gputypes.BufferBindingTypeStorage)
	layouts := make([]*PipelineLayout, 2)
	for i, group0 := range []*BindGroupLayout{uniform, storage} {
		layouts[i], err = device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{group0, storage}})
		if err != nil {
			t.Fatalf("CreatePipelineLayout: %v", err)
		}
		defer layouts[i].Release()
	}

	probe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "probe", Layout: layouts[0], Module: module, EntryPoint: "cs_main"})
	if err != nil {
		skipWithoutComputeShaders(t, err)
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	probe.Release()

	var wg sync.WaitGroup
	errs := make(chan error, 8*10)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 10; i++ {
				pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "concurrent", Layout: layouts[(g+i)%2], Module: module, EntryPoint: "cs_main"})
				if err != nil {
					errs <- err
					continue
				}
				pipe.Release()
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("CreateComputePipeline: %v", err)
	}
}

const subsetVaryingsShader = `
struct VOut {
	@builtin(position) pos: vec4<f32>,
	@location(0) a: vec4<f32>,
	@location(1) b: vec4<f32>,
}
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
	var o: VOut;
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	o.pos = vec4<f32>(x, y, 0.5, 1.0);
	o.a = vec4<f32>(1.0, 0.0, 0.0, 1.0);
	o.b = vec4<f32>(0.0, 1.0, 0.0, 1.0);
	return o;
}
@fragment fn fs_main(@location(1) b: vec4<f32>) -> @location(0) vec4<f32> {
	return b;
}
`

// TestFragmentReadsASubsetOfTheVertexOutputs draws with a fragment stage
// that reads the second of two vertex outputs only.
func TestFragmentReadsASubsetOfTheVertexOutputs(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "subset-varyings", WGSL: subsetVaryingsShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "subset-varyings",
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	tex, view := pixTexture(t, device, "subset-varyings", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	staging := readbackBuffer(t, device, "subset-varyings")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) {
		pass.SetPipeline(pipe)
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, tex, staging)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "fragment reading location 1", readback(t, staging), 0, 255, 0, 255)
	assertNoValidation(t, h)
}
