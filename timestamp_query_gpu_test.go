//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

const stampBusyWGSL = `
@group(0) @binding(0) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var h = id.x;
    for (var i = 0u; i < 2000u; i++) {
        h = (h ^ (h >> 16u)) * 2246822519u + i;
    }
    out[id.x] = h;
}
`

// TestTimestampQueries times a compute pass that does some work and a render
// pass that only clears, with timestamps at both ends of each, and checks
// each pass ends after it begins and takes a believable time.
func TestTimestampQueries(t *testing.T) {
	device, h := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureTimestampQuery))
	qs, err := device.CreateQuerySet(&QuerySetDescriptor{Label: "stamps", Type: QueryTypeTimestamp, Count: 4})
	if err != nil {
		t.Fatalf("CreateQuerySet: %v", err)
	}
	defer qs.Release()

	const invocations = 64 * 1024
	out, err := device.CreateBuffer(&BufferDescriptor{Label: "out", Size: invocations * 4, Usage: BufferUsageStorage})
	if err != nil {
		t.Fatalf("CreateBuffer(out): %v", err)
	}
	defer out.Release()
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: stampBusyWGSL})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipeline, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Module: module, EntryPoint: "main"})
	if err != nil {
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipeline.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: pipeline.GetBindGroupLayout(0), Entries: []BindGroupEntry{{Binding: 0, Buffer: out}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()
	target, view := pixTexture(t, device, "target", TextureUsageRenderAttachment)
	defer releaseTextureView(target, view)

	resolve, err := device.CreateBuffer(&BufferDescriptor{Label: "resolve", Size: 32, Usage: BufferUsageQueryResolve | BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer(resolve): %v", err)
	}
	defer resolve.Release()
	read, err := device.CreateBuffer(&BufferDescriptor{Label: "read", Size: 32, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer(read): %v", err)
	}
	defer read.Release()

	index := func(i uint32) *uint32 { return &i }
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cp, err := enc.BeginComputePass(&ComputePassDescriptor{TimestampWrites: &ComputePassTimestampWrites{
		QuerySet: qs, BeginningOfPassWriteIndex: index(0), EndOfPassWriteIndex: index(1),
	}})
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cp.SetPipeline(pipeline)
	cp.SetBindGroup(0, group, nil)
	cp.Dispatch(invocations/64, 1, 1)
	if err := cp.End(); err != nil {
		t.Fatalf("End(compute): %v", err)
	}
	rp, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
		TimestampWrites:  &RenderPassTimestampWrites{QuerySet: qs, BeginningOfPassWriteIndex: index(2), EndOfPassWriteIndex: index(3)},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	if err := rp.End(); err != nil {
		t.Fatalf("End(render): %v", err)
	}
	enc.ResolveQuerySet(qs, 0, 4, resolve, 0)
	enc.CopyBufferToBuffer(resolve, 0, read, 0, 32)
	finishAndSubmit(t, device, enc)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := read.Map(ctx, MapModeRead, 0, 32); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := read.MappedRange(0, 32)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	var stamps [4]uint64
	for i := range stamps {
		stamps[i] = binary.LittleEndian.Uint64(rng.Bytes()[i*8:])
	}
	if err := read.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}

	period := float64(device.Queue().TimestampPeriod())
	ns := func(a, b uint64) float64 { return float64(b-a) * period }
	t.Logf("ticks %v, %.3f ns a tick: compute %.1f µs, render %.1f µs", stamps, period, ns(stamps[0], stamps[1])/1e3, ns(stamps[2], stamps[3])/1e3)
	for i, s := range stamps {
		if s == 0 {
			t.Errorf("timestamp %d was not written", i)
		}
	}
	if stamps[1] <= stamps[0] {
		t.Errorf("the compute pass ends at %d, not after it begins at %d", stamps[1], stamps[0])
	} else if d := ns(stamps[0], stamps[1]); d > 1e9 {
		t.Errorf("the compute pass took %.0f ms", d/1e6)
	}
	if stamps[3] < stamps[2] {
		t.Errorf("the render pass ends at %d, before it begins at %d", stamps[3], stamps[2])
	}
	assertNoValidation(t, h)
}
