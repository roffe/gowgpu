//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"errors"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
)

// A buffer the GC collects while it is mapped keeps its mapping: a slice of
// it holds no reference to the buffer and may still be read.
func TestBufferCollectedWhileMappedKeepsMapping(t *testing.T) {
	inst, adapter, device := newTestDevice(t)
	defer inst.Release()
	defer adapter.Release()
	defer device.Release()

	const size = 1 << 20
	coreBuf, released, data := func() (*core.Buffer, *atomic.Bool, []byte) {
		buf, err := device.CreateBuffer(&BufferDescriptor{
			Label: "collected-while-mapped",
			Size:  size,
			Usage: BufferUsageMapRead | BufferUsageCopyDst,
		})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := buf.Map(ctx, MapModeRead, 0, size); err != nil {
			t.Fatalf("Map: %v", err)
		}
		rng, err := buf.MappedRange(0, size)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		return buf.core, buf.released, rng.Bytes()
	}()

	for i := 0; !released.Load(); i++ {
		if i == 200 {
			t.Fatal("the GC never collected the buffer")
		}
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}
	if coreBuf.IsDestroyed() {
		t.Fatal("the GC destroyed a mapped buffer, whose mapping a slice still points into")
	}
	if got := coreBuf.CurrentMapState(); got != core.BufferMapStateMapped {
		t.Fatalf("map state after the GC = %v, want mapped", got)
	}
	var sum byte
	for _, b := range data {
		sum |= b
	}
	if sum != 0 {
		t.Errorf("the mapping of a new buffer holds %#x, want zeros", sum)
	}
}

// busyWGSL keeps one GPU thread busy for data[1] rounds, long enough for a
// map that waits for the submission to see its short context end first.
const busyWGSL = `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;

@compute @workgroup_size(1)
fn main() {
	var x = data[0];
	for (var i = 0u; i < data[1]; i++) {
		x = x * 1664525u + 1013904223u;
	}
	data[0] = x;
}
`

// submitBusy submits busyCommands and returns the submission's index.
func submitBusy(t *testing.T, device *Device, rounds uint32, dst *Buffer) uint64 {
	t.Helper()
	idx, err := device.Queue().Submit(busyCommands(t, device, rounds, dst))
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	return idx
}

// busyCommands records a dispatch that runs for rounds rounds and then a
// copy of its result into dst. The test releases what it uses when it ends.
// It skips the test on an adapter without compute shaders.
func busyCommands(t *testing.T, device *Device, rounds uint32, dst *Buffer) *CommandBuffer {
	t.Helper()
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "busy", WGSL: busyWGSL})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(module.Release)
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	t.Cleanup(layout.Release)
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{layout}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	t.Cleanup(pl.Release)
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "busy", Layout: pl, Module: module, EntryPoint: "main"})
	if err != nil {
		if strings.Contains(err.Error(), "downlevel") {
			t.Skipf("no compute shaders: %v", err)
		}
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	t.Cleanup(pipe.Release)
	data, err := device.CreateBuffer(&BufferDescriptor{Label: "busy", Size: 8, Usage: BufferUsageStorage | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(data.Release)
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: data}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(group.Release)
	init := make([]byte, 8)
	binary.LittleEndian.PutUint32(init[4:], rounds)
	if err := device.Queue().WriteBuffer(data, 0, init); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

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
	if dst != nil {
		enc.CopyBufferToBuffer(data, 0, dst, 0, 4)
	}
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return cb
}

// A Map whose context ends before the GPU finishes leaves the buffer
// unmapped, also once the GPU is done, instead of mapping it behind the
// caller's back.
func TestBufferMapContextEndCancelsTheMap(t *testing.T) {
	inst, adapter, device := newTestDevice(t)
	defer inst.Release()
	defer adapter.Release()
	defer device.Release()

	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "map-ctx", Size: 4, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	submitBusy(t, device, 50_000_000, buf)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Millisecond)
	defer cancel()
	err = buf.Map(ctx, MapModeRead, 0, 4)
	if err == nil {
		t.Skip("the GPU finished before the context ended")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Map = %v, want context.DeadlineExceeded", err)
	}
	device.Poll(PollWait)
	if got := buf.MapState(); got != MapStateUnmapped {
		t.Fatalf("state once the GPU is done = %v, want unmapped", got)
	}
	if err := buf.Map(context.Background(), MapModeRead, 0, 4); err != nil {
		t.Fatalf("Map again: %v", err)
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
}

// A Map waits for the submission its map waits for alone: a Submit on
// another goroutine goes on meanwhile, and the work it submits is not
// waited for.
func TestBufferMapWaitsForItsOwnSubmission(t *testing.T) {
	inst, adapter, device := newTestDevice(t)
	defer inst.Release()
	defer adapter.Release()
	defer device.Release()

	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "map-own", Size: 4, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	later := busyCommands(t, device, 50_000_000, nil)
	start := time.Now()
	first := submitBusy(t, device, 50_000_000, buf)

	type result struct {
		err       error
		completed uint64
	}
	mapped := make(chan result, 1)
	go func() {
		err := buf.Map(context.Background(), MapModeRead, 0, 4)
		mapped <- result{err, device.Queue().Poll()}
	}()
	for buf.MapState() == MapStateUnmapped {
		time.Sleep(time.Millisecond)
	}
	time.Sleep(10 * time.Millisecond)
	submitStart := time.Now()
	second, err := device.Queue().Submit(later)
	if err != nil {
		t.Fatalf("Submit: %v", err)
	}
	submitTook := time.Since(submitStart)
	r := <-mapped
	if r.err != nil {
		t.Fatalf("Map: %v", r.err)
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	device.Poll(PollWait)
	busy := time.Since(start) / 2

	if r.completed < first {
		t.Errorf("Map returned with submission %d done, before its own, %d", r.completed, first)
	}
	if r.completed >= second {
		t.Errorf("Map waited for submission %d, made after the map", second)
	}
	if submitTook > busy/2 {
		if submitWaitsForWork(t, device) {
			t.Skip("the driver holds a Submit until the work before it is done")
		}
		t.Errorf("a Submit made while Map waited took %v, a dispatch %v", submitTook, busy)
	}
}

// submitWaitsForWork reports whether a Submit made while the GPU runs a
// dispatch submitted before it returns only once the dispatch is done, as
// lavapipe's does.
func submitWaitsForWork(t *testing.T, device *Device) bool {
	t.Helper()
	busy, later := busyCommands(t, device, 50_000_000, nil), busyCommands(t, device, 1, nil)
	start := time.Now()
	if _, err := device.Queue().Submit(busy); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	time.Sleep(10 * time.Millisecond)
	submitStart := time.Now()
	if _, err := device.Queue().Submit(later); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	submitTook := time.Since(submitStart)
	device.Poll(PollWait)
	return submitTook > time.Since(start)/2
}

// Release fails a map still pending, and afterwards Poll and Map report
// the device gone instead of reaching the destroyed HAL device.
func TestDeviceReleaseFailsPendingMaps(t *testing.T) {
	inst, adapter, device := newTestDevice(t)
	defer inst.Release()
	defer adapter.Release()

	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "released-map", Size: 4, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	if err := device.Queue().WriteBuffer(buf, 0, make([]byte, 4)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	if _, err := device.Queue().Submit(); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	pending, err := buf.MapAsync(MapModeRead, 0, 4)
	if err != nil {
		t.Fatalf("MapAsync: %v", err)
	}
	defer pending.Release()

	device.Release()

	if done, err := pending.Status(); !done || !errors.Is(err, ErrMapDeviceLost) {
		t.Errorf("map pending at Release = %v, %v, want done with ErrMapDeviceLost", done, err)
	}
	if device.Poll(PollPoll) || device.Poll(PollWait) {
		t.Error("Poll after Release found maps to resolve")
	}
	if err := buf.Map(context.Background(), MapModeRead, 0, 4); !errors.Is(err, ErrReleased) {
		t.Errorf("Map after Release = %v, want ErrReleased", err)
	}
}

// An encoder made before Release records nothing more after it and fails
// to finish, in a pass or not, instead of recording into the command
// buffers of the destroyed device.
func TestDeviceReleaseStopsEncoders(t *testing.T) {
	inst, adapter, device := newTestDevice(t)
	defer inst.Release()
	defer adapter.Release()

	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "released-dst", Size: 4, Usage: BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	src, err := device.CreateBuffer(&BufferDescriptor{Label: "released-src", Size: 4, Usage: BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer src.Release()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size:          Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageRenderAttachment,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer view.Release()

	inPass, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := inPass.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
		View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
	}}})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	copying, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}

	device.Release()

	pass.SetViewport(gputypes.Viewport{Width: 4, Height: 4, MaxDepth: 1})
	_ = pass.End()
	if _, err := inPass.Finish(); err == nil {
		t.Error("Finish of an encoder in a pass at Release succeeded")
	}
	copying.CopyBufferToBuffer(src, 0, buf, 0, 4)
	if _, err := copying.Finish(); err == nil {
		t.Error("Finish of an encoder made before Release succeeded")
	}
}
