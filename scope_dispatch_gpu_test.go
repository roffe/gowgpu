//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
)

// GPU tests of the usage scopes a compute pass has, one per dispatch, of the
// buffers a bind group binds, and of a submit with more than one command
// buffer. Same setup as scope_transitions_gpu_test.go: the validation layer,
// when it is on the instance, fails them on any message.

// TestSubmitTwoCommandBuffersRenderThenSample: cb1 renders into S, cb2
// samples S, both in ONE Submit. cb2's ColorTarget -> Resource barrier has
// to run between cb1 and cb2, so each command buffer of a submit gets its
// own barrier command buffer right ahead of it, as Rust wgpu-core's
// pre_submit_for_command_buffers does; one for the whole submit, ahead of
// cb1, would put S in SHADER_READ_ONLY_OPTIMAL before cb1 draws into it,
// which the validation layer flags on the draw.
func TestSubmitTwoCommandBuffersRenderThenSample(t *testing.T) {
	device, h := newValidatedDevice(t)
	blit := newBlitRig(t, device)

	src, srcView := pixTexture(t, device, "two-cb-src", TextureUsageRenderAttachment|TextureUsageTextureBinding)
	defer releaseTextureView(src, srcView)
	dst, dstView := pixTexture(t, device, "two-cb-dst", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(dst, dstView)
	group := blit.group(t, device, srcView)
	st := readbackBuffer(t, device, "two-cb-st")

	enc1, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc1, srcView, gputypes.LoadOpClear, red, nil)
	cb1, err := enc1.Finish()
	if err != nil {
		t.Fatalf("Finish cb1: %v", err)
	}

	enc2, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc2, dstView, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) {
		pass.SetPipeline(blit.pipeline)
		pass.SetBindGroup(0, group, nil)
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc2, dst, st)
	cb2, err := enc2.Finish()
	if err != nil {
		t.Fatalf("Finish cb2: %v", err)
	}

	if _, err := device.Queue().Submit(cb1, cb2); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	expectPixels(t, "U sampled in cb2 from S rendered in cb1, one Submit", readback(t, st), 255, 0, 0, 255)
	assertNoValidation(t, h)
}

const loadShader = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
	out[id.y * 4u + id.x] = textureLoad(src, vec2<i32>(id.xy), 0);
}
`

// computeRig is a storage-store pipeline and a load-into-buffer pipeline on
// the same texture.
type computeRig struct {
	store, load *ComputePipeline
	groupA      *BindGroup // T as write-only storage texture
	groupB      *BindGroup // T sampled + out storage buffer
	tex         *Texture
	out         *Buffer
}

// computePipeline makes a compute pipeline over one bind group layout, or
// skips the test on an adapter without compute shaders.
func computePipeline(t *testing.T, device *Device, label, wgsl string, entries []gputypes.BindGroupLayoutEntry) (*ComputePipeline, *BindGroupLayout) {
	t.Helper()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: entries})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout(%s): %v", label, err)
	}
	t.Cleanup(layout.Release)
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{layout}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout(%s): %v", label, err)
	}
	t.Cleanup(pl.Release)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: label, WGSL: wgsl})
	if err != nil {
		t.Fatalf("CreateShaderModule(%s): %v", label, err)
	}
	t.Cleanup(module.Release)
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: label, Layout: pl, Module: module, EntryPoint: "cs_main"})
	if err != nil {
		if strings.Contains(err.Error(), "ComputeShaders") {
			t.Skipf("no compute shaders on this adapter: %v", err)
		}
		t.Fatalf("CreateComputePipeline(%s): %v", label, err)
	}
	t.Cleanup(pipe.Release)
	return pipe, layout
}

func newComputeRig(t *testing.T, device *Device) *computeRig {
	t.Helper()
	store, layoutA := computePipeline(t, device, "scope-store", storeShader, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, StorageTexture: &gputypes.StorageTextureBindingLayout{
			Access: gputypes.StorageTextureAccessWriteOnly, Format: gputypes.TextureFormatRGBA8Unorm, ViewDimension: gputypes.TextureViewDimension2D}},
	})
	load, layoutB := computePipeline(t, device, "scope-load", loadShader, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 1, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
	})

	tex, view := pixTexture(t, device, "scope-pingpong", TextureUsageRenderAttachment|TextureUsageStorageBinding|TextureUsageTextureBinding)
	t.Cleanup(func() { releaseTextureView(tex, view) })
	out, err := device.CreateBuffer(&BufferDescriptor{Label: "scope-out", Size: 256, Usage: BufferUsageStorage | BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer out: %v", err)
	}
	t.Cleanup(out.Release)
	groupA, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layoutA, Entries: []BindGroupEntry{{Binding: 0, TextureView: view}}})
	if err != nil {
		t.Fatalf("CreateBindGroup A: %v", err)
	}
	t.Cleanup(groupA.Release)
	groupB, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layoutB, Entries: []BindGroupEntry{
		{Binding: 0, TextureView: view},
		{Binding: 1, Buffer: out, Offset: 0, Size: 256},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup B: %v", err)
	}
	t.Cleanup(groupB.Release)
	return &computeRig{store: store, load: load, groupA: groupA, groupB: groupB, tex: tex, out: out}
}

// readFloats maps buf and returns its first n float32s.
func readFloats(t *testing.T, buf *Buffer, n int) []float32 {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, MapModeRead, 0, uint64(n*4)); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, uint64(n*4))
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	data := rng.Bytes()
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

func expectFloats4(t *testing.T, what string, f []float32, want [4]float32) {
	t.Helper()
	for i := 0; i+3 < len(f); i += 4 {
		if f[i] != want[0] || f[i+1] != want[1] || f[i+2] != want[2] || f[i+3] != want[3] {
			t.Errorf("%s: element %d = %v, want %v", what, i/4, f[i:i+4], want)
			return
		}
	}
}

// TestComputePassScopeIsPerDispatch: dispatch 1 stores blue into T through
// a storage binding, dispatch 2 (another bind group, another pipeline) reads
// T as a sampled texture into a storage buffer. In ONE compute pass that is
// valid WebGPU, every dispatch is a usage scope of its own (Rust wgpu-core
// compute.rs State::flush_states), and gets a StorageWrite -> Resource
// barrier between the dispatches; one scope for the whole pass would call
// it a conflict. In two passes it is the same work.
func TestComputePassScopeIsPerDispatch(t *testing.T) {
	device, h := newValidatedDevice(t)
	rig := newComputeRig(t, device)
	st := readbackBuffer(t, device, "per-dispatch-st")

	for _, tc := range []struct {
		name    string
		onePass bool
	}{
		{"one pass, two dispatches", true},
		{"two passes", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			cpass, err := enc.BeginComputePass(nil)
			if err != nil {
				t.Fatalf("BeginComputePass: %v", err)
			}
			cpass.SetPipeline(rig.store)
			cpass.SetBindGroup(0, rig.groupA, nil)
			cpass.Dispatch(pixSide, pixSide, 1)
			if !tc.onePass {
				if err := cpass.End(); err != nil {
					t.Fatalf("ComputePass.End: %v", err)
				}
				cpass, err = enc.BeginComputePass(nil)
				if err != nil {
					t.Fatalf("BeginComputePass 2: %v", err)
				}
			}
			cpass.SetPipeline(rig.load)
			cpass.SetBindGroup(0, rig.groupB, nil)
			cpass.Dispatch(pixSide, pixSide, 1)
			if err := cpass.End(); err != nil {
				t.Fatalf("ComputePass.End: %v", err)
			}
			enc.CopyBufferToBuffer(rig.out, 0, st, 0, 256)
			cb, err := enc.Finish()
			if err != nil {
				t.Fatalf("Finish rejected a valid program: %v", err)
			}
			idx := rig.tex.coreTexture.TrackingData().Index()
			if got := cb.core.TextureScope().StartUsage(idx); got != track.TextureUsesStorageWrite {
				t.Errorf("T start usage = %v, want StorageWrite", got)
			}
			if got := cb.core.TextureScope().EndUsage(idx); got != track.TextureUsesResource {
				t.Errorf("T end usage = %v, want Resource", got)
			}
			if _, err := device.Queue().Submit(cb); err != nil {
				t.Fatalf("Submit: %v", err)
			}
			expectFloats4(t, "T loaded after the store", readFloats(t, st, 64), [4]float32{0, 0, 1, 1})
			assertNoValidation(t, h)
		})
	}
}

// TestComputeWriteWriteBetweenDispatches: the same storage-writing bind
// group dispatched twice in one pass. The second dispatch's scope sees
// StorageWrite -> StorageWrite, which is not hardware-ordered, and records
// a barrier between the dispatches (the core test
// TestRecordPassTextureUsage_ComputePassTransitionsInline counts it); under
// synchronization validation a missing one is a write-after-write hazard.
func TestComputeWriteWriteBetweenDispatches(t *testing.T) {
	device, h := newValidatedDevice(t)
	rig := newComputeRig(t, device)
	st := readbackBuffer(t, device, "ww-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cpass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cpass.SetPipeline(rig.store)
	cpass.SetBindGroup(0, rig.groupA, nil)
	cpass.Dispatch(pixSide, pixSide, 1)
	cpass.Dispatch(pixSide, pixSide, 1)
	if err := cpass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	finishAndSubmit(t, device, enc)

	enc, err = device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cpass, err = enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cpass.SetPipeline(rig.load)
	cpass.SetBindGroup(0, rig.groupB, nil)
	cpass.Dispatch(pixSide, pixSide, 1)
	if err := cpass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	enc.CopyBufferToBuffer(rig.out, 0, st, 0, 256)
	finishAndSubmit(t, device, enc)
	expectFloats4(t, "T after two stores", readFloats(t, st, 64), [4]float32{0, 0, 1, 1})
	assertNoValidation(t, h)
}

const fillShader = `
struct Params { v: vec4<f32> };
@group(0) @binding(0) var<storage, read_write> data: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(1) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
	data[id.x] = params.v;
}
`

// bufferRig is a compute pipeline that writes a uniform's value into every
// element of a storage buffer, with the buffers bound: group writes storage
// from uniform, alt writes another buffer from storage, read as a uniform.
type bufferRig struct {
	fill             *ComputePipeline
	group, alt       *BindGroup
	storage, uniform *Buffer
}

func newBufferRig(t *testing.T, device *Device) *bufferRig {
	t.Helper()
	fill, layout := computePipeline(t, device, "scope-fill", fillShader, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
		{Binding: 1, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
	})
	storage, err := device.CreateBuffer(&BufferDescriptor{Label: "scope-storage", Size: 256, Usage: BufferUsageStorage | BufferUsageUniform | BufferUsageVertex | BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer storage: %v", err)
	}
	t.Cleanup(storage.Release)
	uniform, err := device.CreateBuffer(&BufferDescriptor{Label: "scope-uniform", Size: 256, Usage: BufferUsageUniform | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer uniform: %v", err)
	}
	t.Cleanup(uniform.Release)
	other, err := device.CreateBuffer(&BufferDescriptor{Label: "scope-other", Size: 256, Usage: BufferUsageStorage})
	if err != nil {
		t.Fatalf("CreateBuffer other: %v", err)
	}
	t.Cleanup(other.Release)
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{
		{Binding: 0, Buffer: storage, Size: 256},
		{Binding: 1, Buffer: uniform, Size: 256},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(group.Release)
	alt, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{
		{Binding: 0, Buffer: other, Size: 256},
		{Binding: 1, Buffer: storage, Size: 256},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup alt: %v", err)
	}
	t.Cleanup(alt.Release)
	return &bufferRig{fill: fill, group: group, alt: alt, storage: storage, uniform: uniform}
}

// TestComputeBindGroupReplacedBeforeDispatchIsNoConflict: a group that
// writes S is bound and replaced, before the dispatch, by one of the same
// layout that reads S as a uniform. Only the groups a dispatch runs with are
// in its scope, so S is Uniform there and nothing conflicts; a scope filled
// at SetBindGroup would see StorageWrite against Uniform.
func TestComputeBindGroupReplacedBeforeDispatchIsNoConflict(t *testing.T) {
	device, h := newValidatedDevice(t)
	rig := newBufferRig(t, device)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cpass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cpass.SetPipeline(rig.fill)
	cpass.SetBindGroup(0, rig.group, nil)
	cpass.SetBindGroup(0, rig.alt, nil)
	cpass.Dispatch(16, 1, 1)
	if err := cpass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	sIdx := rig.storage.core.TrackingData().Index()
	if got := cb.core.BufferScope().StartUsage(sIdx); got != track.BufferUsesUniform {
		t.Errorf("S start usage = %v, want Uniform (the replaced group is not in the scope)", got)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	assertNoValidation(t, h)
}

// TestBindGroupBuffersInDispatchScope: a bind group's buffers go in the
// dispatch's usage scope like its textures (Rust wgpu-core merge_bind_group):
// the storage buffer as StorageWrite, the uniform as Uniform. A render pass
// of the same encoder that then takes the storage buffer as a vertex buffer
// is transitioned from the write, ahead of the pass, so the pass is split
// off into a command buffer of its own; without the tracking the vertex
// read would count as the buffer's first use.
func TestBindGroupBuffersInDispatchScope(t *testing.T) {
	device, _ := newValidatedDevice(t)
	rig := newBufferRig(t, device)
	target, targetView := pixTexture(t, device, "bufscope-target", TextureUsageRenderAttachment)
	defer releaseTextureView(target, targetView)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	defer enc.DiscardEncoding()
	cpass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cpass.SetPipeline(rig.fill)
	cpass.SetBindGroup(0, rig.group, nil)
	cpass.Dispatch(16, 1, 1)
	if err := cpass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	tracker := enc.core.Mutable().BufferScope()
	sIdx := rig.storage.core.TrackingData().Index()
	uIdx := rig.uniform.core.TrackingData().Index()
	if got := tracker.EndUsage(sIdx); got != track.BufferUsesStorageWrite {
		t.Errorf("storage buffer usage after the dispatch = %v, want StorageWrite", got)
	}
	if got := tracker.EndUsage(uIdx); got != track.BufferUsesUniform {
		t.Errorf("uniform buffer usage after the dispatch = %v, want Uniform", got)
	}

	clearPass(t, enc, targetView, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) {
		pass.SetVertexBuffer(0, rig.storage, 0)
	})
	if got := tracker.StartUsage(sIdx); got != track.BufferUsesStorageWrite {
		t.Errorf("storage buffer start usage = %v, want StorageWrite (the compute write is its first use)", got)
	}
	if got := tracker.EndUsage(sIdx); got != track.BufferUsesVertex {
		t.Errorf("storage buffer end usage = %v, want Vertex", got)
	}
	// The initial command buffer, the transit one with the StorageWrite ->
	// Vertex barrier, and the pass's own; nothing reopened after the pass.
	if got := enc.core.CBListLen(); got != 3 {
		t.Errorf("HAL command buffers after the split pass = %d, want 3", got)
	}
	if enc.core.IsCBOpen() {
		t.Error("a command buffer is open after the split pass, want none until the next command")
	}
}

// TestComputeWriteThenCopyBufferSameEncoder: a dispatch fills the storage
// buffer from the uniform, and the same encoder copies it out. The copy is a
// usage scope of its own, so it is no conflict with the write, and the
// StorageWrite -> CopySrc barrier between them makes the copy see the
// values.
func TestComputeWriteThenCopyBufferSameEncoder(t *testing.T) {
	device, h := newValidatedDevice(t)
	rig := newBufferRig(t, device)
	st := readbackBuffer(t, device, "fill-st")

	var params [16]byte
	for i, v := range [4]float32{1, 2, 3, 4} {
		binary.LittleEndian.PutUint32(params[i*4:], math.Float32bits(v))
	}
	if err := device.Queue().WriteBuffer(rig.uniform, 0, params[:]); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cpass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cpass.SetPipeline(rig.fill)
	cpass.SetBindGroup(0, rig.group, nil)
	cpass.Dispatch(16, 1, 1)
	if err := cpass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	enc.CopyBufferToBuffer(rig.storage, 0, st, 0, 256)
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	sIdx := rig.storage.core.TrackingData().Index()
	if got := cb.core.BufferScope().StartUsage(sIdx); got != track.BufferUsesStorageWrite {
		t.Errorf("storage buffer start usage = %v, want StorageWrite", got)
	}
	if got := cb.core.BufferScope().EndUsage(sIdx); got != track.BufferUsesCopySrc {
		t.Errorf("storage buffer end usage = %v, want CopySrc", got)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	expectFloats4(t, "storage buffer filled then copied in one encoder", readFloats(t, st, 64), [4]float32{1, 2, 3, 4})
	assertNoValidation(t, h)
}
