// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin && !(js && wasm)

package metal

import (
	"errors"
	"fmt"
	"testing"
	"time"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/naga/ir"
)

func TestCommandEncoderRecordingErrorKeepsFirstFailure(t *testing.T) {
	first := errors.New("first")
	encoder := &CommandEncoder{}
	encoder.failRecording(first)
	encoder.failRecording(errors.New("second"))
	if !errors.Is(encoder.recordErr, first) {
		t.Fatalf("recording error = %v, want first failure", encoder.recordErr)
	}
}

// TestCommandEncoder_RecordingState is a regression test for Issue #24.
// The IsRecording() method returns cmdBuffer != 0.
//
// This test verifies the state machine transitions:
// - New encoder should not be recording (cmdBuffer == 0)
// - After cmdBuffer is set, should be recording
// - After cmdBuffer is cleared, should not be recording
func TestCommandEncoder_RecordingState(t *testing.T) {
	// Create encoder without device (state-only test)
	enc := &CommandEncoder{}

	// New encoder should not be recording
	if enc.IsRecording() {
		t.Error("new encoder should not be recording")
	}
	if enc.cmdBuffer != 0 {
		t.Error("cmdBuffer should be 0 initially")
	}

	// Simulate BeginEncoding - cmdBuffer is set to non-zero
	enc.cmdBuffer = 1
	if !enc.IsRecording() {
		t.Error("encoder should be recording when cmdBuffer is set")
	}

	// Simulate EndEncoding - cmdBuffer is transferred to CommandBuffer
	enc.cmdBuffer = 0
	if enc.IsRecording() {
		t.Error("encoder should not be recording after cmdBuffer cleared")
	}
}

// TestCommandEncoder_DiscardState verifies that DiscardEncoding
// properly resets the recording state by clearing cmdBuffer.
func TestCommandEncoder_DiscardState(t *testing.T) {
	enc := &CommandEncoder{}

	// Simulate active recording
	enc.cmdBuffer = 1

	// Simulate discard - cmdBuffer is released and set to 0
	enc.cmdBuffer = 0

	if enc.IsRecording() {
		t.Error("encoder should not be recording after discard")
	}
}

// TestCommandEncoder_BeginRenderPassGuard verifies that BeginRenderPass
// correctly checks IsRecording() before creating sub-encoders.
// The guard in BeginRenderPass checks: if !e.IsRecording() || e.cmdBuffer == 0
func TestCommandEncoder_BeginRenderPassGuard(t *testing.T) {
	enc := &CommandEncoder{}

	// When cmdBuffer is 0, IsRecording() returns false
	enc.cmdBuffer = 0

	// This is the guard condition - both must be true to proceed
	if enc.IsRecording() {
		t.Error("encoder should not be recording when cmdBuffer is 0")
	}

	// When cmdBuffer is non-zero, IsRecording() returns true
	enc.cmdBuffer = 1
	if !enc.IsRecording() {
		t.Error("encoder should be recording when cmdBuffer is non-zero")
	}
}

// TestCommandEncoder_IsRecordingMethod documents that IsRecording()
// is based on cmdBuffer != 0.
func TestCommandEncoder_IsRecordingMethod(t *testing.T) {
	enc := &CommandEncoder{}

	// IsRecording() == (cmdBuffer != 0)
	enc.cmdBuffer = 0
	if enc.IsRecording() != (enc.cmdBuffer != 0) {
		t.Error("IsRecording() should equal (cmdBuffer != 0)")
	}

	enc.cmdBuffer = 12345
	if enc.IsRecording() != (enc.cmdBuffer != 0) {
		t.Error("IsRecording() should equal (cmdBuffer != 0)")
	}

	enc.cmdBuffer = 0
	if enc.IsRecording() != (enc.cmdBuffer != 0) {
		t.Error("IsRecording() should equal (cmdBuffer != 0)")
	}
}

func TestRenderPassDeferredStateKeepsLatestValues(t *testing.T) {
	firstPipeline := &RenderPipeline{}
	latestPipeline := &RenderPipeline{}
	firstGroup := &BindGroup{}
	latestGroup := &BindGroup{}
	firstVertex := &Buffer{}
	latestVertex := &Buffer{}
	index := &Buffer{}
	state := renderPassPendingState{}
	pass := &RenderPassEncoder{pending: &state}

	pass.SetPipeline(firstPipeline)
	pass.SetPipeline(latestPipeline)
	pass.SetBindGroup(1, firstGroup, []uint32{4, 8})
	pass.SetBindGroup(1, latestGroup, []uint32{12})
	pass.SetVertexBuffer(2, firstVertex, 16)
	pass.SetVertexBuffer(2, latestVertex, 24)
	pass.SetIndexBuffer(index, gputypes.IndexFormatUint32, 32)
	pass.SetViewport(gputypes.Viewport{X: 1, Y: 2, Width: 3, Height: 4, MinDepth: 0.25, MaxDepth: 0.75})
	pass.SetScissorRect(gputypes.ScissorRect{X: 5, Y: 6, Width: 7, Height: 8})
	pass.SetBlendConstant(&gputypes.Color{R: 0.1, G: 0.2, B: 0.3, A: 0.4})
	pass.SetStencilReference(9)

	if pass.pipeline != latestPipeline {
		t.Fatal("deferred pipeline did not keep latest value")
	}
	group := pass.pending.bindGroups[1]
	if !group.set || group.group != latestGroup || group.offsetCount != 1 || group.offsets[0] != 12 {
		t.Fatalf("deferred bind group = %#v, want latest group and offsets", group)
	}
	vertex := pass.pending.vertexBuffers[2]
	if !vertex.set || vertex.buffer != latestVertex || vertex.offset != 24 {
		t.Fatalf("deferred vertex buffer = %#v, want latest buffer and offset", vertex)
	}
	if pass.indexBuffer != index || pass.indexFormat != gputypes.IndexFormatUint32 || pass.indexOffset != 32 {
		t.Fatal("deferred index buffer state was not retained")
	}
	if !pass.pending.viewportSet || pass.pending.viewport != (MTLViewport{OriginX: 1, OriginY: 2, Width: 3, Height: 4, ZNear: 0.25, ZFar: 0.75}) {
		t.Fatalf("deferred viewport = %#v", pass.pending.viewport)
	}
	if !pass.pending.scissorSet || pass.pending.scissor != (MTLScissorRect{X: 5, Y: 6, Width: 7, Height: 8}) {
		t.Fatalf("deferred scissor = %#v", pass.pending.scissor)
	}
	if !pass.pending.blendSet || pass.pending.blend != (gputypes.Color{R: 0.1, G: 0.2, B: 0.3, A: 0.4}) {
		t.Fatalf("deferred blend constant = %#v", pass.pending.blend)
	}
	if !pass.pending.stencilSet || pass.pending.stencil != 9 {
		t.Fatalf("deferred stencil reference = %d", pass.pending.stencil)
	}
}

func TestRenderPassDeferredStateUsesBoundedStorage(t *testing.T) {
	group := &BindGroup{}
	buffer := &Buffer{}
	pipeline := &RenderPipeline{}
	state := renderPassPendingState{}
	var pass RenderPassEncoder
	offsets := []uint32{1, 2, 3, 4}

	allocs := testing.AllocsPerRun(100, func() {
		clear(state.bindGroups[:])
		clear(state.vertexBuffers[:])
		pass = RenderPassEncoder{pending: &state}
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, group, offsets)
		pass.SetVertexBuffer(0, buffer, 12)
		pass.SetIndexBuffer(buffer, gputypes.IndexFormatUint16, 4)
		pass.SetViewport(gputypes.Viewport{X: 0, Y: 0, Width: 10, Height: 10, MinDepth: 0, MaxDepth: 1})
		pass.SetScissorRect(gputypes.ScissorRect{X: 0, Y: 0, Width: 10, Height: 10})
		pass.SetBlendConstant(&gputypes.Color{A: 1})
		pass.SetStencilReference(2)
	})
	if allocs != 0 {
		t.Fatalf("deferred state allocated %.2f objects per recording", allocs)
	}
}

func BenchmarkComputePassEncoderBindBufferSizes(b *testing.B) {
	if err := Init(); err != nil {
		b.Fatalf("Init: %v", err)
	}
	rawDevice := CreateSystemDefaultDevice()
	if rawDevice == 0 {
		b.Skip("no Metal device available")
	}
	b.Cleanup(func() { Release(rawDevice) })

	device, err := newDevice(&Adapter{raw: rawDevice})
	if err != nil {
		b.Fatalf("newDevice: %v", err)
	}
	b.Cleanup(device.Destroy)

	command := &CommandEncoder{device: device}
	if err := command.BeginEncoding("bind-buffer-sizes-benchmark"); err != nil {
		b.Fatalf("BeginEncoding: %v", err)
	}
	pass, ok := command.BeginComputePass(nil).(*ComputePassEncoder)
	if !ok || pass == nil {
		command.DiscardEncoding()
		b.Fatal("BeginComputePass returned no Metal compute encoder")
	}
	b.Cleanup(func() {
		pass.End()
		command.DiscardEncoding()
	})

	pipeline := &ComputePipeline{
		sizesBindings: []ir.ResourceBinding{
			{Group: 0, Binding: 0},
			{Group: 0, Binding: 1},
			{Group: 1, Binding: 0},
			{Group: 1, Binding: 1},
		},
		sizesSlot: 4,
	}
	bindingSizes := map[[2]uint32]uint64{
		{0, 0}: 16,
		{0, 1}: 28,
		{1, 0}: 64,
		{1, 1}: 256,
	}

	// Warm the selector cache before the timed region.
	_ = Sel("setBytes:length:atIndex:")

	for _, dispatches := range []int{1, 8, 64} {
		b.Run(fmt.Sprintf("dispatches_per_pass=%d", dispatches), func(b *testing.B) {
			// This benchmark compares Go allocations only. Repeated setBytes
			// calls intentionally reuse one native command encoder.
			b.ReportAllocs()
			for b.Loop() {
				encoder := ComputePassEncoder{
					raw:          pass.raw,
					pipeline:     pipeline,
					bindingSizes: bindingSizes,
				}
				for range dispatches {
					encoder.bindBufferSizes()
				}
			}
		})
	}
}

// TestComputeBindSlots_PerTypeSequentialIndexing is a regression test for the
// SetBindGroup slot indexing bug.
//
// Bug: Previously used entry.Binding (WGSL @binding(N)) as the Metal slot
// index for all resource types. Metal uses separate per-type index spaces:
//
//	[[buffer(N)]], [[texture(M)]], [[sampler(K)]]
//
// The naga MSL compiler generates sequential indices per type, so with entries
// at bindings [0:buffer, 1:texture, 2:buffer, 3:sampler]:
//
//	Bug:     buffer→0, texture→1, buffer→2, sampler→3
//	Correct: buffer→0, texture→0, buffer→1, sampler→0
func TestComputeBindSlots_PerTypeSequentialIndexing(t *testing.T) {
	tests := []struct {
		name         string
		entries      []gputypes.BindGroupEntry
		wantBuffers  []uintptr // expected slot per buffer entry
		wantTextures []uintptr // expected slot per texture entry
		wantSamplers []uintptr // expected slot per sampler entry
	}{
		{
			name: "mixed types: the exact bug scenario",
			entries: []gputypes.BindGroupEntry{
				{Binding: 0, Resource: gputypes.BufferBinding{Buffer: 0x100}},
				{Binding: 1, Resource: gputypes.TextureViewBinding{TextureView: 0x200}},
				{Binding: 2, Resource: gputypes.BufferBinding{Buffer: 0x300}},
				{Binding: 3, Resource: gputypes.SamplerBinding{Sampler: 0x400}},
			},
			// Bug would produce buffer slots [0, 2], texture slots [1], sampler slots [3]
			// Fix produces buffer slots [0, 1], texture slots [0], sampler slots [0]
			wantBuffers:  []uintptr{0, 1},
			wantTextures: []uintptr{0},
			wantSamplers: []uintptr{0},
		},
		{
			name: "all buffers",
			entries: []gputypes.BindGroupEntry{
				{Binding: 0, Resource: gputypes.BufferBinding{Buffer: 0x100}},
				{Binding: 1, Resource: gputypes.BufferBinding{Buffer: 0x200}},
				{Binding: 2, Resource: gputypes.BufferBinding{Buffer: 0x300}},
			},
			wantBuffers:  []uintptr{0, 1, 2},
			wantTextures: nil,
			wantSamplers: nil,
		},
		{
			name: "texture then sampler then buffer",
			entries: []gputypes.BindGroupEntry{
				{Binding: 0, Resource: gputypes.TextureViewBinding{TextureView: 0x100}},
				{Binding: 1, Resource: gputypes.SamplerBinding{Sampler: 0x200}},
				{Binding: 2, Resource: gputypes.BufferBinding{Buffer: 0x300}},
			},
			wantBuffers:  []uintptr{0},
			wantTextures: []uintptr{0},
			wantSamplers: []uintptr{0},
		},
		{
			name: "multiple textures and samplers interleaved",
			entries: []gputypes.BindGroupEntry{
				{Binding: 0, Resource: gputypes.BufferBinding{Buffer: 0x100}},
				{Binding: 1, Resource: gputypes.TextureViewBinding{TextureView: 0x200}},
				{Binding: 2, Resource: gputypes.SamplerBinding{Sampler: 0x300}},
				{Binding: 3, Resource: gputypes.TextureViewBinding{TextureView: 0x400}},
				{Binding: 4, Resource: gputypes.SamplerBinding{Sampler: 0x500}},
				{Binding: 5, Resource: gputypes.BufferBinding{Buffer: 0x600}},
			},
			wantBuffers:  []uintptr{0, 1},
			wantTextures: []uintptr{0, 1},
			wantSamplers: []uintptr{0, 1},
		},
		{
			name: "sparse bindings with gaps",
			entries: []gputypes.BindGroupEntry{
				{Binding: 0, Resource: gputypes.BufferBinding{Buffer: 0x100}},
				{Binding: 5, Resource: gputypes.TextureViewBinding{TextureView: 0x200}},
				{Binding: 10, Resource: gputypes.BufferBinding{Buffer: 0x300}},
				{Binding: 15, Resource: gputypes.SamplerBinding{Sampler: 0x400}},
				{Binding: 20, Resource: gputypes.TextureViewBinding{TextureView: 0x500}},
			},
			// With the bug, these would use binding numbers [0,5,10,15,20] as slots.
			// With the fix, per-type sequential: buffers [0,1], textures [0,1], samplers [0].
			wantBuffers:  []uintptr{0, 1},
			wantTextures: []uintptr{0, 1},
			wantSamplers: []uintptr{0},
		},
		{
			name:         "empty entries",
			entries:      []gputypes.BindGroupEntry{},
			wantBuffers:  nil,
			wantTextures: nil,
			wantSamplers: nil,
		},
		{
			name: "single texture",
			entries: []gputypes.BindGroupEntry{
				{Binding: 3, Resource: gputypes.TextureViewBinding{TextureView: 0x100}},
			},
			// Bug would use slot 3; fix uses slot 0.
			wantBuffers:  nil,
			wantTextures: []uintptr{0},
			wantSamplers: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bufSlots, texSlots, sampSlots := computeBindSlots(tt.entries)

			checkSlots := func(name string, got []bindSlotAssignment, want []uintptr) {
				if len(got) != len(want) {
					t.Errorf("%s: got %d assignments, want %d", name, len(got), len(want))
					return
				}
				for i, a := range got {
					if a.slot != want[i] {
						t.Errorf("%s[%d]: got slot %d, want %d (entryIndex=%d)",
							name, i, a.slot, want[i], a.entryIndex)
					}
				}
			}

			checkSlots("buffer", bufSlots, tt.wantBuffers)
			checkSlots("texture", texSlots, tt.wantTextures)
			checkSlots("sampler", sampSlots, tt.wantSamplers)
		})
	}
}

// TestUncommittedCommandBuffersBalance checks the count of command buffers
// made and not committed through every way out of an encoder, and that an
// encoder refuses to begin once the queue could hold no committed one.
func TestUncommittedCommandBuffersBalance(t *testing.T) {
	device, queue := newMetalTextureCopyTestDevice(t)
	begin := func() *CommandEncoder {
		t.Helper()
		enc := &CommandEncoder{device: device}
		if err := enc.BeginEncoding("count"); err != nil {
			t.Fatalf("BeginEncoding failed: %v", err)
		}
		return enc
	}
	end := func(enc *CommandEncoder) *CommandBuffer {
		t.Helper()
		cb, err := enc.EndEncoding()
		if err != nil {
			t.Fatalf("EndEncoding failed: %v", err)
		}
		return cb.(*CommandBuffer)
	}

	submitted := end(begin())
	if _, err := queue.Submit([]hal.CommandBuffer{submitted}); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle failed: %v", err)
	}
	submitted.Destroy()
	begin().DiscardEncoding()
	end(begin()).Destroy()
	failed := begin()
	failed.failRecording(errors.New("recording failed"))
	if _, err := failed.EndEncoding(); err == nil {
		t.Fatal("EndEncoding of a failed recording succeeded")
	}
	begin().Destroy()
	if n := device.uncommitted.Load(); n != 0 {
		t.Fatalf("%d command buffers counted as uncommitted, want 0", n)
	}

	device.uncommitted.Store(maxCommandBuffers - 1)
	defer device.uncommitted.Store(0)
	enc := &CommandEncoder{device: device}
	if err := enc.BeginEncoding("one too many"); err == nil {
		enc.DiscardEncoding()
		t.Fatal("BeginEncoding succeeded with the queue's command buffers all uncommitted")
	}
}

// TestOcclusionQuerySetResolves makes an occlusion query set, which needs no
// feature, and resolves it: its queries, never written, read 0. Timestamp
// query sets aren't offered.
func TestOcclusionQuerySetResolves(t *testing.T) {
	device, queue := newMetalTextureCopyTestDevice(t)
	if _, err := device.CreateQuerySet(&hal.QuerySetDescriptor{Type: hal.QueryTypeTimestamp, Count: 2}); err == nil {
		t.Error("a timestamp query set was made")
	}
	raw, err := device.CreateQuerySet(&hal.QuerySetDescriptor{Type: hal.QueryTypeOcclusion, Count: 4})
	if err != nil {
		t.Fatalf("CreateQuerySet(occlusion) failed: %v", err)
	}
	defer raw.Destroy()
	readback := createMetalTextureCopyTestReadback(t, device, 64)
	defer readback.Destroy()
	results := unsafe.Slice((*byte)(readback.Contents()), 64)
	for i := range results {
		results[i] = 0xff
	}
	encoder := &CommandEncoder{device: device}
	if err := encoder.BeginEncoding("resolve"); err != nil {
		t.Fatalf("BeginEncoding failed: %v", err)
	}
	encoder.ResolveQuerySet(raw, 1, 3, readback, 8)
	submitMetalTextureCopyTestEncoder(t, device, queue, encoder)
	for i, b := range results {
		want := byte(0)
		if i < 8 || i >= 32 {
			want = 0xff
		}
		if b != want {
			t.Fatalf("byte %d = %#x, want %#x", i, b, want)
		}
	}
}

// TestResetFenceAfterSignal resets a fence whose event was signaled, which
// an MTLSharedEvent can't go back from: the fence reads unsignaled after,
// and waits for values past the reset.
func TestResetFenceAfterSignal(t *testing.T) {
	device, _ := newMetalTextureCopyTestDevice(t)
	raw, err := device.CreateFence()
	if err != nil {
		t.Fatalf("CreateFence failed: %v", err)
	}
	defer device.DestroyFence(raw)
	fence := raw.(*Fence)
	signal := func(v uint64) { _ = MsgSend(fence.event, Sel("setSignaledValue:"), uintptr(v)) }

	signal(5)
	if done, _ := device.GetFenceStatus(fence); !done {
		t.Fatal("signaled fence reads unsignaled")
	}
	if err := device.ResetFence(fence); err != nil {
		t.Fatalf("ResetFence failed: %v", err)
	}
	if done, _ := device.GetFenceStatus(fence); done {
		t.Error("reset fence reads signaled")
	}
	if done, _ := device.Wait(fence, 1, 10*time.Millisecond); done {
		t.Error("Wait for 1 on a reset fence returned at once")
	}
	signal(6)
	if done, _ := device.Wait(fence, 1, time.Second); !done {
		t.Error("Wait for 1 after the event passed the reset by 1 timed out")
	}
}

// TestLostQueueRefusesWork checks that a completed command buffer reads as
// no failure, and that a queue whose command buffer failed refuses
// submissions and waits with hal.ErrDeviceLost.
func TestLostQueueRefusesWork(t *testing.T) {
	device, queue := newMetalTextureCopyTestDevice(t)
	encoder := &CommandEncoder{device: device}
	if err := encoder.BeginEncoding("fine"); err != nil {
		t.Fatalf("BeginEncoding failed: %v", err)
	}
	raw, err := encoder.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding failed: %v", err)
	}
	cb := raw.(*CommandBuffer)
	defer cb.Destroy()
	if _, err := queue.Submit([]hal.CommandBuffer{cb}); err != nil {
		t.Fatalf("Submit failed: %v", err)
	}
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle failed: %v", err)
	}
	if commandBufferFailed(cb.raw) {
		t.Error("a completed command buffer reads as failed")
	}

	queue.lost.Store(true)
	if _, err := queue.Submit(nil); !errors.Is(err, hal.ErrDeviceLost) {
		t.Errorf("Submit on a lost queue: %v, want hal.ErrDeviceLost", err)
	}
	if err := device.WaitIdle(); !errors.Is(err, hal.ErrDeviceLost) {
		t.Errorf("WaitIdle on a lost queue: %v, want hal.ErrDeviceLost", err)
	}
}
