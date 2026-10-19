//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
)

// Encoders that mix compute dispatches, render passes and copies, and a
// command buffer whose texture went away before Submit.

// A texture released after Finish frees its tracker index at once
// (releaseTextureTracking), so a texture created next gets the same index
// while the finished command buffer's tracker still holds it under the old
// texture's usages. Submit has to reject that command buffer before its
// tracker is merged, or the new texture's device state would become the old
// texture's end state and its first real use would get a barrier from a
// layout it was never in.
func TestReleasedTextureIndexReuseBeforeSubmit(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	desc := &TextureDescriptor{
		Size:          Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		// No TextureBinding: nothing queues a zeroing write for it.
		Usage: TextureUsageRenderAttachment | TextureUsageCopySrc,
	}
	a, err := device.CreateTexture(desc)
	if err != nil {
		t.Fatalf("CreateTexture(a): %v", err)
	}
	aView, err := device.CreateTextureView(a, nil)
	if err != nil {
		t.Fatalf("CreateTextureView(a): %v", err)
	}
	aIdx := a.coreTexture.TrackingData().Index()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	runColorPass(t, enc, aView, gputypes.LoadOpClear, nil)
	cb := mustFinish(t, enc)
	if got := cb.core.TextureScope().EndUsage(aIdx); got != track.TextureUsesColorTarget {
		t.Fatalf("command buffer end usage of a = %v, want ColorTarget", got)
	}

	aView.Release()
	a.Release()

	b, err := device.CreateTexture(desc)
	if err != nil {
		t.Fatalf("CreateTexture(b): %v", err)
	}
	defer b.Release()
	bIdx := b.coreTexture.TrackingData().Index()
	if bIdx != aIdx {
		t.Skipf("b got index %d, a had %d: no reuse to probe", bIdx, aIdx)
	}
	tracker := device.core.Tracker()
	if got := tracker.TextureUsage(bIdx); got != track.TextureUsesUninitialized {
		t.Fatalf("device state of the new texture before Submit = %v, want Uninitialized", got)
	}

	if _, err := device.Queue().Submit(cb); err == nil {
		t.Errorf("Submit of a command buffer that renders to a released texture succeeded")
	} else {
		t.Logf("Submit rejected as expected: %v", err)
	}
	if got := tracker.TextureUsage(bIdx); got != track.TextureUsesUninitialized {
		t.Errorf("device state of the new texture after the rejected Submit = %v, want Uninitialized (the old texture's usages leaked onto the reused index)", got)
	}
}

// TestComputeStoreThenSampleInRenderPassOneEncoder: a compute dispatch
// stores into S through a storage binding (the encoder's first scope, S
// starts StorageWrite), a render pass samples S into U (split off, transit
// StorageWrite -> Resource ahead of it) and U is copied out, all in one
// encoder. S is zeroed by the pending writes of the same submit, so the
// submit barrier is Resource -> StorageWrite after them.
func TestComputeStoreThenSampleInRenderPassOneEncoder(t *testing.T) {
	device, h := newValidatedDevice(t)
	blit := newBlitRig(t, device)
	store, storeLayout := computePipeline(t, device, "mixed-store", storeShader, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, StorageTexture: &gputypes.StorageTextureBindingLayout{
			Access: gputypes.StorageTextureAccessWriteOnly, Format: gputypes.TextureFormatRGBA8Unorm, ViewDimension: gputypes.TextureViewDimension2D}},
	})

	s, sView := pixTexture(t, device, "mixed-s", TextureUsageStorageBinding|TextureUsageTextureBinding)
	defer releaseTextureView(s, sView)
	u, uView := pixTexture(t, device, "mixed-u", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(u, uView)
	storeGroup, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: storeLayout, Entries: []BindGroupEntry{{Binding: 0, TextureView: sView}}})
	if err != nil {
		t.Fatalf("CreateBindGroup(store): %v", err)
	}
	defer storeGroup.Release()
	sampleGroup := blit.group(t, device, sView)
	st := readbackBuffer(t, device, "mixed-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cpass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cpass.SetPipeline(store)
	cpass.SetBindGroup(0, storeGroup, nil)
	cpass.Dispatch(pixSide, pixSide, 1)
	if err := cpass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	clearPass(t, enc, uView, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) {
		pass.SetPipeline(blit.pipeline)
		pass.SetBindGroup(0, sampleGroup, nil)
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, u, st)
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	sIdx := s.coreTexture.TrackingData().Index()
	if got := cb.core.TextureScope().StartUsage(sIdx); got != track.TextureUsesStorageWrite {
		t.Errorf("S start usage = %v, want StorageWrite", got)
	}
	if got := cb.core.TextureScope().EndUsage(sIdx); got != track.TextureUsesResource {
		t.Errorf("S end usage = %v, want Resource", got)
	}
	if got := len(cb.core.HalBufferList()); got != 4 {
		t.Errorf("HAL command buffers = %d, want 4 (compute, transit, pass, copy)", got)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	expectPixels(t, "U sampled from S stored by compute earlier in the encoder", readback(t, st), 0, 0, 255, 255)
	assertNoValidation(t, h)
}

// TestRenderThenComputeLoadOneEncoder: the encoder's first pass clears T
// red (not split, its command buffer stays open), a compute dispatch then
// loads T through a sampled binding, so the dispatch's scope records
// ColorTarget -> Resource inline in that same command buffer ahead of the
// dispatch, and the result buffer is copied out.
func TestRenderThenComputeLoadOneEncoder(t *testing.T) {
	device, h := newValidatedDevice(t)
	rig := newComputeRig(t, device)
	view, err := device.CreateTextureView(rig.tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer view.Release()
	st := readbackBuffer(t, device, "mixed-rc-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, red, nil)
	cpass, err := enc.BeginComputePass(nil)
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
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	idx := rig.tex.coreTexture.TrackingData().Index()
	if got := cb.core.TextureScope().StartUsage(idx); got != track.TextureUsesColorTarget {
		t.Errorf("T start usage = %v, want ColorTarget", got)
	}
	if got := cb.core.TextureScope().EndUsage(idx); got != track.TextureUsesResource {
		t.Errorf("T end usage = %v, want Resource", got)
	}
	if got := len(cb.core.HalBufferList()); got != 1 {
		t.Errorf("HAL command buffers = %d, want 1 (first pass is not split, compute and copy share its buffer)", got)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	expectFloats4(t, "T loaded by compute after the pass", readFloats(t, st, 64), [4]float32{1, 0, 0, 1})
	assertNoValidation(t, h)
}
