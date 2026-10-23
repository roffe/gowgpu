//go:build !(js && wasm)

package core

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func TestDeviceTracker_New(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	if dt == nil {
		t.Fatal("NewDeviceTracker returned nil")
	}
	if dt.Textures() == nil {
		t.Fatal("Textures() returned nil")
	}
	if dt.Textures().Size() != 0 {
		t.Errorf("Initial texture tracker size = %d, want 0", dt.Textures().Size())
	}
}

func TestDeviceTracker_InsertRemoveTexture(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)

	dt.InsertTexture(idx, track.TextureUsesColorTarget)
	if dt.Textures().Size() != 1 {
		t.Errorf("Size after insert = %d, want 1", dt.Textures().Size())
	}
	if !dt.Textures().IsTracked(idx) {
		t.Error("Texture should be tracked after insert")
	}

	dt.RemoveTexture(idx)
	if dt.Textures().IsTracked(idx) {
		t.Error("Texture should not be tracked after remove")
	}
}

func TestDeviceTracker_MergeTextureScope_NilSafe(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	transitions := dt.MergeTextureScope(nil)
	if len(transitions) != 0 {
		t.Errorf("Nil scope should produce 0 transitions, got %d", len(transitions))
	}
}

func TestDeviceTracker_MergeTextureScope_Transition(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)

	// Register texture with initial color target usage
	dt.InsertTexture(idx, track.TextureUsesColorTarget)

	// Command buffer uses it as copy source
	scope := track.NewEncoderTextureTracker()
	scope.SetSingle(idx, track.TextureUsesCopySrc)

	transitions := dt.MergeTextureScope(scope)

	if len(transitions) != 1 {
		t.Fatalf("Expected 1 transition, got %d", len(transitions))
	}
	if transitions[0].Usage.From != track.TextureUsesColorTarget {
		t.Errorf("From = %d, want ColorTarget", transitions[0].Usage.From)
	}
	if transitions[0].Usage.To != track.TextureUsesCopySrc {
		t.Errorf("To = %d, want CopySrc", transitions[0].Usage.To)
	}
}

func TestDeviceTracker_MergeTextureScope_NoTransition(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)

	// Same ordered state
	dt.InsertTexture(idx, track.TextureUsesResource)
	scope := track.NewEncoderTextureTracker()
	scope.SetSingle(idx, track.TextureUsesResource)

	transitions := dt.MergeTextureScope(scope)
	if len(transitions) != 0 {
		t.Errorf("Expected 0 transitions for same ordered state, got %d", len(transitions))
	}
}

func TestDeviceTracker_MultipleTextures(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx0 := track.TrackerIndex(0)
	idx1 := track.TrackerIndex(1)
	idx2 := track.TrackerIndex(2)

	dt.InsertTexture(idx0, track.TextureUsesColorTarget)
	dt.InsertTexture(idx1, track.TextureUsesResource)
	dt.InsertTexture(idx2, track.TextureUsesResource)

	scope := track.NewEncoderTextureTracker()
	scope.SetSingle(idx0, track.TextureUsesCopySrc)     // transition needed
	scope.SetSingle(idx1, track.TextureUsesResource)    // same ordered, skip
	scope.SetSingle(idx2, track.TextureUsesColorTarget) // transition needed

	transitions := dt.MergeTextureScope(scope)

	if len(transitions) != 2 {
		t.Fatalf("Expected 2 transitions, got %d", len(transitions))
	}
}

// =============================================================================
// ADR-060: Submit-time barrier generation (texture tracker wiring)
// =============================================================================

// TestDeviceTracker_BarrierCBFromTransitions_SkipsNilTextures verifies that
// BarrierCBFromTransitions safely skips destroyed textures (nil resolver).
func TestDeviceTracker_BarrierCBFromTransitions_SkipsNilTextures(t *testing.T) {
	t.Parallel()

	transitions := []track.TexturePendingTransition{
		{
			Index: track.TrackerIndex(0),
			Usage: track.TextureStateTransition{
				From: track.TextureUsesColorTarget,
				To:   track.TextureUsesCopySrc,
			},
		},
		{
			Index: track.TrackerIndex(1),
			Usage: track.TextureStateTransition{
				From: track.TextureUsesResource,
				To:   track.TextureUsesColorTarget,
			},
		},
	}

	// Resolver returns nil for index 0 (destroyed) and a stub for index 1.
	encoder := &noopEncoder{}
	cb, err := BarrierCBFromTransitions(encoder, transitions, func(idx track.TrackerIndex) hal.Texture {
		if idx == 1 {
			return &noopTexture{}
		}
		return nil
	})

	if err != nil {
		t.Fatalf("BarrierCBFromTransitions error: %v", err)
	}
	if cb == nil {
		t.Fatal("Expected non-nil command buffer")
	}
	// Verify that TransitionTextures was called with 1 barrier (not 2)
	if encoder.transitionCount != 1 {
		t.Errorf("TransitionTextures barrier count = %d, want 1 (skipped nil texture)", encoder.transitionCount)
	}
}

// TestDeviceTracker_BarrierCBFromTransitions_EmptyAfterFiltering verifies that
// when all textures resolve to nil, TransitionTextures is not called.
func TestDeviceTracker_BarrierCBFromTransitions_EmptyAfterFiltering(t *testing.T) {
	t.Parallel()

	transitions := []track.TexturePendingTransition{
		{
			Index: track.TrackerIndex(0),
			Usage: track.TextureStateTransition{
				From: track.TextureUsesColorTarget,
				To:   track.TextureUsesCopySrc,
			},
		},
	}

	encoder := &noopEncoder{}
	cb, err := BarrierCBFromTransitions(encoder, transitions, func(_ track.TrackerIndex) hal.Texture {
		return nil
	})

	if err != nil {
		t.Fatalf("BarrierCBFromTransitions error: %v", err)
	}
	if cb == nil {
		t.Fatal("Expected non-nil command buffer even with all nil textures")
	}
	if encoder.transitionCount != 0 {
		t.Errorf("TransitionTextures should not be called when all textures are nil, got %d", encoder.transitionCount)
	}
}

// =============================================================================
// ADR-060: Submit-time barrier injection flow tests
// =============================================================================

// TestDeviceTracker_SubmitFlowSingleCB simulates the full submit-time barrier
// injection flow with one command buffer:
// 1. Create device with DeviceTracker
// 2. Allocate a texture with TrackerIndex
// 3. Insert texture into device tracker (initial state)
// 4. Build a command buffer tracker (simulating a render pass attachment)
// 5. Merge it into the device tracker
// 6. Verify transitions are generated when state changes
func TestDeviceTracker_SubmitFlowSingleCB(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)

	// Texture starts as UNINITIALIZED (fresh from swapchain acquire).
	dt.InsertTexture(idx, track.TextureUsesUninitialized)

	// Command buffer uses it as COLOR_TARGET (render pass attachment).
	scope := track.NewEncoderTextureTracker()
	scope.SetSingle(idx, track.TextureUsesColorTarget)

	// Merge scope — should produce UNINITIALIZED -> COLOR_TARGET transition.
	transitions := dt.MergeTextureScope(scope)
	if len(transitions) != 1 {
		t.Fatalf("Expected 1 transition, got %d", len(transitions))
	}
	if transitions[0].Usage.From != track.TextureUsesUninitialized {
		t.Errorf("From = %d, want Uninitialized", transitions[0].Usage.From)
	}
	if transitions[0].Usage.To != track.TextureUsesColorTarget {
		t.Errorf("To = %d, want ColorTarget", transitions[0].Usage.To)
	}

	// Device tracker should now be in ColorTarget state.
	if dt.Textures().GetUsage(idx) != track.TextureUsesColorTarget {
		t.Error("Device tracker should be in ColorTarget state")
	}

	// Second submit with same usage — no barrier (same ordered state).
	scope2 := track.NewEncoderTextureTracker()
	scope2.SetSingle(idx, track.TextureUsesColorTarget)
	transitions2 := dt.MergeTextureScope(scope2)
	if len(transitions2) != 0 {
		t.Errorf("Expected 0 transitions for same ordered state, got %d", len(transitions2))
	}
}

// TestDeviceTracker_SubmitFlowMultipleCBs simulates merging scopes from
// multiple command buffers in one submit, verifying that the device tracker
// accumulates state changes correctly.
func TestDeviceTracker_SubmitFlowMultipleCBs(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idxA := track.TrackerIndex(0)
	idxB := track.TrackerIndex(1)

	// Both textures start as RESOURCE (previously used for sampling).
	dt.InsertTexture(idxA, track.TextureUsesResource)
	dt.InsertTexture(idxB, track.TextureUsesResource)

	// First CB: uses A as ColorTarget
	scope1 := track.NewEncoderTextureTracker()
	scope1.SetSingle(idxA, track.TextureUsesColorTarget)

	// Second CB: uses B as CopyDst
	scope2 := track.NewEncoderTextureTracker()
	scope2.SetSingle(idxB, track.TextureUsesCopyDst)

	// Merge both scopes (simulating Submit loop)
	trans1 := dt.MergeTextureScope(scope1)
	trans2 := dt.MergeTextureScope(scope2)

	all := make([]track.TexturePendingTransition, 0, len(trans1)+len(trans2))
	all = append(all, trans1...)
	all = append(all, trans2...)
	if len(all) != 2 {
		t.Fatalf("Expected 2 transitions from 2 CBs, got %d", len(all))
	}

	// Verify final states
	if dt.Textures().GetUsage(idxA) != track.TextureUsesColorTarget {
		t.Error("Texture A should be in ColorTarget state")
	}
	if dt.Textures().GetUsage(idxB) != track.TextureUsesCopyDst {
		t.Error("Texture B should be in CopyDst state")
	}
}

// TestDeviceTracker_SubmitFlowEmptyScope verifies that an empty scope
// produces no transitions and does not affect the device tracker.
func TestDeviceTracker_SubmitFlowEmptyScope(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)
	dt.InsertTexture(idx, track.TextureUsesColorTarget)

	scope := track.NewEncoderTextureTracker()
	if !scope.IsEmpty() {
		t.Fatal("New scope should be empty")
	}

	transitions := dt.MergeTextureScope(scope)
	if len(transitions) != 0 {
		t.Errorf("Empty scope should produce 0 transitions, got %d", len(transitions))
	}

	// State should be unchanged
	if dt.Textures().GetUsage(idx) != track.TextureUsesColorTarget {
		t.Error("Device tracker state should be unchanged after empty scope merge")
	}
}

// TestDeviceTracker_SubmitFlowNewTextureAutoInsert verifies that when a scope
// references a texture not yet in the device tracker, it is auto-inserted
// without producing a transition.
func TestDeviceTracker_SubmitFlowNewTextureAutoInsert(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(42)

	// Scope references a texture not yet tracked
	scope := track.NewEncoderTextureTracker()
	scope.SetSingle(idx, track.TextureUsesColorTarget)

	// Should auto-insert without transition
	transitions := dt.MergeTextureScope(scope)
	if len(transitions) != 0 {
		t.Errorf("Expected 0 transitions for auto-insert, got %d", len(transitions))
	}

	// Texture should now be tracked
	if !dt.Textures().IsTracked(idx) {
		t.Error("Texture should be tracked after auto-insert")
	}
	if dt.Textures().GetUsage(idx) != track.TextureUsesColorTarget {
		t.Errorf("Usage = %d, want ColorTarget", dt.Textures().GetUsage(idx))
	}
}

// TestDeviceTracker_BarrierCBFromTransitions_FullFlow verifies the complete
// barrier CB generation: transitions -> HAL barriers -> command buffer.
func TestDeviceTracker_BarrierCBFromTransitions_FullFlow(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx0 := track.TrackerIndex(0)
	idx1 := track.TrackerIndex(1)

	dt.InsertTexture(idx0, track.TextureUsesResource)
	dt.InsertTexture(idx1, track.TextureUsesResource)

	scope := track.NewEncoderTextureTracker()
	scope.SetSingle(idx0, track.TextureUsesColorTarget) // Resource->ColorTarget
	scope.SetSingle(idx1, track.TextureUsesResource)    // same, skip

	transitions := dt.MergeTextureScope(scope)
	if len(transitions) != 1 {
		t.Fatalf("Expected 1 transition, got %d", len(transitions))
	}

	tex0 := &noopTexture{}
	tex1 := &noopTexture{}
	encoder := &noopEncoder{}
	cb, err := BarrierCBFromTransitions(encoder, transitions, func(idx track.TrackerIndex) hal.Texture {
		switch idx {
		case 0:
			return tex0
		case 1:
			return tex1
		default:
			return nil
		}
	})

	if err != nil {
		t.Fatalf("BarrierCBFromTransitions: %v", err)
	}
	if cb == nil {
		t.Fatal("Expected non-nil barrier command buffer")
	}
	if encoder.transitionCount != 1 {
		t.Errorf("TransitionTextures called with %d barriers, want 1", encoder.transitionCount)
	}
}

// noopEncoder is a minimal hal.CommandEncoder for testing barrier generation.
type noopEncoder struct {
	transitionCount       int
	bufferTransitionCount int
}

func (e *noopEncoder) BeginEncoding(_ string) error            { return nil }
func (e *noopEncoder) EndEncoding() (hal.CommandBuffer, error) { return &noopCB{}, nil }
func (e *noopEncoder) DiscardEncoding()                        {}
func (e *noopEncoder) ResetAll(_ []hal.CommandBuffer)          {}
func (e *noopEncoder) Destroy()                                {}
func (e *noopEncoder) CopyBufferToBuffer(_, _ hal.Buffer, _ []hal.BufferCopy) {
}
func (e *noopEncoder) CopyTextureToTexture(_ hal.Texture, _ hal.Texture, _ []hal.TextureCopy) {
}
func (e *noopEncoder) CopyBufferToTexture(_ hal.Buffer, _ hal.Texture, _ []hal.BufferTextureCopy) {
}
func (e *noopEncoder) CopyTextureToBuffer(_ hal.Texture, _ hal.Buffer, _ []hal.BufferTextureCopy) {
}
func (e *noopEncoder) ClearBuffer(_ hal.Buffer, _ uint64, _ uint64) {}
func (e *noopEncoder) BeginRenderPass(_ *hal.RenderPassDescriptor) hal.RenderPassEncoder {
	return nil
}
func (e *noopEncoder) BeginComputePass(_ *hal.ComputePassDescriptor) hal.ComputePassEncoder {
	return nil
}
func (e *noopEncoder) TransitionBuffers(barriers []hal.BufferBarrier) {
	e.bufferTransitionCount = len(barriers)
}
func (e *noopEncoder) TransitionTextures(barriers []hal.TextureBarrier) {
	e.transitionCount = len(barriers)
}
func (e *noopEncoder) ResolveQuerySet(_ hal.QuerySet, _ uint32, _ uint32, _ hal.Buffer, _ uint64) {
}
func (e *noopEncoder) BuildAccelerationStructures(_ []hal.BuildAccelerationStructureDescriptor) {}
func (e *noopEncoder) PlaceAccelerationStructureBarrier(_ hal.AccelerationStructureBarrier)     {}
func (e *noopEncoder) CopyAccelerationStructure(_, _ hal.AccelerationStructure, _ gputypes.AccelerationStructureCopyMode) {
}
func (e *noopEncoder) ReadAccelerationStructureCompactSize(_ hal.AccelerationStructure, _ hal.Buffer, _ uint64) {
}

// noopCB implements hal.CommandBuffer for testing.
type noopCB struct{}

func (c *noopCB) Destroy() {}

// noopTexture implements hal.Texture for testing.
type noopTexture struct{}

func (t *noopTexture) NativeHandle() uint64                { return 0 }
func (t *noopTexture) Destroy()                            {}
func (t *noopTexture) CurrentUsage() gputypes.TextureUsage { return 0 }
func (t *noopTexture) AddPendingRef()                      {}
func (t *noopTexture) DecPendingRef()                      {}

// noopHALBuffer implements hal.Buffer for testing.
type noopHALBuffer struct{}

func (b *noopHALBuffer) NativeHandle() uint64 { return 0 }
func (b *noopHALBuffer) Destroy()             {}

// =============================================================================
// Task #13: Buffer Tracker Integration Tests
// =============================================================================

func TestDeviceTracker_NewHasBufferTracker(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	if dt.Buffers() == nil {
		t.Fatal("Buffers() returned nil")
	}
	if dt.Buffers().Size() != 0 {
		t.Errorf("Initial buffer tracker size = %d, want 0", dt.Buffers().Size())
	}
}

func TestDeviceTracker_InsertRemoveBuffer(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)

	dt.InsertBuffer(idx, track.BufferUsesVertex)
	if dt.Buffers().Size() != 1 {
		t.Errorf("Size after insert = %d, want 1", dt.Buffers().Size())
	}
	if !dt.Buffers().IsTracked(idx) {
		t.Error("Buffer should be tracked after insert")
	}

	dt.RemoveBuffer(idx)
	if dt.Buffers().IsTracked(idx) {
		t.Error("Buffer should not be tracked after remove")
	}
}

func TestDeviceTracker_MergeBufferScope_NilSafe(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	transitions := dt.MergeBufferScope(nil)
	if len(transitions) != 0 {
		t.Errorf("Nil scope should produce 0 transitions, got %d", len(transitions))
	}
}

func TestDeviceTracker_MergeBufferScope_Transition(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)

	// Register buffer with initial vertex usage
	dt.InsertBuffer(idx, track.BufferUsesVertex)

	// Command buffer uses it as copy destination
	scope := track.NewEncoderBufferTracker()
	scope.SetSingle(idx, track.BufferUsesCopyDst)

	transitions := dt.MergeBufferScope(scope)

	if len(transitions) != 1 {
		t.Fatalf("Expected 1 transition, got %d", len(transitions))
	}
	if transitions[0].Usage.From != track.BufferUsesVertex {
		t.Errorf("From = %d, want Vertex", transitions[0].Usage.From)
	}
	if transitions[0].Usage.To != track.BufferUsesCopyDst {
		t.Errorf("To = %d, want CopyDst", transitions[0].Usage.To)
	}
}

func TestDeviceTracker_MergeBufferScope_NoTransition_SameState(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)

	dt.InsertBuffer(idx, track.BufferUsesVertex)
	scope := track.NewEncoderBufferTracker()
	scope.SetSingle(idx, track.BufferUsesVertex)

	transitions := dt.MergeBufferScope(scope)
	if len(transitions) != 0 {
		t.Errorf("Expected 0 transitions for same state, got %d", len(transitions))
	}
}

func TestDeviceTracker_MergeBufferScope_NewBufferAutoInsert(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(42)

	scope := track.NewEncoderBufferTracker()
	scope.SetSingle(idx, track.BufferUsesUniform)

	transitions := dt.MergeBufferScope(scope)
	if len(transitions) != 0 {
		t.Errorf("Expected 0 transitions for auto-insert, got %d", len(transitions))
	}
	if !dt.Buffers().IsTracked(idx) {
		t.Error("Buffer should be tracked after auto-insert")
	}
	if dt.Buffers().GetUsage(idx) != track.BufferUsesUniform {
		t.Errorf("Usage = %d, want Uniform", dt.Buffers().GetUsage(idx))
	}
}

// TestBuffer_TrackerIndex_Allocated verifies that NewBuffer allocates a
// real TrackerIndex from the device's buffer allocator.
func TestBuffer_TrackerIndex_Allocated(t *testing.T) {
	t.Parallel()

	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")

	buf := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageVertex, 256, "buf")
	td := buf.TrackingData()
	if td == nil {
		t.Fatal("TrackingData() returned nil")
	}
	if !td.Index().IsValid() {
		t.Error("Tracker index should be valid (allocated from device)")
	}
}

// TestBuffer_TrackerIndex_Freed verifies that releasing tracking data
// frees the index for reuse.
func TestBuffer_TrackerIndex_Freed(t *testing.T) {
	t.Parallel()

	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")

	buf := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageVertex, 256, "buf")
	td := buf.TrackingData()
	idx := td.Index()
	if !idx.IsValid() {
		t.Fatal("Expected valid index")
	}

	// Release should make the index recyclable
	td.Release()
	if !td.IsReleased() {
		t.Error("TrackingData should be released")
	}

	// Allocate again — should reuse the freed index
	buf2 := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageVertex, 256, "buf2")
	td2 := buf2.TrackingData()
	if td2.Index() != idx {
		t.Errorf("Expected reused index %d, got %d", idx, td2.Index())
	}
}

// TestSubmit_MergesBufferScope verifies that the DeviceTracker merges
// buffer scopes and produces transitions.
func TestSubmit_MergesBufferScope(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)

	// Buffer starts as vertex
	dt.InsertBuffer(idx, track.BufferUsesVertex)

	// Scope: now used as copy source
	scope := track.NewEncoderBufferTracker()
	scope.SetSingle(idx, track.BufferUsesCopySrc)

	transitions := dt.MergeBufferScope(scope)
	if len(transitions) != 1 {
		t.Fatalf("Expected 1 transition, got %d", len(transitions))
	}

	// After merge, device tracker should have the new state
	if dt.Buffers().GetUsage(idx) != track.BufferUsesCopySrc {
		t.Error("Device tracker should be in CopySrc state after merge")
	}
}

// TestSubmit_BufferTransition_GeneratesBarrier verifies that buffer
// transitions are correctly converted to HAL barriers.
func TestSubmit_BufferTransition_GeneratesBarrier(t *testing.T) {
	t.Parallel()

	dt := NewDeviceTracker()
	idx := track.TrackerIndex(0)

	dt.InsertBuffer(idx, track.BufferUsesVertex)

	scope := track.NewEncoderBufferTracker()
	scope.SetSingle(idx, track.BufferUsesCopyDst)

	transitions := dt.MergeBufferScope(scope)
	if len(transitions) != 1 {
		t.Fatalf("Expected 1 transition, got %d", len(transitions))
	}

	buf0 := &noopHALBuffer{}
	encoder := &noopEncoder{}
	cb, err := BarrierCBFromAllTransitions(
		encoder,
		nil, // no texture transitions
		transitions,
		nil, // no texture resolver
		func(idx track.TrackerIndex) hal.Buffer {
			if idx == 0 {
				return buf0
			}
			return nil
		},
	)
	if err != nil {
		t.Fatalf("BarrierCBFromAllTransitions: %v", err)
	}
	if cb == nil {
		t.Fatal("Expected non-nil barrier command buffer")
	}
	if encoder.bufferTransitionCount != 1 {
		t.Errorf("TransitionBuffers called with %d barriers, want 1", encoder.bufferTransitionCount)
	}
}

// TestBarrierCBFromAllTransitions_BothTypes verifies that a single barrier
// CB can contain both texture and buffer barriers.
func TestBarrierCBFromAllTransitions_BothTypes(t *testing.T) {
	t.Parallel()

	texTransitions := []track.TexturePendingTransition{
		{
			Index: track.TrackerIndex(0),
			Usage: track.TextureStateTransition{
				From: track.TextureUsesResource,
				To:   track.TextureUsesColorTarget,
			},
		},
	}
	bufTransitions := []track.PendingTransition{
		{
			Index: track.TrackerIndex(0),
			Usage: track.StateTransition{
				From: track.BufferUsesVertex,
				To:   track.BufferUsesCopyDst,
			},
		},
	}

	encoder := &noopEncoder{}
	cb, err := BarrierCBFromAllTransitions(
		encoder,
		texTransitions,
		bufTransitions,
		func(_ track.TrackerIndex) hal.Texture { return &noopTexture{} },
		func(_ track.TrackerIndex) hal.Buffer { return &noopHALBuffer{} },
	)
	if err != nil {
		t.Fatalf("BarrierCBFromAllTransitions: %v", err)
	}
	if cb == nil {
		t.Fatal("Expected non-nil barrier command buffer")
	}
	if encoder.transitionCount != 1 {
		t.Errorf("texture transitions = %d, want 1", encoder.transitionCount)
	}
	if encoder.bufferTransitionCount != 1 {
		t.Errorf("buffer transitions = %d, want 1", encoder.bufferTransitionCount)
	}
}

// =============================================================================
// Task #14: Usage Conflict Validation Tests
// =============================================================================

// TestBeginRenderPass_TextureConflict_Error verifies that one texture used
// as color attachment and as depth/stencil attachment of one render pass is
// refused before the HAL render pass begins: by the attachment checks, as
// no format is both.
func TestBeginRenderPass_TextureConflict_Error(t *testing.T) {
	t.Parallel()

	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")

	alloc := device.TrackerIndices().Textures()
	td := track.NewTrackingData(alloc)
	tex := &Texture{
		raw:          NewSnatchable[hal.Texture](nil),
		format:       gputypes.TextureFormatRGBA8Unorm,
		usage:        gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding,
		device:       device,
		trackingData: td,
	}
	view := &TextureView{Parent: tex}

	halEncoder := &passOrderEncoder{}
	enc, err := device.CreateCommandEncoderWithHAL(halEncoder, "test")
	if err != nil {
		t.Fatalf("CreateCommandEncoderWithHAL: %v", err)
	}

	desc := &RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{
			{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore},
		},
		DepthStencilAttachment: &RenderPassDepthStencilAttachment{
			View:         view,
			DepthLoadOp:  gputypes.LoadOpClear,
			DepthStoreOp: gputypes.StoreOpStore,
		},
	}
	_, passErr := enc.BeginRenderPass(desc)
	var invalid *RenderPassValidationError
	if !errors.As(passErr, &invalid) {
		t.Fatalf("BeginRenderPass = %v, want RenderPassValidationError", passErr)
	}
	if len(halEncoder.events) != 0 {
		t.Errorf("conflicting pass reached the HAL encoder: %v", halEncoder.events)
	}
}

// passOrderEncoder records the order of HAL calls: barriers, render pass
// begins and command buffer begins and ends. Each command buffer it closes
// says which one it is, so the order Finish hands them out in can be checked.
type passOrderEncoder struct {
	noopEncoder
	events   []string
	barriers []hal.TextureBarrier
	closed   int
}

// orderedCB is the n-th command buffer a passOrderEncoder closed.
type orderedCB struct {
	noopCB
	n int
}

func (e *passOrderEncoder) BeginEncoding(_ string) error {
	e.events = append(e.events, "begin")
	return nil
}

func (e *passOrderEncoder) EndEncoding() (hal.CommandBuffer, error) {
	e.events = append(e.events, "end")
	cb := &orderedCB{n: e.closed}
	e.closed++
	return cb, nil
}

func (e *passOrderEncoder) TransitionTextures(barriers []hal.TextureBarrier) {
	e.events = append(e.events, "barrier")
	e.barriers = append(e.barriers, barriers...)
}

func (e *passOrderEncoder) BeginRenderPass(_ *hal.RenderPassDescriptor) hal.RenderPassEncoder {
	e.events = append(e.events, "begin-pass")
	return nil
}

// cbOrder returns the closing order of the command buffers in list.
func cbOrder(list []hal.CommandBuffer) []int {
	order := make([]int, 0, len(list))
	for _, cb := range list {
		if o, ok := cb.(*orderedCB); ok {
			order = append(order, o.n)
		}
	}
	return order
}

// TestBeginRenderPass_EarlierUsageIsTransitionedNotConflict verifies the
// WebGPU usage-scope rule for render passes: a texture used differently by
// an EARLIER command of the encoder (sampled, uploaded, copied) is not a
// conflict. The attachment is transitioned into COLOR_TARGET, and because
// the barrier is only known once the pass has ended (its bind groups are
// part of the scope), it goes in a command buffer inserted ahead of the
// pass's own: the pass is recorded in a command buffer of its own, and the
// barrier is the second of three in the finished list.
func TestBeginRenderPass_EarlierUsageIsTransitionedNotConflict(t *testing.T) {
	t.Parallel()

	for _, earlier := range []track.TextureUses{
		track.TextureUsesResource,
		track.TextureUsesCopyDst,
		track.TextureUsesCopySrc,
	} {
		halDevice := &mockHALDevice{}
		device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")
		td := track.NewTrackingData(device.TrackerIndices().Textures())
		tex := &Texture{
			raw:          NewSnatchable[hal.Texture](&noopTexture{}),
			format:       gputypes.TextureFormatRGBA8Unorm,
			usage:        gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding,
			device:       device,
			trackingData: td,
		}

		halEncoder := &passOrderEncoder{}
		enc, err := device.CreateCommandEncoderWithHAL(halEncoder, "test")
		if err != nil {
			t.Fatalf("CreateCommandEncoderWithHAL: %v", err)
		}
		enc.Mutable().TextureScope().SetSingle(td.Index(), earlier)

		pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
			ColorAttachments: []RenderPassColorAttachment{
				{View: &TextureView{Parent: tex}, LoadOp: gputypes.LoadOpLoad, StoreOp: gputypes.StoreOpStore},
			},
		})
		if err != nil {
			t.Fatalf("earlier %v: BeginRenderPass: %v", earlier, err)
		}
		if err := pass.End(); err != nil {
			t.Fatalf("End: %v", err)
		}
		// The initial recording is closed, the pass gets its own command
		// buffer, then the transit one with the barrier is recorded and
		// swapped ahead of it. Nothing is reopened: the next command that
		// records would open a fresh one, Finish does not.
		want := "end,begin,begin-pass,end,begin,barrier,end"
		if got := strings.Join(halEncoder.events, ","); got != want {
			t.Errorf("earlier %v: HAL call order = %q, want %q", earlier, got, want)
		}
		if len(halEncoder.barriers) != 1 ||
			halEncoder.barriers[0].Usage.OldUsage != earlier.ToTextureUsage() ||
			halEncoder.barriers[0].Usage.NewUsage != gputypes.TextureUsageRenderAttachment {
			t.Errorf("earlier %v: barriers = %+v, want %v -> RenderAttachment", earlier, halEncoder.barriers, earlier)
		}
		tracker := enc.Mutable().TextureScope()
		if got := tracker.StartUsage(td.Index()); got != earlier {
			t.Errorf("start usage = %v, want %v", got, earlier)
		}
		if got := tracker.EndUsage(td.Index()); got != track.TextureUsesColorTarget {
			t.Errorf("end usage = %v, want ColorTarget", got)
		}
		cb, err := enc.Finish()
		if err != nil {
			t.Fatalf("Finish: %v", err)
		}
		// Closed in the order initial (0), pass (1), transit (2);
		// submitted with the transit ahead of the pass, and no empty
		// command buffer after it.
		if got := cbOrder(cb.HalBufferList()); !slices.Equal(got, []int{0, 2, 1}) {
			t.Errorf("earlier %v: command buffer order = %v, want [0 2 1] (transit ahead of the pass)", earlier, got)
		}
	}
}

// TestBeginRenderPass_FirstPassIsNotSplit verifies that an encoder's first
// pass stays in the command buffer the encoder started with: nothing was
// used before it, so nothing can need a transition ahead of it, and the
// submit-time barrier covers the first usages.
func TestBeginRenderPass_FirstPassIsNotSplit(t *testing.T) {
	t.Parallel()

	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")
	td := track.NewTrackingData(device.TrackerIndices().Textures())
	tex := &Texture{
		raw:          NewSnatchable[hal.Texture](&noopTexture{}),
		format:       gputypes.TextureFormatRGBA8Unorm,
		usage:        gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding,
		device:       device,
		trackingData: td,
	}

	halEncoder := &passOrderEncoder{}
	enc, err := device.CreateCommandEncoderWithHAL(halEncoder, "test")
	if err != nil {
		t.Fatalf("CreateCommandEncoderWithHAL: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{
			{View: &TextureView{Parent: tex}, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore},
		},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	if got := strings.Join(halEncoder.events, ","); got != "begin-pass" {
		t.Errorf("HAL call order = %q, want just begin-pass", got)
	}
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if got := cbOrder(cb.HalBufferList()); !slices.Equal(got, []int{0}) {
		t.Errorf("command buffer order = %v, want [0]", got)
	}
	if got := enc.Mutable(); got != nil {
		t.Errorf("finished encoder still holds its scopes")
	}
	if got := cb.TextureScope().StartUsage(td.Index()); got != track.TextureUsesColorTarget {
		t.Errorf("start usage = %v, want ColorTarget", got)
	}
}

// TestRecordPassTextureUsage_RenderPassTransitionsAheadOfPass verifies that
// a texture a bind group of a render pass binds, sampled, after an earlier
// command of the encoder wrote it, is transitioned ahead of the pass, and
// that a sampled texture that is also the pass's attachment is a conflict.
func TestRecordPassTextureUsage_RenderPassTransitionsAheadOfPass(t *testing.T) {
	t.Parallel()

	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")
	newTex := func() *Texture {
		return &Texture{
			raw:          NewSnatchable[hal.Texture](&noopTexture{}),
			format:       gputypes.TextureFormatRGBA8Unorm,
			usage:        gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding,
			device:       device,
			trackingData: track.NewTrackingData(device.TrackerIndices().Textures()),
		}
	}
	target, sampled := newTex(), newTex()
	sampledHAL := &noopTexture{}

	halEncoder := &passOrderEncoder{}
	enc, err := device.CreateCommandEncoderWithHAL(halEncoder, "test")
	if err != nil {
		t.Fatalf("CreateCommandEncoderWithHAL: %v", err)
	}
	// An earlier upload into the sampled texture.
	enc.Mutable().TextureScope().SetSingle(sampled.TrackingData().Index(), track.TextureUsesCopyDst)

	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{
			{View: &TextureView{Parent: target}, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore},
		},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	if err := enc.RecordPassTextureUsage(sampled, sampledHAL, track.TextureSelector{}, track.TextureUsesResource); err != nil {
		t.Fatalf("RecordPassTextureUsage(sampled): %v", err)
	}
	// Bound again by another group of the same pass: the same usage merges.
	if err := enc.RecordPassTextureUsage(sampled, sampledHAL, track.TextureSelector{}, track.TextureUsesResource); err != nil {
		t.Fatalf("RecordPassTextureUsage(sampled again): %v", err)
	}
	var conflict *track.TextureUsageConflictError
	if err := enc.RecordPassTextureUsage(target, nil, track.TextureSelector{}, track.TextureUsesResource); !errors.As(err, &conflict) {
		t.Fatalf("sampling the pass's own attachment = %v, want TextureUsageConflictError", err)
	}
	if n := len(halEncoder.barriers); n != 0 {
		t.Fatalf("%d barriers recorded while the pass was open, want none", n)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	if len(halEncoder.barriers) != 1 ||
		halEncoder.barriers[0].Texture != sampledHAL ||
		halEncoder.barriers[0].Usage.OldUsage != gputypes.TextureUsageCopyDst ||
		halEncoder.barriers[0].Usage.NewUsage != gputypes.TextureUsageTextureBinding {
		t.Errorf("barriers = %+v, want one CopyDst -> TextureBinding on the sampled texture", halEncoder.barriers)
	}
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if got := cbOrder(cb.HalBufferList()); !slices.Equal(got, []int{0, 2, 1}) {
		t.Errorf("command buffer order = %v, want [0 2 1] (transit ahead of the pass)", got)
	}
	tracker := cb.TextureScope()
	if got := tracker.EndUsage(sampled.TrackingData().Index()); got != track.TextureUsesResource {
		t.Errorf("sampled end usage = %v, want Resource", got)
	}
	if got := tracker.StartUsage(target.TrackingData().Index()); got != track.TextureUsesColorTarget {
		t.Errorf("target start usage = %v, want ColorTarget", got)
	}
}

// TestRecordPassTextureUsage_ComputePassTransitionsInline verifies that a
// storage texture a compute pass binds is transitioned at the dispatch that
// uses it: each dispatch is a usage scope of its own, a compute pass can
// take barriers, so they go inline ahead of the dispatch without splitting
// the command buffer, and a write after a write gets one too.
func TestRecordPassTextureUsage_ComputePassTransitionsInline(t *testing.T) {
	t.Parallel()

	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")
	tex := &Texture{
		raw:          NewSnatchable[hal.Texture](&noopTexture{}),
		format:       gputypes.TextureFormatRGBA8Unorm,
		usage:        gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding,
		device:       device,
		trackingData: track.NewTrackingData(device.TrackerIndices().Textures()),
	}
	texHAL := &noopTexture{}

	halEncoder := &passOrderEncoder{}
	enc, err := device.CreateCommandEncoderWithHAL(halEncoder, "test")
	if err != nil {
		t.Fatalf("CreateCommandEncoderWithHAL: %v", err)
	}
	// Rendered earlier in this encoder.
	enc.Mutable().TextureScope().SetSingle(tex.TrackingData().Index(), track.TextureUsesColorTarget)

	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	if err := enc.RecordPassTextureUsage(tex, texHAL, track.TextureSelector{}, track.TextureUsesStorageWrite); err != nil {
		t.Fatalf("RecordPassTextureUsage: %v", err)
	}
	if n := len(halEncoder.barriers); n != 0 {
		t.Fatalf("%d barriers recorded at bind time, want none before the dispatch", n)
	}
	pass.Dispatch(1, 1, 1)
	if len(halEncoder.barriers) != 1 ||
		halEncoder.barriers[0].Usage.OldUsage != gputypes.TextureUsageRenderAttachment ||
		halEncoder.barriers[0].Usage.NewUsage != gputypes.TextureUsageStorageBinding {
		t.Fatalf("barriers = %+v, want one RenderAttachment -> StorageBinding recorded at the dispatch", halEncoder.barriers)
	}
	// Bound again, by the next dispatch's group: a write after a write still
	// needs a barrier.
	if err := enc.RecordPassTextureUsage(tex, texHAL, track.TextureSelector{}, track.TextureUsesStorageWrite); err != nil {
		t.Fatalf("RecordPassTextureUsage again: %v", err)
	}
	pass.Dispatch(1, 1, 1)
	if len(halEncoder.barriers) != 2 {
		t.Fatalf("barriers = %d, want 2", len(halEncoder.barriers))
	}
	// A dispatch with nothing bound records nothing.
	pass.Dispatch(1, 1, 1)
	if len(halEncoder.barriers) != 2 {
		t.Fatalf("barriers = %d after an empty dispatch, want still 2", len(halEncoder.barriers))
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	if got := strings.Join(halEncoder.events, ","); got != "barrier,barrier" {
		t.Errorf("HAL call order = %q, want the barriers inline, no command buffer split", got)
	}
	if got := enc.Mutable().TextureScope().EndUsage(tex.TrackingData().Index()); got != track.TextureUsesStorageWrite {
		t.Errorf("end usage = %v, want StorageWrite", got)
	}
}

// TestRenderPass_VertexBufferScopeIsPerPass verifies that vertex/index buffer
// usages are validated per pass and applied to the command buffer tracker
// when the pass ends, so a later pass (or a copy) can use the buffer
// differently.
func TestRenderPass_VertexBufferScopeIsPerPass(t *testing.T) {
	t.Parallel()

	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")
	buf := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageVertex|gputypes.BufferUsageCopyDst, 256, "vb")
	t.Cleanup(buf.Destroy)
	idx := buf.TrackingData().Index()

	enc, err := device.CreateCommandEncoderWithHAL(&passOrderEncoder{}, "test")
	if err != nil {
		t.Fatalf("CreateCommandEncoderWithHAL: %v", err)
	}
	// An earlier copy wrote the buffer.
	enc.Mutable().BufferScope().SetSingle(idx, track.BufferUsesCopyDst)

	pass, err := enc.BeginRenderPass(testPass(""))
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	pass.SetVertexBuffer(0, buf, 0)
	pass.SetIndexBuffer(buf, gputypes.IndexFormatUint16, 0)
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	if encErr := enc.Error(); encErr != nil {
		t.Fatalf("encoder error after vertex+index use of a previously copied buffer: %v", encErr)
	}
	tracker := enc.Mutable().BufferScope()
	if got := tracker.StartUsage(idx); got != track.BufferUsesCopyDst {
		t.Errorf("start usage = %v, want CopyDst", got)
	}
	if got := tracker.EndUsage(idx); got != track.BufferUsesVertex|track.BufferUsesIndex {
		t.Errorf("end usage = %v, want Vertex|Index", got)
	}
}

// TestBeginRenderPass_NoConflict_SameUsage verifies that using a texture as
// COLOR_TARGET in two color attachments is allowed (same exclusive usage).
func TestBeginRenderPass_NoConflict_SameUsage(t *testing.T) {
	t.Parallel()

	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")

	alloc := device.TrackerIndices().Textures()
	td := track.NewTrackingData(alloc)

	tex := &Texture{
		raw:          NewSnatchable[hal.Texture](nil),
		format:       gputypes.TextureFormatRGBA8Unorm,
		usage:        gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding,
		device:       device,
		trackingData: td,
	}
	view := &TextureView{Parent: tex}

	enc, err := device.CreateCommandEncoder("test")
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}

	// Two color attachments pointing to the same texture (same exclusive usage).
	// This should NOT produce a conflict.
	desc := &RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{
			{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore},
			{View: view, LoadOp: gputypes.LoadOpLoad, StoreOp: gputypes.StoreOpStore},
		},
	}
	_, passErr := enc.BeginRenderPass(desc)
	if passErr != nil {
		t.Fatalf("Unexpected error for same-usage: %v", passErr)
	}
}

// TestBuffer_UsageConflict_Error verifies that using a buffer as both
// STORAGE_WRITE and UNIFORM in the same command buffer produces an error.
func TestBuffer_UsageConflict_Error(t *testing.T) {
	t.Parallel()

	scope := track.NewBufferUsageScope()
	idx := track.TrackerIndex(0)

	// First usage: storage write (exclusive)
	if err := scope.SetUsage(idx, track.BufferUsesStorageWrite); err != nil {
		t.Fatalf("First SetUsage (StorageWrite) should succeed: %v", err)
	}

	// Second usage: uniform (read-only) — incompatible with write
	err := scope.SetUsage(idx, track.BufferUsesUniform)
	if err == nil {
		t.Fatal("Expected usage conflict error for StorageWrite + Uniform, got nil")
	}
}

// TestBuffer_NoConflict_ReadOnly verifies that multiple read-only buffer
// usages in the same scope are compatible.
func TestBuffer_NoConflict_ReadOnly(t *testing.T) {
	t.Parallel()

	scope := track.NewBufferUsageScope()
	idx := track.TrackerIndex(0)

	// Vertex + Index are both read-only — should be compatible
	if err := scope.SetUsage(idx, track.BufferUsesVertex); err != nil {
		t.Fatalf("Vertex usage should succeed: %v", err)
	}
	if err := scope.SetUsage(idx, track.BufferUsesIndex); err != nil {
		t.Fatalf("Index usage should succeed (both read-only): %v", err)
	}

	// Verify merged usage
	usage := scope.GetUsage(idx)
	if usage&track.BufferUsesVertex == 0 {
		t.Error("Expected Vertex flag in merged usage")
	}
	if usage&track.BufferUsesIndex == 0 {
		t.Error("Expected Index flag in merged usage")
	}
}
