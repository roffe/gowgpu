//go:build !(js && wasm)

package track

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// TextureUses represents internal texture usage states for tracking.
// These are more granular than gputypes.TextureUsage for precise barrier
// insertion and usage conflict detection.
//
// The flags mirror Rust wgpu-types TextureUses (wgpu-types/src/texture.rs:215).
// Two composite constants partition the flags:
//   - Ordered  — hardware guarantees access order; no barrier when state unchanged
//   - Exclusive — require sole access; barrier always needed on transition
//
// Reference: wgpu-core track/texture.rs (ResourceUses impl for TextureUses)
type TextureUses uint32

// Texture usage flags for state tracking.
const (
	TextureUsesNone              TextureUses = 0
	TextureUsesUninitialized     TextureUses = 1 << 0 // Unknown/junk contents
	TextureUsesPresent           TextureUses = 1 << 1 // Ready for surface presentation
	TextureUsesCopySrc           TextureUses = 1 << 2 // Source of a hardware copy
	TextureUsesCopyDst           TextureUses = 1 << 3 // Destination of a hardware copy
	TextureUsesResource          TextureUses = 1 << 4 // Read-only sampled or fetched
	TextureUsesColorTarget       TextureUses = 1 << 5 // Render pass color target
	TextureUsesDepthStencilRead  TextureUses = 1 << 6 // Read-only depth/stencil
	TextureUsesDepthStencilWrite TextureUses = 1 << 7 // Read-write depth/stencil
	TextureUsesStorageRead       TextureUses = 1 << 8 // Storage texture read-only
	TextureUsesStorageWrite      TextureUses = 1 << 9 // Storage texture write-only
)

// textureUsesInclusive is the combination of states that may coexist.
// Reference: wgpu-types TextureUses::INCLUSIVE
const textureUsesInclusive = TextureUsesCopySrc | TextureUsesResource |
	TextureUsesDepthStencilRead | TextureUsesStorageRead

// textureUsesExclusive is the combination of states that require sole access.
// Reference: wgpu-types TextureUses::EXCLUSIVE
const textureUsesExclusive = TextureUsesCopyDst | TextureUsesColorTarget |
	TextureUsesDepthStencilWrite | TextureUsesStorageWrite | TextureUsesPresent

// textureUsesOrdered is the combination of usages guaranteed to be ordered
// by hardware. If a texture stays in an ordered state between operations, no
// barrier is needed.
// Reference: wgpu-types TextureUses::ORDERED
const textureUsesOrdered = textureUsesInclusive | TextureUsesColorTarget |
	TextureUsesDepthStencilWrite | TextureUsesStorageRead

// IsReadOnly returns true if the usage contains only read-only operations.
func (u TextureUses) IsReadOnly() bool {
	return u&textureUsesExclusive == 0
}

// IsEmpty returns true if no usage flags are set.
func (u TextureUses) IsEmpty() bool {
	return u == TextureUsesNone
}

// Contains returns true if all flags in other are present in u.
func (u TextureUses) Contains(other TextureUses) bool {
	return u&other == other
}

// AllOrdered returns true if all set flags are in the ordered set.
// When all usages are ordered, hardware guarantees access ordering.
// Reference: wgpu-core track/mod.rs ResourceUses::all_ordered()
func (u TextureUses) AllOrdered() bool {
	return u&^textureUsesOrdered == 0
}

// IsExclusive returns true if any exclusive usage flag is set.
// Reference: wgpu-core track/mod.rs ResourceUses::any_exclusive()
func (u TextureUses) IsExclusive() bool {
	return u&textureUsesExclusive != 0
}

// IsCompatible returns true if two usages can coexist without a barrier.
// Read-only (inclusive) usages are compatible with each other.
// Any exclusive usage requires sole access unless the exact same flag.
//
// Implements the Rust wgpu-core rule: any(inclusive) XOR one(exclusive).
// Reference: wgpu-core track/mod.rs invalid_resource_state()
func (u TextureUses) IsCompatible(other TextureUses) bool {
	if u.IsEmpty() || other.IsEmpty() {
		return true
	}
	combined := u | other
	// If any exclusive bit is set in the combined state, only a single
	// usage flag may be active (power of two = exactly one bit set).
	if combined.IsExclusive() {
		return isPowerOfTwo(uint32(combined))
	}
	// All inclusive — compatible.
	return true
}

// isPowerOfTwo returns true when exactly one bit is set.
func isPowerOfTwo(v uint32) bool {
	return v != 0 && (v&(v-1)) == 0
}

// SkipBarrier returns true if transitioning from old to new does NOT
// require a barrier.
//
// A barrier can be skipped when:
//  1. The state did not change, AND
//  2. All usages in the state are ordered by hardware
//
// Reference: wgpu-core track/mod.rs skip_barrier()
func SkipBarrier(old, next TextureUses) bool {
	return old == next && old.AllOrdered()
}

// ToHAL converts internal uses to the HAL's fine-grained state, which has
// the same bits.
func (u TextureUses) ToHAL() hal.TextureUses {
	return hal.TextureUses(u)
}

// ToTextureUsage converts internal uses to gputypes.TextureUsage for HAL.
//
// A read-only depth/stencil attachment maps to RenderAttachment, not
// TextureBinding, as the coarse usage of an attachment. Uninitialized maps
// to nothing: the backend takes a usage of 0 as the undefined state a new
// texture is in. The coarse usage is for backends that go by it alone; a
// barrier carries the fine state too (IntoHAL), which is what tells a
// read-only depth attachment from a written one.
func (u TextureUses) ToTextureUsage() gputypes.TextureUsage {
	var result gputypes.TextureUsage

	if u&TextureUsesCopySrc != 0 {
		result |= gputypes.TextureUsageCopySrc
	}
	if u&TextureUsesCopyDst != 0 {
		result |= gputypes.TextureUsageCopyDst
	}
	if u&TextureUsesResource != 0 {
		result |= gputypes.TextureUsageTextureBinding
	}
	if u&(TextureUsesStorageRead|TextureUsesStorageWrite) != 0 {
		result |= gputypes.TextureUsageStorageBinding
	}
	if u&(TextureUsesColorTarget|TextureUsesDepthStencilWrite|TextureUsesDepthStencilRead) != 0 {
		result |= gputypes.TextureUsageRenderAttachment
	}

	return result
}

// TextureTracker tracks texture usage states for a device, per subresource.
// It is the device-level source of truth: each submitted command buffer's
// tracker is merged into it, which produces the barriers into the states
// the command buffer starts in.
//
// Reference: wgpu-core track/texture.rs DeviceTextureTracker
type TextureTracker struct {
	set      stateSet
	metadata ResourceMetadata
}

// NewTextureTracker creates a new texture tracker.
func NewTextureTracker() *TextureTracker {
	return &TextureTracker{metadata: NewResourceMetadata()}
}

func (t *TextureTracker) ensureSize(size int) {
	t.set.ensureSize(size)
}

// InsertSingle tracks a texture with every subresource in usage.
func (t *TextureTracker) InsertSingle(index TrackerIndex, usage TextureUses) {
	t.ensureSize(int(index) + 1)
	t.set.setSimple(index, usage)
	t.metadata.SetOwned(index, true)
}

// Remove stops tracking a texture.
func (t *TextureTracker) Remove(index TrackerIndex) {
	if int(index) < len(t.set.simple) {
		t.set.remove(index)
		t.metadata.SetOwned(index, false)
	}
}

// GetUsage returns the usage of a texture, every usage one of its
// subresources is in when they differ.
func (t *TextureTracker) GetUsage(index TrackerIndex) TextureUses {
	if !t.IsTracked(index) {
		return TextureUsesNone
	}
	return t.set.union(index)
}

// SetUsage puts every subresource of a tracked texture in usage.
func (t *TextureTracker) SetUsage(index TrackerIndex, usage TextureUses) {
	if t.IsTracked(index) {
		t.set.setSimple(index, usage)
	}
}

// IsTracked returns true if the texture is being tracked.
func (t *TextureTracker) IsTracked(index TrackerIndex) bool {
	return int(index) < len(t.set.simple) && t.metadata.IsOwned(index)
}

// Size returns the number of tracked textures.
func (t *TextureTracker) Size() int {
	return t.metadata.Count()
}

// SetSubresourceUsage puts the subresources sel selects of the texture at
// index, whose subresources are full, in usage, and appends the
// transitions from the states they were in to dst. A texture the tracker
// does not track yet has not been on the GPU, and takes no transition.
//
// Reference: wgpu-core track/texture.rs DeviceTextureTracker::set_single
func (t *TextureTracker) SetSubresourceUsage(index TrackerIndex, full, sel TextureSelector, usage TextureUses, dst []TexturePendingTransition) []TexturePendingTransition {
	src, ok := selection(full, sel, usage)
	if !ok {
		return dst
	}
	return t.apply(index, src, src, dst)
}

// apply brings the texture at index from its state to start, appending the
// transitions to dst, and leaves it in end. An untracked texture is
// inserted in end without transitions.
func (t *TextureTracker) apply(index TrackerIndex, start, end stateSource, dst []TexturePendingTransition) []TexturePendingTransition {
	t.ensureSize(int(index) + 1)
	if !t.metadata.IsOwned(index) {
		t.set.insert(index, end)
		t.metadata.SetOwned(index, true)
		return dst
	}
	if start.simple() && end.simple() && t.setSimple(index, start.uses, end.uses, &dst) {
		return dst
	}
	dst = t.set.barrier(index, start, dst)
	t.set.update(index, end, nil, 0)
	return dst
}

// Merge merges usage from scope into tracker, returning needed transitions.
// Each returned TexturePendingTransition should be converted to a
// hal.TextureBarrier and emitted before the commands of the scope.
//
// Reference: wgpu-core track/texture.rs set_from_usage_scope_and_drain_transitions
func (t *TextureTracker) Merge(scope *TextureUsageScope) []TexturePendingTransition {
	return t.MergeInto(scope, nil)
}

// MergeInto is Merge appending the transitions to dst.
func (t *TextureTracker) MergeInto(scope *TextureUsageScope, dst []TexturePendingTransition) []TexturePendingTransition {
	for i, uses := range scope.set.simple {
		index, valid := trackerIndexAt(i)
		if !valid || !scope.metadata.IsOwned(index) {
			continue
		}
		if uses != usesComplex && t.setSimple(index, uses, uses, &dst) {
			continue
		}
		src := sourceOf(&scope.set, index)
		dst = t.apply(index, src, src, dst)
	}
	return dst
}

// setSimple is apply for a texture that is tracked and in one state, from
// start to end, which are one state each: it appends the transition to dst
// and reports true. It does nothing and reports false for anything else.
func (t *TextureTracker) setSimple(index TrackerIndex, start, end TextureUses, dst *[]TexturePendingTransition) bool {
	if !t.IsTracked(index) {
		return false
	}
	cur := t.set.simple[index]
	if cur == usesComplex {
		return false
	}
	if !SkipBarrier(cur, start) {
		*dst = append(*dst, TexturePendingTransition{Index: index, Usage: TextureStateTransition{From: cur, To: start}})
	}
	t.set.simple[index] = end
	return true
}

// MergeEncoder merges a submitted command buffer's tracker into the device
// tracker. For every subresource the command buffer uses, a transition from
// the device's state to the command buffer's start state is returned
// (unless SkipBarrier allows omitting it), and the device state becomes the
// command buffer's end state. Textures the device does not track yet are
// inserted with the end state and need no barrier.
//
// Reference: wgpu-core track/texture.rs DeviceTextureTracker::set_from_tracker_and_drain_transitions
func (t *TextureTracker) MergeEncoder(enc *EncoderTextureTracker) []TexturePendingTransition {
	return t.MergeEncoderInto(enc, nil)
}

// MergeEncoderInto is MergeEncoder appending the transitions to dst, so a
// queue that merges every submit can keep the list.
func (t *TextureTracker) MergeEncoderInto(enc *EncoderTextureTracker, dst []TexturePendingTransition) []TexturePendingTransition {
	if enc == nil {
		return dst
	}
	for i, end := range enc.end.simple {
		index, valid := trackerIndexAt(i)
		if !valid || !enc.metadata.IsOwned(index) {
			continue
		}
		if start := enc.start.simple[i]; start != usesComplex && end != usesComplex && t.setSimple(index, start, end, &dst) {
			continue
		}
		dst = t.apply(index, sourceOf(&enc.start, index), sourceOf(&enc.end, index), dst)
	}
	return dst
}

// TextureSnapshot is the state a tracker had for one texture, from Save.
type TextureSnapshot struct {
	state   snapshot
	tracked bool
}

// Save returns the state the tracker has for the texture at index.
func (t *TextureTracker) Save(index TrackerIndex) TextureSnapshot {
	if !t.IsTracked(index) {
		return TextureSnapshot{}
	}
	return TextureSnapshot{state: t.set.snapshot(index), tracked: true}
}

// Restore puts the texture at index back in the state Save returned.
func (t *TextureTracker) Restore(index TrackerIndex, snap TextureSnapshot) {
	if !snap.tracked {
		t.Remove(index)
		return
	}
	t.ensureSize(int(index) + 1)
	t.set.restore(index, snap.state)
	t.metadata.SetOwned(index, true)
}

// TextureUsageScope tracks texture usage within ONE WebGPU usage scope: a
// render pass, a compute dispatch, or a single copy command. Adding a usage
// that is incompatible with an earlier usage of the same subresource in the
// scope is a validation error; different subresources of one texture may
// be used differently. Usages across different scopes are not merged here;
// the command buffer's EncoderTextureTracker transitions between them.
//
// Reference: wgpu-core track/texture.rs TextureUsageScope
type TextureUsageScope struct {
	set      stateSet
	metadata ResourceMetadata
}

// NewTextureUsageScope creates a new usage scope.
func NewTextureUsageScope() *TextureUsageScope {
	return &TextureUsageScope{metadata: NewResourceMetadata()}
}

// SetUsage adds a usage of every subresource of a texture to the scope.
// Returns an error if a subresource already has an incompatible usage.
func (s *TextureUsageScope) SetUsage(index TrackerIndex, usage TextureUses) error {
	s.set.ensureSize(int(index) + 1)
	if !s.metadata.IsOwned(index) {
		s.set.setSimple(index, usage)
		s.metadata.SetOwned(index, true)
		return nil
	}
	cur := s.set.simple[index]
	if cur == usesComplex {
		return s.set.merge(index, stateSource{uses: usage})
	}
	merged := cur | usage
	if conflicting(merged) {
		return &TextureUsageConflictError{Index: index, Existing: cur, New: usage}
	}
	s.set.simple[index] = merged
	return nil
}

// SetSubresourceUsage adds a usage of the subresources sel selects of the
// texture at index, whose subresources are full, to the scope. Returns an
// error if one of them already has an incompatible usage.
//
// Reference: wgpu-core track/texture.rs TextureUsageScope::merge_single
func (s *TextureUsageScope) SetSubresourceUsage(index TrackerIndex, full, sel TextureSelector, usage TextureUses) error {
	src, ok := selection(full, sel, usage)
	switch {
	case !ok:
		return nil
	case src.simple():
		return s.SetUsage(index, usage)
	}
	s.set.ensureSize(int(index) + 1)
	if !s.metadata.IsOwned(index) {
		s.set.insert(index, src)
		s.metadata.SetOwned(index, true)
		return nil
	}
	return s.set.merge(index, src)
}

// ReplaceUsage unconditionally replaces the usage recorded for a texture,
// for every subresource. It is intended only after preflight validation,
// or after an explicit transition where the scope must describe the state
// after the encoded barrier.
func (s *TextureUsageScope) ReplaceUsage(index TrackerIndex, usage TextureUses) {
	s.set.ensureSize(int(index) + 1)
	s.set.setSimple(index, usage)
	s.metadata.SetOwned(index, true)
}

// GetUsage returns the usages of a texture in this scope, of all its
// subresources the scope uses.
func (s *TextureUsageScope) GetUsage(index TrackerIndex) TextureUses {
	if !s.IsUsed(index) {
		return TextureUsesNone
	}
	return s.set.union(index)
}

// IsUsed returns true if the texture is used in this scope.
func (s *TextureUsageScope) IsUsed(index TrackerIndex) bool {
	return int(index) < len(s.set.simple) && s.metadata.IsOwned(index)
}

// IsEmpty returns true if no textures are tracked in this scope.
func (s *TextureUsageScope) IsEmpty() bool {
	return s.metadata.Count() == 0
}

// Clear resets the scope for reuse.
func (s *TextureUsageScope) Clear() {
	s.set.clear()
	s.metadata.Clear()
}

// TexturePendingTransition represents a texture state transition that needs
// a barrier: the subresources Selector selects of the texture at Index, all
// of them for the zero selector.
//
// Reference: wgpu-core track/mod.rs PendingTransition<TextureUses>
type TexturePendingTransition struct {
	Index    TrackerIndex
	Selector TextureSelector
	Usage    TextureStateTransition
}

// TextureStateTransition represents a from-to state change for a texture.
type TextureStateTransition struct {
	From TextureUses
	To   TextureUses
}

// NeedsBarrier returns true if this transition requires a barrier.
func (t TextureStateTransition) NeedsBarrier() bool {
	return !SkipBarrier(t.From, t.To)
}

// IntoHAL converts a pending transition to a HAL texture barrier on its
// subresources, with the coarse usages and the fine states behind them.
// The caller provides the hal.Texture handle.
func (p TexturePendingTransition) IntoHAL(texture hal.Texture) hal.TextureBarrier {
	return hal.TextureBarrier{
		Texture: texture,
		Range:   p.Selector.HALRange(),
		Usage: hal.TextureUsageTransition{
			OldUsage: p.Usage.From.ToTextureUsage(),
			NewUsage: p.Usage.To.ToTextureUsage(),
			Old:      p.Usage.From.ToHAL(),
			New:      p.Usage.To.ToHAL(),
		},
	}
}

// TextureUsageConflictError is returned when incompatible texture usages
// of the same subresources are detected within the same scope.
type TextureUsageConflictError struct {
	Index    TrackerIndex
	Selector TextureSelector
	Existing TextureUses
	New      TextureUses
}

// Error implements the error interface.
func (e *TextureUsageConflictError) Error() string {
	return fmt.Sprintf("track: texture %d usage conflict on %v: existing %d incompatible with %d",
		e.Index, e.Selector, e.Existing, e.New)
}
