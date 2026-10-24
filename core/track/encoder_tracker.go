//go:build !(js && wasm)

package track

import (
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// Usage scopes versus command-buffer trackers
//
// WebGPU only defines usage conflicts inside a single usage scope: one render
// pass, one compute dispatch, or one copy command. Different commands of the
// same command encoder may use a resource with different usages; the
// implementation transitions the resource between them.
//
// The track package therefore has two kinds of per-command-buffer structures,
// exactly like Rust wgpu-core (track/mod.rs, "Use Case"):
//
//   - TextureUsageScope / BufferUsageScope: one usage scope. Adding a usage
//     merges it with earlier usages of the same scope and fails on conflict.
//   - EncoderTextureTracker / EncoderBufferTracker: the command buffer's
//     tracker. It never reports conflicts. It records the first usage of each
//     resource in the command buffer (start) and the current usage (end), and
//     returns the transitions needed between consecutive usage scopes.
//
// Textures are tracked per subresource, mip level, array layer and, for a
// format with both, depth or stencil aspect: a texture whose subresources
// are in different states keeps a state per range of layers of each mip
// level, which holds the state of each aspect. The usage scopes check each
// aspect on its own. The trackers that make barriers move both aspects of
// a range together where a command uses one, see TextureUses.filled.
//
// At submit time the device tracker emits barriers from its current state to
// each command buffer's start state, then adopts the command buffer's end
// state (see TextureTracker.MergeEncoder / BufferTracker.MergeEncoder).
//
// Reference: wgpu-core track/mod.rs
//
//	CommandBuffer <- UsageScope = insert(buffer.start, buffer.end, scope)
//	                              OR barrier(buffer.end, scope) + update(buffer.end, scope)
//	Device <- CommandBuffer = insert(device.start, device.end, buffer.start, buffer.end)
//	                          OR barrier(device.end, buffer.start) + update(device.end, buffer.end)

// EncoderTextureTracker tracks texture states across the commands of one
// command buffer, per subresource like TextureTracker.
//
// The zero value is ready to use; Clear keeps the arrays for the next
// command buffer.
//
// Reference: wgpu-core track/texture.rs TextureTracker (start_set + end_set)
type EncoderTextureTracker struct {
	start    stateSet
	end      stateSet
	metadata ResourceMetadata
}

// NewEncoderTextureTracker creates an empty command-buffer texture tracker.
func NewEncoderTextureTracker() *EncoderTextureTracker {
	return &EncoderTextureTracker{metadata: NewResourceMetadata()}
}

// SetSingle records that the next command uses every subresource of the
// texture at index with usage, and returns the transitions that take them
// there. See SetSubresourceUsage.
func (t *EncoderTextureTracker) SetSingle(index TrackerIndex, usage TextureUses) []TexturePendingTransition {
	return t.SetSubresourceUsage(index, TextureSelector{}, TextureSelector{}, usage, nil)
}

// SetSubresourceUsage records that the next command uses the subresources
// sel selects of the texture at index, whose subresources are full, with
// usage, and appends to dst the transitions that take them there.
//
// On the first use of a subresource in this command buffer the usage
// becomes both its start and its end state, and no transition is returned:
// the barrier into the start state is generated at submit time from the
// device tracker. On later uses a transition from the current end state is
// returned unless the state is unchanged and hardware-ordered
// (SkipBarrier). The end state is updated either way. It never fails; conflicts are a usage-scope concern.
//
// Reference: wgpu-core track/texture.rs TextureTracker::set_single
func (t *EncoderTextureTracker) SetSubresourceUsage(index TrackerIndex, full, sel TextureSelector, usage TextureUses, dst []TexturePendingTransition) []TexturePendingTransition {
	src, ok := selection(full, sel, usage)
	if !ok {
		return dst
	}
	return t.apply(index, src, src, 0, true, dst)
}

// apply takes the texture at index into the state src brings, which is its
// start state where the command buffer has not used it yet (startUses when
// that is not 0), appending the transitions to dst when emit is set.
//
// Reference: wgpu-core track/texture.rs insert_or_barrier_update
func (t *EncoderTextureTracker) apply(index TrackerIndex, src, start stateSource, startUses TextureUses, emit bool, dst []TexturePendingTransition) []TexturePendingTransition {
	src, start, startUses = src.filled(), start.filled(), startUses.filled()
	t.ensureSize(int(index) + 1)
	if !t.metadata.IsOwned(index) {
		t.start.insert(index, start)
		t.end.insert(index, src)
		t.metadata.SetOwned(index, true)
		return dst
	}
	if cur := t.end.simple[index]; cur != usesComplex && src.simple() {
		if emit && !SkipBarrier(cur, src.uses) {
			dst = append(dst, TexturePendingTransition{Index: index, Usage: TextureStateTransition{From: cur, To: src.uses}})
		}
		t.end.simple[index] = src.uses
		return dst
	}
	if emit {
		dst = t.end.barrier(index, src, dst)
	}
	t.end.update(index, src, &t.start, startUses)
	return dst
}

// SetFromScope applies every usage of a finished usage scope (for example a
// render pass) to the tracker and returns the transitions that must be
// recorded before that scope's commands execute.
//
// Reference: wgpu-core track/texture.rs TextureTracker::set_from_usage_scope
func (t *EncoderTextureTracker) SetFromScope(scope *TextureUsageScope) []TexturePendingTransition {
	return t.SetFromScopeInto(scope, nil)
}

// SetFromScopeInto is SetFromScope appending to dst, so an encoder that
// records a pass every frame can keep the list.
func (t *EncoderTextureTracker) SetFromScopeInto(scope *TextureUsageScope, dst []TexturePendingTransition) []TexturePendingTransition {
	if scope == nil {
		return dst
	}
	for i, uses := range scope.set.simple {
		index, valid := trackerIndexAt(i)
		if !valid || !scope.metadata.IsOwned(index) {
			continue
		}
		if uses != usesComplex {
			dst = t.apply(index, stateSource{uses: uses}, stateSource{uses: uses}, 0, true, dst)
			continue
		}
		src := sourceOf(&scope.set, index)
		dst = t.apply(index, src, src, 0, true, dst)
	}
	return dst
}

// RecordTransition records a barrier the caller has already encoded
// explicitly (CommandEncoder.TransitionTextures) on the subresources sel
// selects of the texture at index, whose subresources are full. Their end
// state becomes to. The ones this command buffer has not used yet start in
// from, so the submit-time barrier brings them into the state the explicit
// barrier expects. A zero from (undefined contents) has no start state to
// request; to is used instead.
func (t *EncoderTextureTracker) RecordTransition(index TrackerIndex, full, sel TextureSelector, from, to TextureUses) {
	if from.IsEmpty() {
		from = to
	}
	src, ok := selection(full, sel, to)
	if !ok {
		return
	}
	start, _ := selection(full, sel, from)
	t.apply(index, src, start, start.uses, false, nil)
}

// StartUsage returns the first usage of the texture in this command buffer,
// every one of its subresources had when they differ, or TextureUsesNone if
// the texture is not used.
func (t *EncoderTextureTracker) StartUsage(index TrackerIndex) TextureUses {
	if !t.IsUsed(index) {
		return TextureUsesNone
	}
	return t.start.union(index)
}

// EndUsage returns the usage the command buffer leaves the texture in,
// every one its subresources are in when they differ, or TextureUsesNone if
// the texture is not used.
func (t *EncoderTextureTracker) EndUsage(index TrackerIndex) TextureUses {
	if !t.IsUsed(index) {
		return TextureUsesNone
	}
	return t.end.union(index)
}

// IsUsed returns true if the texture is used in this command buffer.
func (t *EncoderTextureTracker) IsUsed(index TrackerIndex) bool {
	return int(index) < len(t.end.simple) && t.metadata.IsOwned(index)
}

// IsEmpty returns true if no textures are used in this command buffer.
func (t *EncoderTextureTracker) IsEmpty() bool {
	return t.metadata.Count() == 0
}

// Clear forgets every texture, keeping the arrays for the next command
// buffer.
func (t *EncoderTextureTracker) Clear() {
	t.start.clear()
	t.end.clear()
	t.metadata.Clear()
}

func (t *EncoderTextureTracker) ensureSize(size int) {
	t.start.ensureSize(size)
	t.end.ensureSize(size)
}

// EncoderBufferTracker tracks buffer states across the commands of one
// command buffer. See EncoderTextureTracker for the semantics.
//
// Reference: wgpu-core track/buffer.rs BufferTracker (start + end)
type EncoderBufferTracker struct {
	start    []BufferUses
	end      []BufferUses
	metadata ResourceMetadata
}

// NewEncoderBufferTracker creates an empty command-buffer buffer tracker.
func NewEncoderBufferTracker() *EncoderBufferTracker {
	return &EncoderBufferTracker{
		start:    make([]BufferUses, 0, 32),
		end:      make([]BufferUses, 0, 32),
		metadata: NewResourceMetadata(),
	}
}

// SkipBufferBarrier returns true if a buffer moving from old to next needs no
// barrier: the state is unchanged and contains only hardware-ordered
// (read-only) usages.
//
// Reference: wgpu-core track/mod.rs skip_barrier with the INCLUSIVE set
func SkipBufferBarrier(old, next BufferUses) bool {
	return old == next && old.IsReadOnly()
}

// SetSingle records that the next command uses the buffer with usage. See
// EncoderTextureTracker.SetSingle.
//
// Reference: wgpu-core track/buffer.rs BufferTracker::set_single
func (t *EncoderBufferTracker) SetSingle(index TrackerIndex, usage BufferUses) (PendingTransition, bool) {
	t.ensureSize(int(index) + 1)
	if !t.metadata.IsOwned(index) {
		t.start[index] = usage
		t.end[index] = usage
		t.metadata.SetOwned(index, true)
		return PendingTransition{}, false
	}
	current := t.end[index]
	t.end[index] = usage
	if SkipBufferBarrier(current, usage) {
		return PendingTransition{}, false
	}
	return PendingTransition{
		Index: index,
		Usage: StateTransition{From: current, To: usage},
	}, true
}

// SetFromScope applies every usage of a finished usage scope to the tracker
// and returns the transitions needed before that scope's commands.
//
// Reference: wgpu-core track/buffer.rs BufferTracker::set_from_usage_scope
func (t *EncoderBufferTracker) SetFromScope(scope *BufferUsageScope) []PendingTransition {
	return t.SetFromScopeInto(scope, nil)
}

// SetFromScopeInto is SetFromScope appending to dst.
func (t *EncoderBufferTracker) SetFromScopeInto(scope *BufferUsageScope, dst []PendingTransition) []PendingTransition {
	if scope == nil {
		return dst
	}
	for i := range scope.states {
		index, valid := trackerIndexAt(i)
		if !valid || !scope.metadata.IsOwned(index) {
			continue
		}
		if transition, ok := t.SetSingle(index, scope.states[i].usage); ok {
			dst = append(dst, transition)
		}
	}
	return dst
}

// StartUsage returns the first usage of the buffer in this command buffer.
func (t *EncoderBufferTracker) StartUsage(index TrackerIndex) BufferUses {
	if !t.IsUsed(index) {
		return BufferUsesNone
	}
	return t.start[index]
}

// EndUsage returns the usage the command buffer leaves the buffer in.
func (t *EncoderBufferTracker) EndUsage(index TrackerIndex) BufferUses {
	if !t.IsUsed(index) {
		return BufferUsesNone
	}
	return t.end[index]
}

// IsUsed returns true if the buffer is used in this command buffer.
func (t *EncoderBufferTracker) IsUsed(index TrackerIndex) bool {
	return int(index) < len(t.end) && t.metadata.IsOwned(index)
}

// IsEmpty returns true if no buffers are used in this command buffer.
func (t *EncoderBufferTracker) IsEmpty() bool {
	return t.metadata.Count() == 0
}

// Clear forgets every buffer, keeping the arrays for the next command
// buffer.
func (t *EncoderBufferTracker) Clear() {
	t.start = t.start[:0]
	t.end = t.end[:0]
	t.metadata.Clear()
}

func (t *EncoderBufferTracker) ensureSize(size int) {
	for len(t.end) < size {
		t.start = append(t.start, BufferUsesNone)
		t.end = append(t.end, BufferUsesNone)
	}
}

// trackerIndexAt converts a dense state-slice position into a TrackerIndex.
// Positions outside the TrackerIndex range (never produced by the allocator)
// are reported invalid instead of being truncated.
func trackerIndexAt(i int) (TrackerIndex, bool) {
	if i < 0 || uint64(i) > uint64(^TrackerIndex(0)-1) {
		return 0, false
	}
	return TrackerIndex(i), true
}

// TextureUsesFromUsage converts a public texture usage, as passed to an
// explicit barrier, into tracker states. RenderAttachment maps to the depth
// or color target state depending on the texture format; StorageBinding maps
// to the exclusive StorageWrite state so later uses are always fenced.
//
// Reference: wgpu-core conv.rs map_texture_usage
func TextureUsesFromUsage(usage gputypes.TextureUsage, depthStencil bool) TextureUses {
	return TextureUses(hal.TextureUsesFromUsage(usage, depthStencil))
}
