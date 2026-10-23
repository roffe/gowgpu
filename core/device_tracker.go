//go:build !(js && wasm)

package core

import (
	"sync"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/hal"
)

// DeviceTracker holds device-level resource usage trackers.
// During queue submit, each command buffer's per-resource usage scope is
// merged into this tracker, producing the barrier list that must be
// recorded before the command buffer executes.
//
// This is the Go equivalent of Rust wgpu-core's DeviceTracker
// (device/resource.rs). Covers both textures and buffers.
//
// Reference: wgpu-core track/mod.rs DeviceTracker
type DeviceTracker struct {
	// mu serializes the methods below: Submit merges into the trackers, and
	// a buffer can be destroyed on another goroutine (Release, a GC cleanup)
	// and remove its entry.
	mu       sync.Mutex
	textures *track.TextureTracker
	buffers  *track.BufferTracker
}

// NewDeviceTracker creates a device tracker with empty trackers.
func NewDeviceTracker() *DeviceTracker {
	return &DeviceTracker{
		textures: track.NewTextureTracker(),
		buffers:  track.NewBufferTracker(),
	}
}

// Textures returns the device-level texture tracker.
func (dt *DeviceTracker) Textures() *track.TextureTracker {
	return dt.textures
}

// Buffers returns the device-level buffer tracker.
func (dt *DeviceTracker) Buffers() *track.BufferTracker {
	return dt.buffers
}

// MergeTextureScope merges a command buffer's texture tracker (see
// CoreCommandBuffer.TextureScope) into the device tracker and returns the
// barriers that bring each texture from its device state into the state the
// command buffer first uses it in. Afterwards the device state of every
// texture is the state the command buffer leaves it in.
//
// Each returned TexturePendingTransition must be converted to a
// hal.TextureBarrier (via IntoHAL) and recorded into a command encoder
// before the command buffer executes.
//
// Reference: wgpu-core device/queue.rs pre_submit_for_command_buffers
func (dt *DeviceTracker) MergeTextureScope(cmdTracker *track.EncoderTextureTracker) []track.TexturePendingTransition {
	return dt.MergeTextureScopeInto(cmdTracker, nil)
}

// MergeTextureScopeInto is MergeTextureScope appending to dst.
func (dt *DeviceTracker) MergeTextureScopeInto(cmdTracker *track.EncoderTextureTracker, dst []track.TexturePendingTransition) []track.TexturePendingTransition {
	if cmdTracker == nil {
		return dst
	}
	dt.mu.Lock()
	defer dt.mu.Unlock()
	return dt.textures.MergeEncoderInto(cmdTracker, dst)
}

// MergeBufferScope merges a command buffer's buffer tracker (see
// CoreCommandBuffer.BufferScope) into the device tracker and returns the
// barriers into each buffer's first usage in the command buffer. Afterwards
// the device state of every buffer is the command buffer's last usage.
//
// Each returned PendingTransition must be converted to a hal.BufferBarrier
// (via IntoHAL) and recorded into a command encoder before the command
// buffer executes.
//
// Reference: wgpu-core device/queue.rs pre_submit_for_command_buffers
func (dt *DeviceTracker) MergeBufferScope(cmdTracker *track.EncoderBufferTracker) []track.PendingTransition {
	return dt.MergeBufferScopeInto(cmdTracker, nil)
}

// MergeBufferScopeInto is MergeBufferScope appending to dst.
func (dt *DeviceTracker) MergeBufferScopeInto(cmdTracker *track.EncoderBufferTracker, dst []track.PendingTransition) []track.PendingTransition {
	if cmdTracker == nil {
		return dst
	}
	dt.mu.Lock()
	defer dt.mu.Unlock()
	return dt.buffers.MergeEncoderInto(cmdTracker, dst)
}

// SetBufferUsage moves a buffer to usage outside any command buffer, as a
// queue write does, and returns the barrier from the state the GPU last used
// it in, if one is needed. A buffer the device does not track yet has not
// been on the GPU, so it is inserted without one.
//
// Reference: wgpu-core device/queue.rs write_buffer, trackers.buffers.set_single
func (dt *DeviceTracker) SetBufferUsage(index track.TrackerIndex, usage track.BufferUses) (track.PendingTransition, bool) {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	if !dt.buffers.IsTracked(index) {
		dt.buffers.InsertSingle(index, usage)
		return track.PendingTransition{}, false
	}
	current := dt.buffers.GetUsage(index)
	dt.buffers.SetUsage(index, usage)
	if track.SkipBufferBarrier(current, usage) {
		return track.PendingTransition{}, false
	}
	return track.PendingTransition{Index: index, Usage: track.StateTransition{From: current, To: usage}}, true
}

// InsertBuffer registers a buffer in the device tracker with its initial usage.
func (dt *DeviceTracker) InsertBuffer(index track.TrackerIndex, usage track.BufferUses) {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	dt.buffers.InsertSingle(index, usage)
}

// RemoveBuffer removes a buffer from the device tracker.
func (dt *DeviceTracker) RemoveBuffer(index track.TrackerIndex) {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	dt.buffers.Remove(index)
}

// TrackPresentTexture records a swapchain texture's transition to Present
// state. This is called after all user command buffers are processed, to
// generate the final COLOR_ATTACHMENT_OPTIMAL -> PRESENT_SRC barrier.
//
// The trackerIndex must be a valid index for this texture in the device
// tracker. If the texture is not yet tracked, it will be added.
//
// Returns the barrier (from -> Present) or nil if no barrier is needed.
func (dt *DeviceTracker) TrackPresentTexture(trackerIndex track.TrackerIndex) *track.TexturePendingTransition {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	scope := track.NewTextureUsageScope()
	_ = scope.SetUsage(trackerIndex, track.TextureUsesPresent)
	transitions := dt.textures.Merge(scope)
	if len(transitions) == 0 {
		return nil
	}
	return &transitions[0]
}

// TextureUsage returns the usage the device tracker last left a texture in,
// or TextureUsesNone for one it does not track.
func (dt *DeviceTracker) TextureUsage(index track.TrackerIndex) track.TextureUses {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	return dt.textures.GetUsage(index)
}

// InsertTexture registers a texture in the device tracker with its initial
// usage. Called when a texture is first used (e.g., acquired from swapchain).
func (dt *DeviceTracker) InsertTexture(index track.TrackerIndex, usage track.TextureUses) {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	dt.textures.InsertSingle(index, usage)
}

// RemoveTexture removes a texture from the device tracker.
// Called when a texture is destroyed.
func (dt *DeviceTracker) RemoveTexture(index track.TrackerIndex) {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	dt.textures.Remove(index)
}

// TextureState returns the usage the device tracker holds for a texture and
// whether it tracks it, for RestoreTexture.
func (dt *DeviceTracker) TextureState(index track.TrackerIndex) (track.TextureUses, bool) {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	return dt.textures.GetUsage(index), dt.textures.IsTracked(index)
}

// RestoreTexture puts a texture's state back to what TextureState returned.
func (dt *DeviceTracker) RestoreTexture(index track.TrackerIndex, usage track.TextureUses, tracked bool) {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	if tracked {
		dt.textures.InsertSingle(index, usage)
	} else {
		dt.textures.Remove(index)
	}
}

// BufferState returns the usage the device tracker holds for a buffer and
// whether it tracks it, for RestoreBuffer.
func (dt *DeviceTracker) BufferState(index track.TrackerIndex) (track.BufferUses, bool) {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	return dt.buffers.GetUsage(index), dt.buffers.IsTracked(index)
}

// RestoreBuffer puts a buffer's state back to what BufferState returned.
func (dt *DeviceTracker) RestoreBuffer(index track.TrackerIndex, usage track.BufferUses, tracked bool) {
	dt.mu.Lock()
	defer dt.mu.Unlock()
	if tracked {
		dt.buffers.InsertSingle(index, usage)
	} else {
		dt.buffers.Remove(index)
	}
}

// BarrierCBFromTransitions creates a HAL command buffer containing texture
// barriers for the given transitions. The caller provides a HAL command
// encoder (from the encoder pool) and a function to resolve tracker indices
// to HAL textures.
//
// The caller MUST pass a non-empty transitions slice. Calling with an empty
// slice is a programming error (the caller should check len before calling).
//
// Returns the recorded command buffer, or an error if encoding fails.
// The caller must ensure the returned command buffer is included in the
// Submit call's command buffer list.
func BarrierCBFromTransitions(
	halEncoder hal.CommandEncoder,
	transitions []track.TexturePendingTransition,
	resolveTexture func(track.TrackerIndex) hal.Texture,
) (hal.CommandBuffer, error) {
	return BarrierCBFromAllTransitions(halEncoder, transitions, nil, resolveTexture, nil)
}

// BarrierResolver maps tracker indices to the HAL resources of a submit's
// command buffers. A nil result means the resource is gone and its barrier
// is skipped.
type BarrierResolver interface {
	ResolveTexture(track.TrackerIndex) hal.Texture
	ResolveBuffer(track.TrackerIndex) hal.Buffer
}

// funcResolver adapts the function form of the resolvers.
type funcResolver struct {
	texture func(track.TrackerIndex) hal.Texture
	buffer  func(track.TrackerIndex) hal.Buffer
}

func (r funcResolver) ResolveTexture(i track.TrackerIndex) hal.Texture {
	if r.texture == nil {
		return nil
	}
	return r.texture(i)
}

func (r funcResolver) ResolveBuffer(i track.TrackerIndex) hal.Buffer {
	if r.buffer == nil {
		return nil
	}
	return r.buffer(i)
}

// BarrierCBFromAllTransitions creates a HAL command buffer containing both
// texture and buffer barriers. This is the generalized version that handles
// all resource types in a single preamble command buffer.
//
// Either textureTransitions or bufferTransitions (or both) must be non-empty.
//
// Returns the recorded command buffer, or an error if encoding fails.
func BarrierCBFromAllTransitions(
	halEncoder hal.CommandEncoder,
	textureTransitions []track.TexturePendingTransition,
	bufferTransitions []track.PendingTransition,
	resolveTexture func(track.TrackerIndex) hal.Texture,
	resolveBuffer func(track.TrackerIndex) hal.Buffer,
) (hal.CommandBuffer, error) {
	return BarrierCB(halEncoder, textureTransitions, bufferTransitions, funcResolver{resolveTexture, resolveBuffer}, nil)
}

// BarrierScratch holds the barrier lists BarrierCB hands the HAL, between
// calls, so that a queue submitting every frame allocates none. The backends
// copy what they keep, and the lists are emptied after each use.
type BarrierScratch struct {
	Textures []hal.TextureBarrier
	Buffers  []hal.BufferBarrier
}

// BarrierCB is BarrierCBFromAllTransitions with a BarrierResolver, which a
// queue implements itself, and a scratch the queue keeps; scratch may be nil.
func BarrierCB(
	halEncoder hal.CommandEncoder,
	textureTransitions []track.TexturePendingTransition,
	bufferTransitions []track.PendingTransition,
	resolve BarrierResolver,
	scratch *BarrierScratch,
) (hal.CommandBuffer, error) {
	if scratch == nil {
		scratch = &BarrierScratch{}
	}
	defer scratch.clear()

	// Begin a short-lived encoding for the barrier commands
	if err := halEncoder.BeginEncoding("barrier-inject"); err != nil {
		return nil, err
	}

	// Emit texture barriers
	if len(textureTransitions) > 0 {
		barriers := scratch.Textures[:0]
		for _, t := range textureTransitions {
			tex := resolve.ResolveTexture(t.Index)
			if tex == nil {
				continue // destroyed texture — skip
			}
			barriers = append(barriers, t.IntoHAL(tex))
		}
		scratch.Textures = barriers
		if len(barriers) > 0 {
			halEncoder.TransitionTextures(barriers)
		}
	}

	// Emit buffer barriers
	if len(bufferTransitions) > 0 {
		barriers := scratch.Buffers[:0]
		for _, t := range bufferTransitions {
			buf := resolve.ResolveBuffer(t.Index)
			if buf == nil {
				continue // destroyed buffer — skip
			}
			barriers = append(barriers, t.IntoHAL(buf))
		}
		scratch.Buffers = barriers
		if len(barriers) > 0 {
			halEncoder.TransitionBuffers(barriers)
		}
	}

	return halEncoder.EndEncoding()
}

// clear empties the lists, so they pin no resource between submits.
func (s *BarrierScratch) clear() {
	clear(s.Textures)
	clear(s.Buffers)
	s.Textures, s.Buffers = s.Textures[:0], s.Buffers[:0]
}
