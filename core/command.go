//go:build !(js && wasm)

package core

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// ComputePassDescriptor describes how to create a compute pass.
type ComputePassDescriptor struct {
	// Label is an optional debug name for the compute pass.
	Label string

	// TimestampWrites are timestamp queries to write at pass boundaries (optional).
	TimestampWrites *ComputePassTimestampWrites
}

// ComputePassTimestampWrites describes timestamp query writes for a compute pass.
type ComputePassTimestampWrites struct {
	// QuerySet is the query set to write timestamps to.
	QuerySet QuerySetID

	// BeginningOfPassWriteIndex is the query index for pass start.
	// Use nil to skip.
	BeginningOfPassWriteIndex *uint32

	// EndOfPassWriteIndex is the query index for pass end.
	// Use nil to skip.
	EndOfPassWriteIndex *uint32
}

// =============================================================================
// HAL-Integrated Command Encoder (CORE-005)
// =============================================================================

// CommandEncoderStatus represents the current state of a command encoder.
//
// State machine transitions:
//
//	Recording -> (BeginRenderPass/BeginComputePass) -> Locked
//	Locked    -> (EndRenderPass/EndComputePass)     -> Recording
//	Recording -> Finish()                           -> Finished
//	Finished  -> (submitted to queue)               -> Consumed
//	Any state -> (error)                            -> Error
type CommandEncoderStatus int32

const (
	// CommandEncoderStatusRecording - ready to record commands.
	CommandEncoderStatusRecording CommandEncoderStatus = iota

	// CommandEncoderStatusLocked - a pass is in progress.
	CommandEncoderStatusLocked

	// CommandEncoderStatusFinished - encoding complete, ready for submit.
	CommandEncoderStatusFinished

	// CommandEncoderStatusError - an error occurred.
	CommandEncoderStatusError

	// CommandEncoderStatusConsumed - submitted to queue.
	CommandEncoderStatusConsumed
)

// Common state names shared between CommandEncoderStatus and CommandEncoderPassState.
const (
	stateRecording = "Recording"
	stateFinished  = "Finished"
	stateError     = "Error"
)

// String returns a human-readable representation of the status.
func (s CommandEncoderStatus) String() string {
	switch s {
	case CommandEncoderStatusRecording:
		return stateRecording
	case CommandEncoderStatusLocked:
		return "Locked"
	case CommandEncoderStatusFinished:
		return stateFinished
	case CommandEncoderStatusError:
		return stateError
	case CommandEncoderStatusConsumed:
		return "Consumed"
	default:
		return fmt.Sprintf("Unknown(%d)", s)
	}
}

// CommandBufferMutable holds mutable state during encoding.
//
// This tracks resources used within a command buffer for validation
// and synchronization purposes.
type CommandBufferMutable struct {
	// textureTracker records, for every texture this command buffer uses,
	// the first usage (start) and the current usage (end). It never reports
	// usage conflicts: conflicts only exist inside one usage scope (a render
	// pass, a compute pass or a single copy command), which is validated
	// separately. Between usage scopes the tracker yields the transitions to
	// record. At submit the device TextureTracker emits barriers into the
	// start states and adopts the end states.
	//
	// Reference: wgpu-core command/mod.rs CommandBufferMutable.trackers.textures
	textureTracker *track.EncoderTextureTracker

	// bufferTracker is the buffer counterpart of textureTracker.
	//
	// Reference: wgpu-core command/mod.rs CommandBufferMutable.trackers.buffers
	bufferTracker *track.EncoderBufferTracker

	// pass is the usage scope of the pass being recorded. An encoder has one
	// pass open at a time, so one scope set serves them all: cleared when a
	// pass begins, applied to the trackers when it ends.
	pass passScope

	// activePass is the current pass encoder (if any).
	// This is either *CoreRenderPassEncoder or *CoreComputePassEncoder.
	activePass any
}

// passScope is the usage scope of one pass: its attachments, the textures
// its bind groups bind and its vertex and index buffers, with the HAL
// handles the transitions into the pass are recorded against. The lists are
// kept across passes, so recording one allocates nothing.
//
// Reference: wgpu-core command/render.rs State.scope (UsageScope per pass)
type passScope struct {
	textures *track.TextureUsageScope
	buffers  *track.BufferUsageScope
	// texHAL and bufHAL hold the HAL handle of each resource in the scope,
	// by tracker index, for the barriers.
	texHAL []scopeTexture
	bufHAL []scopeBuffer
	// Scratch for the transitions into the scope and their barriers.
	texTransitions []track.TexturePendingTransition
	bufTransitions []track.PendingTransition
	texBarriers    []hal.TextureBarrier
	bufBarriers    []hal.BufferBarrier
	// split is set on a render pass recorded in a HAL command buffer of its
	// own, so that the barriers into it can go in one ahead of it.
	split bool
	// drawValidations are the validated indirect draws of a render pass,
	// whose validation goes ahead of it with the barriers.
	drawValidations []drawValidation
	// zeros are the textures a render pass reads whose contents an earlier
	// pass discarded, zeroed ahead of it with the barriers, and the
	// transitions and barriers into the passes that zero them.
	zeros           []passZero
	zeroTransitions []track.TexturePendingTransition
	zeroBarriers    []hal.TextureBarrier
}

type scopeTexture struct {
	index   track.TrackerIndex
	texture hal.Texture
}

type scopeBuffer struct {
	index  track.TrackerIndex
	buffer hal.Buffer
}

// clear empties the scope for the next pass.
func (p *passScope) clear() {
	p.textures.Clear()
	p.buffers.Clear()
	clear(p.texHAL)
	clear(p.bufHAL)
	p.texHAL, p.bufHAL = p.texHAL[:0], p.bufHAL[:0]
	clear(p.drawValidations)
	p.drawValidations = p.drawValidations[:0]
	clear(p.zeros)
	p.zeros = p.zeros[:0]
	p.clearScratch()
	p.split = false
}

// clearScratch empties the transition and barrier lists, so they pin no
// resource between passes.
func (p *passScope) clearScratch() {
	clear(p.texBarriers)
	clear(p.bufBarriers)
	p.texTransitions, p.bufTransitions = p.texTransitions[:0], p.bufTransitions[:0]
	p.texBarriers, p.bufBarriers = p.texBarriers[:0], p.bufBarriers[:0]
	p.clearZeroScratch()
}

// addTexture merges a usage of the subresources sel selects of a texture,
// whose subresources are full, into the scope. An incompatible usage of one
// of them earlier in the scope is a conflict. A texture the scope does not
// know yet is listed with its HAL handle; nil means no barrier can be
// recorded for it.
func (p *passScope) addTexture(index track.TrackerIndex, full, sel track.TextureSelector, texture hal.Texture, usage track.TextureUses) error {
	known := p.textures.IsUsed(index)
	if err := p.textures.SetSubresourceUsage(index, full, sel, usage); err != nil {
		return err
	}
	if !known {
		p.texHAL = append(p.texHAL, scopeTexture{index: index, texture: texture})
	}
	return nil
}

// addView is addTexture for the subresources of an attachment view's parent
// texture. Views without a parent texture or without a valid tracker index
// are not tracked.
// The view names the HAL texture it is of when the public layer knows a
// better handle than the parent's (a surface texture); otherwise the
// parent's is used.
func (p *passScope) addView(view *TextureView, usage track.TextureUses, guard SnatchGuard) error {
	if view == nil || view.Parent == nil {
		return nil
	}
	td := view.Parent.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return nil
	}
	texture := view.Texture
	if texture == nil {
		texture = view.Parent.Raw(guard)
	}
	return p.addTexture(td.Index(), view.Parent.Subresources(), view.Subresources(), texture, usage)
}

// addBuffer is addTexture for a buffer.
func (p *passScope) addBuffer(index track.TrackerIndex, buffer hal.Buffer, usage track.BufferUses) error {
	known := p.buffers.IsUsed(index)
	if err := p.buffers.SetUsage(index, usage); err != nil {
		return err
	}
	if !known {
		p.bufHAL = append(p.bufHAL, scopeBuffer{index: index, buffer: buffer})
	}
	return nil
}

// texture returns the HAL handle of the texture at index in the scope.
func (p *passScope) texture(index track.TrackerIndex) hal.Texture {
	for i := range p.texHAL {
		if p.texHAL[i].index == index {
			return p.texHAL[i].texture
		}
	}
	return nil
}

// buffer returns the HAL handle of the buffer at index in the scope.
func (p *passScope) buffer(index track.TrackerIndex) hal.Buffer {
	for i := range p.bufHAL {
		if p.bufHAL[i].index == index {
			return p.bufHAL[i].buffer
		}
	}
	return nil
}

// recordBarriers records the scope's transitions on halEncoder, against the
// HAL handles the scope holds. A resource without one is skipped.
func (p *passScope) recordBarriers(halEncoder hal.CommandEncoder) {
	p.texBarriers = p.texBarriers[:0]
	for _, t := range p.texTransitions {
		if tex := p.texture(t.Index); tex != nil {
			p.texBarriers = append(p.texBarriers, t.IntoHAL(tex))
		}
	}
	if len(p.texBarriers) > 0 {
		halEncoder.TransitionTextures(p.texBarriers)
	}
	p.bufBarriers = p.bufBarriers[:0]
	for _, t := range p.bufTransitions {
		if buf := p.buffer(t.Index); buf != nil {
			p.bufBarriers = append(p.bufBarriers, t.IntoHAL(buf))
		}
	}
	if len(p.bufBarriers) > 0 {
		halEncoder.TransitionBuffers(p.bufBarriers)
	}
}

// scopePool keeps the trackers and pass scope of spent command buffers,
// emptied, for the next command encoder. A fresh set is a dozen allocations,
// and more as the arrays grow to the tracker indices a frame's resources
// have.
type scopePool struct {
	mu   sync.Mutex
	free []*CommandBufferMutable
}

// maxFreeScopes caps how many sets of scopes a device keeps for reuse.
const maxFreeScopes = 32

// get returns a kept set of scopes, or new ones. p may be nil.
func (p *scopePool) get() *CommandBufferMutable {
	if p != nil {
		p.mu.Lock()
		if n := len(p.free); n > 0 {
			m := p.free[n-1]
			p.free[n-1] = nil
			p.free = p.free[:n-1]
			p.mu.Unlock()
			return m
		}
		p.mu.Unlock()
	}
	return newCommandBufferMutable()
}

// newCommandBufferMutable makes an empty set of trackers and pass scope.
func newCommandBufferMutable() *CommandBufferMutable {
	return &CommandBufferMutable{
		textureTracker: track.NewEncoderTextureTracker(),
		bufferTracker:  track.NewEncoderBufferTracker(),
		pass: passScope{
			textures: track.NewTextureUsageScope(),
			buffers:  track.NewBufferUsageScope(),
		},
	}
}

// put empties the scopes and keeps them. p may be nil.
func (p *scopePool) put(m *CommandBufferMutable) {
	if p == nil || m == nil || m.textureTracker == nil || m.bufferTracker == nil ||
		m.pass.textures == nil || m.pass.buffers == nil {
		return
	}
	m.textureTracker.Clear()
	m.bufferTracker.Clear()
	m.pass.clear()
	m.activePass = nil
	p.mu.Lock()
	if len(p.free) < maxFreeScopes {
		p.free = append(p.free, m)
	}
	p.mu.Unlock()
}

// BufferUses tracks how a buffer is used within a command buffer.
type BufferUses uint32

const (
	// BufferUsesNone indicates no usage.
	BufferUsesNone BufferUses = 0
	// BufferUsesVertex indicates vertex buffer usage.
	BufferUsesVertex BufferUses = 1 << iota
	// BufferUsesIndex indicates index buffer usage.
	BufferUsesIndex
	// BufferUsesUniform indicates uniform buffer usage.
	BufferUsesUniform
	// BufferUsesStorage indicates storage buffer usage.
	BufferUsesStorage
	// BufferUsesIndirect indicates indirect buffer usage.
	BufferUsesIndirect
	// BufferUsesCopySrc indicates copy source usage.
	BufferUsesCopySrc
	// BufferUsesCopyDst indicates copy destination usage.
	BufferUsesCopyDst
)

// TextureUses tracks how a texture is used within a command buffer.
type TextureUses uint32

const (
	// TextureUsesNone indicates no usage.
	TextureUsesNone TextureUses = 0
	// TextureUsesSampled indicates sampled texture usage.
	TextureUsesSampled TextureUses = 1 << iota
	// TextureUsesStorage indicates storage texture usage.
	TextureUsesStorage
	// TextureUsesRenderAttachment indicates render attachment usage.
	TextureUsesRenderAttachment
	// TextureUsesCopySrc indicates copy source usage.
	TextureUsesCopySrc
	// TextureUsesCopyDst indicates copy destination usage.
	TextureUsesCopyDst
)

// CoreCommandEncoder records GPU commands for submission.
//
// This is the HAL-integrated command encoder that bridges core command
// recording to HAL command encoders. The state machine ensures commands
// are recorded in the correct order and validates encoder state transitions.
//
// Multi-CB support: A single CoreCommandEncoder can produce multiple HAL
// command buffers via OpenPass/CloseCB/CloseAndSwap/CloseAndPushFront.
// All CBs are submitted together in one HAL queue submit, enabling
// barrier CB insertion before/after render pass CBs in the same submit.
//
// Reference: Rust wgpu-core InnerCommandEncoder (command/mod.rs:530-738)
// which holds a Vec<CommandBuffer> list, not just one.
//
// CoreCommandEncoder is thread-safe for concurrent access.
type CoreCommandEncoder struct {
	// raw is the HAL encoder wrapped for safe destruction.
	raw Snatchable[hal.CommandEncoder]

	// device is the parent device.
	device *Device

	// pass, cpass and cb are the encoder's render pass, compute pass and
	// command buffer. An encoder has one of each open at a time and is
	// single-use, so they live here and a frame allocates none of them.
	pass  CoreRenderPassEncoder
	cpass CoreComputePassEncoder
	cb    CoreCommandBuffer

	// status is the current encoder status (atomic for lock-free reads).
	status atomic.Int32

	// mu protects mutable state.
	mu sync.Mutex

	// mutable holds the mutable encoding state.
	mutable *CommandBufferMutable

	// error holds the error that caused the Error state.
	error error

	// label is the debug label for this encoder.
	label string

	// cbList accumulates command buffers from multiple open/close cycles.
	// All CBs in this list are submitted together in one HAL queue submit.
	// This enables inserting barrier CBs before/after render pass CBs.
	//
	// Reference: Rust wgpu-core InnerCommandEncoder.list (command/mod.rs:551)
	cbList []hal.CommandBuffer

	// cbListOpen is true when the HAL encoder is in the "recording" state
	// for a multi-CB pass (opened via OpenPass). When false, the encoder
	// is between passes and not recording.
	//
	// Reference: Rust wgpu-core InnerCommandEncoder.is_open (command/mod.rs:561)
	cbListOpen bool
}

// CreateCommandEncoder creates a new command encoder on this device.
//
// The encoder is created in the Recording state, ready to record commands.
//
// Parameters:
//   - label: Debug label for the encoder.
//
// Returns the encoder and nil on success.
// Returns nil and an error if the device is destroyed or HAL creation fails.
func (d *Device) CreateCommandEncoder(label string) (*CoreCommandEncoder, error) {
	// 1. Check device validity
	if err := d.checkValid(); err != nil {
		return nil, err
	}

	// 2. Acquire snatch guard for HAL access
	guard := d.snatchLock.Read()
	defer guard.Release()

	halDevice := d.raw.Get(guard)
	if halDevice == nil {
		return nil, ErrDeviceDestroyed
	}

	// 3. Create HAL command encoder
	halEncoder, err := (*halDevice).CreateCommandEncoder(&hal.CommandEncoderDescriptor{
		Label: label,
	})
	if err != nil {
		return nil, &CreateCommandEncoderError{
			Kind:     CreateCommandEncoderErrorHAL,
			Label:    label,
			HALError: err,
		}
	}

	// 4. Begin encoding
	if err := halEncoder.BeginEncoding(label); err != nil {
		return nil, &CreateCommandEncoderError{
			Kind:     CreateCommandEncoderErrorHAL,
			Label:    label,
			HALError: fmt.Errorf("failed to begin encoding: %w", err),
		}
	}

	// 5. Create core encoder
	enc := &CoreCommandEncoder{}
	enc.init(d, halEncoder, label)
	return enc, nil
}

// CreateCommandEncoderWithHAL creates a core command encoder using a pre-existing
// HAL encoder that is already in recording state (BeginEncoding already called).
// This is used by the public API encoder pool to reuse HAL encoders across frames,
// matching Rust wgpu-core's CommandAllocator pattern where encoders are acquired
// from a pool rather than created fresh each time.
//
// The caller is responsible for ensuring the HAL encoder is valid and recording.
func (d *Device) CreateCommandEncoderWithHAL(halEncoder hal.CommandEncoder, label string) (*CoreCommandEncoder, error) {
	enc := &CoreCommandEncoder{}
	if err := d.InitCommandEncoderWithHAL(enc, halEncoder, label); err != nil {
		return nil, err
	}
	return enc, nil
}

// InitCommandEncoderWithHAL is CreateCommandEncoderWithHAL into enc, which the
// caller owns and may reuse once the command buffer it produced is spent.
func (d *Device) InitCommandEncoderWithHAL(enc *CoreCommandEncoder, halEncoder hal.CommandEncoder, label string) error {
	if err := d.checkValid(); err != nil {
		return err
	}
	enc.init(d, halEncoder, label)
	return nil
}

// init readies e to record with halEncoder, which is recording the first
// command buffer. Whatever e did before is spent: its command buffer handed
// its scopes back, its lists were moved out.
func (e *CoreCommandEncoder) init(d *Device, halEncoder hal.CommandEncoder, label string) {
	if e.device != nil && e.device != d {
		e.device.encoders.remove(e)
	}
	e.raw.Set(halEncoder)
	e.device = d
	d.encoders.add(e)
	e.mutable = d.scopes.get()
	e.error = nil
	e.label = label
	// The last command buffer's HAL list is spent by now; keep its room.
	if spent := e.cb.halBuffers; spent != nil {
		clear(spent)
		e.cbList = spent[:0]
	} else {
		e.cbList = e.cbList[:0]
	}
	e.cbListOpen = true
	e.pass = CoreRenderPassEncoder{}
	e.cpass = CoreComputePassEncoder{}
	e.cb = CoreCommandBuffer{}
	e.status.Store(int32(CommandEncoderStatusRecording))

	trackResource(uintptr(unsafe.Pointer(e)), "CommandEncoder") //nolint:gosec // debug tracking uses pointer as unique ID
}

// TakeHALEncoder extracts the HAL command encoder from this core encoder,
// removing it from snatchable ownership. This is used by the encoder pool
// to reclaim the HAL encoder after Finish() for recycling. The HAL encoder
// is returned to the pool after GPU completion and ResetAll.
//
// Must be called after Finish() (encoder in Finished state). Returns nil if
// the encoder has already been snatched or was never set.
func (e *CoreCommandEncoder) TakeHALEncoder() hal.CommandEncoder {
	guard := e.device.snatchLock.Write()
	defer guard.Release()

	ptr := e.raw.Snatch(guard)
	if ptr == nil {
		return nil
	}
	return *ptr
}

// RawEncoder returns the underlying HAL command encoder for direct HAL access.
// Requires the device's snatch lock to be held. Returns nil if the encoder
// has been snatched or the device is destroyed.
func (e *CoreCommandEncoder) RawEncoder() hal.CommandEncoder {
	guard := e.device.snatchLock.Read()
	defer guard.Release()
	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		return nil
	}
	return *halEncoder
}

// Discard ends the encoder's recording without a command buffer, throwing
// away what it recorded.
func (e *CoreCommandEncoder) Discard() {
	e.device.encoders.remove(e)
	if raw := e.RawEncoder(); raw != nil {
		raw.DiscardEncoding()
	}
}

// abandon stops the encoder of a device being destroyed: its HAL encoder
// goes with the HAL device, so it and an open pass record nothing more,
// and Finish reports the device destroyed.
func (e *CoreCommandEncoder) abandon() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.raw.Snatch(ExclusiveSnatchGuard{})
	e.pass.ended, e.pass.raw = true, nil
	e.cpass.ended, e.cpass.raw = true, nil
	e.setError(ErrDeviceDestroyed)
}

// liveEncoders are the command encoders recording on a device.
type liveEncoders struct {
	mu sync.Mutex
	m  map[*CoreCommandEncoder]struct{}
}

func (l *liveEncoders) add(e *CoreCommandEncoder) {
	if l == nil {
		return
	}
	l.mu.Lock()
	if l.m == nil {
		l.m = make(map[*CoreCommandEncoder]struct{})
	}
	l.m[e] = struct{}{}
	l.mu.Unlock()
}

func (l *liveEncoders) remove(e *CoreCommandEncoder) {
	if l == nil {
		return
	}
	l.mu.Lock()
	delete(l.m, e)
	l.mu.Unlock()
}

// abandonAll abandons every encoder still recording.
func (l *liveEncoders) abandonAll() {
	if l == nil {
		return
	}
	l.mu.Lock()
	live := l.m
	l.m = nil
	l.mu.Unlock()
	for e := range live {
		e.abandon()
	}
}

// Status returns the current encoder status.
func (e *CoreCommandEncoder) Status() CommandEncoderStatus {
	return CommandEncoderStatus(e.status.Load())
}

// Label returns the encoder's debug label.
func (e *CoreCommandEncoder) Label() string {
	return e.label
}

// Device returns the parent device.
func (e *CoreCommandEncoder) Device() *Device {
	return e.device
}

// Mutable returns the mutable encoding state. This is exposed for testing
// and for advanced usage where the buffer/texture scopes need to be accessed
// directly (e.g., by the public API layer for bind group tracking).
func (e *CoreCommandEncoder) Mutable() *CommandBufferMutable {
	return e.mutable
}

// TextureScope returns the command buffer's texture tracker (first and last
// usage of every texture, see CommandBufferMutable.textureTracker). Returns
// nil if mutable is nil.
func (m *CommandBufferMutable) TextureScope() *track.EncoderTextureTracker {
	if m == nil {
		return nil
	}
	return m.textureTracker
}

// BufferScope returns the command buffer's buffer tracker. Returns nil if
// mutable is nil.
func (m *CommandBufferMutable) BufferScope() *track.EncoderBufferTracker {
	if m == nil {
		return nil
	}
	return m.bufferTracker
}

// Error returns the error that caused the Error state, or nil.
func (e *CoreCommandEncoder) Error() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.error
}

// BeginRenderPass begins a render pass.
//
// The encoder must be in the Recording state.
// After this call, the encoder transitions to the Locked state.
//
// Returns the render pass encoder and nil on success.
// Returns nil and an error if the encoder is not in Recording state.
func (e *CoreCommandEncoder) BeginRenderPass(desc *RenderPassDescriptor) (*CoreRenderPassEncoder, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.Status() != CommandEncoderStatusRecording {
		return nil, e.statusError("begin render pass")
	}

	// Validate descriptor
	if desc == nil {
		err := fmt.Errorf("render pass descriptor is nil")
		e.setError(err)
		return nil, err
	}

	// Validate MRT rules: attachment count, sample count consistency,
	// dimension consistency. This produces the RenderPassContext used
	// for draw-time pipeline compatibility checks.
	pass := &e.pass
	*pass = CoreRenderPassEncoder{encoder: e, device: e.device}
	if valErr := validateRenderPassDescriptor(desc, e.device.Limits, e.device.Features, &pass.passCtx); valErr != nil {
		e.setError(valErr)
		return nil, valErr
	}

	// Convert to HAL descriptor. The backends read it until the pass ends, so
	// it lives that long and then goes back to the pool.
	pass.halDesc = halPassPool.Get().(*halPassDescriptor)
	halDesc := pass.halDesc.fill(e, desc)

	guard := e.device.snatchLock.Read()
	defer guard.Release()

	// The pass is one usage scope. Its attachments are validated against
	// each other before anything is recorded: the same texture as colour and
	// depth target is a conflict, while what earlier commands of this encoder
	// did with a texture belongs to other scopes and is transitioned instead.
	// This runs before the HAL pass begins: a pass begun and then dropped is
	// never ended, and Metal leaks its render pass descriptor together with
	// every attachment texture it holds on to.
	//
	// Reference: wgpu-core command/render.rs RenderPassInfo::start
	// (scope.textures.merge_single for each attachment)
	if conflictErr := e.beginPassScope(desc, guard); conflictErr != nil {
		pass.halDesc.release()
		pass.halDesc = nil
		e.setError(conflictErr)
		return nil, conflictErr
	}

	// Get HAL encoder
	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		err := ErrResourceDestroyed
		e.setError(err)
		return nil, err
	}

	// A pass that may need barriers ahead of it, because this encoder used
	// a texture or buffer before, is recorded in a HAL command buffer of its
	// own: which barriers is known only once its bind groups are, at End,
	// and they go in a command buffer inserted before it then (see
	// recordPassTransitions). An encoder's first pass needs none: every
	// usage in it is a first usage, which the submit-time barrier covers.
	if m := e.mutable; m != nil && (!m.textureTracker.IsEmpty() || !m.bufferTracker.IsEmpty()) {
		if err := e.closeIfOpenLocked(*halEncoder); err != nil {
			e.setError(err)
			return nil, err
		}
		if err := e.openLocked(*halEncoder, e.label); err != nil {
			e.setError(err)
			return nil, err
		}
		m.pass.split = true
	} else if err := e.ensureOpenLocked(*halEncoder); err != nil {
		e.setError(err)
		return nil, err
	}

	// Begin HAL render pass
	halPass := (*halEncoder).BeginRenderPass(halDesc)

	// Transition to locked state
	e.status.Store(int32(CommandEncoderStatusLocked))

	pass.raw = halPass
	pass.passContext = &pass.passCtx
	if ds := desc.DepthStencilAttachment; ds != nil && ds.View != nil {
		pass.depthReadOnly, pass.stencilReadOnly = ds.DepthReadOnly, ds.StencilReadOnly
	}
	pass.width, pass.height = passExtent(desc)
	if e.mutable != nil {
		pass.scope = &e.mutable.pass
	}
	e.mutable.activePass = pass

	return pass, nil
}

// EndRenderPass ends the current render pass.
//
// The encoder must be in the Locked state with an active render pass.
// After this call, the encoder transitions back to the Recording state.
//
// This is called internally by CoreRenderPassEncoder.End().
func (e *CoreCommandEncoder) EndRenderPass(pass *CoreRenderPassEncoder) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.Status() != CommandEncoderStatusLocked {
		return e.statusError("end render pass")
	}
	if e.mutable.activePass != pass {
		return fmt.Errorf("wrong pass being ended")
	}

	// End HAL render pass (already called by CoreRenderPassEncoder.End())

	// The pass's usage scope reaches the command buffer's trackers now, and
	// the transitions into it are recorded ahead of the pass.
	err := e.recordPassTransitions()

	// Return to recording state
	e.status.Store(int32(CommandEncoderStatusRecording))
	e.mutable.activePass = nil

	if err != nil {
		e.setError(err)
	}
	return err
}

// recordPassTransitions applies the render pass's usage scope to the command
// buffer's trackers and records the transitions into the pass: from what
// earlier commands of this encoder left its textures and buffers in (a copy
// into a texture it samples, a render into one it samples, a copy into its
// vertex buffer) to what the pass uses them as. Barriers cannot go inside a
// render pass, and which ones the pass needs is only known now, so a pass
// recorded in a HAL command buffer of its own (BeginRenderPass) gets them in
// one inserted ahead of it, Rust wgpu-core's transit command buffer. Nothing
// is open afterwards: the next command that records opens a fresh one
// (ensureOpenLocked), so an encoder that ends with the pass submits no empty
// command buffer after it. Caller holds e.mu.
//
// Reference: wgpu-core command/render.rs (insert_barriers_from_scope on the
// transit encoder, then close_and_swap; CommandEncoder::open is lazy)
func (e *CoreCommandEncoder) recordPassTransitions() error {
	m := e.mutable
	if m == nil {
		return nil
	}
	scope := &m.pass
	scope.zeroTransitionsOf(m.textureTracker, scope.zeros)
	scope.texTransitions = m.textureTracker.SetFromScopeInto(scope.textures, scope.texTransitions[:0])
	scope.bufTransitions = m.bufferTracker.SetFromScopeInto(scope.buffers, scope.bufTransitions[:0])
	defer scope.clearScratch()
	needed := len(scope.texTransitions) > 0 || len(scope.bufTransitions) > 0 || len(scope.drawValidations) > 0 || len(scope.zeros) > 0

	guard := e.device.snatchLock.Read()
	defer guard.Release()
	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		return ErrResourceDestroyed
	}

	if !scope.split && !needed {
		// An encoder's first pass: every usage in it is a first usage, which
		// the submit-time barriers take care of.
		return nil
	}

	// Close the pass's command buffer and put the barriers in one ahead of
	// it. Recording goes on in a fresh one once something needs it.
	if err := e.closeIfOpenLocked(*halEncoder); err != nil {
		return err
	}
	if !needed {
		return nil
	}
	if err := e.openLocked(*halEncoder, "(wgpu internal) pass transit"); err != nil {
		return err
	}
	scope.recordZeros(*halEncoder, scope.zeros)
	scope.recordBarriers(*halEncoder)
	scope.recordDrawValidations(*halEncoder)
	return e.closeAndSwapLocked(*halEncoder)
}

// BeginComputePass begins a compute pass.
//
// The encoder must be in the Recording state.
// After this call, the encoder transitions to the Locked state.
//
// Returns the compute pass encoder and nil on success.
// Returns nil and an error if the encoder is not in Recording state.
func (e *CoreCommandEncoder) BeginComputePass(desc *CoreComputePassDescriptor) (*CoreComputePassEncoder, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.Status() != CommandEncoderStatusRecording {
		return nil, e.statusError("begin compute pass")
	}

	// Convert to HAL descriptor
	halDesc := &hal.ComputePassDescriptor{}
	if desc != nil {
		halDesc.Label = desc.Label
	}

	// Get HAL encoder
	guard := e.device.snatchLock.Read()
	defer guard.Release()

	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		err := ErrResourceDestroyed
		e.setError(err)
		return nil, err
	}

	// Nothing may be open after a split render pass; the compute pass needs
	// a command buffer to go in.
	if err := e.ensureOpenLocked(*halEncoder); err != nil {
		e.setError(err)
		return nil, err
	}
	if desc != nil && desc.TimestampWrites != nil && desc.TimestampWrites.QuerySet != nil {
		if raw := desc.TimestampWrites.QuerySet.Raw(guard); raw != nil {
			halDesc.TimestampWrites = &hal.ComputePassTimestampWrites{
				QuerySet:                  raw,
				BeginningOfPassWriteIndex: desc.TimestampWrites.BeginningOfPassWriteIndex,
				EndOfPassWriteIndex:       desc.TimestampWrites.EndOfPassWriteIndex,
			}
		}
	}

	// Each dispatch of the pass is a usage scope of its own, see
	// CoreComputePassEncoder.scope; it starts empty.
	pass := &e.cpass
	*pass = CoreComputePassEncoder{
		encoder: e,
		device:  e.device,
	}
	if e.mutable != nil {
		e.mutable.pass.clear()
		pass.scope = &e.mutable.pass
	}

	// Begin HAL compute pass
	pass.raw = (*halEncoder).BeginComputePass(halDesc)

	// Transition to locked state
	e.status.Store(int32(CommandEncoderStatusLocked))
	e.mutable.activePass = pass

	return pass, nil
}

// EndComputePass ends the current compute pass.
//
// The encoder must be in the Locked state with an active compute pass.
// After this call, the encoder transitions back to the Recording state.
//
// This is called internally by CoreComputePassEncoder.End().
func (e *CoreCommandEncoder) EndComputePass(pass *CoreComputePassEncoder) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.Status() != CommandEncoderStatusLocked {
		return e.statusError("end compute pass")
	}
	if e.mutable.activePass != pass {
		return fmt.Errorf("wrong pass being ended")
	}

	// End HAL compute pass (already called by CoreComputePassEncoder.End())

	// Return to recording state
	e.status.Store(int32(CommandEncoderStatusRecording))
	e.mutable.activePass = nil

	return nil
}

// Finish completes encoding and returns a command buffer.
//
// The encoder must be in the Recording state (not in a pass).
// After this call, the encoder transitions to the Finished state.
//
// The open recording is closed, and every HAL command buffer the encoder
// made, the one it started with and those its render passes and OpenPass/
// CloseCB/CloseAndSwap/CloseAndPushFront split off, goes in the returned
// CoreCommandBuffer, in order.
//
// Returns the command buffer and nil on success.
// Returns nil and an error if the encoder is not in Recording state.
func (e *CoreCommandEncoder) Finish() (*CoreCommandBuffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.device.encoders.remove(e)

	if e.Status() != CommandEncoderStatusRecording {
		return nil, e.statusError("finish")
	}

	// Get HAL encoder
	guard := e.device.snatchLock.Read()
	defer guard.Release()

	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		return nil, ErrResourceDestroyed
	}

	if err := e.closeIfOpenLocked(*halEncoder); err != nil {
		e.setError(err)
		return nil, err
	}

	e.status.Store(int32(CommandEncoderStatusFinished))
	untrackResource(uintptr(unsafe.Pointer(e))) //nolint:gosec // debug tracking uses pointer as unique ID

	cb := &e.cb
	*cb = CoreCommandBuffer{
		device:     e.device,
		mutable:    e.mutable,
		label:      e.label,
		halBuffers: e.cbList,
	}
	// The scopes belong to the command buffer now, which hands them back to
	// the device once they are spent (ReleaseScopes), and so does the HAL
	// list, which init takes back for the next recording.
	e.mutable = nil
	// Set raw to the first CB for backward compatibility with Raw().
	if len(e.cbList) > 0 {
		cb.raw = e.cbList[0]
	}
	e.cbList = nil
	return cb, nil
}

// =============================================================================
// Multi-CB Encoder Methods (Rust InnerCommandEncoder parity)
// =============================================================================

// OpenPass starts recording a new command buffer for a render or compute pass.
// If a recording is already open, it is closed first (CloseIfOpen pattern).
// The label is passed to HAL BeginEncoding.
//
// This enables the multi-CB pattern where barrier CBs are inserted before/after
// render pass CBs within the same encoder, all submitted together.
//
// The encoder must be in the Recording state (not in a pass or finished).
//
// Reference: Rust wgpu-core InnerCommandEncoder::open_pass (command/mod.rs:711-723)
func (e *CoreCommandEncoder) OpenPass(label string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.Status() != CommandEncoderStatusRecording {
		return e.statusError("open pass")
	}

	guard := e.device.snatchLock.Read()
	defer guard.Release()

	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		return ErrResourceDestroyed
	}

	// Close any open recording before starting a new one.
	if err := e.closeIfOpenLocked(*halEncoder); err != nil {
		e.setError(err)
		return err
	}

	// Begin recording a new command buffer.
	if err := e.openLocked(*halEncoder, label); err != nil {
		e.setError(err)
		return err
	}

	return nil
}

// openLocked begins recording a new HAL command buffer. Caller holds e.mu
// and has closed the previous recording.
func (e *CoreCommandEncoder) openLocked(halEncoder hal.CommandEncoder, label string) error {
	if err := halEncoder.BeginEncoding(label); err != nil {
		return err
	}
	e.cbListOpen = true
	return nil
}

// ensureOpenLocked opens a HAL command buffer when none is: a split render
// pass closes its own on End and leaves it to the next command that records
// to open a fresh one, so an encoder that ends with the pass has no empty
// command buffer after it. Caller holds e.mu.
//
// Reference: wgpu-core command/mod.rs CommandEncoder::open
func (e *CoreCommandEncoder) ensureOpenLocked(halEncoder hal.CommandEncoder) error {
	if e.cbListOpen {
		return nil
	}
	return e.openLocked(halEncoder, e.label)
}

// RecordingEncoder returns the HAL command encoder with a command buffer
// open to record on, for a command the public layer records on the HAL
// itself (a copy, a barrier, a buffer clear). RawEncoder only returns the
// encoder, which may have nothing open after a split render pass. Returns
// nil if the encoder has been snatched, or sets the encoder's error and
// returns nil if the HAL cannot begin a command buffer.
func (e *CoreCommandEncoder) RecordingEncoder() hal.CommandEncoder {
	e.mu.Lock()
	defer e.mu.Unlock()
	guard := e.device.snatchLock.Read()
	defer guard.Release()
	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		return nil
	}
	if err := e.ensureOpenLocked(*halEncoder); err != nil {
		e.setError(err)
		return nil
	}
	return *halEncoder
}

// CloseCB ends the current command buffer recording and pushes it to the
// end of the CB list. The HAL encoder transitions to the closed state.
//
// The encoder must have an open recording (cbListOpen == true).
//
// Reference: Rust wgpu-core InnerCommandEncoder::close (command/mod.rs:641-650)
func (e *CoreCommandEncoder) CloseCB() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.cbListOpen {
		return fmt.Errorf("core: CloseCB: no command buffer is currently open")
	}

	guard := e.device.snatchLock.Read()
	defer guard.Release()

	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		return ErrResourceDestroyed
	}

	halCmdBuffer, err := (*halEncoder).EndEncoding()
	if err != nil {
		e.setError(err)
		return err
	}

	e.cbList = append(e.cbList, halCmdBuffer)
	e.cbListOpen = false
	return nil
}

// CloseAndSwap ends the current CB and inserts it BEFORE the last element
// in the CB list. This is used for inserting barrier CBs before render pass
// CBs: the render pass CB is already at the end of the list, and the barrier
// CB needs to go right before it.
//
// The encoder must have an open recording AND at least one CB already in the list.
//
// Reference: Rust wgpu-core InnerCommandEncoder::close_and_swap (command/mod.rs:599-607)
func (e *CoreCommandEncoder) CloseAndSwap() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.cbListOpen {
		return fmt.Errorf("core: CloseAndSwap: no command buffer is currently open")
	}
	if len(e.cbList) == 0 {
		return fmt.Errorf("core: CloseAndSwap: CB list is empty, need at least one CB to swap before")
	}

	guard := e.device.snatchLock.Read()
	defer guard.Release()

	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		return ErrResourceDestroyed
	}

	if err := e.closeAndSwapLocked(*halEncoder); err != nil {
		e.setError(err)
		return err
	}
	return nil
}

// closeAndSwapLocked is CloseAndSwap with e.mu held and the checks done.
func (e *CoreCommandEncoder) closeAndSwapLocked(halEncoder hal.CommandEncoder) error {
	halCmdBuffer, err := halEncoder.EndEncoding()
	if err != nil {
		return err
	}

	// Insert before the last element: [A, B, C] -> [A, B, NEW, C]
	// Matches Rust: self.list.insert(self.list.len() - 1, new)
	insertPos := len(e.cbList) - 1
	e.cbList = append(e.cbList, nil)                   // grow by one
	copy(e.cbList[insertPos+1:], e.cbList[insertPos:]) // shift last element right
	e.cbList[insertPos] = halCmdBuffer

	e.cbListOpen = false
	return nil
}

// CloseAndPushFront ends the current CB and inserts it at position 0 in the
// CB list. This is used for submit-time device tracker barriers that must
// execute before all other commands in the submission.
//
// The encoder must have an open recording.
//
// Reference: Rust wgpu-core InnerCommandEncoder::close_and_push_front (command/mod.rs:620-628)
func (e *CoreCommandEncoder) CloseAndPushFront() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.cbListOpen {
		return fmt.Errorf("core: CloseAndPushFront: no command buffer is currently open")
	}

	guard := e.device.snatchLock.Read()
	defer guard.Release()

	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		return ErrResourceDestroyed
	}

	halCmdBuffer, err := (*halEncoder).EndEncoding()
	if err != nil {
		e.setError(err)
		return err
	}

	// Insert at position 0: [A, B] -> [NEW, A, B]
	// Matches Rust: self.list.insert(0, new)
	e.cbList = append([]hal.CommandBuffer{halCmdBuffer}, e.cbList...)

	e.cbListOpen = false
	return nil
}

// CloseIfOpen closes the current CB if one is being recorded, appending it
// to the end of the CB list. If no recording is open, this is a no-op.
//
// This is safe to call at any time and is used by Finish() to finalize
// any open recording before returning the accumulated CB list.
//
// Reference: Rust wgpu-core InnerCommandEncoder::close_if_open (command/mod.rs:662-671)
func (e *CoreCommandEncoder) CloseIfOpen() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	guard := e.device.snatchLock.Read()
	defer guard.Release()

	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		if e.cbListOpen {
			return ErrResourceDestroyed
		}
		return nil
	}

	return e.closeIfOpenLocked(*halEncoder)
}

// closeIfOpenLocked is the lock-free inner implementation of CloseIfOpen.
// Caller must hold e.mu and provide a valid HAL encoder.
func (e *CoreCommandEncoder) closeIfOpenLocked(halEncoder hal.CommandEncoder) error {
	if !e.cbListOpen {
		return nil
	}

	halCmdBuffer, err := halEncoder.EndEncoding()
	if err != nil {
		return err
	}

	e.cbList = append(e.cbList, halCmdBuffer)
	e.cbListOpen = false
	return nil
}

// CBListLen returns the number of command buffers accumulated so far.
// This is useful for testing and debugging.
func (e *CoreCommandEncoder) CBListLen() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.cbList)
}

// ClosedBuffers returns the HAL command buffers the encoder has closed so
// far, for a caller that discards the recording and resets the HAL encoder
// with them. Nil once Finish has handed them to the command buffer.
func (e *CoreCommandEncoder) ClosedBuffers() []hal.CommandBuffer {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.cbList
}

// IsCBOpen returns whether a command buffer is currently being recorded
// in the multi-CB path. This is useful for testing.
func (e *CoreCommandEncoder) IsCBOpen() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.cbListOpen
}

// MarkConsumed marks the encoder as consumed after submission.
//
// This is called by the queue after successful submission.
func (e *CoreCommandEncoder) MarkConsumed() {
	e.status.Store(int32(CommandEncoderStatusConsumed))
}

// SetError records a deferred error on this encoder.
//
// The error transitions the encoder to the Error state and will be returned
// by Finish(). This implements the WebGPU deferred error pattern where
// encoding-phase errors are collected and surfaced at Finish() time.
func (e *CoreCommandEncoder) SetError(err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.setError(err)
}

// setError transitions to error state, keeping the first error: the one
// that made the encoder invalid, which Finish reports. Caller must hold e.mu.
func (e *CoreCommandEncoder) setError(err error) {
	if e.error == nil {
		e.error = err
	}
	e.status.Store(int32(CommandEncoderStatusError))
}

// statusError returns an error for invalid status.
// When the encoder is in the Error state, the deferred error is attached
// as the Cause so that errors.Is/errors.As work through the chain.
func (e *CoreCommandEncoder) statusError(operation string) error {
	ese := &EncoderStateError{
		Operation: operation,
		Status:    e.Status(),
	}
	if e.Status() == CommandEncoderStatusError {
		ese.Cause = e.error
	}
	return ese
}

// halPassDescriptor is the HAL descriptor of one render pass with room for
// its attachments, kept in a pool between passes.
type halPassDescriptor struct {
	desc   hal.RenderPassDescriptor
	color  []hal.RenderPassColorAttachment
	depth  hal.RenderPassDepthStencilAttachment
	stamps hal.RenderPassTimestampWrites
}

var halPassPool = sync.Pool{New: func() any { return new(halPassDescriptor) }}

// fill converts desc into the HAL descriptor.
func (h *halPassDescriptor) fill(e *CoreCommandEncoder, desc *RenderPassDescriptor) *hal.RenderPassDescriptor {
	h.desc = hal.RenderPassDescriptor{Label: desc.Label}

	h.color = h.color[:0]
	for _, ca := range desc.ColorAttachments {
		halCA := hal.RenderPassColorAttachment{
			LoadOp:     ca.LoadOp,
			StoreOp:    ca.StoreOp,
			ClearValue: ca.ClearValue,
		}
		if ca.View != nil {
			halCA.View = ca.View.HAL
		}
		if ca.ResolveTarget != nil {
			halCA.ResolveTarget = ca.ResolveTarget.HAL
		}
		h.color = append(h.color, halCA)
	}
	h.desc.ColorAttachments = h.color

	if ds := desc.DepthStencilAttachment; ds != nil {
		h.depth = hal.RenderPassDepthStencilAttachment{
			DepthLoadOp:       ds.DepthLoadOp,
			DepthStoreOp:      ds.DepthStoreOp,
			DepthClearValue:   ds.DepthClearValue,
			DepthReadOnly:     ds.DepthReadOnly,
			StencilLoadOp:     ds.StencilLoadOp,
			StencilStoreOp:    ds.StencilStoreOp,
			StencilClearValue: ds.StencilClearValue,
			StencilReadOnly:   ds.StencilReadOnly,
		}
		// A read-only channel, and one the format lacks, takes no ops; the
		// backends load and keep it, as wgpu-core resolves both to
		// ResolvedPassChannel::ReadOnly.
		if ds.View != nil && ds.View.Parent != nil {
			f := ds.View.ViewFormat()
			h.depth.DepthReadOnly = ds.DepthReadOnly || !f.HasDepth()
			h.depth.StencilReadOnly = ds.StencilReadOnly || !f.HasStencil()
		}
		if h.depth.DepthReadOnly {
			h.depth.DepthLoadOp, h.depth.DepthStoreOp = gputypes.LoadOpLoad, gputypes.StoreOpStore
		}
		if h.depth.StencilReadOnly {
			h.depth.StencilLoadOp, h.depth.StencilStoreOp = gputypes.LoadOpLoad, gputypes.StoreOpStore
		}
		if ds.View != nil {
			h.depth.View = ds.View.HAL
		}
		h.desc.DepthStencilAttachment = &h.depth
	}

	if writes := desc.TimestampWrites; writes != nil && writes.QuerySet != nil {
		guard := e.device.snatchLock.Read()
		raw := writes.QuerySet.Raw(guard)
		guard.Release()
		if raw != nil {
			h.stamps = hal.RenderPassTimestampWrites{
				QuerySet:                  raw,
				BeginningOfPassWriteIndex: writes.BeginningOfPassWriteIndex,
				EndOfPassWriteIndex:       writes.EndOfPassWriteIndex,
			}
			h.desc.TimestampWrites = &h.stamps
		}
	}

	return &h.desc
}

// release puts h back in the pool, holding on to nothing.
func (h *halPassDescriptor) release() {
	clear(h.color)
	h.depth = hal.RenderPassDepthStencilAttachment{}
	h.stamps = hal.RenderPassTimestampWrites{}
	h.desc = hal.RenderPassDescriptor{}
	halPassPool.Put(h)
}

// beginPassScope starts the usage scope of a render pass with its
// attachments: colour attachments and resolve targets as COLOR_TARGET, the
// depth/stencil attachment as DEPTH_STENCIL_WRITE (or _READ when both
// aspects are read-only). The textures its bind groups bind join the scope
// as they are bound (RecordPassTextureUsage), its vertex and index buffers
// as they are set.
//
// A render pass is ONE usage scope: incompatible usages of the same texture
// inside it (the same texture as colour and depth/stencil attachment) are a
// validation error. Usages by earlier commands of the encoder are other
// usage scopes and are not checked here.
func (e *CoreCommandEncoder) beginPassScope(desc *RenderPassDescriptor, guard SnatchGuard) error {
	if e.mutable == nil {
		return nil
	}
	scope := &e.mutable.pass
	scope.clear()
	for i := range desc.ColorAttachments {
		ca := &desc.ColorAttachments[i]
		if err := scope.addView(ca.View, track.TextureUsesColorTarget, guard); err != nil {
			return fmt.Errorf("color attachment %d: %w", i, err)
		}
		if err := scope.addView(ca.ResolveTarget, track.TextureUsesColorTarget, guard); err != nil {
			return fmt.Errorf("resolve target %d: %w", i, err)
		}
	}
	if ds := desc.DepthStencilAttachment; ds != nil {
		if err := scope.addView(ds.View, depthStencilUsage(ds), guard); err != nil {
			return fmt.Errorf("depth/stencil attachment: %w", err)
		}
	}
	return nil
}

// depthStencilUsage returns the usage a depth/stencil attachment makes of
// its texture, aspect by aspect: DEPTH_STENCIL_READ for a read-only one,
// DEPTH_STENCIL_WRITE otherwise. The usage of a format with one aspect is
// that aspect's. A view without a parent is taken to have both aspects,
// used together: written unless both are read-only.
//
// Reference: Dawn CommandEncoder.cpp BeginRenderPass, TextureRangeUsedAs per
// aspect of a depth/stencil attachment with one aspect read-only
func depthStencilUsage(ds *RenderPassDepthStencilAttachment) track.TextureUses {
	usage := func(readOnly bool) track.TextureUses {
		if readOnly {
			return track.TextureUsesDepthStencilRead
		}
		return track.TextureUsesDepthStencilWrite
	}
	if ds.View == nil || ds.View.Parent == nil {
		return usage(ds.DepthReadOnly && ds.StencilReadOnly)
	}
	f := ds.View.Parent.Format()
	depth, stencil := hasDepthAspect(f), hasStencilAspect(f)
	switch {
	case depth && stencil:
		return track.DepthStencilUses(usage(ds.DepthReadOnly), usage(ds.StencilReadOnly))
	case stencil:
		return usage(ds.StencilReadOnly)
	}
	return usage(ds.DepthReadOnly)
}

// RecordPassTextureUsage records that the usage scope being recorded uses
// the subresources sel selects of the texture with usage, as a bind group
// binds a view of them: sampled, or as a storage texture. In a render pass
// the scope is the pass; in a compute pass it is the next dispatch. Within
// the scope the usage has to agree with the other uses of those
// subresources in it (an attachment that is also sampled is a conflict);
// what earlier commands of the encoder did with them is transitioned
// instead. halTexture is the handle the barrier is recorded against.
//
// A render pass cannot take barriers, so its transitions are recorded ahead
// of it once it ends (see recordPassTransitions). A compute pass can, so
// each dispatch records its own right before it (see
// CoreComputePassEncoder.Dispatch).
//
// Textures without valid tracking data are skipped, and so is a call outside
// any pass: nothing binds a texture there.
//
// Reference: wgpu-core command/render.rs merge_bind_group (the pass scope),
// command/compute.rs State::flush_states (the dispatch scope)
func (e *CoreCommandEncoder) RecordPassTextureUsage(texture *Texture, halTexture hal.Texture, sel track.TextureSelector, usage track.TextureUses) error {
	if e.mutable == nil || texture == nil {
		return nil
	}
	td := texture.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	m := e.mutable
	if m.activePass == nil {
		return nil
	}
	return m.pass.addTexture(td.Index(), texture.Subresources(), sel, halTexture, usage)
}

// RecordPassBufferUsage is RecordPassTextureUsage for a buffer a bind group
// binds: uniform, or storage read-only or read-write. A storage buffer a
// dispatch writes and the next one reads, or a render pass then takes as a
// vertex buffer, is transitioned between the two scopes; inside one scope a
// buffer written and read is a conflict. halBuffer is the handle the barrier
// is recorded against.
//
// Reference: wgpu-core command/render.rs merge_bind_group,
// command/compute.rs State::flush_states (scope.buffers)
func (e *CoreCommandEncoder) RecordPassBufferUsage(buffer *Buffer, halBuffer hal.Buffer, usage track.BufferUses) error {
	if e.mutable == nil || buffer == nil {
		return nil
	}
	td := buffer.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return nil
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	m := e.mutable
	if m.activePass == nil {
		return nil
	}
	return m.pass.addBuffer(td.Index(), halBuffer, usage)
}

// RecordTextureTransition records a texture barrier the caller has already
// encoded explicitly on the HAL encoder (public CommandEncoder.
// TransitionTextures) on the subresources rng takes in. The command buffer
// tracker adopts newUsage as their current state, so later commands
// transition from the state the explicit barrier left them in instead of
// repeating that barrier. For subresources the command buffer has not used
// before, oldUsage is the state it expects on entry (its start state).
//
// Textures without valid tracking data are skipped.
//
// Reference: wgpu-core command/transition_resources.rs (explicit transitions
// go through the command buffer tracker)
func (e *CoreCommandEncoder) RecordTextureTransition(texture *Texture, rng hal.TextureRange, oldUsage, newUsage gputypes.TextureUsage) {
	if e.mutable == nil || e.mutable.textureTracker == nil || texture == nil {
		return
	}
	td := texture.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return
	}
	depthStencil := texture.Format().IsDepthStencil()
	full := texture.Subresources()
	e.mutable.textureTracker.RecordTransition(td.Index(), full, track.SelectorFromRange(full, rng),
		track.TextureUsesFromUsage(oldUsage, depthStencil),
		track.TextureUsesFromUsage(newUsage, depthStencil))
}

// =============================================================================
// Core Render Pass Encoder
// =============================================================================

// RenderPassDescriptor describes a render pass.
type RenderPassDescriptor struct {
	// Label is an optional debug name.
	Label string

	// ColorAttachments are the color render targets.
	ColorAttachments []RenderPassColorAttachment

	// DepthStencilAttachment is the depth/stencil target (optional).
	DepthStencilAttachment *RenderPassDepthStencilAttachment

	// TimestampWrites are timestamp queries to write at pass boundaries (optional).
	TimestampWrites *RenderPassTimestampWrites
}

// RenderPassColorAttachment describes a color attachment.
type RenderPassColorAttachment struct {
	// View is the texture view to render to.
	View *TextureView

	// ResolveTarget is the MSAA resolve target (optional).
	ResolveTarget *TextureView

	// LoadOp specifies what to do at pass start.
	LoadOp gputypes.LoadOp

	// StoreOp specifies what to do at pass end.
	StoreOp gputypes.StoreOp

	// ClearValue is the clear color (used if LoadOp is Clear).
	ClearValue gputypes.Color
}

// RenderPassDepthStencilAttachment describes a depth/stencil attachment.
type RenderPassDepthStencilAttachment struct {
	// View is the texture view to use.
	View *TextureView

	// DepthLoadOp specifies what to do with depth at pass start.
	DepthLoadOp gputypes.LoadOp

	// DepthStoreOp specifies what to do with depth at pass end.
	DepthStoreOp gputypes.StoreOp

	// DepthClearValue is the depth clear value.
	DepthClearValue float32

	// DepthReadOnly makes the depth aspect read-only.
	DepthReadOnly bool

	// StencilLoadOp specifies what to do with stencil at pass start.
	StencilLoadOp gputypes.LoadOp

	// StencilStoreOp specifies what to do with stencil at pass end.
	StencilStoreOp gputypes.StoreOp

	// StencilClearValue is the stencil clear value.
	StencilClearValue uint32

	// StencilReadOnly makes the stencil aspect read-only.
	StencilReadOnly bool
}

// RenderPassTimestampWrites describes timestamp queries at render pass boundaries.
type RenderPassTimestampWrites struct {
	// QuerySet is the timestamp query set that receives the writes.
	QuerySet *QuerySet

	// BeginningOfPassWriteIndex is the query index to write at pass start.
	// If nil, no timestamp is written at the beginning of the pass.
	BeginningOfPassWriteIndex *uint32

	// EndOfPassWriteIndex is the query index to write at pass end.
	// If nil, no timestamp is written at the end of the pass.
	EndOfPassWriteIndex *uint32
}

// CoreRenderPassEncoder records render commands within a pass.
//
// This is the HAL-integrated render pass encoder that bridges core
// render commands to HAL render pass encoder.
type CoreRenderPassEncoder struct {
	// raw is the HAL render pass encoder.
	raw hal.RenderPassEncoder

	// encoder is the parent command encoder.
	encoder *CoreCommandEncoder

	// device is the parent device.
	device *Device

	// pipeline is the currently bound render pipeline.
	pipeline *RenderPipeline

	// ended indicates whether End() has been called.
	ended bool

	// passContext stores the resolved attachment configuration of this
	// render pass. Used for draw-time validation: when SetPipeline is
	// called, the pipeline's passContext is checked against this to
	// ensure format and sample count compatibility.
	//
	// Matches Rust wgpu-core command/render.rs RenderPassInfo.context.
	passContext *RenderPassContext

	// depthReadOnly and stencilReadOnly are the depth/stencil attachment's
	// read-only aspects, which no pipeline of the pass may write.
	depthReadOnly, stencilReadOnly bool

	// width and height are the render extent, which holds the scissor rect.
	width, height uint32

	// passCtx holds passContext, and halDesc the HAL descriptor the backend
	// reads until End, so a pass allocates neither.
	passCtx RenderPassContext
	halDesc *halPassDescriptor

	// scope is this render pass's usage scope, the encoder's: its
	// attachments, the textures its bind groups bind, its vertex and index
	// buffers. Conflicts are validated inside the pass only; on End the
	// scope is applied to the command buffer's trackers.
	//
	// Reference: wgpu-core command/render.rs State.scope (UsageScope per pass)
	scope *passScope
}

// RawPass returns the underlying HAL render pass encoder for direct HAL access.
func (p *CoreRenderPassEncoder) RawPass() hal.RenderPassEncoder {
	return p.raw
}

// recordBufferUsage records a buffer usage in this render pass's usage
// scope, with the HAL buffer the transition into the pass is recorded
// against. Buffers without valid tracking data are skipped. Returns an error
// if the buffer already has an incompatible usage in this pass.
//
// Reference: wgpu-core command/render.rs (scope.buffers.merge_single)
func (p *CoreRenderPassEncoder) recordBufferUsage(buffer *Buffer, halBuffer hal.Buffer, usage track.BufferUses) error {
	if p.scope == nil || buffer == nil {
		return nil
	}
	td := buffer.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return nil
	}
	return p.scope.addBuffer(td.Index(), halBuffer, usage)
}

// SetPipeline sets the render pipeline.
//
// Draw-time validation (WebGPU spec §10.3): the pipeline's fragment target
// count, formats, and sample count must match the current render pass's
// color attachments. If they do not match, a validation error is recorded
// on the parent command encoder.
//
// Matches Rust wgpu-core command/render.rs set_pipeline → context.check_compatible.
func (p *CoreRenderPassEncoder) SetPipeline(pipeline *RenderPipeline) {
	if p.ended {
		return
	}

	// Draw-time validation: pipeline pass context must be compatible
	// with the render pass context.
	if pipeline != nil && pipeline.PassContext() != nil && p.passContext != nil {
		if err := p.passContext.CheckCompatible(pipeline.PassContext(), pipeline.label); err != nil {
			p.encoder.SetError(fmt.Errorf("SetPipeline: %w", err))
			return
		}
	}

	p.pipeline = pipeline
	// Note: HAL SetPipeline pending (requires core.RenderPipeline with HAL).
	// if p.raw != nil && pipeline.Raw() != nil {
	//     p.raw.SetPipeline(pipeline.Raw())
	// }
}

// CheckPipeline checks a render pipeline, made for the attachments ctx
// describes, writing depth or stencil when it says so, against the pass:
// the same color formats, depth/stencil format and sample count, and no
// write to a read-only aspect.
//
// Reference: wgpu-core command/render.rs set_pipeline (check_compatible,
// IncompatibleDepthAccess, IncompatibleStencilAccess)
func (p *CoreRenderPassEncoder) CheckPipeline(ctx *RenderPassContext, writesDepth, writesStencil bool, label string) error {
	if p.passContext != nil {
		if err := p.passContext.CheckCompatible(ctx, label); err != nil {
			return err
		}
	}
	switch {
	case writesDepth && p.depthReadOnly:
		return fmt.Errorf("pipeline %q writes depth, which the pass has read-only", label)
	case writesStencil && p.stencilReadOnly:
		return fmt.Errorf("pipeline %q writes stencil, which the pass has read-only", label)
	}
	return nil
}

// SetVertexBuffer sets a vertex buffer.
func (p *CoreRenderPassEncoder) SetVertexBuffer(slot uint32, buffer *Buffer, offset uint64) {
	if p.ended || buffer == nil {
		return
	}
	var halBuffer hal.Buffer
	if p.device != nil {
		guard := p.device.snatchLock.Read()
		defer guard.Release()
		halBuffer = buffer.Raw(guard)
	}
	// Record the usage in this pass's usage scope (vertex buffers are
	// read-only); it reaches the command buffer tracker on End.
	if err := p.recordBufferUsage(buffer, halBuffer, track.BufferUsesVertex); err != nil {
		p.encoder.SetError(fmt.Errorf("SetVertexBuffer slot %d: %w", slot, err))
	}
	if p.raw != nil && halBuffer != nil {
		p.raw.SetVertexBuffer(slot, halBuffer, offset)
	}
}

// SetIndexBuffer sets the index buffer.
func (p *CoreRenderPassEncoder) SetIndexBuffer(buffer *Buffer, format gputypes.IndexFormat, offset uint64) {
	if p.ended || buffer == nil {
		return
	}
	var halBuffer hal.Buffer
	if p.device != nil {
		guard := p.device.snatchLock.Read()
		defer guard.Release()
		halBuffer = buffer.Raw(guard)
	}
	// Record the usage in this pass's usage scope (index buffers are
	// read-only); it reaches the command buffer tracker on End.
	if err := p.recordBufferUsage(buffer, halBuffer, track.BufferUsesIndex); err != nil {
		p.encoder.SetError(fmt.Errorf("SetIndexBuffer: %w", err))
	}
	if p.raw != nil && halBuffer != nil {
		p.raw.SetIndexBuffer(halBuffer, format, offset)
	}
}

// SetViewport sets the viewport.
func (p *CoreRenderPassEncoder) SetViewport(vp gputypes.Viewport) {
	if p.ended {
		return
	}
	var maxDimension uint32
	if p.device != nil {
		maxDimension = p.device.Limits.MaxTextureDimension2D
	}
	if maxDimension == 0 {
		maxDimension = gputypes.DefaultLimits().MaxTextureDimension2D
	}
	if err := checkViewport(vp, maxDimension); err != nil {
		p.encoder.SetError(fmt.Errorf("SetViewport: %w", err))
		return
	}
	if p.raw != nil {
		p.raw.SetViewport(vp)
	}
}

// SetScissorRect sets the scissor rectangle.
func (p *CoreRenderPassEncoder) SetScissorRect(rect gputypes.ScissorRect) {
	if p.ended {
		return
	}
	if err := checkScissor(rect, p.width, p.height); err != nil {
		p.encoder.SetError(fmt.Errorf("SetScissorRect: %w", err))
		return
	}
	if p.raw != nil {
		p.raw.SetScissorRect(rect)
	}
}

// SetBlendConstant sets the blend constant color.
func (p *CoreRenderPassEncoder) SetBlendConstant(color *gputypes.Color) {
	if p.ended {
		return
	}
	if p.raw != nil {
		p.raw.SetBlendConstant(color)
	}
}

// SetStencilReference sets the stencil reference value.
func (p *CoreRenderPassEncoder) SetStencilReference(reference uint32) {
	if p.ended {
		return
	}
	if p.raw != nil {
		p.raw.SetStencilReference(reference)
	}
}

// Draw draws primitives.
func (p *CoreRenderPassEncoder) Draw(args gputypes.DrawArgs) {
	if p.ended {
		return
	}
	if p.raw != nil {
		p.raw.Draw(args)
	}
}

// DrawIndexed draws indexed primitives.
func (p *CoreRenderPassEncoder) DrawIndexed(args gputypes.DrawIndexedArgs) {
	if p.ended {
		return
	}
	if p.raw != nil {
		p.raw.DrawIndexed(args)
	}
}

// DrawIndirect draws primitives with GPU-generated parameters.
func (p *CoreRenderPassEncoder) DrawIndirect(buffer *Buffer, offset uint64) {
	p.MultiDrawIndirect(buffer, offset, 1)
}

func (p *CoreRenderPassEncoder) MultiDrawIndirect(buffer *Buffer, offset uint64, drawCount uint32) {
	if p.ended || drawCount == 0 {
		return
	}
	if p.raw != nil && buffer != nil {
		guard := p.device.snatchLock.Read()
		defer guard.Release()
		halBuffer := buffer.Raw(guard)
		if err := p.recordBufferUsage(buffer, halBuffer, track.BufferUsesIndirect); err != nil {
			p.encoder.SetError(fmt.Errorf("DrawIndirect: %w", err))
			return
		}
		if halBuffer != nil {
			p.raw.DrawIndirect(halBuffer, offset, drawCount)
		}
	}
}

// DrawIndexedIndirect draws one indexed primitive with GPU-generated parameters.
func (p *CoreRenderPassEncoder) DrawIndexedIndirect(buffer *Buffer, offset uint64) {
	p.MultiDrawIndexedIndirect(buffer, offset, 1)
}

// MultiDrawIndexedIndirect draws consecutive indexed primitives with
// GPU-generated parameters.
func (p *CoreRenderPassEncoder) MultiDrawIndexedIndirect(buffer *Buffer, offset uint64, drawCount uint32) {
	if p.ended || drawCount == 0 {
		return
	}
	if p.raw != nil && buffer != nil {
		guard := p.device.snatchLock.Read()
		defer guard.Release()
		halBuffer := buffer.Raw(guard)
		if err := p.recordBufferUsage(buffer, halBuffer, track.BufferUsesIndirect); err != nil {
			p.encoder.SetError(fmt.Errorf("DrawIndexedIndirect: %w", err))
			return
		}
		if halBuffer != nil {
			p.raw.DrawIndexedIndirect(halBuffer, offset, drawCount)
		}
	}
}

// DrawIndirectCount draws primitives with a GPU-provided draw count.
func (p *CoreRenderPassEncoder) DrawIndirectCount(
	indirectBuffer *Buffer, indirectOffset uint64,
	countBuffer *Buffer, countOffset uint64, maxDrawCount uint32,
) {
	p.forwardIndirectCount(indirectBuffer, countBuffer, indirectOffset, countOffset, maxDrawCount,
		func(halIndirect, halCount hal.Buffer, indirectOffset, countOffset uint64, maxDrawCount uint32) {
			p.raw.DrawIndirectCount(halIndirect, indirectOffset, halCount, countOffset, maxDrawCount)
		})
}

// DrawIndexedIndirectCount draws indexed primitives with a GPU-provided draw count.
func (p *CoreRenderPassEncoder) DrawIndexedIndirectCount(
	indirectBuffer *Buffer, indirectOffset uint64,
	countBuffer *Buffer, countOffset uint64, maxDrawCount uint32,
) {
	p.forwardIndirectCount(indirectBuffer, countBuffer, indirectOffset, countOffset, maxDrawCount,
		func(halIndirect, halCount hal.Buffer, indirectOffset, countOffset uint64, maxDrawCount uint32) {
			p.raw.DrawIndexedIndirectCount(halIndirect, indirectOffset, halCount, countOffset, maxDrawCount)
		})
}

// End ends the render pass.
func (p *CoreRenderPassEncoder) End() error {
	if p.ended {
		return nil
	}
	p.ended = true

	if p.raw != nil {
		p.raw.End()
	}
	if p.halDesc != nil {
		p.halDesc.release()
		p.halDesc = nil
	}

	return p.encoder.EndRenderPass(p)
}

// =============================================================================
// Core Compute Pass Encoder
// =============================================================================

// CoreComputePassDescriptor describes a compute pass for HAL-integrated API.
type CoreComputePassDescriptor struct {
	// Label is an optional debug name.
	Label string

	// TimestampWrites are timestamp queries to write at the pass's
	// boundaries, which take the same shape as a render pass's.
	TimestampWrites *RenderPassTimestampWrites
}

// CoreComputePassEncoder records compute commands within a pass.
//
// This is the HAL-integrated compute pass encoder that bridges core
// compute commands to HAL compute pass encoder.
type CoreComputePassEncoder struct {
	// raw is the HAL compute pass encoder.
	raw hal.ComputePassEncoder

	// encoder is the parent command encoder.
	encoder *CoreCommandEncoder

	// device is the parent device.
	device *Device

	// pipeline is the currently bound compute pipeline.
	pipeline *ComputePipeline

	// ended indicates whether End() has been called.
	ended bool

	// scope is the usage scope of the dispatch being recorded, the
	// encoder's: the textures and buffers its bind groups bind
	// (RecordPassTextureUsage, RecordPassBufferUsage) and its indirect
	// buffer. Conflicts are checked inside one dispatch only, a texture one
	// dispatch writes and the next samples is transitioned between them:
	// Dispatch applies the scope to the command buffer's trackers, records
	// the barriers inline ahead of the dispatch, a compute pass can take
	// them, and empties the scope for the next one.
	//
	// Reference: wgpu-core command/compute.rs State::flush_states (scope
	// merged and drained per dispatch)
	scope *passScope
}

// RawPass returns the underlying HAL compute pass encoder for direct HAL access.
func (p *CoreComputePassEncoder) RawPass() hal.ComputePassEncoder {
	return p.raw
}

// FlushScope applies the usage scope of the dispatch being recorded to the
// command buffer's trackers and records the transitions into it on the HAL
// encoder, ahead of the dispatch, then empties the scope. Dispatch and
// DispatchIndirect do this themselves; a caller that dispatches on the HAL
// pass directly (the indirect dispatch validation) calls it first.
func (p *CoreComputePassEncoder) FlushScope() {
	p.flushScope()
}

// flushScope is FlushScope.
func (p *CoreComputePassEncoder) flushScope() {
	e := p.encoder
	scope := p.scope
	if e == nil || scope == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	m := e.mutable
	if m == nil || (len(scope.texHAL) == 0 && len(scope.bufHAL) == 0) {
		return
	}
	defer scope.clear()
	scope.texTransitions = m.textureTracker.SetFromScopeInto(scope.textures, scope.texTransitions[:0])
	scope.bufTransitions = m.bufferTracker.SetFromScopeInto(scope.buffers, scope.bufTransitions[:0])
	if len(scope.texTransitions) == 0 && len(scope.bufTransitions) == 0 {
		return
	}
	guard := e.device.snatchLock.Read()
	defer guard.Release()
	if halEncoder := e.raw.Get(guard); halEncoder != nil {
		scope.recordBarriers(*halEncoder)
	}
}

// recordBufferUsage puts a buffer in the dispatch's usage scope. Buffers
// without valid tracking data are skipped.
func (p *CoreComputePassEncoder) recordBufferUsage(buffer *Buffer, halBuffer hal.Buffer, usage track.BufferUses) error {
	if p.scope == nil || p.encoder == nil || buffer == nil {
		return nil
	}
	td := buffer.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return nil
	}
	p.encoder.mu.Lock()
	defer p.encoder.mu.Unlock()
	return p.scope.addBuffer(td.Index(), halBuffer, usage)
}

// ReplaceRawPass replaces the underlying HAL compute pass encoder.
// Used by indirect dispatch validation which ends/re-begins the HAL pass
// to insert barriers between the validation dispatch and the user dispatch.
// The caller is responsible for ensuring the old pass was properly ended.
func (p *CoreComputePassEncoder) ReplaceRawPass(newPass hal.ComputePassEncoder) {
	p.raw = newPass
}

// SetPipeline sets the compute pipeline.
func (p *CoreComputePassEncoder) SetPipeline(pipeline *ComputePipeline) {
	if p.ended {
		return
	}
	p.pipeline = pipeline
	// Note: HAL SetPipeline pending (requires core.ComputePipeline with HAL).
}

// Dispatch dispatches compute work. The dispatch's usage scope, what its
// bind groups bind, reaches the command buffer's trackers first, and the
// barriers into it go in ahead of the dispatch (see scope).
func (p *CoreComputePassEncoder) Dispatch(x, y, z uint32) {
	if p.ended {
		return
	}
	p.flushScope()
	if p.raw != nil {
		p.raw.Dispatch(x, y, z)
	}
}

// DispatchIndirect dispatches compute work with GPU-generated parameters.
// The indirect buffer joins the dispatch's usage scope as INDIRECT, so a
// buffer a dispatch wrote and the next one reads its arguments from is
// transitioned between them.
func (p *CoreComputePassEncoder) DispatchIndirect(buffer *Buffer, offset uint64) {
	if p.ended || buffer == nil {
		return
	}
	// The HAL handle for the scope, without holding the snatch guard into
	// flushScope, which takes its own.
	if p.scope != nil && p.device != nil {
		guard := p.device.snatchLock.Read()
		halBuffer := buffer.Raw(guard)
		guard.Release()
		if err := p.recordBufferUsage(buffer, halBuffer, track.BufferUsesIndirect); err != nil {
			p.encoder.SetError(fmt.Errorf("DispatchIndirect: %w", err))
			return
		}
	}
	p.flushScope()
	if p.raw != nil {
		guard := p.device.snatchLock.Read()
		defer guard.Release()
		halBuffer := buffer.Raw(guard)
		if halBuffer != nil {
			p.raw.DispatchIndirect(halBuffer, offset)
		}
	}
}

// End ends the compute pass.
func (p *CoreComputePassEncoder) End() error {
	if p.ended {
		return nil
	}
	p.ended = true

	if p.raw != nil {
		p.raw.End()
	}

	return p.encoder.EndComputePass(p)
}

// =============================================================================
// Core Command Buffer
// =============================================================================

// CoreCommandBuffer is a finished command recording ready for submission.
//
// This is created by CoreCommandEncoder.Finish() and can be submitted
// to a queue for execution.
//
// A CoreCommandBuffer may contain multiple HAL command buffers when the
// encoder used multi-CB recording (OpenPass/CloseCB/CloseAndSwap/CloseAndPushFront).
// All CBs are submitted together in one HAL queue submit call.
//
// Reference: Rust wgpu-core BakedCommands.encoder.list (command/mod.rs:742-749)
type CoreCommandBuffer struct {
	// raw is the primary HAL command buffer (first in the list, or the only one).
	// Maintained for backward compatibility with Raw().
	raw hal.CommandBuffer

	// halBuffers holds all HAL command buffers, in submission order. Set by
	// Finish; raw equals halBuffers[0].
	//
	// Reference: Rust wgpu-core InnerCommandEncoder.list (command/mod.rs:551)
	halBuffers []hal.CommandBuffer

	// device is the parent device.
	device *Device

	// mutable holds the resource tracking state from encoding.
	mutable *CommandBufferMutable

	// label is the debug label.
	label string
}

// Raw returns the primary underlying HAL command buffer.
// For multi-CB encoders, this returns the first CB in the list.
// Use HalBufferList() to get all CBs for submission.
func (cb *CoreCommandBuffer) Raw() hal.CommandBuffer {
	return cb.raw
}

// HalBufferList returns all HAL command buffers in submission order: the
// one the encoder started with and those its render passes split off.
//
// This is used by Queue.Submit() to flatten multiple CBs from a single
// encoder into the combined submission list.
//
// Reference: Rust wgpu-core BakedCommands.encoder.list
func (cb *CoreCommandBuffer) HalBufferList() []hal.CommandBuffer {
	if len(cb.halBuffers) > 0 {
		return cb.halBuffers
	}
	if cb.raw == nil {
		return nil
	}
	return []hal.CommandBuffer{cb.raw}
}

// Device returns the parent device.
func (cb *CoreCommandBuffer) Device() *Device {
	return cb.device
}

// Label returns the debug label.
func (cb *CoreCommandBuffer) Label() string {
	return cb.label
}

// ReleaseScopes hands the usage scopes back to the device for the next
// command encoder, once Submit has merged them into the device tracker or
// the command buffer is dropped without a submit. TextureScope and
// BufferScope return nil afterwards.
func (cb *CoreCommandBuffer) ReleaseScopes() {
	if cb == nil || cb.mutable == nil {
		return
	}
	m := cb.mutable
	cb.mutable = nil
	if cb.device != nil {
		cb.device.scopes.put(m)
	}
}

// TextureScope returns the command buffer's texture tracker: the first and
// the last usage of every texture the command buffer uses. At submit time it
// is merged into the device-level TextureTracker (DeviceTracker.
// MergeTextureScope), which emits barriers into the first usages and adopts
// the last usages.
//
// Returns nil if the command buffer has no tracker (e.g. ID-based API).
func (cb *CoreCommandBuffer) TextureScope() *track.EncoderTextureTracker {
	if cb.mutable == nil {
		return nil
	}
	return cb.mutable.textureTracker
}

// BufferScope returns the command buffer's buffer tracker (first and last
// usage of every buffer). See TextureScope.
//
// Returns nil if the command buffer has no tracker (e.g. ID-based API).
func (cb *CoreCommandBuffer) BufferScope() *track.EncoderBufferTracker {
	if cb.mutable == nil {
		return nil
	}
	return cb.mutable.bufferTracker
}

// =============================================================================
// ID-Based API (Backward Compatibility)
// =============================================================================

// ComputePassEncoder records compute commands within a compute pass.
// It wraps hal.ComputePassEncoder with validation and ID-based resource lookup.
type ComputePassEncoder struct {
	raw    hal.ComputePassEncoder
	device *Device
	ended  bool
}

// SetPipeline sets the active compute pipeline for subsequent dispatch calls.
// The pipeline must have been created on the same device as this encoder.
//
// Returns an error if the pipeline ID is invalid.
func (e *ComputePassEncoder) SetPipeline(pipeline ComputePipelineID) error {
	if e.ended {
		return fmt.Errorf("compute pass has already ended")
	}

	hub := GetGlobal().Hub()
	rawPipeline, err := hub.GetComputePipeline(pipeline)
	if err != nil {
		return fmt.Errorf("invalid compute pipeline: %w", err)
	}

	// Note: HAL integration pending. When core.ComputePipeline has HAL,
	// convert rawPipeline to hal.ComputePipeline and call e.raw.SetPipeline.
	_ = rawPipeline
	// e.raw.SetPipeline(halPipeline)

	return nil
}

// SetBindGroup sets a bind group for the given index.
// The bind group provides resources (buffers, textures, samplers) to shaders.
//
// Parameters:
//   - index: The bind group index (0, 1, 2, or 3).
//   - group: The bind group ID to bind.
//   - offsets: Dynamic offsets for dynamic uniform/storage buffers (can be nil).
//
// Returns an error if the bind group ID is invalid or if the encoder has ended.
func (e *ComputePassEncoder) SetBindGroup(index uint32, group BindGroupID, offsets []uint32) error {
	if e.ended {
		return fmt.Errorf("compute pass has already ended")
	}

	// WebGPU spec: max 4 bind groups (0-3)
	if index > 3 {
		return fmt.Errorf("bind group index %d exceeds maximum (3)", index)
	}

	hub := GetGlobal().Hub()
	rawGroup, err := hub.GetBindGroup(group)
	if err != nil {
		return fmt.Errorf("invalid bind group: %w", err)
	}

	// Note: HAL integration pending. When core.BindGroup has HAL,
	// convert rawGroup to hal.BindGroup and call e.raw.SetBindGroup.
	_ = rawGroup
	// e.raw.SetBindGroup(index, halGroup, offsets)

	return nil
}

// Dispatch dispatches compute work.
// This executes the compute shader with the specified number of workgroups.
//
// Parameters:
//   - x, y, z: The number of workgroups to dispatch in each dimension.
//
// Each workgroup runs the compute shader's workgroup_size threads.
// The total threads = x * y * z * workgroup_size.
//
// Note: This method does not return an error. Dispatch errors are deferred
// to command buffer submission time, matching the WebGPU error model.
func (e *ComputePassEncoder) Dispatch(x, y, z uint32) {
	if e.ended {
		// Record error for deferred validation
		return
	}

	if e.raw != nil {
		e.raw.Dispatch(x, y, z)
	}
}

// DispatchIndirect dispatches compute work with GPU-generated parameters.
// The dispatch parameters are read from the specified buffer.
//
// Parameters:
//   - buffer: The buffer containing DispatchIndirectArgs at the given offset.
//   - offset: The byte offset into the buffer (must be 4-byte aligned).
//
// The buffer must contain the following structure at the offset:
//
//	struct DispatchIndirectArgs {
//	    x: u32,     // Number of workgroups in X
//	    y: u32,     // Number of workgroups in Y
//	    z: u32,     // Number of workgroups in Z
//	}
//
// Returns an error if the buffer ID is invalid or the offset is not aligned.
func (e *ComputePassEncoder) DispatchIndirect(buffer BufferID, offset uint64) error {
	if e.ended {
		return fmt.Errorf("compute pass has already ended")
	}

	// Indirect dispatch requires 4-byte alignment
	if offset%4 != 0 {
		return fmt.Errorf("indirect dispatch offset must be 4-byte aligned, got %d", offset)
	}

	hub := GetGlobal().Hub()
	rawBuffer, err := hub.GetBuffer(buffer)
	if err != nil {
		return fmt.Errorf("invalid buffer: %w", err)
	}

	// Note: HAL integration pending. When core.Buffer lookup returns HAL buffer,
	// convert rawBuffer to hal.Buffer and call e.raw.DispatchIndirect.
	_ = rawBuffer
	// e.raw.DispatchIndirect(halBuffer, offset)

	return nil
}

// End finishes the compute pass.
// After this call, the encoder cannot be used again.
// Any subsequent method calls will return errors.
func (e *ComputePassEncoder) End() {
	if e.ended {
		return
	}

	e.ended = true

	if e.raw != nil {
		e.raw.End()
	}
}

// CommandEncoderState tracks the state of a command encoder.
type CommandEncoderState int

const (
	// CommandEncoderStateRecording means the encoder is actively recording commands.
	CommandEncoderStateRecording CommandEncoderState = iota

	// CommandEncoderStateEnded means the encoder has finished and produced a command buffer.
	CommandEncoderStateEnded

	// CommandEncoderStateError means the encoder encountered an error.
	CommandEncoderStateError
)

// CommandEncoderImpl provides command encoder functionality.
// It wraps hal.CommandEncoder with validation and ID-based resource lookup.
type CommandEncoderImpl struct {
	raw    hal.CommandEncoder
	device *Device
	state  CommandEncoderState
	label  string
}

// BeginComputePass begins a new compute pass within this command encoder.
// The returned ComputePassEncoder is used to record compute commands.
//
// Parameters:
//   - desc: Optional descriptor with label and timestamp writes.
//     Pass nil for default settings.
//
// The compute pass must be ended with End() before:
//   - Beginning another pass (compute or render)
//   - Finishing the command encoder
//
// Returns the compute pass encoder and any error encountered.
func (e *CommandEncoderImpl) BeginComputePass(desc *ComputePassDescriptor) (*ComputePassEncoder, error) {
	if e.state != CommandEncoderStateRecording {
		return nil, fmt.Errorf("command encoder is not in recording state")
	}

	// Convert core descriptor to HAL descriptor
	halDesc := &hal.ComputePassDescriptor{}
	if desc != nil {
		halDesc.Label = desc.Label

		if desc.TimestampWrites != nil {
			// Note: QuerySet HAL integration pending.
			// Skipping timestamp writes until core.QuerySet has HAL.
			halDesc.TimestampWrites = nil
		}
	}

	// Begin the compute pass on the underlying HAL encoder
	var rawPass hal.ComputePassEncoder
	if e.raw != nil {
		rawPass = e.raw.BeginComputePass(halDesc)
	}

	return &ComputePassEncoder{
		raw:    rawPass,
		device: e.device,
		ended:  false,
	}, nil
}

// DeviceCreateCommandEncoder creates a new command encoder for recording GPU commands.
// This is the entry point for recording command buffers.
//
// Parameters:
//   - id: The device ID to create the encoder on.
//   - label: Optional debug label for the encoder.
//
// Returns the command encoder ID and any error encountered.
func DeviceCreateCommandEncoder(id DeviceID, label string) (CommandEncoderID, error) {
	hub := GetGlobal().Hub()

	// Verify the device exists
	_, err := hub.GetDevice(id)
	if err != nil {
		return CommandEncoderID{}, fmt.Errorf("invalid device: %w", err)
	}

	// Create a placeholder command encoder
	// In a full implementation, this would create the HAL command encoder
	encoder := CommandEncoder{}
	encoderID := hub.RegisterCommandEncoder(encoder)

	return encoderID, nil
}

// CommandEncoderFinish finishes recording and returns a command buffer.
// The command encoder cannot be used after this call.
//
// Parameters:
//   - id: The command encoder ID to finish.
//
// Returns the command buffer ID and any error encountered.
func CommandEncoderFinish(id CommandEncoderID) (CommandBufferID, error) {
	hub := GetGlobal().Hub()

	// Verify the encoder exists
	_, err := hub.GetCommandEncoder(id)
	if err != nil {
		return CommandBufferID{}, fmt.Errorf("invalid command encoder: %w", err)
	}

	// Note: This is the ID-based API. HAL integration is in CoreCommandEncoder.Finish().

	// Create a placeholder command buffer (ID-based API does not have HAL).
	cmdBuffer := CommandBuffer{}
	cmdBufferID := hub.RegisterCommandBuffer(cmdBuffer)

	// Unregister the encoder (it's consumed)
	_, _ = hub.UnregisterCommandEncoder(id)

	return cmdBufferID, nil
}
