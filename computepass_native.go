//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"fmt"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// ComputePassEncoder records compute dispatch commands.
//
// Created by CommandEncoder.BeginComputePass().
// Must be ended with End() before the CommandEncoder can be finished.
//
// NOT thread-safe.
type ComputePassEncoder struct {
	core    *core.CoreComputePassEncoder
	encoder *CommandEncoder
	// currentPipelineBindGroupCount tracks the bind group count of the
	// currently set pipeline. Used by SetBindGroup to validate that the
	// group index is within the pipeline layout bounds. Zero means no
	// pipeline has been set yet.
	currentPipelineBindGroupCount uint32
	// pipelineSet tracks whether SetPipeline has been called.
	// Dispatch commands require a pipeline to be set first.
	pipelineSet bool
	// currentPipeline tracks the currently set pipeline for state restoration
	// after indirect dispatch validation. Set by SetPipeline.
	currentPipeline *ComputePipeline
	// assignedBindGroups tracks the bind groups set at each slot for state
	// restoration after indirect dispatch validation.
	assignedBindGroups [MaxBindGroups]*BindGroup
	// assignedDynOffsets tracks the dynamic offsets for each bind group slot
	// for state restoration after indirect dispatch validation.
	assignedDynOffsets [MaxBindGroups][]uint32
	// binder tracks bind group assignments and validates compatibility
	// at dispatch time, matching Rust wgpu-core's Binder pattern.
	binder binder
}

// reset readies p, which lives in its encoder, for a new pass. The binder's
// lists keep their room.
func (p *ComputePassEncoder) reset(corePass *core.CoreComputePassEncoder, enc *CommandEncoder) {
	b := p.binder
	*p = ComputePassEncoder{core: corePass, encoder: enc}
	b.reset()
	p.binder = b
}

// SetPipeline sets the active compute pipeline.
func (p *ComputePassEncoder) SetPipeline(pipeline *ComputePipeline) {
	if pipeline == nil {
		p.encoder.setError(fmt.Errorf("wgpu: ComputePass.SetPipeline: pipeline is nil"))
		return
	}
	p.currentPipelineBindGroupCount = pipeline.bindGroupCount
	p.pipelineSet = true
	p.currentPipeline = pipeline
	p.binder.updateExpectations(pipeline.bindGroupLayouts)
	p.binder.updateLateBufferBindingsFromPipeline(pipeline.lateSizedBufferGroups)
	p.encoder.trackPipeline(&pipeline.mark, pipeline.ref)
	raw := p.core.RawPass()
	if raw != nil && pipeline.hal != nil {
		raw.SetPipeline(pipeline.hal)
	}
}

// SetBindGroup sets a bind group for the given index.
func (p *ComputePassEncoder) SetBindGroup(index uint32, group *BindGroup, offsets []uint32) {
	if err := validateSetBindGroup("ComputePass", index, group, offsets, p.currentPipelineBindGroupCount, p.encoder.limits()); err != nil {
		p.encoder.setError(err)
		return
	}
	p.binder.assign(index, group.layout)
	p.binder.assignBindGroup(index, group)
	p.assignedBindGroups[index] = group
	p.assignedDynOffsets[index] = append(p.assignedDynOffsets[index][:0], offsets...)
	// Keeps the group and its buffers alive and lists the group for
	// submit-time validation, once per encoder. Its textures and buffers
	// enter a usage scope only when a dispatch runs with it (recordDispatch),
	// so a group bound and replaced before the dispatch conflicts with
	// nothing.
	p.encoder.trackBindGroup(group)
	raw := p.core.RawPass()
	if raw != nil && group.hal != nil {
		raw.SetBindGroup(index, group.hal, offsets)
	}
}

// validateDispatchState checks that a pipeline has been set and all bind groups
// are compatible before a dispatch call.
// Returns true if validation passes, false if an error was recorded.
//
// Each validation failure wraps a typed sentinel error so that callers can
// use errors.Is() to identify the failure category programmatically.
// Matches Rust wgpu-core State::is_ready (command/compute.rs:278-284).
func (p *ComputePassEncoder) validateDispatchState(method string) bool {
	if !p.pipelineSet {
		p.encoder.setError(fmt.Errorf("wgpu: ComputePass.%s: no pipeline set (call SetPipeline first): %w",
			method, ErrDispatchMissingPipeline))
		return false
	}
	if err := p.binder.checkCompatibility(); err != nil {
		sentinel := ErrDispatchMissingBindGroup
		if errors.Is(err, errBindGroupIncompatible) {
			sentinel = ErrDispatchIncompatibleBindGroup
		}
		p.encoder.setError(fmt.Errorf("wgpu: ComputePass.%s: %w: %w", method, sentinel, err))
		return false
	}
	// Late buffer binding size validation: check that bound buffers are large enough
	// for bindings with MinBindingSize == 0. Matches Rust wgpu-core's is_ready()
	// call to check_late_buffer_bindings before dispatch (compute.rs:278-285).
	if err := p.binder.checkLateBufferBindings(); err != nil {
		p.encoder.setError(fmt.Errorf("wgpu: ComputePass.%s: %w: %w", method, ErrDispatchLateBufferTooSmall, err))
		return false
	}
	return true
}

// Dispatch dispatches compute work.
func (p *ComputePassEncoder) Dispatch(x, y, z uint32) {
	if !p.validateDispatchState("Dispatch") {
		return
	}

	// VAL-009: Validate workgroup counts against device limits.
	// Matches Rust wgpu-core compute.rs:853-870.
	// (0, 0, 0) is allowed as a no-op per spec.
	limit := p.encoder.device.core.Limits.MaxComputeWorkgroupsPerDimension
	if x > limit || y > limit || z > limit {
		p.encoder.setError(fmt.Errorf(
			"wgpu: ComputePass.Dispatch: workgroup count (%d, %d, %d) exceeds device limit %d: %w",
			x, y, z, limit, ErrDispatchWorkgroupCountExceeded))
		return
	}

	if !p.recordDispatch("ComputePass.Dispatch") {
		return
	}
	p.core.Dispatch(x, y, z)
}

// recordDispatch puts the textures and buffers of the bind groups the next
// dispatch runs with, the ones in the pipeline's layout, in the dispatch's
// usage scope: each dispatch is a scope of its own, validated on its own,
// and the core dispatch transitions what an earlier dispatch, pass or copy
// of this encoder left in another state. Returns false, with the conflict
// recorded on the encoder, if two of the groups use one resource in
// incompatible ways (a storage buffer one writes and another reads).
//
// Reference: wgpu-core command/compute.rs State::flush_states (binder.
// list_active merged into the scope)
func (p *ComputePassEncoder) recordDispatch(op string) bool {
	enc := p.encoder
	if enc.core == nil {
		return true
	}
	wasErr := enc.core.Status() == core.CommandEncoderStatusError
	for i := uint32(0); i < p.binder.maxSlots && i < MaxBindGroups; i++ {
		enc.recordDispatchBindGroup(p.assignedBindGroups[i], op)
	}
	return wasErr || enc.core.Status() != core.CommandEncoderStatusError
}

// DispatchIndirect dispatches compute work with GPU-generated parameters.
//
// When indirect dispatch validation is available (created at device init),
// a pre-dispatch validation shader checks workgroup counts against
// maxComputeWorkgroupsPerDimension and clamps to (0,0,0) if exceeded.
// This prevents GPU hangs/TDR from invalid indirect buffers.
//
// Matches Rust wgpu-core compute.rs:878-1074 (dispatch_indirect).
func (p *ComputePassEncoder) DispatchIndirect(buffer *Buffer, offset uint64) {
	if !p.validateDispatchState("DispatchIndirect") {
		return
	}
	if buffer == nil {
		p.encoder.setError(fmt.Errorf("wgpu: ComputePass.DispatchIndirect: buffer is nil"))
		return
	}
	// VAL-B3: Validate indirect buffer has INDIRECT usage.
	// Matches Rust wgpu-core compute.rs:896 (check_usage(BufferUsages::INDIRECT)).
	if buffer.Usage()&BufferUsageIndirect == 0 {
		p.encoder.setError(fmt.Errorf(
			"wgpu: ComputePass.DispatchIndirect: buffer %q missing BufferUsageIndirect usage: %w",
			buffer.Label(), ErrDispatchIndirectBufferUsage))
		return
	}
	// VAL-B3: Validate indirect buffer offset is 4-byte aligned.
	// Matches Rust wgpu-core compute.rs:899 (offset % 4 != 0).
	if offset%4 != 0 {
		p.encoder.setError(fmt.Errorf(
			"wgpu: ComputePass.DispatchIndirect: offset %d is not 4-byte aligned: %w",
			offset, ErrDispatchIndirectOffsetAlignment))
		return
	}
	// VAL-B3: Validate indirect args fit within buffer.
	// DispatchIndirect args: 3 x uint32 = 12 bytes. Matches Rust compute.rs:903-909.
	if offset+12 > buffer.Size() {
		p.encoder.setError(fmt.Errorf(
			"wgpu: ComputePass.DispatchIndirect: offset %d + 12 bytes exceeds buffer size %d: %w",
			offset, buffer.Size(), ErrDispatchIndirectBufferOverrun))
		return
	}
	p.encoder.trackBuffer(buffer)

	// The dispatch's usage scope: its bind groups, and the indirect buffer.
	if !p.recordDispatch("ComputePass.DispatchIndirect") {
		return
	}

	// FEAT-COMPUTE-004: GPU-side indirect dispatch validation.
	// If the device has an IndirectValidation pipeline, run a pre-dispatch
	// validation shader that checks workgroup counts and either copies valid
	// values or zeroes them out. The actual DispatchIndirect then reads from
	// the validated destination buffer instead of the user's original buffer.
	//
	// Matches Rust wgpu-core compute.rs:921-1059.
	iv := p.encoder.device.core.IndirectValidation()
	raw := p.core.RawPass()
	if iv != nil && raw != nil {
		// The validation shader reads the user's buffer as storage; the
		// HAL dispatch reads core's own validated copy, which core does not
		// track. The scope's barriers go in ahead of both.
		if err := p.encoder.core.RecordPassBufferUsage(buffer.coreBuffer(), buffer.halBuffer(), track.BufferUsesStorageRead); err != nil {
			p.encoder.setError(fmt.Errorf("wgpu: ComputePass.DispatchIndirect: buffer usage conflict: %w", err))
			return
		}
		p.core.FlushScope()
		p.dispatchIndirectValidated(buffer, offset, iv, raw)
		return
	}

	// Fallback: no validation available, dispatch directly. Core puts the
	// buffer in the scope as INDIRECT and flushes it.
	p.core.DispatchIndirect(buffer.coreBuffer(), offset)
}

// dispatchIndirectValidated runs the GPU-side validation shader before the
// actual indirect dispatch, in a pass of its own, and restores the user's
// pipeline and bind groups after it, matching Rust wgpu-core's approach in
// compute.rs:928-1059.
//
// Flow:
//  1. End the user's pass
//  2. Transition the dst buffer from INDIRECT to STORAGE
//  3. Dispatch(1,1,1) the validation shader in its own pass
//  4. Transition the dst buffer back to INDIRECT
//  5. Begin a new pass with the user's pipeline and bind groups
//  6. DispatchIndirect from the validated dst buffer at offset 0
func (p *ComputePassEncoder) dispatchIndirectValidated(
	buffer *Buffer,
	offset uint64,
	iv *core.IndirectValidation,
	raw hal.ComputePassEncoder,
) {
	halDevice := p.encoder.device.halDevice()
	halSrcBuf := buffer.halBuffer()
	parentEncoder := p.encoder.core.RawEncoder()
	if halDevice == nil || halSrcBuf == nil || parentEncoder == nil {
		p.core.DispatchIndirect(buffer.coreBuffer(), offset)
		return
	}
	srcBindGroup, err := iv.CreateSrcBindGroup(halDevice, halSrcBuf, offset)
	if err != nil {
		// Validation resource creation failed -- fall back to unvalidated dispatch.
		p.core.DispatchIndirect(buffer.coreBuffer(), offset)
		return
	}
	// The group is this dispatch's own. It goes with the encoder's refs, so
	// it's destroyed once the GPU is done with the submit, or right away
	// when the command buffer is dropped without one.
	p.encoder.trackedRefs = append(p.encoder.trackedRefs, core.NewResourceRef("IndirectValidationSrc", func() {
		halDevice.DestroyBindGroup(srcBindGroup)
	}))

	raw.End()
	parentEncoder.TransitionBuffers([]hal.BufferBarrier{
		{
			Buffer: iv.DstBuffer(),
			Usage: hal.BufferUsageTransition{
				OldUsage: gputypes.BufferUsageIndirect,
				NewUsage: gputypes.BufferUsageStorage,
			},
		},
	})

	validationPass := parentEncoder.BeginComputePass(&hal.ComputePassDescriptor{
		Label: "indirect_dispatch_validation",
	})
	validationPass.SetPipeline(iv.Pipeline())
	validationPass.SetBindGroup(0, iv.DstBindGroup(), nil)
	validationPass.SetBindGroup(1, srcBindGroup, nil)
	validationPass.Dispatch(1, 1, 1)
	validationPass.End()

	parentEncoder.TransitionBuffers([]hal.BufferBarrier{
		{
			Buffer: iv.DstBuffer(),
			Usage: hal.BufferUsageTransition{
				OldUsage: gputypes.BufferUsageStorage,
				NewUsage: gputypes.BufferUsageIndirect,
			},
		},
	})

	userPass := parentEncoder.BeginComputePass(&hal.ComputePassDescriptor{
		Label: "indirect_dispatch_user",
	})
	p.restoreComputeState(userPass)
	userPass.DispatchIndirect(iv.DstBuffer(), 0)

	// The pass the core encoder records into from here on.
	p.core.ReplaceRawPass(userPass)
}

// restoreComputeState restores the user's pipeline and bind groups on a new
// HAL compute pass encoder after the validation pass. This is needed because
// the validation pass temporarily replaced the user's pipeline/bind groups.
// Matches Rust wgpu-core compute.rs:1001-1036 (reset state after validation).
func (p *ComputePassEncoder) restoreComputeState(raw hal.ComputePassEncoder) {
	// Restore user's compute pipeline.
	if p.pipelineSet && p.currentPipeline != nil && p.currentPipeline.hal != nil {
		raw.SetPipeline(p.currentPipeline.hal)
	}

	// Restore user's bind groups.
	for i := uint32(0); i < p.currentPipelineBindGroupCount; i++ {
		bg := p.assignedBindGroups[i]
		if bg != nil && bg.hal != nil {
			raw.SetBindGroup(i, bg.hal, p.assignedDynOffsets[i])
		}
	}
}

// End ends the compute pass.
func (p *ComputePassEncoder) End() error {
	return p.core.End()
}
