//go:build !(js && wasm)

package core

import (
	"fmt"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// IndirectDrawValidation checks the records of the indirect draws of a device
// made without FeatureIndirectFirstInstance, where WebGPU makes a draw whose
// record has a first instance a no-op: a compute pass ahead of the render
// pass copies each draw's records into a buffer of the encoder's and zeroes
// the instance count of every record with a first instance, and the draw
// reads the copy.
//
// Reference: wgpu-core indirect_validation/draw.rs
type IndirectDrawValidation struct {
	device            hal.Device
	module            hal.ShaderModule
	layout            hal.BindGroupLayout
	pipelineLayout    hal.PipelineLayout
	draw, drawIndexed hal.ComputePipeline
	storageAlign      uint64
}

// drawValidationWorkgroup is the workgroup size of the validation shader,
// one record an invocation.
const drawValidationWorkgroup = 64

// indirectDrawValidationWGSL binds src so that it ends with the records of a
// draw call and dst so that it holds them all.
const indirectDrawValidationWGSL = `@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

fn validate(stride: u32, first_instance: u32, i: u32) {
    let n = arrayLength(&dst) / stride;
    if (i >= n) {
        return;
    }
    let s = arrayLength(&src) - (n - i) * stride;
    let d = i * stride;
    for (var w = 0u; w < stride; w = w + 1u) {
        dst[d + w] = src[s + w];
    }
    if (src[s + first_instance] != 0u) {
        dst[d + 1u] = 0u;
    }
}

@compute @workgroup_size(64)
fn draw(@builtin(global_invocation_id) id: vec3<u32>) {
    validate(4u, 3u, id.x);
}

@compute @workgroup_size(64)
fn draw_indexed(@builtin(global_invocation_id) id: vec3<u32>) {
    validate(5u, 4u, id.x);
}
`

// newIndirectDrawValidation makes the validation's pipelines, or returns nil
// where the device can't run them.
func newIndirectDrawValidation(halDevice hal.Device, limits gputypes.Limits) *IndirectDrawValidation {
	if halDevice == nil || limits.MaxComputeWorkgroupSizeX < drawValidationWorkgroup {
		return nil
	}
	v := &IndirectDrawValidation{device: halDevice, storageAlign: uint64(max(limits.MinStorageBufferOffsetAlignment, 4))}
	if err := v.build(); err != nil {
		v.Dispose()
		return nil
	}
	return v
}

func (v *IndirectDrawValidation) build() error {
	d := v.device
	var err error
	if v.module, err = d.CreateShaderModule(&hal.ShaderModuleDescriptor{
		Label: "wgpu_indirect_draw_validation", Source: hal.ShaderSource{WGSL: indirectDrawValidationWGSL},
	}); err != nil {
		return fmt.Errorf("shader module: %w", err)
	}
	storage := func(binding uint32, t gputypes.BufferBindingType) gputypes.BindGroupLayoutEntry {
		return gputypes.BindGroupLayoutEntry{Binding: binding, Visibility: gputypes.ShaderStageCompute,
			Buffer: &gputypes.BufferBindingLayout{Type: t, MinBindingSize: 16}}
	}
	if v.layout, err = d.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{
		Label: "wgpu_indirect_draw_validation", Entries: []gputypes.BindGroupLayoutEntry{
			storage(0, gputypes.BufferBindingTypeReadOnlyStorage), storage(1, gputypes.BufferBindingTypeStorage),
		},
	}); err != nil {
		return fmt.Errorf("bind group layout: %w", err)
	}
	if v.pipelineLayout, err = d.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
		Label: "wgpu_indirect_draw_validation", BindGroupLayouts: []hal.BindGroupLayout{v.layout},
	}); err != nil {
		return fmt.Errorf("pipeline layout: %w", err)
	}
	pipeline := func(entryPoint string) (hal.ComputePipeline, error) {
		return d.CreateComputePipeline(&hal.ComputePipelineDescriptor{
			Label: "wgpu_indirect_draw_validation", Layout: v.pipelineLayout,
			Compute: hal.ComputeState{Module: v.module, EntryPoint: entryPoint},
		})
	}
	if v.draw, err = pipeline("draw"); err != nil {
		return fmt.Errorf("draw pipeline: %w", err)
	}
	if v.drawIndexed, err = pipeline("draw_indexed"); err != nil {
		return fmt.Errorf("indexed draw pipeline: %w", err)
	}
	return nil
}

// Dispose destroys the validation's pipelines.
func (v *IndirectDrawValidation) Dispose() {
	if v == nil || v.device == nil {
		return
	}
	d := v.device
	if v.draw != nil {
		d.DestroyComputePipeline(v.draw)
	}
	if v.drawIndexed != nil {
		d.DestroyComputePipeline(v.drawIndexed)
	}
	if v.pipelineLayout != nil {
		d.DestroyPipelineLayout(v.pipelineLayout)
	}
	if v.layout != nil {
		d.DestroyBindGroupLayout(v.layout)
	}
	if v.module != nil {
		d.DestroyShaderModule(v.module)
	}
	v.device = nil
}

// BindGroup makes the bind group of one validated draw call: the size bytes
// of its records at offset in src, bound from offset rounded down to the
// storage offset alignment, and as many at dstOffset in dst. The caller
// destroys it once the GPU is done with the draw.
func (v *IndirectDrawValidation) BindGroup(src hal.Buffer, offset uint64, dst hal.Buffer, dstOffset, size uint64) (hal.BindGroup, error) {
	base := offset - offset%v.storageAlign
	return v.device.CreateBindGroup(&hal.BindGroupDescriptor{
		Label:  "wgpu_indirect_draw_validation",
		Layout: v.layout,
		Entries: []gputypes.BindGroupEntry{
			{Binding: 0, Resource: gputypes.BufferBinding{Buffer: src.NativeHandle(), Offset: base, Size: offset + size - base}},
			{Binding: 1, Resource: gputypes.BufferBinding{Buffer: dst.NativeHandle(), Offset: dstOffset, Size: size}},
		},
	})
}

// StorageAlign is the alignment of the copies' offsets in their buffer.
func (v *IndirectDrawValidation) StorageAlign() uint64 { return v.storageAlign }

// drawValidation is a validated draw call of a render pass: the bind group of
// its records and the buffer they are copied into, recorded ahead of the pass.
type drawValidation struct {
	pipeline   hal.ComputePipeline
	group      hal.BindGroup
	dst        hal.Buffer
	workgroups uint32
}

// recordDrawValidations records the validation of the pass's indirect draws:
// the copies' buffers become storage, a compute pass copies and checks the
// records, and the buffers become indirect for the draws.
func (p *passScope) recordDrawValidations(halEncoder hal.CommandEncoder) {
	if len(p.drawValidations) == 0 {
		return
	}
	barriers := func(old, usage gputypes.BufferUsage) {
		p.bufBarriers = p.bufBarriers[:0]
		for i, v := range p.drawValidations {
			if i == 0 || v.dst != p.drawValidations[i-1].dst {
				p.bufBarriers = append(p.bufBarriers, hal.BufferBarrier{Buffer: v.dst,
					Usage: hal.BufferUsageTransition{OldUsage: old, NewUsage: usage}})
			}
		}
		halEncoder.TransitionBuffers(p.bufBarriers)
	}
	barriers(0, gputypes.BufferUsageStorage)
	pass := halEncoder.BeginComputePass(&hal.ComputePassDescriptor{Label: "(wgpu internal) indirect draw validation"})
	for _, v := range p.drawValidations {
		pass.SetPipeline(v.pipeline)
		pass.SetBindGroup(0, v.group, nil)
		pass.Dispatch(v.workgroups, 1, 1)
	}
	pass.End()
	barriers(gputypes.BufferUsageStorage, gputypes.BufferUsageIndirect)
}

// DrawIndirectValidated records an indirect draw whose records the device
// validates: drawCount records in src, which the validation reads as
// storage, copied into dst with group, which the draw reads, and count, a
// count buffer the draw reads as indirect, or nil. draw records the HAL draw
// from the copy, with the count buffer's HAL buffer.
func (p *CoreRenderPassEncoder) DrawIndirectValidated(src, count *Buffer, drawCount uint32, indexed bool, dst hal.Buffer, group hal.BindGroup,
	draw func(raw hal.RenderPassEncoder, count hal.Buffer)) {
	if p.ended || drawCount == 0 || p.raw == nil || src == nil {
		return
	}
	v := p.device.indirectDrawValidation
	guard := p.device.snatchLock.Read()
	defer guard.Release()
	if err := p.recordBufferUsage(src, src.Raw(guard), track.BufferUsesStorageRead); err != nil {
		p.encoder.SetError(fmt.Errorf("DrawIndirect: %w", err))
		return
	}
	var halCount hal.Buffer
	if count != nil {
		halCount = count.Raw(guard)
		if err := p.recordBufferUsage(count, halCount, track.BufferUsesIndirect); err != nil {
			p.encoder.SetError(fmt.Errorf("DrawIndirectCount: count buffer: %w", err))
			return
		}
	}
	pipeline := v.draw
	if indexed {
		pipeline = v.drawIndexed
	}
	p.scope.drawValidations = append(p.scope.drawValidations, drawValidation{
		pipeline: pipeline, group: group, dst: dst,
		workgroups: (drawCount + drawValidationWorkgroup - 1) / drawValidationWorkgroup,
	})
	draw(p.raw, halCount)
}

// IndirectDrawValidation returns the validation of the device's indirect
// draws, or nil when the device needs none or can't run it.
func (d *Device) IndirectDrawValidation() *IndirectDrawValidation {
	return d.indirectDrawValidation
}
