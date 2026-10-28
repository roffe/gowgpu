// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin && !(js && wasm)

package metal

import (
	"sync/atomic"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/naga/ir"
	"github.com/roffe/naga/msl"
)

// Buffer implements hal.Buffer for Metal.
type Buffer struct {
	raw     ID // id<MTLBuffer>
	size    uint64
	usage   gputypes.BufferUsage
	options MTLResourceOptions
	device  *Device
}

// Destroy releases the buffer.
func (b *Buffer) Destroy() {
	if b.device != nil {
		b.device.DestroyBuffer(b)
	}
}

// NativeHandle returns the raw MTLBuffer handle.
func (b *Buffer) NativeHandle() uint64 { return uint64(b.raw) }

// Contents returns the buffer contents pointer (for mapped buffers).
// Returns unsafe.Pointer to allow safe pointer arithmetic via unsafe.Add
// without triggering go vet "possible misuse of unsafe.Pointer" warnings.
func (b *Buffer) Contents() unsafe.Pointer {
	if b.raw == 0 {
		return nil
	}
	p := MsgSend(b.raw, Sel("contents"))
	// p is Metal's memory, not Go's. Its bits are read as a pointer because
	// go vet rejects every uintptr to unsafe.Pointer conversion.
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// Texture implements hal.Texture for Metal.
type Texture struct {
	raw        ID // id<MTLTexture>
	format     gputypes.TextureFormat
	width      uint32
	height     uint32
	depth      uint32
	mipLevels  uint32
	samples    uint32
	dimension  gputypes.TextureDimension
	usage      gputypes.TextureUsage
	device     *Device
	isExternal bool
	// isShared is true when this texture was created with MTLStorageModeShared
	// (Apple Silicon UMA only). Shared textures support direct CPU writes via
	// replaceRegion:.
	isShared bool
}

// IsHostWritable reports whether this texture was allocated with Shared storage,
// meaning the CPU can write to it directly without a staging buffer or GPU blit.
// Used by pendingWrites to bypass the staging belt on Apple Silicon (UMA).
func (t *Texture) IsHostWritable() bool { return t.isShared }

// CurrentUsage returns 0 — Metal has no explicit resource state tracking.
func (t *Texture) CurrentUsage() gputypes.TextureUsage { return 0 }
func (t *Texture) AddPendingRef()                      {}
func (t *Texture) DecPendingRef()                      {}

// Destroy releases the texture.
func (t *Texture) Destroy() {
	if t.device != nil {
		t.device.DestroyTexture(t)
	}
}

// NativeHandle returns the raw MTLTexture handle.
func (t *Texture) NativeHandle() uint64 { return uint64(t.raw) }

// metalTexture is the Texture behind t, a texture or a surface texture, or
// nil.
func metalTexture(t hal.Texture) *Texture {
	switch t := t.(type) {
	case *Texture:
		return t
	case *SurfaceTexture:
		if t != nil {
			return t.texture
		}
	}
	return nil
}

// TextureView implements hal.TextureView for Metal.
type TextureView struct {
	raw     ID // id<MTLTexture>
	texture *Texture
	device  *Device
}

// Destroy releases the texture view.
func (v *TextureView) Destroy() {
	if v.device != nil {
		v.device.DestroyTextureView(v)
	}
}

// NativeHandle returns the raw MTLTexture handle (view is also a texture).
func (v *TextureView) NativeHandle() uint64 { return uint64(v.raw) }

// Sampler implements hal.Sampler for Metal.
type Sampler struct {
	raw    ID // id<MTLSamplerState>
	device *Device
}

// Destroy releases the sampler.
func (s *Sampler) Destroy() {
	if s.device != nil {
		s.device.DestroySampler(s)
	}
}

// NativeHandle returns the raw MTLSamplerState handle.
func (s *Sampler) NativeHandle() uint64 { return uint64(s.raw) }

// ShaderModule implements hal.ShaderModule for Metal.
//
// WGSL is parsed and lowered to naga IR here. MSL is written later, when the
// pipeline is created, because the slots of the resources and of naga's
// `_buffer_sizes` argument come from the pipeline layout.
//
// Reference: Rust wgpu-hal metal/device.rs:138-145 (load_shader takes
// `layout: &super::PipelineLayout`).
type ShaderModule struct {
	source         hal.ShaderSource
	irModule       *ir.Module // nil unless the module was created from WGSL
	device         *Device
	workgroupSizes map[string][3]uint32 // entry point name -> workgroup size
}

// Destroy releases the shader module.
func (m *ShaderModule) Destroy() {
	if m.device != nil {
		m.device.DestroyShaderModule(m)
	}
}

// BindGroupLayout implements hal.BindGroupLayout for Metal.
type BindGroupLayout struct {
	entries []gputypes.BindGroupLayoutEntry
	device  *Device

	// Per-type resource counts for this layout.
	// Used by PipelineLayout to compute cumulative slot offsets across groups.
	bufferCount  int
	textureCount int
	samplerCount int
}

// Destroy releases the bind group layout.
func (l *BindGroupLayout) Destroy() {
	if l.device != nil {
		l.device.DestroyBindGroupLayout(l)
	}
}

// BindGroup implements hal.BindGroup for Metal.
type BindGroup struct {
	layout  *BindGroupLayout
	entries []gputypes.BindGroupEntry
	device  *Device
}

// Destroy releases the bind group.
func (g *BindGroup) Destroy() {
	if g.device != nil {
		g.device.DestroyBindGroup(g)
	}
}

// GroupSlotOffsets holds the cumulative Metal slot offsets for a single bind group.
// These offsets are the starting [[buffer(N)]], [[texture(M)]], [[sampler(K)]]
// indices for each group, computed from the resource counts of all preceding groups.
//
// Reference: Rust wgpu-hal metal/mod.rs BindGroupLayoutInfo.base_resource_indices.
type GroupSlotOffsets struct {
	Buffers  int
	Textures int
	Samplers int
}

// PipelineLayout implements hal.PipelineLayout for Metal.
type PipelineLayout struct {
	layouts []hal.BindGroupLayout
	device  *Device

	// groupOffsets holds the cumulative per-type slot offsets for each bind group.
	// groupOffsets[i] contains the starting Metal indices for group i,
	// computed by summing resource counts of groups 0..i-1.
	//
	// Reference: Rust wgpu-hal metal/mod.rs PipelineLayout.bind_group_infos.
	groupOffsets []GroupSlotOffsets

	// totalBuffers is how many buffer bindings all bind group layouts declare
	// together. It is also the first free Metal buffer slot, which is where
	// naga's `_buffer_sizes` argument goes.
	//
	// Reference: Rust wgpu-hal metal/device.rs:851-856
	// (info.sizes_buffer = Some(info.counters.buffers)).
	totalBuffers int

	// resources is the slot of each binding, numbered as groupOffsets
	// numbers them, for the MSL of the pipelines made with the layout.
	resources map[ir.ResourceBinding]msl.BindTarget
}

// groupSlots returns where the slots of bind group index start, and false
// when the layout has no group index with room for bg's resources: a group
// a pipeline's layout doesn't have stays bound for a later pipeline, and
// must not land on the slots of another.
func (l *PipelineLayout) groupSlots(index uint32, bg *BindGroup) (GroupSlotOffsets, bool) {
	if l == nil || int(index) >= len(l.groupOffsets) || bg.layout == nil {
		return GroupSlotOffsets{}, false
	}
	var buffers, textures, samplers int
	if bgl, ok := l.layouts[index].(*BindGroupLayout); ok && bgl != nil {
		buffers, textures, samplers = bgl.bufferCount, bgl.textureCount, bgl.samplerCount
	}
	if buffers != bg.layout.bufferCount || textures != bg.layout.textureCount || samplers != bg.layout.samplerCount {
		return GroupSlotOffsets{}, false
	}
	return l.groupOffsets[index], true
}

// Destroy releases the pipeline layout.
func (l *PipelineLayout) Destroy() {
	if l.device != nil {
		l.device.DestroyPipelineLayout(l)
	}
}

// RenderPipeline implements hal.RenderPipeline for Metal.
type RenderPipeline struct {
	raw           ID // id<MTLRenderPipelineState>
	device        *Device
	layout        *PipelineLayout // for SetBindGroup slot offset lookup
	cullMode      MTLCullMode
	frontFace     MTLWinding
	depthClipMode MTLDepthClipMode
	primitive     MTLPrimitiveType // what draws assemble, from the topology
	icbCompatible bool

	depthStencil    ID // id<MTLDepthStencilState>
	depthBias       float32
	depthSlopeScale float32
	depthClamp      float32

	// vertexSizes and fragmentSizes list the bindings whose byte sizes each
	// stage reads from naga's `_buffer_sizes` argument, which the stages
	// take at buffer slot sizesSlot, as ComputePipeline's sizesBindings.
	vertexSizes, fragmentSizes []ir.ResourceBinding
	sizesSlot                  int

	// vertexBuffers is how many vertex buffer slots the pipeline reads.
	vertexBuffers uint32
	// pulledBuffers are the vertex buffer slots whose bound sizes follow
	// vertexSizes in the vertex stage's `_buffer_sizes`.
	pulledBuffers []uint32

	// clampsFragDepth is set when the fragment stage clamps the frag_depth
	// it writes to the range it takes at buffer slot sizesSlot+1.
	clampsFragDepth bool
}

// Destroy releases the render pipeline.
func (p *RenderPipeline) Destroy() {
	if p.device != nil {
		p.device.DestroyRenderPipeline(p)
	}
}

// ComputePipeline implements hal.ComputePipeline for Metal.
type ComputePipeline struct {
	raw           ID // id<MTLComputePipelineState>
	device        *Device
	layout        *PipelineLayout // for SetBindGroup slot offset lookup
	workgroupSize MTLSize         // workgroup size from shader

	// sizesBindings lists the bindings whose byte size the kernel needs, in the
	// field order that naga uses. It is empty if the kernel has no
	// runtime-sized array. sizesSlot is the buffer slot that the MSL was
	// compiled with, and it comes from PipelineLayout.totalBuffers.
	sizesBindings []ir.ResourceBinding
	sizesSlot     int
}

// Destroy releases the compute pipeline.
func (p *ComputePipeline) Destroy() {
	if p.device != nil {
		p.device.DestroyComputePipeline(p)
	}
}

// QuerySet implements hal.QuerySet for Metal: an occlusion query set, whose
// results are a buffer of a uint64 a query.
type QuerySet struct {
	buffer ID // id<MTLBuffer>
	count  uint32
	device *Device
}

// Destroy releases the query set.
func (q *QuerySet) Destroy() {
	if q.device != nil {
		q.device.DestroyQuerySet(q)
	}
}

// Fence implements hal.Fence for Metal using MTLSharedEvent.
//
// MTLSharedEvent (unlike MTLEvent) exposes signaledValue to the CPU,
// enabling proper blocking waits and non-blocking status queries.
// The GPU updates signaledValue when encodeSignalEvent:value: completes.
type Fence struct {
	event  ID // id<MTLSharedEvent>
	device *Device
	// base is the event's value at the last ResetFence: the fence's value
	// is the event's less base.
	base atomic.Uint64
}

// Destroy releases the fence.
func (f *Fence) Destroy() {
	if f.device != nil {
		f.device.DestroyFence(f)
	}
}
