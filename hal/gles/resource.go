// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"cmp"
	"fmt"
	"slices"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/gogpu/naga/glsl"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// Surface and SurfaceTexture are defined in platform-specific files (resource_windows.go, resource_linux.go)

// Buffer implements hal.Buffer for OpenGL.
type Buffer struct {
	id     uint32 // GL buffer object ID
	target uint32 // GL_ARRAY_BUFFER, GL_UNIFORM_BUFFER, etc.
	size   uint64
	usage  gputypes.BufferUsage
	glCtx  *gl.Context
	// mapped is the CPU shadow mappings point into, kept between the
	// mappings of a MapWrite buffer, which only its mappings write: a
	// mapping holds the buffer's contents, as WebGPU's do.
	mapped []byte
	// mapOffset and mapSize are the range of mapped the mapping covers,
	// what Unmap uploads when writeBack is set: for a mapping at creation
	// or of a MapWrite buffer.
	mapOffset, mapSize uint64
	writeBack          bool
}

// Destroy releases the buffer.
func (b *Buffer) Destroy() {
	if b.id != 0 && b.glCtx != nil {
		b.glCtx.DeleteBuffers(b.id)
		b.id = 0
	}
}

// NativeHandle returns the GL buffer object ID.
func (b *Buffer) NativeHandle() uint64 { return uint64(b.id) }

// MapBuffer maps a range of a buffer through its CPU shadow, which a
// MapRead buffer's mapping reads from GL.
func (d *Device) MapBuffer(buffer hal.Buffer, offset, size uint64) (hal.BufferMapping, error) {
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil || offset+size > buf.size {
		return hal.BufferMapping{}, hal.ErrInvalidMapRange
	}
	if !buf.writeBack { // a mapping at creation is ready
		if buf.mapped == nil {
			buf.mapped = make([]byte, buf.size)
		}
		buf.mapOffset, buf.mapSize = offset, size
		buf.writeBack = buf.usage&gputypes.BufferUsageMapWrite != 0
		if buf.usage&gputypes.BufferUsageMapRead != 0 && buf.id != 0 && size > 0 {
			glCtx := d.ctx.Lock()
			glCtx.BindBuffer(gl.COPY_READ_BUFFER, buf.id)
			if p := glCtx.MapBufferRange(gl.COPY_READ_BUFFER, int(offset), int(size), gl.MAP_READ_BIT); p != 0 {
				copy(buf.mapped[offset:], unsafe.Slice(*(**byte)(unsafe.Pointer(&p)), size))
				glCtx.UnmapBuffer(gl.COPY_READ_BUFFER)
			}
			glCtx.BindBuffer(gl.COPY_READ_BUFFER, 0)
			d.ctx.Unlock()
		}
	}
	ptr := unsafe.Pointer(unsafe.SliceData(buf.mapped)) // an empty range at the end has no byte of its own
	if offset < uint64(len(buf.mapped)) {
		ptr = unsafe.Pointer(&buf.mapped[offset])
	}
	return hal.BufferMapping{Ptr: ptr, IsCoherent: true}, nil
}

// UnmapBuffer ends a mapping, uploading the range of a writable one.
func (d *Device) UnmapBuffer(buffer hal.Buffer) error {
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil || buf.mapped == nil {
		return nil
	}
	if buf.writeBack && buf.id != 0 && buf.mapSize > 0 {
		glCtx := d.ctx.Lock()
		glCtx.BindBuffer(buf.target, buf.id)
		glCtx.BufferSubData(buf.target, int(buf.mapOffset), int(buf.mapSize), unsafe.Pointer(&buf.mapped[buf.mapOffset]))
		glCtx.BindBuffer(buf.target, 0)
		d.ctx.Unlock()
	}
	buf.writeBack = false
	if buf.usage&gputypes.BufferUsageMapWrite == 0 {
		buf.mapped = nil
	}
	return nil
}

// Texture implements hal.Texture for OpenGL.
type Texture struct {
	id          uint32 // GL texture object ID
	target      uint32 // GL_TEXTURE_2D, GL_TEXTURE_2D_MULTISAMPLE, etc., or GL_RENDERBUFFER
	format      gputypes.TextureFormat
	glf         glFormat
	dimension   gputypes.TextureDimension
	size        hal.Extent3D
	mipLevels   uint32
	sampleCount uint32 // 1 for regular textures, >1 for MSAA
	usage       gputypes.TextureUsage
	immutable   bool // made with glTexStorage*, which texture views take
	glCtx       *gl.Context

	// baseLevel and maxLevel are the mip range TEXTURE_BASE_LEVEL and
	// TEXTURE_MAX_LEVEL hold, the range of the view bound last, and
	// stencilMode whether DEPTH_STENCIL_TEXTURE_MODE reads the stencil.
	baseLevel, maxLevel int32
	stencilMode         bool
}

// layers is how many array layers (or cube faces) the texture has; a 3D
// texture's slices are not layers.
func (t *Texture) layers() uint32 {
	if t.dimension == gputypes.TextureDimension3D {
		return 1
	}
	return max(1, t.size.DepthOrArrayLayers)
}

// layered reports whether the texture's levels are arrays of images, which
// are attached and written one layer or slice at a time.
func (t *Texture) layered() bool {
	return t.target == gl.TEXTURE_2D_ARRAY || t.target == gl.TEXTURE_3D
}

// mipSize is the width and height of a mip level.
func (t *Texture) mipSize(mip uint32) (width, height uint32) {
	return max(1, t.size.Width>>mip), max(1, t.size.Height>>mip)
}

// CurrentUsage returns 0 — GLES has no explicit resource state tracking.
func (t *Texture) CurrentUsage() gputypes.TextureUsage { return 0 }
func (t *Texture) AddPendingRef()                      {}
func (t *Texture) DecPendingRef()                      {}

// Destroy releases the texture.
func (t *Texture) Destroy() {
	if t.glCtx == nil || t.id == 0 {
		return
	}
	if t.target == gl.RENDERBUFFER {
		t.glCtx.DeleteRenderbuffers(t.id)
	} else {
		t.glCtx.DeleteTextures(t.id)
	}
	t.id = 0
}

// NativeHandle returns the GL texture object ID.
func (t *Texture) NativeHandle() uint64 { return uint64(t.id) }

// TextureView implements hal.TextureView for OpenGL. A view the shaders
// take as its texture is no GL object: its mip range and aspect are set on
// the texture when it is bound. One of another dimension or of some of the
// layers is a texture view, id, where the context has them. A view
// attached to a framebuffer attaches its texture's mip level and layer.
type TextureView struct {
	texture    *Texture
	dimension  gputypes.TextureViewDimension
	aspect     gputypes.TextureAspect
	baseMip    uint32
	mipCount   uint32
	baseLayer  uint32
	layerCount uint32
	id, target uint32          // the texture view and its target, 0 for none
	handle     uint64          // NativeHandle, which bind group entries find the view by
	isSurface  bool            // true for default framebuffer (surface texture)
	surfaceTex *SurfaceTexture // non-nil only when isSurface is true
}

// views are the texture views by their handles.
var (
	views       sync.Map // uint64 -> *TextureView
	lastViewKey atomic.Uint64
)

// Destroy forgets the view's handle.
func (v *TextureView) Destroy() {
	if v.handle != 0 {
		views.Delete(v.handle)
	}
}

// NativeHandle returns the handle that stands for the view in a bind group
// entry, 0 for a surface view.
func (v *TextureView) NativeHandle() uint64 { return v.handle }

// viewByHandle returns the view a bind group entry names, nil for none.
func viewByHandle(handle uint64) *TextureView {
	v, _ := views.Load(handle)
	view, _ := v.(*TextureView)
	return view
}

// Sampler implements hal.Sampler for OpenGL using GL sampler objects (GL 3.3+).
type Sampler struct {
	id    uint32 // GL sampler object ID (0 if sampler objects not supported)
	glCtx *gl.Context
}

// Destroy releases the GL sampler object.
func (s *Sampler) Destroy() {
	if s.id != 0 && s.glCtx != nil {
		s.glCtx.DeleteSamplers(s.id)
		s.id = 0
	}
}

// NativeHandle returns the GL sampler object ID.
func (s *Sampler) NativeHandle() uint64 { return uint64(s.id) }

// ShaderModule implements hal.ShaderModule for OpenGL.
type ShaderModule struct {
	vertexID   uint32 // GL shader object ID for vertex
	fragmentID uint32 // GL shader object ID for fragment
	computeID  uint32 // GL shader object ID for compute
	source     hal.ShaderSource
	glCtx      *gl.Context
}

// Destroy releases the shader module.
func (m *ShaderModule) Destroy() {
	if m.vertexID != 0 && m.glCtx != nil {
		m.glCtx.DeleteShader(m.vertexID)
	}
	if m.fragmentID != 0 && m.glCtx != nil {
		m.glCtx.DeleteShader(m.fragmentID)
	}
	if m.computeID != 0 && m.glCtx != nil {
		m.glCtx.DeleteShader(m.computeID)
	}
}

// BindGroupLayout implements hal.BindGroupLayout for OpenGL.
type BindGroupLayout struct {
	entries []gputypes.BindGroupLayoutEntry
}

// Destroy is a no-op for bind group layouts.
func (l *BindGroupLayout) Destroy() {}

// BindGroup implements hal.BindGroup for OpenGL.
type BindGroup struct {
	layout   *BindGroupLayout
	bindings []groupBinding // by binding number, which dynamic offsets go by
}

// groupBinding is a bind group entry with what binding it takes from its
// layout entry.
type groupBinding struct {
	binding uint32
	class   bindingClass

	buffer       uint32 // GL buffer name
	offset, size int
	dynamic      bool // the buffer takes a dynamic offset

	view    *TextureView
	sampler uint32 // GL sampler name

	// access and imageFormat are how a storage texture's image unit is
	// bound: GL_READ_ONLY, GL_WRITE_ONLY or GL_READ_WRITE, and the layout's
	// format.
	access, imageFormat uint32
}

// newBindGroup resolves desc's entries against its layout.
func newBindGroup(desc *BindGroupDescriptor) (*BindGroup, error) {
	layout, ok := desc.Layout.(*BindGroupLayout)
	if !ok {
		return nil, fmt.Errorf("gles: invalid bind group layout type")
	}
	g := &BindGroup{layout: layout, bindings: make([]groupBinding, 0, len(desc.Entries))}
	for _, e := range desc.Entries {
		b := groupBinding{binding: e.Binding}
		for _, le := range layout.entries {
			if le.Binding == e.Binding {
				b.class = classifyBindGroupEntry(le)
				b.dynamic = le.Buffer != nil && le.Buffer.HasDynamicOffset
				if st := le.StorageTexture; st != nil {
					b.access = imageAccess(st.Access)
					f, _ := glFormats[st.Format]
					b.imageFormat = f.internal
				}
				break
			}
		}
		switch res := e.Resource.(type) {
		case gputypes.BufferBinding:
			b.buffer, b.offset, b.size = uint32(res.Buffer), int(res.Offset), int(res.Size)
		case gputypes.TextureViewBinding:
			if b.view = viewByHandle(res.TextureView); b.view == nil {
				return nil, fmt.Errorf("gles: binding %d: unknown texture view", e.Binding)
			}
			if err := b.view.bindable(b.class); err != nil {
				return nil, fmt.Errorf("gles: binding %d: %w", e.Binding, err)
			}
		case gputypes.SamplerBinding:
			b.sampler = uint32(res.Sampler)
		}
		g.bindings = append(g.bindings, b)
	}
	slices.SortFunc(g.bindings, func(a, b groupBinding) int { return cmp.Compare(a.binding, b.binding) })
	return g, nil
}

// imageAccess is the image unit access of a storage texture's access mode.
func imageAccess(access gputypes.StorageTextureAccess) uint32 {
	switch access {
	case gputypes.StorageTextureAccessReadOnly:
		return gl.READ_ONLY
	case gputypes.StorageTextureAccessReadWrite:
		return gl.READ_WRITE
	}
	return gl.WRITE_ONLY
}

// Destroy is a no-op for bind groups.
func (g *BindGroup) Destroy() {}

// BindGroupLayoutInfo stores per-group binding-to-slot mapping computed at
// PipelineLayout creation time. Matches Rust wgpu-hal BindGroupLayoutInfo.
// Each entry in BindingToSlot is indexed by the WGSL binding number and maps
// to the sequential GL slot index for that resource type. 0xFF means unused.
type BindGroupLayoutInfo struct {
	BindingToSlot []uint8 // indexed by binding number, 0xFF = unused
}

// PipelineLayout implements hal.PipelineLayout for OpenGL.
// Stores pre-computed per-type sequential binding indices (Rust wgpu pattern).
// The BindingMap is used by naga GLSL writer, GroupInfos by SetBindGroup at runtime.
type PipelineLayout struct {
	bindGroupLayouts []*BindGroupLayout
	// groupInfos stores per-group binding-to-slot tables computed from per-type
	// sequential counters (samplers, textures, images, uniform buffers, storage buffers).
	// Matches Rust wgpu-hal/src/gles/mod.rs BindGroupLayoutInfo.
	groupInfos []BindGroupLayoutInfo
	// bindingMap maps (group, binding) to flat GL slot index for naga GLSL writer.
	// Computed simultaneously with groupInfos in CreatePipelineLayout.
	bindingMap map[glsl.BindingMapKey]uint8
}

// Destroy is a no-op for pipeline layouts.
func (l *PipelineLayout) Destroy() {}

// ColorTargetDesc describes per-render-target blend and write mask state.
// Matches Rust wgpu-hal GLES ColorTargetDesc (mod.rs:770-773).
type ColorTargetDesc struct {
	Blend     *gputypes.BlendState
	WriteMask gputypes.ColorWriteMask
}

// RenderPipeline implements hal.RenderPipeline for OpenGL.
type RenderPipeline struct {
	programID uint32 // GL program object ID
	layout    *PipelineLayout
	glCtx     *gl.Context

	// Pipeline state
	primitiveTopology gputypes.PrimitiveTopology
	cullMode          gputypes.CullMode
	frontFace         gputypes.FrontFace
	depthStencil      *hal.DepthStencilState
	multisample       gputypes.MultisampleState
	sampleMask        uint32 // as SetPipelineStateCommand's
	attribs           uint32 // the shader locations its vertex buffers feed

	// unclippedDepth turns GL_DEPTH_CLAMP on, on a context that has it
	// (depthClamp).
	depthClamp, unclippedDepth bool

	// Per-target blend and write mask state for MRT.
	// Matches Rust wgpu-hal GLES RenderPipeline.color_targets (mod.rs:798).
	// Targets that all blend and mask alike are applied with the global
	// calls; perTargetBlend is set when they differ, which takes GL's per
	// draw buffer calls (GL 4.0, ES 3.2, *_draw_buffers_indexed).
	colorTargets   []ColorTargetDesc
	perTargetBlend bool

	// Vertex buffer layouts from the pipeline descriptor.
	// OpenGL requires explicit glVertexAttribPointer calls to configure
	// how vertex data is interpreted. This is stored here so that
	// SetVertexBuffer can configure attributes using the pipeline's layout.
	vertexBuffers []gputypes.VertexBufferLayout

	// samplerBindMap maps texture unit indices to sampler unit indices.
	// Built from naga GLSL TranslationInfo.TextureMappings at pipeline creation.
	// When binding textures, the associated sampler must be bound to the SAME
	// texture unit (not the sampler's own WGSL binding). This is because naga
	// GLSL generates combined sampler2D on the texture's binding.
	// Matches Rust wgpu-hal GLES SamplerBindMap pattern.
	samplerBindMap [maxTextureSlots]int8 // -1 = no sampler, otherwise = sampler glBinding

	// firstInstanceLoc is the location of naga's naga_vs_first_instance
	// uniform, -1 when the vertex shader has none, and firstInstance the
	// value the program has in it.
	firstInstanceLoc int32
	firstInstance    uint32

	// clampsFragDepth is set when the fragment shader clamps the frag_depth
	// it writes to the range, min and max at fragDepthRangeLocs.
	clampsFragDepth    bool
	fragDepthRangeLocs [2]int32
}

// setFirstInstance gives the program in use, p's, the first instance of the
// draw coming: what WGSL's instance_index starts at.
func (p *RenderPipeline) setFirstInstance(ctx *gl.Context, first uint32) {
	if p != nil && p.firstInstanceLoc >= 0 && p.firstInstance != first {
		ctx.Uniform1ui(p.firstInstanceLoc, first)
		p.firstInstance = first
	}
}

const maxTextureSlots = 32

// samplerBindMapFor maps each texture unit to the unit of the sampler the
// combined sampler2D in infos samples it with, -1 for none, by the GL slots
// of layout.
func samplerBindMapFor(layout *PipelineLayout, infos ...glsl.TranslationInfo) (m [maxTextureSlots]int8) {
	for i := range m {
		m[i] = -1
	}
	for _, info := range infos {
		for _, tm := range info.TextureMappings {
			if tm.SamplerBinding == nil {
				continue
			}
			texKey := glsl.BindingMapKey{Group: tm.TextureBinding.Group, Binding: tm.TextureBinding.Binding}
			samplerKey := glsl.BindingMapKey{Group: tm.SamplerBinding.Group, Binding: tm.SamplerBinding.Binding}
			texUnit, texOk := layout.bindingMap[texKey]
			samplerUnit, samplerOk := layout.bindingMap[samplerKey]
			if texOk && samplerOk && texUnit < maxTextureSlots {
				m[texUnit] = int8(samplerUnit)
			}
		}
	}
	return m
}

// Destroy releases the render pipeline.
func (p *RenderPipeline) Destroy() {
	if p.programID != 0 && p.glCtx != nil {
		p.glCtx.DeleteProgram(p.programID)
		p.programID = 0
	}
}

// ComputePipeline implements hal.ComputePipeline for OpenGL.
type ComputePipeline struct {
	programID      uint32
	layout         *PipelineLayout
	glCtx          *gl.Context
	samplerBindMap [maxTextureSlots]int8 // as RenderPipeline's
}

// Destroy releases the compute pipeline.
func (p *ComputePipeline) Destroy() {
	if p.programID != 0 && p.glCtx != nil {
		p.glCtx.DeleteProgram(p.programID)
		p.programID = 0
	}
}

// glFence holds a GL sync object paired with its submission value.
// Matches Rust wgpu-hal/src/gles/fence.rs GLFence struct.
type glFence struct {
	sync  uintptr // GL sync object handle from glFenceSync
	value uint64  // submission index this fence was signaled with
}

// Fence implements hal.Fence using GL sync objects (glFenceSync).
// Tracks pending GL sync objects and polls their completion status.
// Matches Rust wgpu-hal/src/gles/fence.rs Fence struct.
type Fence struct {
	lastCompleted atomic.Uint64 // highest known completed value
	signaled      uint64        // the value Signal was given last
	pending       []glFence     // GL sync objects awaiting completion
	glCtx         *gl.Context
}

// NewFence creates a new fence.
func NewFence(glCtx *gl.Context) *Fence {
	return &Fence{
		glCtx: glCtx,
	}
}

// Signal inserts a GL fence sync object into the command stream at the given value.
// Must be called on the GL thread BEFORE glFlush — the flush sends both the
// preceding commands and this fence to the GPU together.
// Returns an error if glFenceSync fails (typically OOM).
// Matches Rust wgpu-hal/src/gles/fence.rs Fence::signal.
func (f *Fence) Signal(value uint64) error {
	f.signaled = value
	if f.glCtx == nil || !f.glCtx.SupportsFenceSync() {
		f.lastCompleted.Store(value)
		return nil
	}
	sync := f.glCtx.FenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)
	if sync == 0 {
		return fmt.Errorf("gles: glFenceSync failed (OOM)")
	}
	f.pending = append(f.pending, glFence{sync: sync, value: value})
	return nil
}

// Signaled is the value Signal was given last: the queue's latest submission.
func (f *Fence) Signaled() uint64 { return f.signaled }

// GetLatest polls pending sync objects and returns the highest completed value.
// Matches Rust wgpu-hal/src/gles/fence.rs Fence::get_latest.
func (f *Fence) GetLatest() uint64 {
	maxValue := f.lastCompleted.Load()

	if f.glCtx == nil || !f.glCtx.SupportsFenceSync() {
		return maxValue
	}

	for _, gf := range f.pending {
		if gf.value <= maxValue {
			continue // already known complete
		}
		status := f.glCtx.GetSyncStatus(gf.sync)
		if status == gl.SIGNALED {
			maxValue = gf.value
		} else {
			// Anything after the first unsignalled is guaranteed unsignalled.
			break
		}
	}

	// Cache the latest value to avoid redundant queries.
	for {
		old := f.lastCompleted.Load()
		if maxValue <= old {
			break
		}
		if f.lastCompleted.CompareAndSwap(old, maxValue) {
			break
		}
	}

	return maxValue
}

// Maintain cleans up completed sync objects.
// Matches Rust wgpu-hal/src/gles/fence.rs Fence::maintain.
func (f *Fence) Maintain() {
	if f.glCtx == nil || !f.glCtx.SupportsFenceSync() {
		return
	}

	latest := f.GetLatest()

	// Delete completed sync objects.
	n := 0
	for _, gf := range f.pending {
		if gf.value <= latest {
			f.glCtx.DeleteSync(gf.sync)
		} else {
			f.pending[n] = gf
			n++
		}
	}
	f.pending = f.pending[:n]
}

// Wait waits for the fence to reach the specified value with timeout.
// Matches Rust wgpu-hal/src/gles/fence.rs Fence::wait.
func (f *Fence) Wait(waitValue uint64, timeout time.Duration) bool {
	if f.lastCompleted.Load() >= waitValue {
		return true
	}

	if f.glCtx == nil || !f.glCtx.SupportsFenceSync() {
		return f.lastCompleted.Load() >= waitValue
	}

	// Find a matching pending fence.
	var target *glFence
	for i := range f.pending {
		if f.pending[i].value >= waitValue {
			target = &f.pending[i]
			break
		}
	}
	if target == nil {
		return false // value not yet signaled
	}

	status := f.glCtx.ClientWaitSync(target.sync, gl.SYNC_FLUSH_COMMANDS_BIT, uint64(max(timeout, 0).Nanoseconds()))

	signaled := status == gl.ALREADY_SIGNALED || status == gl.CONDITION_SATISFIED
	if signaled {
		// Update last completed.
		for {
			old := f.lastCompleted.Load()
			if waitValue <= old {
				break
			}
			if f.lastCompleted.CompareAndSwap(old, waitValue) {
				break
			}
		}
	}
	return signaled
}

// GetValue returns the current fence value (non-blocking poll).
func (f *Fence) GetValue() uint64 {
	return f.GetLatest()
}

// Reset resets the fence to the unsignaled state.
func (f *Fence) Reset() {
	f.lastCompleted.Store(0)
	// Clean up any pending sync objects.
	if f.glCtx != nil {
		for _, gf := range f.pending {
			f.glCtx.DeleteSync(gf.sync)
		}
	}
	f.pending = f.pending[:0]
}

// Destroy releases all fence resources.
// Matches Rust wgpu-hal/src/gles/fence.rs Fence::destroy.
func (f *Fence) Destroy() {
	if f.glCtx != nil {
		for _, gf := range f.pending {
			f.glCtx.DeleteSync(gf.sync)
		}
	}
	f.pending = nil
}

// QuerySet implements hal.QuerySet for OpenGL.
// Stores GL query object IDs and the target type (GL_TIMESTAMP or GL_ANY_SAMPLES_PASSED).
// Matches Rust wgpu-hal/src/gles/mod.rs QuerySet struct.
type QuerySet struct {
	queries []uint32 // GL query object IDs
	target  uint32   // GL_TIMESTAMP or GL_ANY_SAMPLES_PASSED_CONSERVATIVE
	glCtx   *gl.Context
}

// Target returns the GL query target type (GL_TIMESTAMP or GL_ANY_SAMPLES_PASSED).
func (q *QuerySet) Target() uint32 { return q.target }

// Destroy releases all GL query objects.
func (q *QuerySet) Destroy() {
	if q.glCtx != nil && len(q.queries) > 0 {
		q.glCtx.DeleteQueries(int32(len(q.queries)), &q.queries[0])
	}
	q.queries = nil
}

// CreateQuerySet creates a query set with GL query objects.
// Supports QueryTypeTimestamp on desktop GL (GL 3.3, ARB_timer_query).
// Matches Rust wgpu-hal/src/gles/device.rs create_query_set.
func (d *Device) CreateQuerySet(desc *hal.QuerySetDescriptor) (hal.QuerySet, error) {
	if desc == nil {
		return nil, fmt.Errorf("gles: query set descriptor is nil")
	}

	glCtx := d.ctx.Lock()
	defer d.ctx.Unlock()

	// Determine GL query target.
	var target uint32
	switch desc.Type {
	case hal.QueryTypeTimestamp:
		if !glCtx.SupportsTimestampQueries() {
			return nil, hal.ErrTimestampsNotSupported
		}
		target = gl.TIMESTAMP
	default:
		return nil, fmt.Errorf("gles: unsupported query type %d", desc.Type)
	}

	// Create GL query objects.
	queries := make([]uint32, desc.Count)
	for i := uint32(0); i < desc.Count; i++ {
		q := glCtx.GenQueries(1)
		if q == 0 {
			// Clean up already-created queries on failure.
			for j := uint32(0); j < i; j++ {
				glCtx.DeleteQueries(1, &queries[j])
			}
			return nil, fmt.Errorf("gles: failed to create query object %d/%d", i, desc.Count)
		}
		queries[i] = q
	}

	return &QuerySet{
		queries: queries,
		target:  target,
		glCtx:   glCtx,
	}, nil
}

// DestroyQuerySet destroys a query set and releases GL query objects.
func (d *Device) DestroyQuerySet(qs hal.QuerySet) {
	q, ok := qs.(*QuerySet)
	if !ok || q == nil || d.ctx == nil {
		return
	}
	_ = d.ctx.Lock()
	defer d.ctx.Unlock()
	q.Destroy()
}

// NativeHandle returns 0 (no single native handle for a query set).
func (q *QuerySet) NativeHandle() uint64 { return 0 }
