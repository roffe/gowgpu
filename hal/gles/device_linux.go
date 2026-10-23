// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

package gles

import (
	"fmt"
	"time"

	"github.com/gogpu/naga/glsl"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// Device implements hal.Device for OpenGL on Linux.
// Holds a shared *AdapterContext (owned by Instance or Surface). All GL
// operations acquire the context lock, do MakeCurrent, execute, and unlock.
type Device struct {
	ctx             *AdapterContext
	fence           *Fence // the queue's, which stands for the work in WaitIdle on the helper context
	displayHandle   uintptr
	windowHandle    uintptr
	vao             uint32        // persistent VAO (Core Profile requires one bound)
	fbs             *framebuffers // made with the VAO, on the queue's context
	maxTextureUnits int32         // GL_MAX_TEXTURE_IMAGE_UNITS (queried at init)
	maxMSAA         int32         // GL_MAX_SAMPLES (validate MSAA in CreateTexture)
	formats         formatCaps
	depthClamp      bool // GL_DEPTH_CLAMP, which unclippedDepth turns on
	// independentBlend is blending and write masks per draw buffer (GL 4.0,
	// ES 3.2, OES/EXT_draw_buffers_indexed).
	independentBlend bool
	anisotropy       bool // TEXTURE_MAX_ANISOTROPY
	// fixedIndexRestart is primitive restart at the largest index (ES 3.0,
	// GL 4.3, ARB_ES3_compatibility); without it strips restart through
	// glPrimitiveRestartIndex.
	fixedIndexRestart bool
	// drawIndirect is glDraw*Indirect (GL 4.0, ARB_draw_indirect, ES 3.1).
	drawIndirect bool

	// glslVersion is the target GLSL version for shader compilation, detected
	// from the adapter's GL_SHADING_LANGUAGE_VERSION at Open time.
	// Propagated to naga GLSL writer for correct #version directive and
	// version-gated features (layout(binding=N) needs GLSL >= 420).
	glslVersion glsl.Version

	// shaderBindingLayout is true when the driver supports layout(binding=N)
	// in shaders (GLSL >= 420 desktop or >= 310 ES). When false, bindings must
	// be assigned at runtime after linking via glGetUniformBlockIndex etc.
	// Mirrors Rust wgpu-hal PrivateCapabilities::SHADER_BINDING_LAYOUT.
	shaderBindingLayout bool
}

// CreateBuffer creates a GPU buffer.
func (d *Device) CreateBuffer(desc *BufferDescriptor) (hal.Buffer, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: buffer descriptor is nil in GLES.CreateBuffer — core validation gap")
	}

	glCtx := d.ctx.Lock()
	defer d.ctx.Unlock()

	id := glCtx.GenBuffers(1)

	// Determine GL buffer target from usage
	target := uint32(gl.ARRAY_BUFFER)
	switch {
	case desc.Usage&gputypes.BufferUsageIndex != 0:
		target = gl.ELEMENT_ARRAY_BUFFER
	case desc.Usage&gputypes.BufferUsageUniform != 0:
		target = gl.UNIFORM_BUFFER
	case desc.Usage&gputypes.BufferUsageCopySrc != 0, desc.Usage&gputypes.BufferUsageCopyDst != 0:
		target = gl.COPY_READ_BUFFER
	}

	// Determine usage hint.
	// Rust wgpu-hal GLES uses DYNAMIC_DRAW for all writable buffers (device.rs:600):
	// "Some vendors take usage very literally and STATIC_DRAW will freeze us with
	// an empty buffer." On Intel, STATIC_DRAW causes glBufferSubData to be silently
	// ignored for per-frame uniform updates, resulting in invisible text (BUG-GLES-TEXT-001).
	usage := uint32(gl.DYNAMIC_DRAW)
	if desc.Usage&gputypes.BufferUsageMapRead != 0 {
		usage = gl.DYNAMIC_READ
	}

	glCtx.BindBuffer(target, id)
	glCtx.BufferData(target, int(desc.Size), 0, usage)
	glCtx.BindBuffer(target, 0)

	buf := &Buffer{
		id:     id,
		target: target,
		size:   desc.Size,
		usage:  desc.Usage,
		glCtx:  glCtx,
	}

	switch {
	case desc.MappedAtCreation:
		buf.mapped = make([]byte, desc.Size)
		buf.mapSize = desc.Size
		buf.writeBack = true
	case desc.Usage&gputypes.BufferUsageMapRead != 0:
		// Read mappings aren't written back, so nothing else would zero it.
		(&ClearBufferCommand{buffer: buf, size: desc.Size}).Execute(glCtx)
	}

	return buf, nil
}

// DestroyBuffer destroys a GPU buffer.
// Caller must not hold the AdapterContext lock; this method acquires it so
// glDeleteBuffers runs with a current context after Unlock unmakes between ops.
func (d *Device) DestroyBuffer(buffer hal.Buffer) {
	b, ok := buffer.(*Buffer)
	if !ok || d.ctx == nil {
		return
	}
	_ = d.ctx.Lock()
	defer d.ctx.Unlock()
	b.Destroy()
}

// CreateTexture creates a GPU texture.
func (d *Device) CreateTexture(desc *TextureDescriptor) (hal.Texture, error) {
	glCtx := d.ctx.Lock()
	defer d.ctx.Unlock()
	t, err := createTexture(glCtx, d.formats, d.maxMSAA, desc)
	if err != nil {
		return nil, err
	}
	return t, nil
}

// DestroyTexture destroys a GPU texture.
func (d *Device) DestroyTexture(texture hal.Texture) {
	t, ok := texture.(*Texture)
	if !ok || d.ctx == nil {
		return
	}
	_ = d.ctx.Lock()
	defer d.ctx.Unlock()
	t.Destroy()
}

// CreateTextureView creates a view into a texture.
// Accepts both *Texture and *SurfaceTexture (default framebuffer).
func (d *Device) CreateTextureView(texture hal.Texture, desc *TextureViewDescriptor) (hal.TextureView, error) {
	// Surface texture (default framebuffer) — return a view with no GL texture.
	if st, ok := texture.(*SurfaceTexture); ok {
		return &TextureView{
			isSurface:  true,
			surfaceTex: st,
		}, nil
	}

	t, ok := texture.(*Texture)
	if !ok {
		return nil, fmt.Errorf("gles: invalid texture type")
	}
	v, err := newTextureView(t, desc)
	if err != nil {
		return nil, err
	}
	return v, nil
}

// DestroyTextureView destroys a texture view.
func (d *Device) DestroyTextureView(view hal.TextureView) {
	if v, ok := view.(*TextureView); ok {
		v.Destroy()
	}
}

// CreateSampler creates a texture sampler using GL sampler objects (GL 3.3+).
func (d *Device) CreateSampler(desc *SamplerDescriptor) (hal.Sampler, error) {
	glCtx := d.ctx.Lock()
	defer d.ctx.Unlock()

	if desc == nil {
		return &Sampler{glCtx: glCtx}, nil
	}
	id := configureSampler(glCtx, desc, d.anisotropy)
	return &Sampler{
		id:    id,
		glCtx: glCtx,
	}, nil
}

// DestroySampler destroys a sampler.
func (d *Device) DestroySampler(sampler hal.Sampler) {
	s, ok := sampler.(*Sampler)
	if !ok || d.ctx == nil {
		return
	}
	_ = d.ctx.Lock()
	defer d.ctx.Unlock()
	s.Destroy()
}

// CreateBindGroupLayout creates a bind group layout.
func (d *Device) CreateBindGroupLayout(desc *BindGroupLayoutDescriptor) (hal.BindGroupLayout, error) {
	return &BindGroupLayout{
		entries: desc.Entries,
	}, nil
}

// DestroyBindGroupLayout destroys a bind group layout.
func (d *Device) DestroyBindGroupLayout(layout hal.BindGroupLayout) {}

// CreateBindGroup creates a bind group.
func (d *Device) CreateBindGroup(desc *BindGroupDescriptor) (hal.BindGroup, error) {
	g, err := newBindGroup(desc)
	if err != nil {
		return nil, err
	}
	return g, nil
}

// DestroyBindGroup destroys a bind group.
func (d *Device) DestroyBindGroup(group hal.BindGroup) {}

// CreatePipelineLayout creates a pipeline layout.
// Computes per-type sequential binding indices following the Rust wgpu-hal pattern
// (wgpu-hal/src/gles/device.rs:1154-1221). Five resource type counters (samplers,
// textures, images, uniform buffers, storage buffers) are incremented sequentially
// across all bind group layouts, producing a flat GL slot index per binding.
func (d *Device) CreatePipelineLayout(desc *PipelineLayoutDescriptor) (hal.PipelineLayout, error) {
	layouts := make([]*BindGroupLayout, len(desc.BindGroupLayouts))
	for i, l := range desc.BindGroupLayouts {
		layout, ok := l.(*BindGroupLayout)
		if !ok {
			return nil, fmt.Errorf("gles: invalid bind group layout at index %d", i)
		}
		layouts[i] = layout
	}

	bindingMap, groupInfos := computeBindingMap(layouts)

	return &PipelineLayout{
		bindGroupLayouts: layouts,
		groupInfos:       groupInfos,
		bindingMap:       bindingMap,
	}, nil
}

// DestroyPipelineLayout destroys a pipeline layout.
func (d *Device) DestroyPipelineLayout(layout hal.PipelineLayout) {}

// CreateShaderModule creates a shader module.
func (d *Device) CreateShaderModule(desc *ShaderModuleDescriptor) (hal.ShaderModule, error) {
	// For now, store the source - compilation happens at pipeline creation
	return &ShaderModule{
		source: desc.Source,
		glCtx:  d.ctx.GL(),
	}, nil
}

// DestroyShaderModule destroys a shader module.
func (d *Device) DestroyShaderModule(module hal.ShaderModule) {
	m, ok := module.(*ShaderModule)
	if !ok || d.ctx == nil {
		return
	}
	_ = d.ctx.Lock()
	defer d.ctx.Unlock()
	m.Destroy()
}

// CreateRenderPipeline creates a render pipeline.
func (d *Device) CreateRenderPipeline(desc *RenderPipelineDescriptor) (hal.RenderPipeline, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: render pipeline descriptor is nil in GLES.CreateRenderPipeline — core validation gap")
	}
	glCtx := d.ctx.Lock()
	defer d.ctx.Unlock()
	p, err := d.createRenderPipeline(glCtx, desc)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// DestroyRenderPipeline destroys a render pipeline.
func (d *Device) DestroyRenderPipeline(pipeline hal.RenderPipeline) {
	p, ok := pipeline.(*RenderPipeline)
	if !ok || d.ctx == nil {
		return
	}
	_ = d.ctx.Lock()
	defer d.ctx.Unlock()
	p.Destroy()
}

// CreateComputePipeline creates a compute pipeline.
func (d *Device) CreateComputePipeline(desc *ComputePipelineDescriptor) (hal.ComputePipeline, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: compute pipeline descriptor is nil in GLES.CreateComputePipeline — core validation gap")
	}
	glCtx := d.ctx.Lock()
	defer d.ctx.Unlock()
	p, err := d.createComputePipeline(glCtx, desc)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// DestroyComputePipeline destroys a compute pipeline.
func (d *Device) DestroyComputePipeline(pipeline hal.ComputePipeline) {
	p, ok := pipeline.(*ComputePipeline)
	if !ok || d.ctx == nil {
		return
	}
	_ = d.ctx.Lock()
	defer d.ctx.Unlock()
	p.Destroy()
}

// CreateQuerySet creates a query set.
// TODO: implement using GL_EXT_disjoint_timer_query for timestamp support.
func (d *Device) CreateQuerySet(_ *hal.QuerySetDescriptor) (hal.QuerySet, error) {
	return nil, hal.ErrTimestampsNotSupported
}

// DestroyQuerySet destroys a query set.
// Stub — add Lock when GL query objects are implemented.
func (d *Device) DestroyQuerySet(_ hal.QuerySet) {
}

// CreateCommandEncoder creates a command encoder.
// VAO is lazily created on first call — ensures it's allocated on the window
// surface (after Configure), not on the pbuffer (during Open). Mesa d3d12
// gallium invalidates GL objects when switching from pbuffer to window surface.
func (d *Device) CreateCommandEncoder(_ *CommandEncoderDescriptor) (hal.CommandEncoder, error) {
	glCtx := d.ctx.Lock()
	defer d.ctx.Unlock()

	if d.vao == 0 || d.fbs == nil {
		if d.ctx.OnHelper() {
			return nil, errOtherThread
		}
		if d.vao == 0 {
			d.vao = glCtx.GenVertexArrays(1)
			glCtx.BindVertexArray(d.vao)
			hal.Logger().Debug("gles: lazy VAO created", "vao", d.vao)
		}
		if d.fbs == nil {
			d.fbs = newFramebuffers(glCtx)
		}
	}
	return &CommandEncoder{
		glCtx:             glCtx,
		vao:               d.vao,
		fbs:               d.fbs,
		maxTextureUnits:   d.maxTextureUnits,
		fixedIndexRestart: d.fixedIndexRestart,
		drawIndirect:      d.drawIndirect,
	}, nil
}

// CreateFence creates a synchronization fence.
func (d *Device) CreateFence() (hal.Fence, error) {
	return NewFence(d.ctx.GL()), nil
}

// DestroyFence destroys a fence.
func (d *Device) DestroyFence(fence hal.Fence) {
	f, ok := fence.(*Fence)
	if !ok || d.ctx == nil {
		return
	}
	_ = d.ctx.Lock()
	defer d.ctx.Unlock()
	f.Destroy()
}

// Wait waits for a fence to reach the specified value.
func (d *Device) Wait(fence hal.Fence, value uint64, timeout time.Duration) (bool, error) {
	f, ok := fence.(*Fence)
	if !ok {
		return false, fmt.Errorf("gles: invalid fence type")
	}
	// glClientWaitSync needs the context current on this thread.
	if d.ctx != nil {
		d.ctx.Lock()
		defer d.ctx.Unlock()
	}
	return f.Wait(value, timeout), nil
}

// ResetFence resets a fence to the unsignaled state.
func (d *Device) ResetFence(fence hal.Fence) error {
	f, ok := fence.(*Fence)
	if !ok {
		return fmt.Errorf("gles: invalid fence type")
	}
	if d.ctx != nil {
		d.ctx.Lock()
		defer d.ctx.Unlock()
	}
	f.Reset()
	return nil
}

// GetFenceStatus returns true if the fence is signaled (non-blocking).
func (d *Device) GetFenceStatus(fence hal.Fence) (bool, error) {
	f, ok := fence.(*Fence)
	if !ok {
		return false, fmt.Errorf("gles: invalid fence type")
	}
	if d.ctx != nil {
		d.ctx.Lock()
		defer d.ctx.Unlock()
	}
	return f.GetValue() > 0, nil
}

// FreeCommandBuffer is a no-op for GLES.
// GLES doesn't have Vulkan-style command pools - commands are recorded directly.
func (d *Device) FreeCommandBuffer(cmdBuffer hal.CommandBuffer) {
	// GLES command buffers don't need explicit freeing
}

// CreateRenderBundleEncoder is not supported in GLES backend.
func (d *Device) CreateRenderBundleEncoder(desc *hal.RenderBundleEncoderDescriptor) (hal.RenderBundleEncoder, error) {
	return nil, fmt.Errorf("gles: render bundles not supported")
}

// DestroyRenderBundle is not supported in GLES backend.
func (d *Device) DestroyRenderBundle(bundle hal.RenderBundle) {}

// CreateAccelerationStructure returns an error because the GLES backend
// does not support ray tracing. OpenGL ES has no acceleration structure API.
func (d *Device) CreateAccelerationStructure(_ *hal.AccelerationStructureDescriptor) (hal.AccelerationStructure, error) {
	return nil, fmt.Errorf("gles: ray tracing not supported")
}

// DestroyAccelerationStructure is a no-op (GLES has no ray tracing).
func (d *Device) DestroyAccelerationStructure(_ hal.AccelerationStructure) {}

// GetAccelerationStructureBuildSizes returns zero sizes (GLES has no ray tracing).
func (d *Device) GetAccelerationStructureBuildSizes(_ *hal.GetAccelerationStructureBuildSizesDescriptor) hal.AccelerationStructureBuildSizes {
	return hal.AccelerationStructureBuildSizes{}
}

// GetAccelerationStructureDeviceAddress returns 0 (GLES has no ray tracing).
func (d *Device) GetAccelerationStructureDeviceAddress(_ hal.AccelerationStructure) uint64 {
	return 0
}

// TlasInstanceToBytes returns nil (GLES has no ray tracing).
func (d *Device) TlasInstanceToBytes(_ hal.TlasInstance) []byte { return nil }

// WaitIdle waits for all GPU work to complete.
func (d *Device) WaitIdle() error {
	if d.ctx != nil && d.ctx.GL() != nil {
		glCtx := d.ctx.Lock()
		defer d.ctx.Unlock()
		if d.ctx.OnHelper() {
			// glFinish here would finish the helper context's commands,
			// none: the queue's last fence stands for the work, sync
			// objects being shared.
			if d.fence != nil {
				v := d.fence.Signaled()
				for deadline := time.Now().Add(10 * time.Second); d.fence.GetLatest() < v && time.Now().Before(deadline); {
					d.fence.Wait(v, 100*time.Millisecond)
				}
			}
			return nil
		}
		glCtx.Finish()
	}
	return nil
}

// Destroy releases the device.
func (d *Device) Destroy() {
	if d.ctx == nil || (d.vao == 0 && d.fbs == nil) {
		return
	}
	glCtx := d.ctx.Lock()
	defer d.ctx.Unlock()
	if d.vao != 0 {
		glCtx.DeleteVertexArrays(d.vao)
		d.vao = 0
	}
	if d.fbs != nil {
		d.fbs.destroy(glCtx)
		d.fbs = nil
	}
}

// Type aliases for hal descriptors
type (
	BufferDescriptor          = hal.BufferDescriptor
	TextureDescriptor         = hal.TextureDescriptor
	TextureViewDescriptor     = hal.TextureViewDescriptor
	SamplerDescriptor         = hal.SamplerDescriptor
	BindGroupLayoutDescriptor = hal.BindGroupLayoutDescriptor
	BindGroupDescriptor       = hal.BindGroupDescriptor
	PipelineLayoutDescriptor  = hal.PipelineLayoutDescriptor
	ShaderModuleDescriptor    = hal.ShaderModuleDescriptor
	RenderPipelineDescriptor  = hal.RenderPipelineDescriptor
	ComputePipelineDescriptor = hal.ComputePipelineDescriptor
	CommandEncoderDescriptor  = hal.CommandEncoderDescriptor
)
