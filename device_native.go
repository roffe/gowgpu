//go:build !(js && wasm)

package wgpu

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"slices"
	"sync/atomic"
	"time"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	naga "github.com/roffe/naga"
	"github.com/roffe/naga/ir"
)

// Device represents a logical GPU device.
// It is the main interface for creating GPU resources.
//
// Device methods are safe for concurrent use, except Release() which
// must not be called concurrently with other methods.
type Device struct {
	core     *core.Device
	queue    *Queue
	instance *Instance
	released atomic.Bool

	// cmdEncoderPool is the single shared encoder pool for the device.
	// Used by both CreateCommandEncoder (user command encoders) and
	// PendingWrites (internal staging encoders). Matches Rust wgpu-core's
	// single device.command_allocator for both paths (queue.rs:1373).
	//
	// Encoders are acquired from the pool instead of creating expensive
	// GPU resources (DX12 ID3D12CommandAllocator ~64KB, Vulkan VkCommandPool)
	// every frame. After GPU completion, encoders are reset via ResetAll
	// and returned to the pool for reuse.
	//
	// Lifecycle: created before PendingWrites, destroyed after PendingWrites.
	// PendingWrites.destroy() clears its pool reference but does NOT destroy
	// the pool. Device.Release() destroys the pool after all users are done.
	//
	// nil when no HAL device (e.g., core-only path).
	cmdEncoderPool *encoderPool

	// usedLists recycles the submit-time validation lists of spent command
	// buffers.
	usedLists usedListPool

	uncapturedErrorHandler atomic.Pointer[func(*GPUError)]
}

// Queue returns the device's command queue.
func (d *Device) Queue() *Queue {
	return d.queue
}

// Features returns the device's enabled features.
func (d *Device) Features() gputypes.Features {
	return d.core.Features
}

// Limits returns the device's resource limits.
func (d *Device) Limits() gputypes.Limits {
	return d.core.Limits
}

// CreateBuffer creates a GPU buffer.
func (d *Device) CreateBuffer(desc *BufferDescriptor) (_ *Buffer, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: buffer descriptor is nil")
	}

	gpuDesc := &gputypes.BufferDescriptor{
		Label:            desc.Label,
		Size:             desc.Size,
		Usage:            desc.Usage,
		MappedAtCreation: desc.MappedAtCreation,
	}

	coreBuffer, err := d.core.CreateBuffer(gpuDesc)
	if err != nil {
		return nil, err
	}

	// Initialize ResourceRef with onZero callback for refcount-driven destruction.
	// When the last reference drops (either from explicit Release or Phase 2
	// Triage after GPU completion), onZero fires and HAL-destroys the buffer.
	// This matches Rust wgpu's Arc<Buffer> Drop behavior.
	//
	// Clone'd during encoding (SetBindGroup, SetVertexBuffer, CopyBufferToBuffer),
	// Drop'd when GPU completes submission via DestroyQueue.Triage.
	coreBuffer.Ref = core.NewResourceRef("Buffer:"+desc.Label, func() {
		coreBuffer.Destroy()
	})

	buf := &Buffer{core: coreBuffer, device: d, released: new(atomic.Bool)}
	if !coreBuffer.IsInitialized(0, desc.Size) && d.queue != nil && d.queue.pending != nil {
		if err := d.queue.pending.queueZeroBuffer(buf); err != nil {
			Logger().Warn("wgpu: buffer keeps its old memory", "label", desc.Label, "err", err)
		}
	}

	// Safety net: if the buffer is garbage collected without Release(),
	// schedule deferred destruction via DestroyQueue. This prevents
	// resource leaks when callers create per-frame buffers without
	// explicit lifecycle management (BUG-WGPU-RESOURCE-LIFECYCLE-001).
	buf.cleanup = registerBufferCleanup(buf, d, coreBuffer, desc.Label)

	return buf, nil
}

// CreateTexture creates a GPU texture.
func (d *Device) CreateTexture(desc *TextureDescriptor) (_ *Texture, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: texture descriptor is nil")
	}

	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	halDesc := desc.toHAL()

	if err := core.ValidateTextureDescriptor(halDesc, d.core.Limits, d.core.Features); err != nil {
		return nil, err
	}
	if err := d.core.RequireViewFormats(desc.Format, desc.ViewFormats, gputypes.DownlevelFlagsViewFormats); err != nil {
		return nil, fmt.Errorf("wgpu: CreateTexture: %w", err)
	}
	var zero, pass bool
	if d.queue != nil && d.queue.pending != nil {
		zero, pass = zeroableTexture(halDesc, d.queue.pending.usesBatching)
	}
	switch {
	case zero && pass:
		halDesc.Usage |= gputypes.TextureUsageRenderAttachment // for the pass that clears it
	case zero:
		halDesc.Usage |= gputypes.TextureUsageCopyDst // for the copy that zeroes it
	}

	halTexture, err := halDevice.CreateTexture(halDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: failed to create texture: %w", err)
	}

	// Create core.Texture for TrackerIndex allocation. This enables barrier
	// injection: the TrackerIndex is what the usage scopes and the command
	// buffer's tracker record the texture's usages under.
	coreTexture := core.NewTexture(
		halTexture, d.core, desc.Format, halDesc.Dimension,
		desc.Usage,
		gputypes.Extent3D{
			Width:              desc.Size.Width,
			Height:             desc.Size.Height,
			DepthOrArrayLayers: desc.Size.DepthOrArrayLayers,
		},
		halDesc.MipLevelCount, halDesc.SampleCount, desc.Label,
	)

	// The device tracker knows the texture from now on, as uninitialized, so
	// its first use in a submit gets a barrier from UNDEFINED; Rust wgpu-core
	// inserts every new texture as UNINITIALIZED the same way. Merged in at
	// first use instead, a texture nothing had zeroed (no TextureBinding)
	// was copied into or loaded in a render pass in the layout it was made
	// in.
	if tracker := d.core.Tracker(); tracker != nil {
		if td := coreTexture.TrackingData(); td != nil && td.Index().IsValid() {
			tracker.InsertTexture(td.Index(), track.TextureUsesUninitialized)
		}
	}

	tex := &Texture{hal: halTexture, device: d, format: desc.Format, viewFormats: slices.Clone(desc.ViewFormats), coreTexture: coreTexture, zeroable: zero}
	if zero {
		d.queue.pending.queueZero(tex)
	}
	return tex, nil
}

// CreateTextureView creates a view into a texture.
func (d *Device) CreateTextureView(texture *Texture, desc *TextureViewDescriptor) (_ *TextureView, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}
	if texture == nil {
		return nil, fmt.Errorf("wgpu: texture is nil")
	}
	if d.foreign(texture.device) {
		return nil, errForeign("CreateTextureView", "the texture")
	}
	halTexture := texture.resolveHAL()
	if halTexture == nil {
		return nil, ErrReleased
	}

	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	// A view without a descriptor is the texture's default view, which every
	// such view shares (surface textures come and go every frame, so not
	// theirs).
	if desc == nil && texture.surface == nil {
		if hv, viewDesc := texture.sharedDefaultView(halDevice, halTexture); hv != nil {
			return &TextureView{hal: hv, device: d, texture: texture, desc: viewDesc}, nil
		}
	}

	halDesc := &hal.TextureViewDescriptor{}
	if desc != nil {
		halDesc.Label = desc.Label
		halDesc.Format = desc.Format
		halDesc.Dimension = desc.Dimension
		halDesc.Aspect = desc.Aspect
		halDesc.BaseMipLevel = desc.BaseMipLevel
		halDesc.MipLevelCount = desc.MipLevelCount
		halDesc.BaseArrayLayer = desc.BaseArrayLayer
		halDesc.ArrayLayerCount = desc.ArrayLayerCount
	}
	if texture.coreTexture != nil {
		resolved, err := core.ResolveTextureViewDescriptor(texture.coreTexture, texture.viewFormats, halDesc)
		if err != nil {
			return nil, fmt.Errorf("wgpu: CreateTextureView: %w", err)
		}
		*halDesc = resolved
	}

	halView, err := halDevice.CreateTextureView(halTexture, halDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: failed to create texture view: %w", err)
	}

	view := &TextureView{
		hal:          halView,
		device:       d,
		texture:      texture,
		desc:         *halDesc,
		surface:      texture.surface,
		surfaceLease: texture.surfaceLease,
	}
	return view.withRef(halDesc.Label), nil
}

// CreateSampler creates a texture sampler.
func (d *Device) CreateSampler(desc *SamplerDescriptor) (_ *Sampler, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}

	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	halDesc := &hal.SamplerDescriptor{}
	if desc != nil {
		halDesc.Label = desc.Label
		halDesc.AddressModeU = desc.AddressModeU
		halDesc.AddressModeV = desc.AddressModeV
		halDesc.AddressModeW = desc.AddressModeW
		halDesc.MagFilter = desc.MagFilter
		halDesc.MinFilter = desc.MinFilter
		halDesc.MipmapFilter = desc.MipmapFilter
		halDesc.LodMinClamp = desc.LodMinClamp
		halDesc.LodMaxClamp = desc.LodMaxClamp
		halDesc.Compare = desc.Compare
		halDesc.Anisotropy = desc.Anisotropy
	}

	if err := core.ValidateSamplerDescriptor(halDesc, d.core.Features); err != nil {
		return nil, err
	}
	// The anisotropy is clamped to what the platform samples with, as
	// WebGPU says: 16 at most, which every backend takes, and 1 without
	// anisotropic filtering. wgpu-core clamps it the same way.
	switch {
	case !d.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsAnisotropicFiltering):
		halDesc.Anisotropy = min(halDesc.Anisotropy, 1)
	case halDesc.Anisotropy > 16:
		halDesc.Anisotropy = 16
	}

	halSampler, err := halDevice.CreateSampler(halDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: failed to create sampler: %w", err)
	}

	sampler := &Sampler{
		hal:          halSampler,
		device:       d,
		magFilter:    halDesc.MagFilter,
		minFilter:    halDesc.MinFilter,
		mipmapFilter: halDesc.MipmapFilter,
		compare:      halDesc.Compare,
	}
	return sampler.withRef(halDesc.Label), nil
}

// CreateShaderModule creates a shader module.
func (d *Device) CreateShaderModule(desc *ShaderModuleDescriptor) (_ *ShaderModule, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: shader module descriptor is nil")
	}

	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	halDesc := &hal.ShaderModuleDescriptor{
		Label: desc.Label,
		Source: hal.ShaderSource{
			WGSL:  desc.WGSL,
			SPIRV: desc.SPIRV,
		},
	}

	if err := core.ValidateShaderModuleDescriptor(halDesc, d.core.Features); err != nil {
		return nil, err
	}

	halModule, err := halDevice.CreateShaderModule(halDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: failed to create shader module: %w", err)
	}

	sm := &ShaderModule{hal: halModule, device: d}

	// Parse WGSL source to naga IR for shader introspection (late binding validation).
	// Matches Rust wgpu-core which stores the naga Module on ShaderModule for use
	// by Interface::check_stage during pipeline creation.
	// SPIR-V shaders skip this — they go directly to HAL without IR-level introspection.
	if desc.WGSL != "" {
		ast, parseErr := naga.Parse(desc.WGSL)
		if parseErr == nil {
			irModule, lowerErr := naga.Lower(ast)
			if lowerErr == nil {
				sm.irModule = irModule
			}
		}
		// Parse/lower failures are non-fatal here — the HAL already compiled the shader
		// successfully. Late binding validation will be skipped if IR is unavailable.
	}

	return sm, nil
}

// CreateBindGroupLayout creates a bind group layout.
func (d *Device) CreateBindGroupLayout(desc *BindGroupLayoutDescriptor) (_ *BindGroupLayout, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: bind group layout descriptor is nil")
	}

	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	// In binding order, whatever order they come in: layouts with the same
	// entries are the same to the backend and compare equal entry by entry,
	// as wgpu-core keeps them in a map sorted by binding.
	entries := slices.Clone(desc.Entries)
	slices.SortFunc(entries, func(a, b gputypes.BindGroupLayoutEntry) int { return cmp.Compare(a.Binding, b.Binding) })
	halDesc := &hal.BindGroupLayoutDescriptor{
		Label:   desc.Label,
		Entries: entries,
	}

	if err := core.ValidateBindGroupLayoutDescriptor(halDesc, d.core.Limits); err != nil {
		return nil, err
	}

	halLayout, err := halDevice.CreateBindGroupLayout(halDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: failed to create bind group layout: %w", err)
	}

	return &BindGroupLayout{hal: halLayout, device: d, entries: entries}, nil
}

// CreatePipelineLayout creates a pipeline layout.
func (d *Device) CreatePipelineLayout(desc *PipelineLayoutDescriptor) (_ *PipelineLayout, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: pipeline layout descriptor is nil")
	}

	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	halLayouts := make([]hal.BindGroupLayout, len(desc.BindGroupLayouts))
	for i, layout := range desc.BindGroupLayouts {
		if layout == nil {
			return nil, fmt.Errorf("wgpu: bind group layout at index %d is nil", i)
		}
		if layout.released {
			return nil, ErrReleased
		}
		if d.foreign(layout.device) {
			return nil, errForeign("CreatePipelineLayout", fmt.Sprintf("bind group layout %d", i))
		}
		halLayouts[i] = layout.hal
	}

	halDesc := &hal.PipelineLayoutDescriptor{
		Label:            desc.Label,
		BindGroupLayouts: halLayouts,
	}

	if err := core.ValidatePipelineLayoutDescriptor(halDesc, d.core.Limits, d.core.Features); err != nil {
		return nil, err
	}
	groups := make([][]gputypes.BindGroupLayoutEntry, len(desc.BindGroupLayouts))
	for i, layout := range desc.BindGroupLayouts {
		groups[i] = layout.entries
	}
	if err := core.ValidatePipelineLayoutBindings(desc.Label, groups, d.core.Limits); err != nil {
		return nil, err
	}

	halLayout, err := halDevice.CreatePipelineLayout(halDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: failed to create pipeline layout: %w", err)
	}

	// Store a copy of the bind group layouts slice for binder validation.
	bgLayouts := make([]*BindGroupLayout, len(desc.BindGroupLayouts))
	copy(bgLayouts, desc.BindGroupLayouts)

	layout := &PipelineLayout{
		hal:              halLayout,
		device:           d,
		bindGroupCount:   uint32(len(desc.BindGroupLayouts)), //nolint:gosec // layout count fits uint32
		bindGroupLayouts: bgLayouts,
	}
	layout.ref = core.NewResourceRef("PipelineLayout:"+desc.Label, layout.destroyHAL)
	return layout, nil
}

// CreateBindGroup creates a bind group.
func (d *Device) CreateBindGroup(desc *BindGroupDescriptor) (_ *BindGroup, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: bind group descriptor is nil")
	}

	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	if desc.Layout == nil {
		return nil, &core.CreateBindGroupError{
			Kind:  core.CreateBindGroupErrorMissingLayout,
			Label: desc.Label,
		}
	}
	if desc.Layout.released {
		return nil, ErrReleased
	}
	if d.foreign(desc.Layout.device) {
		return nil, errForeign("CreateBindGroup", "the layout")
	}

	halEntries := make([]gputypes.BindGroupEntry, len(desc.Entries))
	for i, entry := range desc.Entries {
		if entry.TextureView != nil && entry.TextureView.resolveHAL() == nil {
			return nil, ErrReleased
		}
		if entry.Sampler != nil && entry.Sampler.released {
			return nil, ErrReleased
		}
		if entry.Buffer != nil && d.foreign(entry.Buffer.device) || entry.TextureView != nil && d.foreign(entry.TextureView.device) ||
			entry.Sampler != nil && d.foreign(entry.Sampler.device) {
			return nil, errForeign("CreateBindGroup", fmt.Sprintf("the resource at binding %d", entry.Binding))
		}
		halEntries[i] = entry.toHAL()
	}

	halDesc := &hal.BindGroupDescriptor{
		Label:   desc.Label,
		Layout:  desc.Layout.hal,
		Entries: halEntries,
	}

	bufferInfos, samplerInfos, textureInfos := buildBindGroupValidationInfos(desc.Entries)

	if err := core.ValidateBindGroupDescriptor(halDesc, desc.Layout.entries, bufferInfos, samplerInfos, textureInfos, d.core.Limits, d.core.Features); err != nil {
		return nil, err
	}

	halGroup, err := halDevice.CreateBindGroup(halDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: failed to create bind group: %w", err)
	}

	// Build late buffer binding info for layout entries with MinBindingSize == 0.
	// These record the actual bound buffer size at bind group creation time,
	// to be validated against shader requirements at draw/dispatch time.
	// Matches Rust wgpu-core's BindGroup.late_buffer_binding_infos population
	// in Device::create_bind_group (binding_model.rs:1187-1189).
	lateInfos := buildLateBufferBindingInfos(desc.Layout.entries, desc.Entries)

	// Collect buffer and texture references for submit-time validation (VAL-A6).
	boundBuffers, boundBufferUses, boundTextures, boundTextureUses := collectBindGroupResources(desc.Layout.entries, desc.Entries)
	viewAndSamplerRefs := cloneViewAndSamplerRefs(desc.Entries)

	// Initialize ResourceRef with onZero callback for refcount-driven destruction.
	// When the last reference drops (either from explicit Release or Phase 2
	// Triage after GPU completion), onZero fires and defers HAL destruction via
	// DestroyQueue. This matches Rust wgpu's Arc<BindGroup> Drop behavior. ADR-056.
	halBG := halGroup
	bgOnZero := func() {
		dq := d.destroyQueue()
		if dq != nil {
			subIdx := d.lastSubmissionIndex()
			dq.Defer(subIdx, "BindGroup", func() {
				halDevice.DestroyBindGroup(halBG)
			})
		} else {
			halDevice.DestroyBindGroup(halBG)
		}
		for _, ref := range viewAndSamplerRefs {
			ref.Drop()
		}
	}

	bg := &BindGroup{
		hal:                    halGroup,
		device:                 d,
		released:               new(atomic.Bool),
		layout:                 desc.Layout,
		lateBufferBindingInfos: lateInfos,
		dynamicBindings:        buildDynamicBindings(desc.Layout.entries, desc.Entries),
		ref:                    core.NewResourceRef("BindGroup:"+desc.Label, bgOnZero),
		boundBuffers:           boundBuffers,
		boundBufferUses:        boundBufferUses,
		boundTextures:          boundTextures,
		boundTextureUses:       boundTextureUses,
	}

	// Safety net: if the bind group is garbage collected without Release(),
	// schedule deferred destruction via DestroyQueue (BUG-WGPU-RESOURCE-LIFECYCLE-001).
	bg.cleanup = registerBindGroupCleanup(bg, d, desc.Label)

	return bg, nil
}

func buildBindGroupValidationInfos(entries []BindGroupEntry) (
	[]core.BindGroupBufferInfo,
	[]core.BindGroupSamplerInfo,
	[]core.BindGroupTextureInfo,
) {
	var bufferInfos []core.BindGroupBufferInfo
	var samplerInfos []core.BindGroupSamplerInfo
	var textureInfos []core.BindGroupTextureInfo
	for _, entry := range entries {
		if entry.Buffer != nil {
			bufferInfos = append(bufferInfos, core.BindGroupBufferInfo{
				Binding:    entry.Binding,
				Usage:      entry.Buffer.Usage(),
				BufferSize: entry.Buffer.Size(),
				Offset:     entry.Offset,
				Size:       entry.Size,
			})
		}
		if entry.Sampler != nil {
			samplerInfos = append(samplerInfos, core.BindGroupSamplerInfo{
				Binding: entry.Binding,
				Desc: &hal.SamplerDescriptor{
					MagFilter:    entry.Sampler.magFilter,
					MinFilter:    entry.Sampler.minFilter,
					MipmapFilter: entry.Sampler.mipmapFilter,
					Compare:      entry.Sampler.compare,
				},
			})
		}
		if v := entry.TextureView; v != nil && v.texture != nil {
			info := core.BindGroupTextureInfo{
				Binding:       entry.Binding,
				Format:        v.texture.Format(),
				ViewFormat:    v.desc.Format,
				ViewDimension: v.desc.Dimension,
				Aspect:        v.desc.Aspect,
				MipLevelCount: v.desc.MipLevelCount,
			}
			if ct := v.texture.coreTexture; ct != nil {
				info.Usage, info.SampleCount = ct.Usage(), ct.SampleCount()
			} else {
				info.Usage, info.SampleCount = gputypes.TextureUsageTextureBinding|gputypes.TextureUsageStorageBinding, 1
			}
			textureInfos = append(textureInfos, info)
		}
	}
	return bufferInfos, samplerInfos, textureInfos
}

func buildLateBufferBindingInfos(layoutEntries []gputypes.BindGroupLayoutEntry, entries []BindGroupEntry) []LateBufferBindingInfo {
	entryMap := buildBindGroupEntryMap(entries)
	var lateInfos []LateBufferBindingInfo
	for _, layoutEntry := range layoutEntries {
		if layoutEntry.Buffer == nil || layoutEntry.Buffer.MinBindingSize != 0 {
			continue
		}
		var boundSize uint64
		if bgEntry, ok := entryMap[layoutEntry.Binding]; ok && bgEntry.Buffer != nil {
			boundSize = bgEntry.Size
			if boundSize == 0 {
				bufSize := bgEntry.Buffer.Size()
				if bgEntry.Offset < bufSize {
					boundSize = bufSize - bgEntry.Offset
				}
			}
		}
		lateInfos = append(lateInfos, LateBufferBindingInfo{
			BindingIndex: layoutEntry.Binding,
			Size:         boundSize,
		})
	}
	return lateInfos
}

// collectBindGroupResources extracts buffer and texture references from bind
// group entries for submit-time validation (VAL-A6), and how each texture is
// used. Matches Rust wgpu-core where bind group creation stores resource
// references that are later checked via trackers.buffers/textures.used_resources()
// in validate_command_buffer.
func collectBindGroupResources(layout []gputypes.BindGroupLayoutEntry, entries []BindGroupEntry) ([]*Buffer, []track.BufferUses, []*Texture, []track.TextureUses) {
	var buffers []*Buffer
	var bufferUses []track.BufferUses
	var textures []*Texture
	var uses []track.TextureUses
	for i := range entries {
		if entries[i].Buffer != nil {
			buffers = append(buffers, entries[i].Buffer)
			bufferUses = append(bufferUses, bindingBufferUse(layout, entries[i].Binding))
		}
		if entries[i].TextureView != nil && entries[i].TextureView.texture != nil {
			textures = append(textures, entries[i].TextureView.texture)
			uses = append(uses, bindingTextureUse(layout, entries[i].Binding))
		}
	}
	return buffers, bufferUses, textures, uses
}

// buildDynamicBindings returns the buffer bindings of entries whose layout
// entry has a dynamic offset, by binding number.
//
// Reference: wgpu-core device/resource.rs create_buffer_binding
// (dynamic_binding_info, sorted by binding)
func buildDynamicBindings(layout []gputypes.BindGroupLayoutEntry, entries []BindGroupEntry) []dynamicBinding {
	var bindings []dynamicBinding
	for i := range entries {
		e := &entries[i]
		if e.Buffer == nil {
			continue
		}
		j := slices.IndexFunc(layout, func(l gputypes.BindGroupLayoutEntry) bool { return l.Binding == e.Binding })
		if j < 0 || layout[j].Buffer == nil || !layout[j].Buffer.HasDynamicOffset {
			continue
		}
		bufSize := e.Buffer.Size()
		size := e.Size
		if size == 0 && e.Offset <= bufSize {
			size = bufSize - e.Offset
		}
		var maxOffset uint64
		if end := e.Offset + size; end >= e.Offset && end <= bufSize {
			maxOffset = bufSize - end
		}
		bindings = append(bindings, dynamicBinding{
			binding:   e.Binding,
			storage:   layout[j].Buffer.Type != gputypes.BufferBindingTypeUniform,
			maxOffset: maxOffset,
		})
	}
	slices.SortFunc(bindings, func(a, b dynamicBinding) int { return cmp.Compare(a.binding, b.binding) })
	return bindings
}

// cloneViewAndSamplerRefs takes a reference on every texture view and
// sampler the entries bind, which the bind group drops when it is destroyed:
// the HAL group keeps their HAL handles.
//
// Reference: wgpu-core binding_model.rs BindGroup.used (Arc<TextureView>,
// Arc<Sampler>)
func cloneViewAndSamplerRefs(entries []BindGroupEntry) []*core.ResourceRef {
	var refs []*core.ResourceRef
	for i := range entries {
		if v := entries[i].TextureView; v != nil && v.ref != nil {
			v.ref.Clone()
			refs = append(refs, v.ref)
		}
		if s := entries[i].Sampler; s != nil && s.ref != nil {
			s.ref.Clone()
			refs = append(refs, s.ref)
		}
	}
	return refs
}

// bindingBufferUse is how the layout entry at binding uses its buffer:
// uniform, or storage read-only or read-write.
//
// Reference: wgpu-core binding_model.rs (BufferBindingType to BufferUses)
func bindingBufferUse(layout []gputypes.BindGroupLayoutEntry, binding uint32) track.BufferUses {
	for i := range layout {
		if layout[i].Binding != binding || layout[i].Buffer == nil {
			continue
		}
		switch layout[i].Buffer.Type {
		case gputypes.BufferBindingTypeStorage:
			return track.BufferUsesStorageWrite
		case gputypes.BufferBindingTypeReadOnlyStorage:
			return track.BufferUsesStorageRead
		}
		return track.BufferUsesUniform
	}
	return track.BufferUsesUniform
}

// bindingTextureUse is how a texture bound at binding is used: sampled,
// unless the layout declares a storage texture there.
func bindingTextureUse(layout []gputypes.BindGroupLayoutEntry, binding uint32) track.TextureUses {
	for i := range layout {
		if layout[i].Binding != binding || layout[i].StorageTexture == nil {
			continue
		}
		if layout[i].StorageTexture.Access == gputypes.StorageTextureAccessReadOnly {
			return track.TextureUsesStorageRead
		}
		return track.TextureUsesStorageWrite
	}
	return track.TextureUsesResource
}

// buildBindGroupEntryMap builds a lookup map from binding index to BindGroupEntry
// for efficient access during late buffer binding info construction.
func buildBindGroupEntryMap(entries []BindGroupEntry) map[uint32]*BindGroupEntry {
	m := make(map[uint32]*BindGroupEntry, len(entries))
	for i := range entries {
		m[entries[i].Binding] = &entries[i]
	}
	return m
}

// CreateRenderPipeline creates a render pipeline.
func (d *Device) CreateRenderPipeline(desc *RenderPipelineDescriptor) (_ *RenderPipeline, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: render pipeline descriptor is nil")
	}

	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	if desc.Layout != nil && desc.Layout.released {
		return nil, ErrReleased
	}
	if desc.Layout != nil && d.foreign(desc.Layout.device) || desc.Vertex.Module != nil && d.foreign(desc.Vertex.Module.device) ||
		desc.Fragment != nil && desc.Fragment.Module != nil && d.foreign(desc.Fragment.Module.device) {
		return nil, errForeign("CreateRenderPipeline", "the layout or a shader module")
	}

	halDesc := desc.toHAL()

	if err := core.ValidateRenderPipelineDescriptor(halDesc, d.core.Limits, d.core.Features); err != nil {
		return nil, err
	}

	// Without a Layout the pipeline gets the one its shaders imply, and
	// owns it.
	layout := desc.Layout
	var autoLayout *PipelineLayout
	if layout == nil {
		stages := []autoStage{{desc.Vertex.Module, desc.Vertex.EntryPoint, ir.StageVertex, ShaderStageVertex}}
		if desc.Fragment != nil {
			stages = append(stages, autoStage{desc.Fragment.Module, desc.Fragment.EntryPoint, ir.StageFragment, ShaderStageFragment})
		}
		var err error
		if autoLayout, err = d.createAutoLayout(desc.Label, stages...); err != nil {
			return nil, err
		}
		layout = autoLayout
		halDesc.Layout = layout.hal
	}
	if err := checkRenderPipelineShaders(desc, layout); err != nil {
		autoLayout.releaseAuto()
		return nil, fmt.Errorf("wgpu: CreateRenderPipeline %q: %w", desc.Label, err)
	}

	halPipeline, err := halDevice.CreateRenderPipeline(halDesc)
	if err != nil {
		autoLayout.releaseAuto()
		return nil, fmt.Errorf("wgpu: failed to create render pipeline: %w", err)
	}

	bgCount := layout.bindGroupCount
	bgLayouts := layout.bindGroupLayouts
	// Check if any color target uses blend constant factors.
	// Matches Rust wgpu-core PipelineFlags::BLEND_CONSTANT (resource.rs:4562-4569).
	var needsBlendConstant bool
	if desc.Fragment != nil {
		for i := range desc.Fragment.Targets {
			if b := desc.Fragment.Targets[i].Blend; b != nil {
				if b.Color.UsesConstant() || b.Alpha.UsesConstant() {
					needsBlendConstant = true
					break
				}
			}
		}
	}

	// Build shader binding sizes across all stages (vertex + fragment).
	// Matches Rust wgpu-core's check_stage calls that accumulate shader_binding_sizes
	// with max across stages for the same binding (validation.rs:1126-1139).
	shaderBindingSizes := mergeShaderBindingSizes(
		desc.Vertex.Module,
		fragmentShaderModule(desc.Fragment),
	)

	lateGroups := makeLateSizedBufferGroups(shaderBindingSizes, bgLayouts)

	// Initialize ResourceRef with onZero callback for refcount-driven destruction.
	// When the last reference drops (either from explicit Release or Phase 2
	// Triage after GPU completion), onZero fires and defers HAL destruction via
	// DestroyQueue. ADR-056: unified resource lifecycle.
	halRP := halPipeline
	layout.ref.Clone()
	rpOnZero := func() {
		dq := d.destroyQueue()
		if dq != nil {
			subIdx := d.lastSubmissionIndex()
			dq.Defer(subIdx, "RenderPipeline", func() {
				halDevice.DestroyRenderPipeline(halRP)
			})
		} else {
			halDevice.DestroyRenderPipeline(halRP)
		}
		layout.ref.Drop()
		autoLayout.releaseAuto()
	}

	pipeline := &RenderPipeline{
		hal:                   halPipeline,
		device:                d,
		bindGroupCount:        bgCount,
		bindGroupLayouts:      bgLayouts,
		vertexSteps:           vertexStepsOf(desc.Vertex.Buffers),
		blendConstantRequired: needsBlendConstant,
		stripIndexFormat:      desc.Primitive.StripIndexFormat,
		lateSizedBufferGroups: lateGroups,
		ref:                   core.NewResourceRef("RenderPipeline:"+desc.Label, rpOnZero),
		label:                 desc.Label,
	}
	pipeline.setAttachments(desc)
	return pipeline, nil
}

// setAttachments records what a render pipeline draws into and whether it
// writes depth or stencil, which the passes that use it have to agree with.
//
// Reference: wgpu-core pipeline.rs RenderPipeline.pass_context and
// PipelineFlags::WRITES_DEPTH / WRITES_STENCIL
func (p *RenderPipeline) setAttachments(desc *RenderPipelineDescriptor) {
	if fs := desc.Fragment; fs != nil {
		p.passContext.ColorFormats = make([]gputypes.TextureFormat, len(fs.Targets))
		for i := range fs.Targets {
			p.passContext.ColorFormats[i] = fs.Targets[i].Format
		}
	}
	p.passContext.SampleCount = max(desc.Multisample.Count, 1)
	if ds := desc.DepthStencil; ds != nil {
		p.passContext.DepthStencilFormat = ds.Format
		p.writesDepth = ds.DepthWriteEnabled
		writes := func(face StencilFaceState) bool {
			keep := func(op gputypes.StencilOperation) bool {
				return op == gputypes.StencilOperationKeep || op == gputypes.StencilOperationUndefined
			}
			return !keep(face.FailOp) || !keep(face.DepthFailOp) || !keep(face.PassOp)
		}
		cull := desc.Primitive.CullMode
		p.writesStencil = ds.StencilWriteMask != 0 &&
			(cull != gputypes.CullModeFront && writes(ds.StencilFront) || cull != gputypes.CullModeBack && writes(ds.StencilBack))
	}
}

// checkRenderPipelineShaders checks the shaders of a render pipeline
// against its layout, vertex buffers and color targets.
func checkRenderPipelineShaders(desc *RenderPipelineDescriptor, layout *PipelineLayout) error {
	if err := checkStageBindings(desc.Vertex.Module, desc.Vertex.EntryPoint, ir.StageVertex, ShaderStageVertex, layout.bindGroupLayouts); err != nil {
		return fmt.Errorf("vertex stage: %w", err)
	}
	if err := checkVertexInputs(desc.Vertex.Module, desc.Vertex.EntryPoint, desc.Vertex.Buffers); err != nil {
		return err
	}
	if fs := desc.Fragment; fs != nil {
		if err := checkStageBindings(fs.Module, fs.EntryPoint, ir.StageFragment, ShaderStageFragment, layout.bindGroupLayouts); err != nil {
			return fmt.Errorf("fragment stage: %w", err)
		}
		if err := checkFragmentOutputs(fs.Module, fs.EntryPoint, fs.Targets); err != nil {
			return err
		}
	}
	return nil
}

// fragmentShaderModule extracts the ShaderModule from a FragmentState, or nil if absent.
func fragmentShaderModule(fs *FragmentState) *ShaderModule {
	if fs == nil {
		return nil
	}
	return fs.Module
}

// mergeShaderBindingSizes merges binding sizes from vertex and fragment shader stages,
// taking the max size when both stages reference the same binding. Matches Rust
// wgpu-core's pattern of calling check_stage for each stage with the same
// shader_binding_sizes map, where Entry::Occupied takes max (validation.rs:1131-1133).
func mergeShaderBindingSizes(
	vertexModule *ShaderModule,
	fragModule *ShaderModule,
) map[ir.ResourceBinding]uint64 {
	result := make(map[ir.ResourceBinding]uint64)

	if vertexModule != nil && vertexModule.irModule != nil {
		for rb, size := range extractShaderBindingSizes(vertexModule.irModule) {
			result[rb] = size
		}
	}
	if fragModule != nil && fragModule.irModule != nil {
		for rb, size := range extractShaderBindingSizes(fragModule.irModule) {
			if existing, ok := result[rb]; !ok || size > existing {
				result[rb] = size
			}
		}
	}

	return result
}

// CreateComputePipeline creates a compute pipeline.
func (d *Device) CreateComputePipeline(desc *ComputePipelineDescriptor) (_ *ComputePipeline, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: compute pipeline descriptor is nil")
	}

	// Validate that the device supports compute shaders (downlevel check).
	// Matches Rust wgpu-core resource.rs:4367 — first validation in create_compute_pipeline_or_error.
	if err := d.core.RequireDownlevelFlags(gputypes.DownlevelFlagsComputeShaders); err != nil {
		return nil, fmt.Errorf("wgpu: CreateComputePipeline: %w", err)
	}

	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	if desc.Layout != nil && desc.Layout.released {
		return nil, ErrReleased
	}
	if desc.Layout != nil && d.foreign(desc.Layout.device) || desc.Module != nil && d.foreign(desc.Module.device) {
		return nil, errForeign("CreateComputePipeline", "the layout or the shader module")
	}

	halDesc := desc.toHAL()

	if err := core.ValidateComputePipelineDescriptor(halDesc); err != nil {
		return nil, err
	}

	// VAL-010: Validate workgroup_size against device limits.
	// Matches Rust wgpu-core validation.rs:1243-1264.
	if desc.Module != nil && desc.Module.irModule != nil {
		if err := d.validateComputeWorkgroupSize(desc.Label, desc.EntryPoint, desc.Module); err != nil {
			return nil, err
		}
	}

	layout := desc.Layout
	var autoLayout *PipelineLayout
	if layout == nil {
		var err error
		if autoLayout, err = d.createAutoLayout(desc.Label, autoStage{desc.Module, desc.EntryPoint, ir.StageCompute, ShaderStageCompute}); err != nil {
			return nil, err
		}
		layout = autoLayout
		halDesc.Layout = layout.hal
	}
	if err := checkStageBindings(desc.Module, desc.EntryPoint, ir.StageCompute, ShaderStageCompute, layout.bindGroupLayouts); err != nil {
		autoLayout.releaseAuto()
		return nil, fmt.Errorf("wgpu: CreateComputePipeline %q: %w", desc.Label, err)
	}

	halPipeline, err := halDevice.CreateComputePipeline(halDesc)
	if err != nil {
		autoLayout.releaseAuto()
		return nil, fmt.Errorf("wgpu: failed to create compute pipeline: %w", err)
	}

	bgCount := layout.bindGroupCount
	bgLayouts := layout.bindGroupLayouts

	// Build shader binding sizes for the compute stage.
	var shaderBindingSizes map[ir.ResourceBinding]uint64
	if desc.Module != nil && desc.Module.irModule != nil {
		shaderBindingSizes = extractShaderBindingSizes(desc.Module.irModule)
	}

	lateGroups := makeLateSizedBufferGroups(shaderBindingSizes, bgLayouts)

	// Initialize ResourceRef with onZero callback for refcount-driven destruction.
	// ADR-056: unified resource lifecycle.
	halCP := halPipeline
	layout.ref.Clone()
	cpOnZero := func() {
		dq := d.destroyQueue()
		if dq != nil {
			subIdx := d.lastSubmissionIndex()
			dq.Defer(subIdx, "ComputePipeline", func() {
				halDevice.DestroyComputePipeline(halCP)
			})
		} else {
			halDevice.DestroyComputePipeline(halCP)
		}
		layout.ref.Drop()
		autoLayout.releaseAuto()
	}

	return &ComputePipeline{
		hal:                   halPipeline,
		device:                d,
		bindGroupCount:        bgCount,
		bindGroupLayouts:      bgLayouts,
		lateSizedBufferGroups: lateGroups,
		ref:                   core.NewResourceRef("ComputePipeline:"+desc.Label, cpOnZero),
	}, nil
}

// validateComputeWorkgroupSize checks shader workgroup_size against device limits.
// VAL-010: Matches Rust wgpu-core validation.rs:1243-1264.
func (d *Device) validateComputeWorkgroupSize(label, entryPoint string, module *ShaderModule) error {
	irMod := module.irModule
	if irMod == nil {
		return nil
	}

	// Find the entry point matching the requested name.
	for i := range irMod.EntryPoints {
		ep := &irMod.EntryPoints[i]
		if ep.Name != entryPoint || ep.Stage != ir.StageCompute {
			continue
		}

		wg := ep.Workgroup
		limits := d.core.Limits

		// Check each dimension for zero.
		dimNames := [3]string{"X", "Y", "Z"}
		for dim := 0; dim < 3; dim++ {
			if wg[dim] == 0 {
				return &core.CreateComputePipelineError{
					Kind:      core.CreateComputePipelineErrorWorkgroupSizeZero,
					Label:     label,
					Dimension: dimNames[dim],
				}
			}
		}

		// Check each dimension against device limits.
		dimLimits := [3]uint32{
			limits.MaxComputeWorkgroupSizeX,
			limits.MaxComputeWorkgroupSizeY,
			limits.MaxComputeWorkgroupSizeZ,
		}
		for dim := 0; dim < 3; dim++ {
			if wg[dim] > dimLimits[dim] {
				return &core.CreateComputePipelineError{
					Kind:      core.CreateComputePipelineErrorWorkgroupSizeExceeded,
					Label:     label,
					Dimension: dimNames[dim],
					Size:      wg[dim],
					Limit:     dimLimits[dim],
				}
			}
		}

		// Check total invocations (x*y*z) against MaxComputeInvocationsPerWorkgroup.
		total := uint64(wg[0]) * uint64(wg[1]) * uint64(wg[2])
		if total > uint64(limits.MaxComputeInvocationsPerWorkgroup) {
			return &core.CreateComputePipelineError{
				Kind:             core.CreateComputePipelineErrorTooManyInvocations,
				Label:            label,
				TotalInvocations: total,
				Limit:            limits.MaxComputeInvocationsPerWorkgroup,
			}
		}

		break
	}

	return nil
}

// CreateCommandEncoder creates a command encoder for recording GPU commands.
//
// When a device-level encoder pool is available (BUG-DX12-004), the HAL encoder
// is acquired from the pool instead of creating a new one. This avoids allocating
// expensive GPU resources (DX12 ID3D12CommandAllocator ~64KB, Vulkan VkCommandPool)
// on every frame. After GPU completion, the encoder is reset and returned to the
// pool for reuse. Matches Rust wgpu-core's CommandAllocator pattern (allocator.rs).
func (d *Device) CreateCommandEncoder(desc *CommandEncoderDescriptor) (_ *CommandEncoder, err error) {
	defer d.reportError(&err)
	if d.released.Load() {
		return nil, ErrReleased
	}

	label := ""
	if desc != nil {
		label = desc.Label
	}

	// When pool is available, acquire a recycled HAL encoder and pass it to core.
	// This bypasses core's internal CreateCommandEncoder which would create a new
	// HAL encoder, and instead uses CreateCommandEncoderWithHAL that accepts
	// a pre-existing encoder already in recording state.
	if d.cmdEncoderPool != nil {
		halEnc, err := d.cmdEncoderPool.acquire()
		if err != nil {
			return nil, fmt.Errorf("wgpu: encoder pool acquire: %w", err)
		}

		if err := halEnc.BeginEncoding(label); err != nil {
			// Failed to begin encoding — return encoder to pool for future use.
			d.cmdEncoderPool.release(halEnc)
			return nil, fmt.Errorf("wgpu: begin encoding: %w", err)
		}

		// The encoder object, with its core encoder, pass and command buffer
		// inside, is one a spent frame gave back, so a steady stream of
		// frames allocates none. The ref and validation lists of spent
		// command buffers come back the same way.
		e := cmdEncoderObjPool.Get().(*CommandEncoder)
		if err := d.core.InitCommandEncoderWithHAL(&e.coreEnc, halEnc, label); err != nil {
			cmdEncoderObjPool.Put(e)
			halEnc.DiscardEncoding()
			d.cmdEncoderPool.release(halEnc)
			return nil, err
		}
		var refs []*core.ResourceRef
		if dq := d.destroyQueue(); dq != nil {
			refs = dq.RecycledRefs()
		}
		if refs == nil {
			refs = make([]*core.ResourceRef, 0, 64)
		}
		lists := d.usedLists.get()
		e.core = &e.coreEnc
		e.device = d
		e.released = false
		e.serial = encoderSerials.Add(1)
		e.halEncoder = halEnc
		e.trackedRefs = refs
		e.usedBuffers, e.usedTextures, e.usedBindGroups, e.usedQuerySets = lists.buffers, lists.textures, lists.bindGroups, lists.querySets
		e.pooled = true
		return e, nil
	}

	// Fallback: no pool available (e.g., non-HAL device). Use core's built-in
	// encoder creation which creates a fresh HAL encoder each time.
	coreEncoder, err := d.core.CreateCommandEncoder(label)
	if err != nil {
		return nil, err
	}

	return &CommandEncoder{core: coreEncoder, device: d, serial: encoderSerials.Add(1)}, nil
}

// CreateFence creates a GPU synchronization fence.
// Fences are primarily used by the HAL internally for synchronization.
// Most callers should use Queue.Submit + Queue.Poll instead.
func (d *Device) CreateFence() (*Fence, error) {
	if d.released.Load() {
		return nil, ErrReleased
	}
	halDevice := d.halDevice()
	if halDevice == nil {
		return nil, ErrReleased
	}

	halFence, err := halDevice.CreateFence()
	if err != nil {
		return nil, fmt.Errorf("wgpu: failed to create fence: %w", err)
	}

	return &Fence{hal: halFence, device: d}, nil
}

// DestroyFence destroys a fence.
// The fence must not be in use by the GPU when destroyed.
//
// Deprecated: Use Fence.Release() instead.
func (d *Device) DestroyFence(f *Fence) {
	if f != nil {
		f.Release()
	}
}

// ResetFence resets a fence to the unsignaled state.
// The fence must not be in use by the GPU.
func (d *Device) ResetFence(f *Fence) error {
	if d.released.Load() {
		return ErrReleased
	}
	if f == nil || f.released {
		return ErrReleased
	}
	halDevice := d.halDevice()
	if halDevice == nil {
		return ErrReleased
	}
	return halDevice.ResetFence(f.hal)
}

// GetFenceStatus returns true if the fence is signaled (non-blocking).
// This is used for polling completion without blocking.
func (d *Device) GetFenceStatus(f *Fence) (bool, error) {
	if d.released.Load() {
		return false, ErrReleased
	}
	if f == nil || f.released {
		return false, ErrReleased
	}
	halDevice := d.halDevice()
	if halDevice == nil {
		return false, ErrReleased
	}
	return halDevice.GetFenceStatus(f.hal)
}

// WaitForFence waits for a fence to reach the specified value.
// Returns true if the fence reached the value, false if timeout expired.
func (d *Device) WaitForFence(f *Fence, value uint64, timeout time.Duration) (bool, error) {
	if d.released.Load() {
		return false, ErrReleased
	}
	if f == nil || f.released {
		return false, ErrReleased
	}
	halDevice := d.halDevice()
	if halDevice == nil {
		return false, ErrReleased
	}
	return halDevice.Wait(f.hal, value, timeout)
}

// FreeCommandBuffer returns a command buffer to the command pool.
// This must be called after the GPU has finished using the command buffer.
// The command buffer handle becomes invalid after this call.
// For multi-CB encoders, all accumulated HAL command buffers are freed.
func (d *Device) FreeCommandBuffer(cb *CommandBuffer) {
	if d.released.Load() || cb == nil {
		return
	}
	// A command buffer that was never submitted still holds its pooled
	// encoder, which owns the HAL command buffers. Freeing those one by one
	// never gave the encoder back (on Vulkan it is a no-op for pooled
	// encoders, so the whole command pool leaked), so hand it back whole.
	if cb.halEncoder != nil {
		cb.Release()
		return
	}
	halDevice := d.halDevice()
	if halDevice == nil {
		return
	}
	for _, buf := range cb.halBufferList() {
		if buf != nil {
			halDevice.FreeCommandBuffer(buf)
		}
	}
}

// PushErrorScope pushes a new error scope onto the device's error scope stack.
func (d *Device) PushErrorScope(filter ErrorFilter) {
	d.core.PushErrorScope(filter)
}

// PopErrorScope pops the most recently pushed error scope.
// Returns the captured error, or nil if no error occurred.
// Panics if no error scope is pushed.
func (d *Device) PopErrorScope() *GPUError {
	return d.core.PopErrorScope()
}

// reportError hands *err, an error a call on d returns, to the innermost
// error scope that captures its kind, or to uncapturedError when none
// does. A lost or released device reports nothing.
//
// Reference: WebGPU "dispatch error"; wgpu's ErrorSink::handle_error
func (d *Device) reportError(err *error) {
	if *err == nil || d == nil || d.core == nil || d.released.Load() || errors.Is(*err, ErrDeviceLost) {
		return
	}
	gpuErr := &GPUError{Type: ErrorFilterValidation, Message: (*err).Error()}
	if errors.Is(*err, ErrOutOfMemory) {
		gpuErr.Type = ErrorFilterOutOfMemory
	}
	if !d.core.ReportError(gpuErr.Type, gpuErr.Message) {
		d.uncapturedError(gpuErr)
	}
}

// uncapturedError gets the errors of d that no error scope captured. The
// calls that made them return them as well.
func (d *Device) uncapturedError(gpuErr *GPUError) {
	if handler := d.uncapturedErrorHandler.Load(); handler != nil {
		(*handler)(gpuErr)
		return
	}
	hal.Logger().Debug("wgpu: uncaptured error", "type", gpuErr.Type, "message", gpuErr.Message)
}

// SetUncapturedErrorHandler sets the function that gets every GPU error no
// error scope captured: validation, out-of-memory and internal errors, each
// with its Type and message. A newer call replaces the older handler, and
// nil removes it and restores the default handling of uncaptured errors.
//
// In a native build the handler runs on the goroutine whose call made the
// error, before that call returns the error as well, so calls on several
// goroutines can run it at the same time. Without a handler an uncaptured
// error goes to the logger SetLogger sets, at debug level.
func (d *Device) SetUncapturedErrorHandler(handler func(*GPUError)) {
	if handler == nil {
		d.uncapturedErrorHandler.Store(nil)
		return
	}
	d.uncapturedErrorHandler.Store(&handler)
}

// foreign reports whether owner, the device an object was made on, is
// another device than d. An object made without a device is anyone's.
//
// Reference: wgpu-core resource.rs same_device
func (d *Device) foreign(owner *Device) bool {
	return d != nil && owner != nil && owner != d
}

// errForeign is the error for what, an object of another device, passed to
// op.
func errForeign(op, what string) error {
	return fmt.Errorf("wgpu: %s: %s belongs to another device", op, what)
}

// WaitIdle waits for all GPU work to complete.
func (d *Device) WaitIdle() error {
	if d.released.Load() {
		return ErrReleased
	}
	return d.waitIdle()
}

// waitIdle drains the HAL and deferred resource queues without consulting the
// public released bit. Release marks the device unavailable before starting
// teardown, but the native device must remain alive while its last submission
// is drained.
func (d *Device) waitIdle() error {
	halDevice := d.halDevice()
	if halDevice == nil {
		return ErrReleased
	}
	if err := halDevice.WaitIdle(); err != nil {
		return err
	}
	d.maintainAfterIdle()
	return nil
}

// maintainAfterIdle recycles staging buffers and destroys deferred GPU resources
// after the HAL reports the device is idle. Without this, WaitIdle only blocks
// on the GPU — DestroyQueue entries scheduled before the wait are not processed
// until the next Submit, keeping old textures alive across resize boundaries.
func (d *Device) maintainAfterIdle() {
	if d.queue == nil || d.queue.hal == nil {
		return
	}
	completed := d.queue.hal.PollCompleted()
	if dq := d.destroyQueue(); dq != nil {
		dq.Triage(completed)
	}
	if d.queue.pending != nil {
		d.queue.pending.mu.Lock()
		d.queue.pending.maintain(completed)
		d.queue.pending.mu.Unlock()
	}
}

// Release releases the device and all associated resources.
// Deferred resource destructions are flushed before the device is destroyed.
// Shutdown order:
//  0. Retire active surface acquisitions
//  1. WaitIdle and maintain completed work
//  2. Discard unsubmitted queue writes
//  3. Triage + FlushAll — deferred callbacks fire (encoders return to pool)
//  4. Destroy encoder pool (HAL device still alive)
//  5. Destroy core + HAL device
//
// WaitIdle is required because FlushAll calls Triage(PollCompleted()),
// but PollCompleted may return a stale index if GPU hasn't finished.
// Without WaitIdle, deferred encoder recycling callbacks don't fire,
// and pool.destroy() destroys encoders whose VkCommandPool is then
// double-freed by hal.Device.Destroy() → vkDestroyCommandPool crash.
//
// Rust avoids this via Arc ownership + maintain loop. In Go we must
// be explicit: WaitIdle ensures PollCompleted returns final index.
func (d *Device) Release() {
	if d == nil || d.released.Swap(true) {
		return
	}
	var configuredSurfaces []*Surface
	if d.instance != nil {
		configuredSurfaces = d.instance.surfacesForDevice(d)
	}
	// Retire active acquisitions while the backend device is still alive.
	// Vulkan keeps the configured swapchain itself and retires it in
	// hal.Device.Destroy; other backends also get a chance to release borrowed
	// drawable state before their device disappears.
	for _, surface := range configuredSurfaces {
		surface.discardForDevice(d)
	}
	if d.instance != nil {
		defer func() {
			d.instance.unregisterDevice(d)
			d.instance = nil
		}()
	}

	// Step 1: Wait for ALL GPU work to finish. Use the internal path because
	// the public device has already been atomically marked released to close new
	// entry points. This ensures PollCompleted() returns the final submission
	// index before pending encoders or resources are destroyed.
	_ = d.waitIdle()

	// Step 2: Pending writes that were never submitted can now be discarded;
	// completed inflight batches were recycled by maintainAfterIdle above.
	if d.queue != nil {
		d.queue.release()
	}

	// Step 3: Flush deferred destructions. WaitIdle above ensures PollCompleted()
	// returns the final submission index, so Triage processes all submissions
	// and deferred encoder recycling callbacks fire correctly.
	if d.core != nil && d.core.DestroyQueueRef() != nil {
		dq := d.core.DestroyQueueRef()
		// Triage with latest completion index (GPU is idle, all done).
		if d.queue != nil && d.queue.hal != nil {
			dq.Triage(d.queue.hal.PollCompleted())
		}
		dq.FlushAll()
	}

	// Step 4: Destroy encoder pool. Each encoder.Destroy() calls
	// vkDestroyCommandPool / ID3D12CommandAllocator.Release on the still-alive
	// HAL device. After this, no native encoder resources remain.
	if d.cmdEncoderPool != nil {
		d.cmdEncoderPool.destroy()
		d.cmdEncoderPool = nil
	}

	// Step 5: Destroy core + HAL device. core.Destroy() calls FlushAll again
	// (idempotent — already flushed) then halDevice.Destroy().
	d.core.Destroy()
	for _, surface := range configuredSurfaces {
		surface.retireDevice(d)
	}
}

// destroyQueue returns the device's DestroyQueue for deferred resource destruction.
// Returns nil if the device has no HAL integration or no DestroyQueue.
func (d *Device) destroyQueue() *core.DestroyQueue {
	if d.core == nil {
		return nil
	}
	return d.core.DestroyQueueRef()
}

// Poll drives the per-device pending-map triage loop.
//
// PollPoll drains any buffer mappings whose associated GPU submissions
// have already completed and returns immediately. This is the fast path
// for game loops and is called automatically from Queue.Submit, so
// beginner code paths never need to call Poll explicitly.
//
// PollWait blocks until the GPU has finished every submission made before
// the call, then drains all buckets. It waits for the last of them, not
// for work submitted meanwhile, and does not hold up Submit. It is
// appropriate for shutdown drains and short scripts that do not run a
// render loop.
//
// The return value reports whether Poll observed any in-flight maps —
// tests use it to assert that Submit-driven auto-polling drained
// everything without needing an explicit Poll call. After Release, Poll
// does nothing and returns false.
func (d *Device) Poll(pollType PollType) bool {
	if d == nil || d.core == nil || d.released.Load() {
		return false
	}
	switch pollType {
	case PollWait:
		_ = d.waitSubmission(context.Background(), d.lastSubmissionIndex())
		// Everything submitted before the wait is done; a map against a
		// submission made since is not, so ask rather than assume all.
		completed := ^uint64(0)
		if d.queue != nil && d.queue.hal != nil {
			completed = d.queue.hal.PollCompleted()
		}
		return d.core.PollMaps(completed)
	default:
		completed := uint64(0)
		if d.queue != nil && d.queue.hal != nil {
			completed = d.queue.hal.PollCompleted()
		}
		return d.core.PollMaps(completed)
	}
}

// submissionWaiter is a backend queue that can wait for one of its
// submissions without its lock and without the work submitted after it.
type submissionWaiter interface {
	WaitSubmission(index uint64, timeout time.Duration) (bool, error)
}

// submissionWaitSlice is how long a wait for a submission blocks in the
// backend before it looks at its context again.
const submissionWaitSlice = 10 * time.Millisecond

// waitSubmission waits until the GPU has finished submission index, or ctx
// is done. A backend that cannot wait for one submission is polled.
func (d *Device) waitSubmission(ctx context.Context, index uint64) error {
	q := d.queue
	if q == nil || q.hal == nil {
		return nil
	}
	if w, ok := q.hal.(submissionWaiter); ok {
		timeout := time.Duration(-1)
		if ctx.Done() != nil {
			timeout = submissionWaitSlice
		}
		for {
			if done, err := w.WaitSubmission(index, timeout); done || err != nil {
				return err
			}
			if err := ctx.Err(); err != nil {
				return err
			}
		}
	}
	if q.hal.PollCompleted() >= index {
		return nil
	}
	delay := 50 * time.Microsecond
	timer := time.NewTimer(delay)
	defer timer.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timer.C:
		}
		if q.hal.PollCompleted() >= index {
			return nil
		}
		delay = min(2*delay, 2*time.Millisecond)
		timer.Reset(delay)
	}
}

// lastSubmissionIndex returns the latest submission index from the queue.
// Used by Release() methods to schedule deferred destruction.
func (d *Device) lastSubmissionIndex() uint64 {
	if d.queue == nil {
		return 0
	}
	return d.queue.LastSubmissionIndex()
}

// halDevice returns the underlying HAL device for direct resource creation.
func (d *Device) halDevice() hal.Device {
	if d.core == nil || !d.core.HasHAL() {
		return nil
	}
	guard := d.core.SnatchLock().Read()
	defer guard.Release()
	return d.core.Raw(guard)
}

// CreateRenderBundleEncoder creates an encoder for reusable render commands.
func (d *Device) CreateRenderBundleEncoder(desc *RenderBundleEncoderDescriptor) (_ *RenderBundleEncoder, err error) {
	defer d.reportError(&err)
	if desc == nil {
		return nil, fmt.Errorf("wgpu: render bundle encoder descriptor is nil")
	}

	if d.released.Load() || d.halDevice() == nil {
		return nil, ErrReleased
	}
	return &RenderBundleEncoder{device: d}, nil
}
