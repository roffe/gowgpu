//go:build js && wasm

package wgpu

import (
	"fmt"
	"syscall/js"
	"time"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/internal/browser"
)

// Device represents a logical GPU device.
// On browser, this wraps a GPUDevice via internal/browser.Device.
type Device struct {
	browser  *browser.Device
	queue    *Queue
	features gputypes.Features
	limits   gputypes.Limits
	released bool

	// errorScopes holds the filters of the pushed error scopes, innermost
	// last.
	errorScopes []ErrorFilter

	// mapsSettled counts the buffer maps that have settled, mapsPolled what
	// it was at the previous Poll.
	mapsSettled, mapsPolled uint64

	uncapturedErrorHandler func(*GPUError)
}

// usable returns ErrReleased for a released device and ErrDeviceLost for a
// lost one.
func (d *Device) usable() error {
	if d.released {
		return ErrReleased
	}
	return d.lostErr()
}

// lostErr returns ErrDeviceLost, with the browser's message, once the
// device is lost.
func (d *Device) lostErr() error {
	if d == nil || d.browser == nil {
		return nil
	}
	if msg, lost := d.browser.Lost(); lost {
		return fmt.Errorf("%w: %s", ErrDeviceLost, msg)
	}
	return nil
}

// Queue returns the device's command queue.
func (d *Device) Queue() *Queue {
	return d.queue
}

// Features returns the device's enabled features.
func (d *Device) Features() gputypes.Features {
	return d.features
}

// Limits returns the device's resource limits.
func (d *Device) Limits() gputypes.Limits {
	return d.limits
}

// CreateBuffer creates a GPU buffer from the given descriptor.
func (d *Device) CreateBuffer(desc *BufferDescriptor) (*Buffer, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: buffer descriptor is nil")
	}
	if desc.Size == 0 {
		return nil, fmt.Errorf("wgpu: CreateBuffer: buffer %q has zero size", desc.Label)
	}
	if desc.Size > d.limits.MaxBufferSize {
		return nil, fmt.Errorf("wgpu: CreateBuffer: buffer %q size %d exceeds maxBufferSize %d", desc.Label, desc.Size, d.limits.MaxBufferSize)
	}
	if desc.Usage == 0 {
		return nil, fmt.Errorf("wgpu: CreateBuffer: buffer %q has no usage", desc.Label)
	}
	if desc.Usage.ContainsUnknownBits() {
		return nil, fmt.Errorf("wgpu: CreateBuffer: buffer %q has unknown usage bits", desc.Label)
	}
	if desc.Usage.Contains(BufferUsageMapRead) && desc.Usage.Contains(BufferUsageMapWrite) {
		return nil, fmt.Errorf("wgpu: CreateBuffer: buffer %q has both MapRead and MapWrite usage", desc.Label)
	}
	// The GPU buffer is a multiple of 4 bytes, which a mapped-at-creation
	// buffer has to be; Size stays what was asked for.
	jsDesc := browser.BuildBufferDescriptor(
		desc.Label,
		(desc.Size+3)&^3,
		uint64(desc.Usage),
		desc.MappedAtCreation,
	)
	bb, err := d.browser.CreateBufferFromDesc(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreateBuffer: %w", err)
	}

	state := MapStateUnmapped
	if desc.MappedAtCreation {
		// When mappedAtCreation is true the buffer starts in the Mapped state
		// with the entire range [0, size) mapped. Record this so that
		// MappedRange / Unmap work without calling MapAsync first.
		bb.SetMappedAtCreation()
		state = MapStateMapped
	}

	return &Buffer{
		browser:  bb,
		device:   d,
		size:     desc.Size,
		usage:    desc.Usage,
		released: false,
		mapState: state,
	}, nil
}

// CreateTexture creates a GPU texture from the given descriptor.
func (d *Device) CreateTexture(desc *TextureDescriptor) (*Texture, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: texture descriptor is nil")
	}
	if f := browser.TextureFormatFeature(desc.Format); f != 0 && !d.features.Contains(f) {
		return nil, fmt.Errorf("wgpu: CreateTexture: format %s needs the %s feature", desc.Format, f)
	}
	jsDesc := browser.BuildTextureDescriptor(
		desc.Label,
		desc.Size.Width, desc.Size.Height, desc.Size.DepthOrArrayLayers,
		desc.MipLevelCount, desc.SampleCount,
		desc.Dimension, desc.Format, desc.Usage,
		desc.ViewFormats,
	)
	bt, err := d.browser.CreateTextureFromDesc(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreateTexture: %w", err)
	}
	return &Texture{
		browser:  bt,
		format:   desc.Format,
		released: false,
	}, nil
}

// CreateTextureView creates a view into a texture.
func (d *Device) CreateTextureView(texture *Texture, desc *TextureViewDescriptor) (*TextureView, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if texture == nil || texture.browser == nil {
		return nil, ErrReleased
	}
	if desc == nil {
		desc = &TextureViewDescriptor{}
	}
	jsDesc := browser.BuildTextureViewDescriptor(
		desc.Label,
		desc.Format, desc.Dimension, desc.Aspect,
		desc.BaseMipLevel, desc.MipLevelCount,
		desc.BaseArrayLayer, desc.ArrayLayerCount,
	)
	bv, err := texture.browser.CreateView(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreateTextureView: %w", err)
	}
	return &TextureView{
		browser:  bv,
		texture:  texture,
		released: false,
	}, nil
}

// CreateSampler creates a texture sampler from the given descriptor.
func (d *Device) CreateSampler(desc *SamplerDescriptor) (*Sampler, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if desc == nil {
		desc = &SamplerDescriptor{}
	}
	jsDesc := browser.BuildSamplerDescriptor(
		desc.Label,
		desc.AddressModeU, desc.AddressModeV, desc.AddressModeW,
		desc.MagFilter, desc.MinFilter, desc.MipmapFilter,
		desc.LodMinClamp, desc.LodMaxClamp,
		desc.Compare, desc.Anisotropy,
	)
	bs, err := d.browser.CreateSamplerFromDesc(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreateSampler: %w", err)
	}
	return &Sampler{
		browser:  bs,
		released: false,
	}, nil
}

// CreateShaderModule creates a shader module from the given descriptor.
// On browser, WGSL code goes directly to the browser's createShaderModule.
// SPIR-V bytecode is not supported in the browser (browser only accepts WGSL).
func (d *Device) CreateShaderModule(desc *ShaderModuleDescriptor) (*ShaderModule, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: shader module descriptor is nil")
	}
	jsDesc := browser.BuildShaderModuleDescriptor(desc.Label, desc.WGSL)
	bm, err := d.browser.CreateShaderModuleFromDesc(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreateShaderModule: %w", err)
	}
	return &ShaderModule{
		browser:  bm,
		released: false,
	}, nil
}

// CreateBindGroupLayout creates a bind group layout from the given descriptor.
func (d *Device) CreateBindGroupLayout(desc *BindGroupLayoutDescriptor) (*BindGroupLayout, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: bind group layout descriptor is nil")
	}
	entries := convertBindGroupLayoutEntries(desc.Entries)
	jsDesc := browser.BuildBindGroupLayoutDescriptor(desc.Label, entries)
	bl, err := d.browser.CreateBindGroupLayoutFromDesc(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreateBindGroupLayout: %w", err)
	}
	return &BindGroupLayout{
		browser:  bl,
		released: false,
	}, nil
}

// CreatePipelineLayout creates a pipeline layout from the given descriptor.
func (d *Device) CreatePipelineLayout(desc *PipelineLayoutDescriptor) (*PipelineLayout, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: pipeline layout descriptor is nil")
	}
	refs := make([]js.Value, len(desc.BindGroupLayouts))
	for i, bgl := range desc.BindGroupLayouts {
		if bgl != nil && bgl.browser != nil {
			refs[i] = bgl.browser.Ref()
		} else {
			refs[i] = js.Undefined()
		}
	}
	jsDesc := browser.BuildPipelineLayoutDescriptor(desc.Label, refs)
	bp, err := d.browser.CreatePipelineLayoutFromDesc(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreatePipelineLayout: %w", err)
	}
	return &PipelineLayout{
		browser:  bp,
		released: false,
	}, nil
}

// CreateBindGroup creates a bind group from the given descriptor.
func (d *Device) CreateBindGroup(desc *BindGroupDescriptor) (*BindGroup, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: bind group descriptor is nil")
	}
	var layoutRef js.Value
	if desc.Layout != nil && desc.Layout.browser != nil {
		layoutRef = desc.Layout.browser.Ref()
	} else {
		layoutRef = js.Undefined()
	}
	entries := convertBindGroupEntries(desc.Entries)
	jsDesc := browser.BuildBindGroupDescriptor(desc.Label, layoutRef, entries)
	bg, err := d.browser.CreateBindGroupFromDesc(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreateBindGroup: %w", err)
	}
	return &BindGroup{
		browser:  bg,
		released: false,
	}, nil
}

// CreateRenderPipeline creates a render pipeline from the given descriptor.
func (d *Device) CreateRenderPipeline(desc *RenderPipelineDescriptor) (*RenderPipeline, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: render pipeline descriptor is nil")
	}
	jsDesc := convertRenderPipelineDescriptor(desc)
	bp, err := d.browser.CreateRenderPipelineFromDesc(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreateRenderPipeline: %w", err)
	}
	return &RenderPipeline{
		browser:  bp,
		released: false,
	}, nil
}

// CreateComputePipeline creates a compute pipeline from the given descriptor.
func (d *Device) CreateComputePipeline(desc *ComputePipelineDescriptor) (*ComputePipeline, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	if desc == nil {
		return nil, fmt.Errorf("wgpu: compute pipeline descriptor is nil")
	}
	var layoutRef js.Value
	if desc.Layout != nil && desc.Layout.browser != nil {
		layoutRef = desc.Layout.browser.Ref()
	} else {
		layoutRef = js.Undefined()
	}
	var moduleRef js.Value
	if desc.Module != nil && desc.Module.browser != nil {
		moduleRef = desc.Module.browser.Ref()
	} else {
		moduleRef = js.Undefined()
	}
	jsDesc := browser.BuildComputePipelineDescriptor(
		desc.Label, layoutRef, moduleRef, desc.EntryPoint, desc.Constants,
	)
	bp, err := d.browser.CreateComputePipelineFromDesc(jsDesc)
	if err != nil {
		return nil, fmt.Errorf("wgpu: CreateComputePipeline: %w", err)
	}
	return &ComputePipeline{
		browser:  bp,
		released: false,
	}, nil
}

// CreateCommandEncoder creates a command encoder for recording GPU commands.
func (d *Device) CreateCommandEncoder(desc *CommandEncoderDescriptor) (*CommandEncoder, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	label := ""
	if desc != nil {
		label = desc.Label
	}
	jsDesc := browser.BuildCommandEncoderDescriptor(label)
	jsEncoder := d.browser.CreateCommandEncoder().Invoke(jsDesc)
	be := browser.NewCommandEncoder(jsEncoder)
	return &CommandEncoder{
		browser:  be,
		released: false,
	}, nil
}

// CreateFence creates a GPU synchronization fence.
// On browser, fences are not needed (browser auto-polls).
// Returns a no-op fence for API compatibility.
func (d *Device) CreateFence() (*Fence, error) {
	if err := d.usable(); err != nil {
		return nil, err
	}
	return &Fence{}, nil
}

// DestroyFence destroys a fence. No-op on browser — fences are JS-managed.
func (d *Device) DestroyFence(f *Fence) {
	if f != nil {
		f.Release()
	}
}

// ResetFence resets a fence. No-op on browser — fences auto-reset.
func (d *Device) ResetFence(_ *Fence) error { return nil }

// GetFenceStatus returns true (signaled). Browser fences resolve via JS Promises.
func (d *Device) GetFenceStatus(_ *Fence) (bool, error) { return true, nil }

// WaitForFence returns immediately on browser — GPU sync is handled by JS event loop.
func (d *Device) WaitForFence(_ *Fence, _ uint64, _ time.Duration) (bool, error) {
	return true, nil
}

// PushErrorScope pushes a new error scope onto the device's error scope stack.
func (d *Device) PushErrorScope(filter ErrorFilter) {
	d.errorScopes = append(d.errorScopes, filter)
	d.browser.PushErrorScope(filter.toJS())
}

// PopErrorScope pops the most recently pushed error scope.
// Returns the captured error, or nil if no error occurred.
// Panics if no error scope is pushed.
//
// The browser reports the scope's error through a promise, so PopErrorScope
// blocks until it settles: call it from main or another goroutine, never
// from a JavaScript callback.
func (d *Device) PopErrorScope() *GPUError {
	n := len(d.errorScopes)
	if n == 0 {
		panic("PopErrorScope: error scope stack is empty: no matching PushErrorScope")
	}
	filter := d.errorScopes[n-1]
	d.errorScopes = d.errorScopes[:n-1]
	jsErr, err := d.browser.PopErrorScope()
	if err != nil {
		panic(fmt.Sprintf("PopErrorScope: %v", err))
	}
	if jsErr.Type() != js.TypeObject {
		return nil
	}
	return &GPUError{Type: filter, Message: jsErr.Get("message").String()}
}

// SetUncapturedErrorHandler sets the function that gets every GPU error no
// error scope captured: validation, out-of-memory and internal errors, each
// with its Type and message. A newer call replaces the older handler, and
// nil removes it and restores the default handling of uncaptured errors.
//
// In the browser the handler runs as a js.FuncOf callback for the device's
// uncapturederror event, which the browser's event loop waits for, so it
// must not block: hand the error to another goroutine for work that waits.
// While a handler is set, a browser that lets the event be canceled leaves
// out its console warning for the error, and a browser may stop reporting
// errors once there are many.
func (d *Device) SetUncapturedErrorHandler(handler func(*GPUError)) {
	d.uncapturedErrorHandler = handler
	if handler == nil {
		d.browser.SetUncapturedErrorListener(nil)
		return
	}
	d.browser.SetUncapturedErrorListener(func(gpuError js.Value) {
		if h := d.uncapturedErrorHandler; h != nil {
			h(&GPUError{Type: errorFilterOf(gpuError), Message: gpuError.Get("message").String()})
		}
	})
}

// errorFilterOf returns the filter that captures gpuError, a GPUError.
func errorFilterOf(gpuError js.Value) ErrorFilter {
	for _, kind := range []struct {
		class  string
		filter ErrorFilter
	}{
		{"GPUOutOfMemoryError", ErrorFilterOutOfMemory},
		{"GPUInternalError", ErrorFilterInternal},
	} {
		if class := js.Global().Get(kind.class); class.Type() == js.TypeFunction && gpuError.InstanceOf(class) {
			return kind.filter
		}
	}
	return ErrorFilterValidation
}

// WaitIdle waits for all GPU work to complete.
//
// On browser it waits for queue.onSubmittedWorkDone(), which also settles
// every map made before it, so it must be called from main or another
// goroutine, never from a JavaScript callback.
func (d *Device) WaitIdle() error {
	if err := d.usable(); err != nil {
		return err
	}
	if err := d.queue.browser.OnSubmittedWorkDone(); err != nil {
		return fmt.Errorf("wgpu: WaitIdle: %w", err)
	}
	return nil
}

// Poll reports whether buffer maps have settled since the previous Poll.
//
// On browser, maps settle in the browser's promise callbacks whenever its
// event loop runs. PollPoll returns at once; PollWait first waits for
// queue.onSubmittedWorkDone(), which settles every map made before it, so
// PollWait must be called from main or another goroutine, never from a
// JavaScript callback.
func (d *Device) Poll(pollType PollType) bool {
	if d == nil || d.browser == nil || d.released {
		return false
	}
	if pollType == PollWait {
		_ = d.queue.browser.OnSubmittedWorkDone()
	}
	settled := d.mapsSettled != d.mapsPolled
	d.mapsPolled = d.mapsSettled
	return settled
}

// FreeCommandBuffer is a no-op on browser — JS GC handles GPU resource cleanup.
// This exists for API compatibility with the native backend where command buffers
// must be explicitly freed after GPU submission completes.
func (d *Device) FreeCommandBuffer(_ *CommandBuffer) {}

// HalDevice returns nil on browser — there is no HAL layer.
// The browser backend talks directly to the browser WebGPU API.
// This exists for API compatibility with the native backend.
func (d *Device) HalDevice() any { return nil }

// Release releases the device and all associated resources.
func (d *Device) Release() {
	if d.released {
		return
	}
	d.released = true
	if d.browser != nil {
		d.browser.SetUncapturedErrorListener(nil)
		d.browser.Destroy()
	}
}

// --- Descriptor conversion helpers ---

// convertBindGroupLayoutEntries converts Go BindGroupLayoutEntry slice to
// browser.BindGroupLayoutEntryJS slice for JS object construction.
func convertBindGroupLayoutEntries(entries []gputypes.BindGroupLayoutEntry) []browser.BindGroupLayoutEntryJS {
	result := make([]browser.BindGroupLayoutEntryJS, len(entries))
	for i, e := range entries {
		entry := browser.BindGroupLayoutEntryJS{
			Binding:    e.Binding,
			Visibility: uint32(e.Visibility),
		}
		if e.Buffer != nil {
			entry.Buffer = &browser.BufferBindingLayoutJS{
				Type:             browser.BufferBindingTypeToJS(e.Buffer.Type),
				HasDynamicOffset: e.Buffer.HasDynamicOffset,
				MinBindingSize:   e.Buffer.MinBindingSize,
			}
		}
		if e.Sampler != nil {
			entry.Sampler = &browser.SamplerBindingLayoutJS{
				Type: browser.SamplerBindingTypeToJS(e.Sampler.Type),
			}
		}
		if e.Texture != nil {
			entry.Texture = &browser.TextureBindingLayoutJS{
				SampleType:    browser.TextureSampleTypeToJS(e.Texture.SampleType),
				ViewDimension: browser.TextureViewDimensionToJS(e.Texture.ViewDimension),
				Multisampled:  e.Texture.Multisampled,
			}
		}
		if e.StorageTexture != nil {
			entry.StorageTexture = &browser.StorageTextureBindingLayoutJS{
				Access:        browser.StorageTextureAccessToJS(e.StorageTexture.Access),
				Format:        browser.TextureFormatToJS(e.StorageTexture.Format),
				ViewDimension: browser.TextureViewDimensionToJS(e.StorageTexture.ViewDimension),
			}
		}
		result[i] = entry
	}
	return result
}

// convertBindGroupEntries converts Go BindGroupEntry slice to
// browser.BindGroupEntryJS slice for JS object construction.
func convertBindGroupEntries(entries []BindGroupEntry) []browser.BindGroupEntryJS {
	result := make([]browser.BindGroupEntryJS, len(entries))
	for i, e := range entries {
		entry := browser.BindGroupEntryJS{
			Binding:        e.Binding,
			BufferRef:      js.Undefined(),
			SamplerRef:     js.Undefined(),
			TextureViewRef: js.Undefined(),
		}
		if e.Buffer != nil && e.Buffer.browser != nil {
			entry.BufferRef = e.Buffer.browser.Ref()
			entry.BufferOffset = e.Offset
			entry.BufferSize = e.Size
		}
		if e.Sampler != nil && e.Sampler.browser != nil {
			entry.SamplerRef = e.Sampler.browser.Ref()
		}
		if e.TextureView != nil && e.TextureView.browser != nil {
			entry.TextureViewRef = e.TextureView.browser.Ref()
		}
		result[i] = entry
	}
	return result
}

// convertRenderPipelineDescriptor converts a Go RenderPipelineDescriptor to
// a JS descriptor object via browser.BuildRenderPipelineDescriptor.
func convertRenderPipelineDescriptor(desc *RenderPipelineDescriptor) js.Value {
	rpd := &browser.RenderPipelineDescriptorJS{
		Label: desc.Label,
	}

	// Layout
	if desc.Layout != nil && desc.Layout.browser != nil {
		rpd.LayoutRef = desc.Layout.browser.Ref()
	} else {
		rpd.LayoutRef = js.Undefined()
	}

	// Vertex state
	rpd.Vertex = browser.VertexStateJS{
		EntryPoint: desc.Vertex.EntryPoint,
	}
	if desc.Vertex.Module != nil && desc.Vertex.Module.browser != nil {
		rpd.Vertex.ModuleRef = desc.Vertex.Module.browser.Ref()
	} else {
		rpd.Vertex.ModuleRef = js.Undefined()
	}
	rpd.Vertex.Buffers = convertVertexBufferLayouts(desc.Vertex.Buffers)

	// Primitive state
	rpd.Primitive = &browser.PrimitiveStateJS{
		Topology:  browser.PrimitiveTopologyToJS(desc.Primitive.Topology),
		FrontFace: browser.FrontFaceToJS(desc.Primitive.FrontFace),
		CullMode:  browser.CullModeToJS(desc.Primitive.CullMode),
	}
	if desc.Primitive.StripIndexFormat != nil {
		rpd.Primitive.StripIndexFormat = browser.IndexFormatToJS(*desc.Primitive.StripIndexFormat)
	}
	if desc.Primitive.UnclippedDepth {
		rpd.Primitive.UnclippedDepth = true
	}

	// Depth-stencil state
	if desc.DepthStencil != nil {
		rpd.DepthStencil = convertDepthStencilState(desc.DepthStencil)
	}

	// Multisample state
	rpd.Multisample = &browser.MultisampleStateJS{
		Count:                  desc.Multisample.Count,
		Mask:                   desc.Multisample.Mask,
		AlphaToCoverageEnabled: desc.Multisample.AlphaToCoverageEnabled,
	}

	// Fragment state
	if desc.Fragment != nil {
		rpd.Fragment = convertFragmentState(desc.Fragment)
	}

	return browser.BuildRenderPipelineDescriptor(rpd)
}

// convertVertexBufferLayouts converts Go VertexBufferLayout slice to JS types.
func convertVertexBufferLayouts(layouts []gputypes.VertexBufferLayout) []browser.VertexBufferLayoutJS {
	result := make([]browser.VertexBufferLayoutJS, len(layouts))
	for i, l := range layouts {
		jsLayout := browser.VertexBufferLayoutJS{
			ArrayStride: l.ArrayStride,
			StepMode:    browser.VertexStepModeToJS(l.StepMode),
		}
		jsLayout.Attributes = make([]browser.VertexAttributeJS, len(l.Attributes))
		for j, a := range l.Attributes {
			jsLayout.Attributes[j] = browser.VertexAttributeJS{
				Format:         browser.VertexFormatToJS(a.Format),
				Offset:         a.Offset,
				ShaderLocation: a.ShaderLocation,
			}
		}
		result[i] = jsLayout
	}
	return result
}

// convertDepthStencilState converts a Go DepthStencilState to JS types.
func convertDepthStencilState(ds *DepthStencilState) *browser.DepthStencilStateJS {
	result := &browser.DepthStencilStateJS{
		Format:              browser.TextureFormatToJS(ds.Format),
		DepthWriteEnabled:   ds.DepthWriteEnabled,
		DepthCompare:        browser.CompareFunctionToJS(ds.DepthCompare),
		StencilReadMask:     ds.StencilReadMask,
		StencilWriteMask:    ds.StencilWriteMask,
		DepthBias:           ds.DepthBias,
		DepthBiasSlopeScale: ds.DepthBiasSlopeScale,
		DepthBiasClamp:      ds.DepthBiasClamp,
	}
	result.StencilFront = convertStencilFaceState(&ds.StencilFront)
	result.StencilBack = convertStencilFaceState(&ds.StencilBack)
	return result
}

// convertStencilFaceState converts a Go StencilFaceState to JS types.
func convertStencilFaceState(s *StencilFaceState) *browser.StencilFaceStateJS {
	return &browser.StencilFaceStateJS{
		Compare:     browser.CompareFunctionToJS(s.Compare),
		FailOp:      stencilOpToJS(s.FailOp),
		DepthFailOp: stencilOpToJS(s.DepthFailOp),
		PassOp:      stencilOpToJS(s.PassOp),
	}
}

// stencilOpToJS converts StencilOperation (gputypes, webgpu.h spec values) to WebGPU JS string.
func stencilOpToJS(op gputypes.StencilOperation) string {
	switch op {
	case StencilOperationKeep:
		return "keep"
	case StencilOperationZero:
		return "zero"
	case StencilOperationReplace:
		return "replace"
	case StencilOperationInvert:
		return "invert"
	case StencilOperationIncrementClamp:
		return "increment-clamp"
	case StencilOperationDecrementClamp:
		return "decrement-clamp"
	case StencilOperationIncrementWrap:
		return "increment-wrap"
	case StencilOperationDecrementWrap:
		return "decrement-wrap"
	default:
		return "keep"
	}
}

// convertFragmentState converts a Go FragmentState to JS types.
func convertFragmentState(fs *FragmentState) *browser.FragmentStateJS {
	result := &browser.FragmentStateJS{
		EntryPoint: fs.EntryPoint,
	}
	if fs.Module != nil && fs.Module.browser != nil {
		result.ModuleRef = fs.Module.browser.Ref()
	} else {
		result.ModuleRef = js.Undefined()
	}

	result.Targets = make([]*browser.ColorTargetStateJS, len(fs.Targets))
	for i, t := range fs.Targets {
		if t.Format == gputypes.TextureFormatUndefined {
			continue
		}
		ct := &browser.ColorTargetStateJS{
			Format:    browser.TextureFormatToJS(t.Format),
			WriteMask: uint32(t.WriteMask),
		}
		if t.Blend != nil {
			ct.Blend = &browser.BlendStateJS{
				Color: browser.BlendComponentJS{
					SrcFactor: browser.BlendFactorToJS(t.Blend.Color.SrcFactor),
					DstFactor: browser.BlendFactorToJS(t.Blend.Color.DstFactor),
					Operation: browser.BlendOperationToJS(t.Blend.Color.Operation),
				},
				Alpha: browser.BlendComponentJS{
					SrcFactor: browser.BlendFactorToJS(t.Blend.Alpha.SrcFactor),
					DstFactor: browser.BlendFactorToJS(t.Blend.Alpha.DstFactor),
					Operation: browser.BlendOperationToJS(t.Blend.Alpha.Operation),
				},
			}
		}
		result.Targets[i] = ct
	}
	return result
}
