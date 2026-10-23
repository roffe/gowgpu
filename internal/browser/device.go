//go:build js && wasm

package browser

import "syscall/js"

// Device wraps a browser GPUDevice with pre-bound creation methods.
//
// Pre-binding JS methods at construction time avoids repeated .Get("methodName")
// calls on every frame. This pattern is used by Ebiten and other Go WASM libraries
// for optimal performance.
//
// Matches Rust wgpu WebDevice which holds the webgpu_sys::GpuDevice value.
type Device struct {
	// ref_ is the GPUDevice JavaScript object.
	ref_ js.Value

	// queue is the device's command queue (GPUQueue).
	queue *Queue

	// features is the device's GPUSupportedFeatures.
	features js.Value

	// limits is the device's GPUSupportedLimits.
	limits js.Value

	// Pre-bound creation methods. Each is the result of
	// device.methodName.bind(device), so calling fnX.Invoke(args...)
	// is equivalent to device.methodName(args...) but without the
	// property lookup overhead on each call.
	fnCreateBuffer          js.Value
	fnCreateTexture         js.Value
	fnCreateShaderModule    js.Value
	fnCreateBindGroupLayout js.Value
	fnCreateBindGroup       js.Value
	fnCreatePipelineLayout  js.Value
	fnCreateRenderPipeline  js.Value
	fnCreateComputePipeline js.Value
	fnCreateCommandEncoder  js.Value
	fnCreateSampler         js.Value
	fnCreateQuerySet        js.Value

	// lostInfo gets the GPUDeviceLostInfo as its info property once the
	// device is lost.
	lostInfo    js.Value
	lostMessage string
	lost        bool

	// onUncapturedError gets the GPUError of each uncapturederror event
	// while uncapturedListener listens for them.
	onUncapturedError  func(gpuError js.Value)
	uncapturedListener js.Func
	listening          bool
}

// NewDevice constructs a Device from a GPUDevice js.Value.
// Pre-binds all creation methods and extracts the queue.
func NewDevice(ref js.Value) *Device {
	d := &Device{
		ref_:     ref,
		features: ref.Get("features"),
		limits:   ref.Get("limits"),
	}

	// Extract the device's queue (GPUDevice.queue is a readonly attribute).
	d.queue = NewQueue(ref.Get("queue"))

	// Pre-bind all creation methods to avoid property lookups on hot paths.
	// pattern: device.createX.bind(device) => fnCreateX
	d.fnCreateBuffer = bindMethod(ref, "createBuffer")
	d.fnCreateTexture = bindMethod(ref, "createTexture")
	d.fnCreateShaderModule = bindMethod(ref, "createShaderModule")
	d.fnCreateBindGroupLayout = bindMethod(ref, "createBindGroupLayout")
	d.fnCreateBindGroup = bindMethod(ref, "createBindGroup")
	d.fnCreatePipelineLayout = bindMethod(ref, "createPipelineLayout")
	d.fnCreateRenderPipeline = bindMethod(ref, "createRenderPipeline")
	d.fnCreateComputePipeline = bindMethod(ref, "createComputePipeline")
	d.fnCreateCommandEncoder = bindMethod(ref, "createCommandEncoder")
	d.fnCreateSampler = bindMethod(ref, "createSampler")
	d.fnCreateQuerySet = bindMethod(ref, "createQuerySet")

	// The lost promise stores its GPUDeviceLostInfo on lostInfo through a
	// builtin, not a Go callback, which could fire after the program exits.
	if lost := ref.Get("lost"); lost.Type() == js.TypeObject {
		d.lostInfo = newJSObject()
		reflectSet := js.Global().Get("Reflect").Get("set")
		lost.Call("then", reflectSet.Call("bind", nil, d.lostInfo, "info"))
	}

	return d
}

// setLost records the loss a GPUDeviceLostInfo describes.
func (d *Device) setLost(info js.Value) {
	d.lost = true
	for _, key := range []string{"message", "reason"} {
		if v := info.Get(key); v.Type() == js.TypeString && v.String() != "" {
			d.lostMessage = v.String()
			return
		}
	}
}

// Lost reports whether the device has been lost, and the browser's message
// for it. A lost device stays lost.
func (d *Device) Lost() (string, bool) {
	if !d.lost && d.lostInfo.Type() == js.TypeObject {
		if info := d.lostInfo.Get("info"); info.Type() == js.TypeObject {
			d.setLost(info)
		}
	}
	return d.lostMessage, d.lost
}

// bindMethod returns methodName.bind(obj) so that Invoke() calls the method
// with the correct `this` context.
func bindMethod(obj js.Value, methodName string) js.Value {
	method := obj.Get(methodName)
	if method.IsUndefined() || method.IsNull() {
		// Method not available — return undefined. Callers will get a
		// clear JS error when they try to invoke it.
		return js.Undefined()
	}
	return method.Call("bind", obj)
}

// Queue returns the device's command queue.
func (d *Device) Queue() *Queue {
	return d.queue
}

// Features returns the device's GPUSupportedFeatures js.Value.
func (d *Device) Features() js.Value {
	return d.features
}

// Limits returns the device's GPUSupportedLimits js.Value.
func (d *Device) Limits() js.Value {
	return d.limits
}

// Ref returns the underlying GPUDevice js.Value.
func (d *Device) Ref() js.Value {
	return d.ref_
}

// CreateBuffer returns the pre-bound createBuffer function.
// Call with: d.CreateBuffer().Invoke(descriptorObj)
func (d *Device) CreateBuffer() js.Value { return d.fnCreateBuffer }

// CreateTexture returns the pre-bound createTexture function.
func (d *Device) CreateTexture() js.Value { return d.fnCreateTexture }

// CreateShaderModule returns the pre-bound createShaderModule function.
func (d *Device) CreateShaderModule() js.Value { return d.fnCreateShaderModule }

// CreateBindGroupLayout returns the pre-bound createBindGroupLayout function.
func (d *Device) CreateBindGroupLayout() js.Value { return d.fnCreateBindGroupLayout }

// CreateBindGroup returns the pre-bound createBindGroup function.
func (d *Device) CreateBindGroup() js.Value { return d.fnCreateBindGroup }

// CreatePipelineLayout returns the pre-bound createPipelineLayout function.
func (d *Device) CreatePipelineLayout() js.Value { return d.fnCreatePipelineLayout }

// CreateRenderPipeline returns the pre-bound createRenderPipeline function.
func (d *Device) CreateRenderPipeline() js.Value { return d.fnCreateRenderPipeline }

// CreateComputePipeline returns the pre-bound createComputePipeline function.
func (d *Device) CreateComputePipeline() js.Value { return d.fnCreateComputePipeline }

// CreateCommandEncoder returns the pre-bound createCommandEncoder function.
func (d *Device) CreateCommandEncoder() js.Value { return d.fnCreateCommandEncoder }

// CreateSampler returns the pre-bound createSampler function.
func (d *Device) CreateSampler() js.Value { return d.fnCreateSampler }

// CreateQuerySet returns the pre-bound createQuerySet function.
func (d *Device) CreateQuerySet() js.Value { return d.fnCreateQuerySet }

// CreateBufferFromDesc creates a GPUBuffer from a JS descriptor object, or
// returns the exception createBuffer throws as an error.
func (d *Device) CreateBufferFromDesc(desc js.Value) (*Buffer, error) {
	ref, err := invoke(d.fnCreateBuffer, desc)
	if err != nil {
		return nil, err
	}
	return NewBuffer(ref), nil
}

// CreateTextureFromDesc creates a GPUTexture from a JS descriptor object, or
// returns the exception createTexture throws as an error.
func (d *Device) CreateTextureFromDesc(desc js.Value) (*Texture, error) {
	ref, err := invoke(d.fnCreateTexture, desc)
	if err != nil {
		return nil, err
	}
	return NewTexture(ref), nil
}

// CreateSamplerFromDesc creates a GPUSampler from a JS descriptor object, or
// returns the exception createSampler throws as an error.
func (d *Device) CreateSamplerFromDesc(desc js.Value) (*Sampler, error) {
	ref, err := invoke(d.fnCreateSampler, desc)
	if err != nil {
		return nil, err
	}
	return NewSampler(ref), nil
}

// CreateShaderModuleFromDesc creates a GPUShaderModule from a JS descriptor object, or
// returns the exception createShaderModule throws as an error.
func (d *Device) CreateShaderModuleFromDesc(desc js.Value) (*ShaderModule, error) {
	ref, err := invoke(d.fnCreateShaderModule, desc)
	if err != nil {
		return nil, err
	}
	return NewShaderModule(ref), nil
}

// CreateBindGroupLayoutFromDesc creates a GPUBindGroupLayout from a JS descriptor object, or
// returns the exception createBindGroupLayout throws as an error.
func (d *Device) CreateBindGroupLayoutFromDesc(desc js.Value) (*BindGroupLayout, error) {
	ref, err := invoke(d.fnCreateBindGroupLayout, desc)
	if err != nil {
		return nil, err
	}
	return NewBindGroupLayout(ref), nil
}

// CreateBindGroupFromDesc creates a GPUBindGroup from a JS descriptor object, or
// returns the exception createBindGroup throws as an error.
func (d *Device) CreateBindGroupFromDesc(desc js.Value) (*BindGroup, error) {
	ref, err := invoke(d.fnCreateBindGroup, desc)
	if err != nil {
		return nil, err
	}
	return NewBindGroup(ref), nil
}

// CreatePipelineLayoutFromDesc creates a GPUPipelineLayout from a JS descriptor object, or
// returns the exception createPipelineLayout throws as an error.
func (d *Device) CreatePipelineLayoutFromDesc(desc js.Value) (*PipelineLayout, error) {
	ref, err := invoke(d.fnCreatePipelineLayout, desc)
	if err != nil {
		return nil, err
	}
	return NewPipelineLayout(ref), nil
}

// CreateRenderPipelineFromDesc creates a GPURenderPipeline from a JS descriptor object, or
// returns the exception createRenderPipeline throws as an error.
func (d *Device) CreateRenderPipelineFromDesc(desc js.Value) (*RenderPipeline, error) {
	ref, err := invoke(d.fnCreateRenderPipeline, desc)
	if err != nil {
		return nil, err
	}
	return NewRenderPipeline(ref), nil
}

// CreateComputePipelineFromDesc creates a GPUComputePipeline from a JS descriptor object, or
// returns the exception createComputePipeline throws as an error.
func (d *Device) CreateComputePipelineFromDesc(desc js.Value) (*ComputePipeline, error) {
	ref, err := invoke(d.fnCreateComputePipeline, desc)
	if err != nil {
		return nil, err
	}
	return NewComputePipeline(ref), nil
}

// PushErrorScope pushes an error scope for filter, a GPUErrorFilter string.
func (d *Device) PushErrorScope(filter string) {
	d.ref_.Call("pushErrorScope", filter)
}

// PopErrorScope pops the innermost error scope and blocks until the browser
// settles it. It returns the scope's GPUError, or null when it caught none,
// and an error when the browser rejects the pop.
func (d *Device) PopErrorScope() (js.Value, error) {
	return AwaitPromise(d.ref_.Call("popErrorScope"))
}

// SetUncapturedErrorListener has fn called with the GPUError of every
// uncapturederror event the device fires, and cancels the event, which
// leaves out the browser's console warning where the event is cancelable.
// With fn nil it stops listening. fn runs in the event's callback, so it
// must not block.
func (d *Device) SetUncapturedErrorListener(fn func(gpuError js.Value)) {
	d.onUncapturedError = fn
	switch {
	case fn != nil && !d.listening:
		d.uncapturedListener = js.FuncOf(func(_ js.Value, args []js.Value) any {
			event := args[0]
			event.Call("preventDefault")
			if d.onUncapturedError != nil {
				d.onUncapturedError(event.Get("error"))
			}
			return nil
		})
		d.ref_.Call("addEventListener", "uncapturederror", d.uncapturedListener)
		d.listening = true
	case fn == nil && d.listening:
		d.ref_.Call("removeEventListener", "uncapturederror", d.uncapturedListener)
		d.uncapturedListener.Release()
		d.listening = false
	}
}

// Destroy calls GPUDevice.destroy() to release GPU resources.
// After this call the device is no longer usable.
func (d *Device) Destroy() {
	destroy := d.ref_.Get("destroy")
	if !destroy.IsUndefined() && !destroy.IsNull() {
		d.ref_.Call("destroy")
	}
}
