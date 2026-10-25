//go:build js && wasm

package wgpu

import (
	"errors"
	"strings"
	"syscall/js"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/internal/browser"
)

func jsEval(expr string) js.Value {
	return js.Global().Get("Function").New("return " + expr).Invoke()
}

func yieldToJS(t *testing.T) {
	t.Helper()
	if _, err := browser.AwaitPromise(jsEval("Promise.resolve()")); err != nil {
		t.Fatal(err)
	}
}

// fakeDevice returns a GPUDevice stand-in whose lost promise is lost and
// whose error scope stack records the filters it is given in pushed.
func fakeDevice(lost js.Value) (dev, pushed js.Value) {
	dev = jsEval(`{
		queue: {submit() {}},
		pushed: [],
		pushErrorScope(filter) { this.pushed.push(filter) },
		popErrorScope() {
			this.pushed.pop()
			return Promise.resolve(this.pushed.length === 1 ? {message: "inner scope error"} : null)
		},
	}`)
	dev.Set("lost", lost)
	return dev, dev.Get("pushed")
}

func newFakeDevice(lost js.Value) (*Device, js.Value) {
	dev, pushed := fakeDevice(lost)
	bd := browser.NewDevice(dev)
	d := &Device{browser: bd, queue: &Queue{browser: bd.Queue()}, limits: gputypes.DefaultLimits()}
	d.queue.device = d
	return d, pushed
}

func TestBrowserErrorScopes(t *testing.T) {
	d, pushed := newFakeDevice(jsEval("new Promise(() => {})"))
	d.PushErrorScope(ErrorFilterOutOfMemory)
	d.PushErrorScope(ErrorFilterValidation)
	if got := js.Global().Get("JSON").Call("stringify", pushed).String(); got != `["out-of-memory","validation"]` {
		t.Fatalf("pushed filters %s, want out-of-memory then validation", got)
	}

	gpuErr := d.PopErrorScope()
	if gpuErr == nil || gpuErr.Type != ErrorFilterValidation || gpuErr.Message != "inner scope error" {
		t.Fatalf("inner PopErrorScope = %+v, want the validation scope's error", gpuErr)
	}
	if gpuErr := d.PopErrorScope(); gpuErr != nil {
		t.Fatalf("outer PopErrorScope = %+v, want nil", gpuErr)
	}

	defer func() {
		if recover() == nil {
			t.Fatal("PopErrorScope without a pushed scope did not panic")
		}
	}()
	d.PopErrorScope()
}

func TestBrowserLostDeviceRefusesWork(t *testing.T) {
	d, _ := newFakeDevice(jsEval(`Promise.resolve({reason: "unknown", message: "GPU process crashed"})`))
	yieldToJS(t)

	if _, err := d.CreateBuffer(&BufferDescriptor{Size: 16, Usage: BufferUsageCopyDst}); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("CreateBuffer on a lost device: %v, want ErrDeviceLost", err)
	}
	if _, err := d.CreateCommandEncoder(nil); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("CreateCommandEncoder on a lost device: %v, want ErrDeviceLost", err)
	}
	if _, err := d.Queue().Submit(); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("Submit on a lost device: %v, want ErrDeviceLost", err)
	}
	if err := d.Queue().WriteBuffer(&Buffer{browser: &browser.Buffer{}}, 0, nil); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("WriteBuffer on a lost device: %v, want ErrDeviceLost", err)
	}
	buf := &Buffer{browser: &browser.Buffer{}, device: d, size: 16, usage: BufferUsageMapRead}
	if _, err := buf.MapAsync(MapModeRead, 0, 16); !errors.Is(err, ErrMapDeviceLost) {
		t.Errorf("MapAsync on a lost device: %v, want ErrMapDeviceLost", err)
	}
}

func TestBrowserExceptionsBecomeErrors(t *testing.T) {
	dev := jsEval(`{
		queue: {},
		createTexture() { throw new TypeError("feature-gated format") },
		createShaderModule() { throw new TypeError("missing code") },
		createRenderPipeline() { throw new TypeError("mask out of range") },
	}`)
	d := &Device{browser: browser.NewDevice(dev), limits: gputypes.DefaultLimits()}

	if _, err := d.CreateTexture(&TextureDescriptor{Size: Extent3D{Width: 1, Height: 1, DepthOrArrayLayers: 1},
		MipLevelCount: 1, SampleCount: 1, Format: gputypes.TextureFormatRGBA8Unorm}); err == nil {
		t.Error("CreateTexture: nil error for a throwing createTexture")
	}
	if _, err := d.CreateShaderModule(&ShaderModuleDescriptor{}); err == nil {
		t.Error("CreateShaderModule: nil error for a throwing createShaderModule")
	}
	if _, err := d.CreateRenderPipeline(&RenderPipelineDescriptor{}); err == nil {
		t.Error("CreateRenderPipeline: nil error for a throwing createRenderPipeline")
	}
}

func TestBrowserCreateTextureNeedsTheFormatsFeature(t *testing.T) {
	dev := jsEval(`{queue: {}, createTexture() { throw new Error("createTexture reached") }}`)
	d := &Device{browser: browser.NewDevice(dev), limits: gputypes.DefaultLimits()}
	_, err := d.CreateTexture(&TextureDescriptor{Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		MipLevelCount: 1, SampleCount: 1, Format: gputypes.TextureFormatBC1RGBAUnorm})
	if err == nil || !strings.Contains(err.Error(), "TextureCompressionBC") {
		t.Fatalf("CreateTexture(BC1) without the feature = %v, want an error naming TextureCompressionBC", err)
	}
	_, err = d.CreateTexture(&TextureDescriptor{Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		MipLevelCount: 1, SampleCount: 1, Format: gputypes.TextureFormatR16Unorm})
	if err == nil || !strings.Contains(err.Error(), "TextureFormatsTier1") {
		t.Fatalf("CreateTexture(R16Unorm) without the feature = %v, want an error naming TextureFormatsTier1", err)
	}
}

// A texture descriptor's sizes, mip level count and sample count of 0 are
// refused before the browser sees them, as native builds refuse them: the
// browser would take them as 1.
func TestBrowserCreateTextureRefusesZeros(t *testing.T) {
	dev := jsEval(`{queue: {}, created: 0, createTexture(desc) { this.created++; return {width: desc.size.width, height: desc.size.height ?? 1, format: desc.format} }}`)
	d := &Device{browser: browser.NewDevice(dev), limits: gputypes.DefaultLimits()}
	valid := TextureDescriptor{Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatRGBA8Unorm, Usage: TextureUsageTextureBinding}
	for _, tc := range []struct {
		name string
		zero func(*TextureDescriptor)
	}{
		{"width", func(d *TextureDescriptor) { d.Size.Width = 0 }},
		{"height", func(d *TextureDescriptor) { d.Size.Height = 0 }},
		{"depthOrArrayLayers", func(d *TextureDescriptor) { d.Size.DepthOrArrayLayers = 0 }},
		{"mipLevelCount", func(d *TextureDescriptor) { d.MipLevelCount = 0 }},
		{"sampleCount", func(d *TextureDescriptor) { d.SampleCount = 0 }},
	} {
		desc := valid
		tc.zero(&desc)
		if _, err := d.CreateTexture(&desc); err == nil {
			t.Errorf("CreateTexture with a %s of 0: nil error", tc.name)
		}
	}
	if n := dev.Get("created").Int(); n != 0 {
		t.Fatalf("createTexture reached %d times for descriptors with zeros", n)
	}
	if _, err := d.CreateTexture(&valid); err != nil {
		t.Fatalf("CreateTexture(%+v): %v", valid, err)
	}
	if n := dev.Get("created").Int(); n != 1 {
		t.Fatalf("createTexture reached %d times for a valid descriptor, want 1", n)
	}
}

func TestBrowserDeviceWithoutLimitsGetsTheAdapters(t *testing.T) {
	adapter := jsEval(`{
		requested: [],
		requestDevice(desc) {
			this.requested.push(desc.requiredLimits)
			return Promise.resolve({queue: {}, lost: new Promise(() => {})})
		},
	}`)
	a := &Adapter{browser: browser.NewAdapter(adapter), limits: gputypes.Limits{MaxBindGroups: 8, MaxStorageBuffersPerShaderStage: 64}}
	for _, desc := range []*DeviceDescriptor{nil, {Label: "no limits"}, {RequiredLimits: gputypes.Limits{MaxBindGroups: 6}}} {
		if _, err := a.RequestDevice(desc); err != nil {
			t.Fatalf("RequestDevice(%+v): %v", desc, err)
		}
	}
	want := `[{"maxBindGroups":8,"maxStorageBuffersPerShaderStage":64},{"maxBindGroups":8,"maxStorageBuffersPerShaderStage":64},{"maxBindGroups":6}]`
	if got := js.Global().Get("JSON").Call("stringify", adapter.Get("requested")).String(); got != want {
		t.Fatalf("requiredLimits %s, want %s", got, want)
	}
}

func TestBrowserUncapturedErrorHandler(t *testing.T) {
	js.Global().Set("GPUOutOfMemoryError", jsEval("class GPUOutOfMemoryError extends Error {}"))
	defer js.Global().Delete("GPUOutOfMemoryError")
	dev := jsEval("new EventTarget()")
	dev.Set("queue", jsEval("{}"))
	d := &Device{browser: browser.NewDevice(dev)}
	dispatch := func(errorClass, message string) (prevented bool) {
		event := js.Global().Get("Event").New("uncapturederror", map[string]any{"cancelable": true})
		event.Set("error", jsEval("(c, m) => new c(m)").Invoke(jsEval(errorClass), message))
		dev.Call("dispatchEvent", event)
		return event.Get("defaultPrevented").Bool()
	}

	var got []*GPUError
	d.SetUncapturedErrorHandler(func(e *GPUError) { t.Fatal("replaced handler called") })
	d.SetUncapturedErrorHandler(func(e *GPUError) { got = append(got, e) })
	if !dispatch("Error", "bad bind group") {
		t.Error("the event's default, the browser's console warning, was not prevented with a handler set")
	}
	dispatch("GPUOutOfMemoryError", "out of memory")
	want := []GPUError{{Type: ErrorFilterValidation, Message: "bad bind group"}, {Type: ErrorFilterOutOfMemory, Message: "out of memory"}}
	if len(got) != len(want) || *got[0] != want[0] || *got[1] != want[1] {
		t.Fatalf("handler got %v, want %v", got, want)
	}

	d.SetUncapturedErrorHandler(nil)
	if dispatch("Error", "after nil") || len(got) != 2 {
		t.Fatal("the handler or its listener is still there after SetUncapturedErrorHandler(nil)")
	}

	d.SetUncapturedErrorHandler(func(e *GPUError) { got = append(got, e) })
	d.Release()
	if dispatch("Error", "after Release") || len(got) != 2 {
		t.Fatal("the handler is still listening after Release")
	}
}

func TestBrowserPopErrorScopeOnALostInstanceReturnsNil(t *testing.T) {
	dev := jsEval(`{
		queue: {},
		pushErrorScope() {},
		popErrorScope() { return Promise.reject(new DOMException("Instance dropped in popErrorScope", "OperationError")) },
	}`)
	d := &Device{browser: browser.NewDevice(dev)}
	d.PushErrorScope(ErrorFilterValidation)
	if gpuErr := d.PopErrorScope(); gpuErr != nil {
		t.Fatalf("PopErrorScope = %v after the browser rejected the pop, want nil", gpuErr)
	}
}

func TestBrowserGetBindGroupLayoutPastMaxBindGroupsIsNil(t *testing.T) {
	dev := jsEval(`{
		queue: {},
		createComputePipeline() { return {getBindGroupLayout(i) { return {group: i} }} },
	}`)
	d := &Device{browser: browser.NewDevice(dev), limits: gputypes.Limits{MaxBindGroups: 4}}
	pipe, err := d.CreateComputePipeline(&ComputePipelineDescriptor{})
	if err != nil {
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	if pipe.GetBindGroupLayout(3) == nil {
		t.Error("GetBindGroupLayout(3) = nil with 4 bind groups")
	}
	if bgl := pipe.GetBindGroupLayout(4); bgl != nil {
		t.Errorf("GetBindGroupLayout(4) = %v with 4 bind groups, want nil", bgl)
	}
}
