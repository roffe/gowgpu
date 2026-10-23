//go:build js && wasm

package browser

import (
	"errors"
	"fmt"
	"runtime"
	"syscall/js"
)

// ErrCanvasContextFailed is returned when canvas.getContext("webgpu") returns null.
// This happens when WebGPU is not available or the canvas is already bound to
// another context type (e.g., "2d" or "webgl2").
//
// Matches Rust wgpu's CreateSurfaceErrorKind::Web for the null-context case.
var ErrCanvasContextFailed = errors.New("wgpu: canvas.getContext(\"webgpu\") returned null; webgpu not available or canvas already in use")

// Surface wraps an HTML canvas element and its GPUCanvasContext.
//
// On the browser, a "surface" is a canvas + the GPUCanvasContext obtained from
// canvas.getContext("webgpu"). The context is configured with a device, format,
// and size, then getCurrentTexture() returns the next frame texture.
//
// Presentation happens automatically when the JS event loop runs -- there is no
// explicit present() call. This matches Rust wgpu WebSurface / WebSurfaceOutputDetail
// where present() and texture_discard() are both no-ops. Present only records
// the texture, so that GetCurrentTexture can wait for the browser to show it.
type Surface struct {
	// canvas is the HTMLCanvasElement (or OffscreenCanvas).
	canvas js.Value

	// context is the GPUCanvasContext from canvas.getContext("webgpu").
	context js.Value

	// gpu is the navigator.gpu reference, used for getPreferredCanvasFormat().
	gpu js.Value

	// Pre-bound context methods to avoid property lookups per frame.
	fnGetCurrentTexture js.Value
	fnConfigure         js.Value

	// Configuration state.
	configured bool
	width      uint32
	height     uint32
	format     string

	// presented is the GPUTexture last passed to Present, and presentTurn
	// gets a turned property once the event loop has run after that Present.
	presented   js.Value
	presentTurn js.Value
}

// NewSurface creates a Surface from a canvas element by obtaining its GPUCanvasContext.
//
// The gpu parameter is the navigator.gpu object (needed for getPreferredCanvasFormat).
// The canvas must be an HTMLCanvasElement or OffscreenCanvas.
//
// Returns ErrCanvasContextFailed if getContext("webgpu") returns null (WebGPU
// not available or canvas already in use). Panics if getContext throws an
// exception (indicates misuse of canvas state).
//
// Matches Rust wgpu ContextWebGpu::create_surface_from_context.
func NewSurface(gpu js.Value, canvas js.Value) (*Surface, error) {
	// Call canvas.getContext("webgpu"). This may return null (not supported
	// or canvas already has another context) or throw (canvas state misuse).
	//
	// See: https://html.spec.whatwg.org/multipage/canvas.html#dom-canvas-getcontext
	context := canvas.Call("getContext", "webgpu")
	if context.IsNull() || context.IsUndefined() {
		return nil, ErrCanvasContextFailed
	}

	s := &Surface{
		canvas:  canvas,
		context: context,
		gpu:     gpu,
	}

	// Pre-bind context methods for per-frame performance.
	s.fnGetCurrentTexture = bindMethod(context, "getCurrentTexture")
	s.fnConfigure = bindMethod(context, "configure")

	return s, nil
}

// Configure sets the surface configuration on the GPUCanvasContext.
//
// This sets the canvas dimensions and calls context.configure() with the
// provided parameters. After Configure, GetCurrentTexture() can be called.
// When configure throws, for a format the browser does not take as a
// canvas format for example, Configure puts the canvas size back and
// returns the exception as an error, and the previous configuration stays.
//
// Matches Rust wgpu SurfaceInterface::configure for WebSurface.
func (s *Surface) Configure(config js.Value, width, height uint32, format string) error {
	// Set canvas dimensions (Rust wgpu does this in configure too).
	oldWidth, oldHeight := s.canvas.Get("width"), s.canvas.Get("height")
	s.canvas.Set("width", width)
	s.canvas.Set("height", height)

	if err := catchJS(func() { s.fnConfigure.Invoke(config) }); err != nil {
		s.canvas.Set("width", oldWidth)
		s.canvas.Set("height", oldHeight)
		return err
	}

	s.configured = true
	s.width = width
	s.height = height
	s.format = format
	s.presented, s.presentTurn = js.Value{}, js.Value{}
	return nil
}

// Unconfigure removes the surface configuration.
// After this call, GetCurrentTexture() will fail until Configure is called again.
func (s *Surface) Unconfigure() {
	unconfigure := s.context.Get("unconfigure")
	if !unconfigure.IsUndefined() && !unconfigure.IsNull() {
		s.context.Call("unconfigure")
	}
	s.configured = false
	s.presented, s.presentTurn = js.Value{}, js.Value{}
}

// GetCurrentTexture returns the current frame texture from the GPUCanvasContext.
//
// The returned GPUTexture is automatically presented when control returns to
// the browser event loop after the command buffer using it is submitted.
// When the canvas still hands out the texture last passed to Present and the
// event loop has not run since, the browser has had no chance to show it:
// GetCurrentTexture then waits for the next animation frame and asks again,
// which returns control to the event loop for a render loop that would
// otherwise never give it back. It waits once only, since a canvas that is
// not displayed keeps its texture, and never in a JavaScript callback, which
// holds up the event loop until it returns.
//
// Returns an error if the surface is not configured, or the exception
// getCurrentTexture throws.
//
// Matches Rust wgpu SurfaceInterface::get_current_texture for WebSurface.
func (s *Surface) GetCurrentTexture() (*Texture, error) {
	if !s.configured {
		return nil, fmt.Errorf("wgpu: surface not configured")
	}

	jsTexture, err := invoke(s.fnGetCurrentTexture)
	if err == nil && s.unshown(jsTexture) && !inJSCallback() {
		if _, err = AwaitPromise(nextAnimationFrame()); err == nil {
			jsTexture, err = invoke(s.fnGetCurrentTexture)
		}
	}
	if err != nil {
		return nil, err
	}
	s.presented, s.presentTurn = js.Value{}, js.Value{}
	if jsTexture.IsNull() || jsTexture.IsUndefined() {
		return nil, fmt.Errorf("wgpu: getCurrentTexture returned null")
	}

	return NewTexture(jsTexture), nil
}

// Present records texture, which GetCurrentTexture returned, as presented.
// A zero timeout, which only runs once the event loop does, marks
// presentTurn turned through Reflect.set, so no Go callback is involved.
func (s *Surface) Present(texture *Texture) {
	s.presented = texture.Ref()
	s.presentTurn = newJSObject()
	markTurned := js.Global().Get("Reflect").Get("set").Call("bind", nil, s.presentTurn, "turned", true, s.presentTurn)
	js.Global().Call("setTimeout", markTurned, 0)
}

// unshown reports whether texture is the one last presented and the event
// loop has not run since, so the browser has had no chance to show it.
func (s *Surface) unshown(texture js.Value) bool {
	return s.presentTurn.Type() == js.TypeObject && texture.Equal(s.presented) && !s.presentTurn.Get("turned").Truthy()
}

// inJSCallback reports whether the calling goroutine runs a js.FuncOf
// callback, which syscall/js calls from its handleEvent. Blocking there
// deadlocks the program, since the event loop waits for the callback.
func inJSCallback() bool {
	pcs := make([]uintptr, 64)
	for {
		n := runtime.Callers(1, pcs)
		if n == len(pcs) {
			pcs = make([]uintptr, 2*len(pcs))
			continue
		}
		frames := runtime.CallersFrames(pcs[:n])
		for {
			frame, more := frames.Next()
			if frame.Function == "syscall/js.handleEvent" {
				return true
			}
			if !more {
				return false
			}
		}
	}
}

// nextAnimationFrame returns a promise that resolves at the browser's next
// animation frame, or after a zero timeout where there are none.
func nextAnimationFrame() js.Value {
	schedule := js.Global().Get("requestAnimationFrame")
	if schedule.Type() != js.TypeFunction {
		schedule = js.Global().Get("setTimeout")
	}
	executor := js.FuncOf(func(_ js.Value, args []js.Value) any {
		schedule.Invoke(args[0])
		return nil
	})
	defer executor.Release()
	return js.Global().Get("Promise").New(executor)
}

// GetPreferredCanvasFormat returns the preferred texture format for the display.
//
// This calls navigator.gpu.getPreferredCanvasFormat() which returns either
// "bgra8unorm" or "rgba8unorm" depending on the platform.
//
// See: https://www.w3.org/TR/webgpu/#dom-gpu-getpreferredcanvasformat
//
// Matches Rust wgpu SurfaceInterface::get_capabilities which reads
// gpu.get_preferred_canvas_format() to order the formats list.
func (s *Surface) GetPreferredCanvasFormat() string {
	if s.gpu.IsUndefined() || s.gpu.IsNull() {
		return "bgra8unorm" // safe fallback
	}
	return s.gpu.Call("getPreferredCanvasFormat").String()
}

// Canvas returns the underlying canvas js.Value.
func (s *Surface) Canvas() js.Value { return s.canvas }

// Context returns the underlying GPUCanvasContext js.Value.
func (s *Surface) Context() js.Value { return s.context }

// Width returns the configured canvas width in pixels.
func (s *Surface) Width() uint32 { return s.width }

// Height returns the configured canvas height in pixels.
func (s *Surface) Height() uint32 { return s.height }

// Format returns the configured texture format string (e.g., "bgra8unorm").
func (s *Surface) Format() string { return s.format }

// Configured reports whether the surface has been configured.
func (s *Surface) Configured() bool { return s.configured }

// Destroy releases the surface. On browser this unconfigures the context.
// Matches Rust wgpu Drop for WebSurface (no-op in Rust, but we unconfigure
// for clean teardown).
func (s *Surface) Destroy() {
	if s.configured {
		s.Unconfigure()
	}
}

// rgba16floatCanvas caches whether this browser configures an rgba16float
// canvas: probeUnknown until ProbeRGBA16FloatCanvas has run.
var rgba16floatCanvas = probeUnknown

const (
	probeUnknown = iota
	probeUnsupported
	probeSupported
)

// ProbeRGBA16FloatCanvas configures a throwaway 1x1 OffscreenCanvas as
// rgba16float with device, once, to find out whether the browser takes it.
// Some browsers list rgba16float as a canvas format and still throw from
// configure (Firefox). Without OffscreenCanvas the support stays unknown.
func ProbeRGBA16FloatCanvas(device js.Value) {
	if rgba16floatCanvas != probeUnknown {
		return
	}
	ctor := js.Global().Get("OffscreenCanvas")
	if ctor.Type() != js.TypeFunction {
		return
	}
	var context js.Value
	if err := catchJS(func() { context = ctor.New(1, 1).Call("getContext", "webgpu") }); err != nil || context.Type() != js.TypeObject {
		return
	}
	config := newJSObject()
	config.Set("device", device)
	config.Set("format", "rgba16float")
	if catchJS(func() { context.Call("configure", config) }) != nil {
		rgba16floatCanvas = probeUnsupported
		return
	}
	context.Call("unconfigure")
	rgba16floatCanvas = probeSupported
}

// RGBA16FloatCanvasSupported reports whether a canvas can be configured as
// rgba16float. It is true until ProbeRGBA16FloatCanvas finds otherwise.
func RGBA16FloatCanvasSupported() bool {
	return rgba16floatCanvas != probeUnsupported
}
