//go:build js && wasm

package browser

import (
	"syscall/js"
	"testing"
)

// fakeCanvas returns a canvas stand-in whose webgpu context throws a
// TypeError from configure for rgba16float, as Firefox does.
func fakeCanvas() js.Value {
	return js.Global().Get("Function").New(`return {
		width: 300,
		height: 150,
		getContext() {
			return {
				configure(c) { if (c.format === "rgba16float") throw new TypeError("rgba16float is not yet supported") },
				unconfigure() {},
				getCurrentTexture() { throw new DOMException("not configured", "InvalidStateError") },
			}
		},
	}`).Invoke()
}

func TestConfigureReturnsTheExceptionAndKeepsTheCanvas(t *testing.T) {
	canvas := fakeCanvas()
	s, err := NewSurface(js.Undefined(), canvas)
	if err != nil {
		t.Fatalf("NewSurface: %v", err)
	}
	config := newJSObject()
	config.Set("format", "rgba16float")
	if err := s.Configure(config, 64, 32, "rgba16float"); err == nil {
		t.Fatal("Configure: nil error for a throwing configure")
	}
	if s.Configured() || canvas.Get("width").Int() != 300 || canvas.Get("height").Int() != 150 {
		t.Fatalf("after a failed Configure: configured %v, canvas %vx%v, want unconfigured 300x150",
			s.Configured(), canvas.Get("width"), canvas.Get("height"))
	}

	config.Set("format", "bgra8unorm")
	if err := s.Configure(config, 64, 32, "bgra8unorm"); err != nil {
		t.Fatalf("Configure: %v", err)
	}
	if _, err := s.GetCurrentTexture(); err == nil {
		t.Fatal("GetCurrentTexture: nil error for a throwing getCurrentTexture")
	}
}

func TestProbeRGBA16FloatCanvas(t *testing.T) {
	defer func(state int) { rgba16floatCanvas = state }(rgba16floatCanvas)
	defer js.Global().Delete("OffscreenCanvas")

	rgba16floatCanvas = probeUnknown
	ProbeRGBA16FloatCanvas(js.Undefined())
	if !RGBA16FloatCanvasSupported() {
		t.Fatal("rgba16float unsupported without a probe that ran")
	}

	js.Global().Set("OffscreenCanvas", js.Global().Get("Function").New("canvas", "return function() { return canvas }").Invoke(fakeCanvas()))
	ProbeRGBA16FloatCanvas(js.Undefined())
	if RGBA16FloatCanvasSupported() {
		t.Fatal("rgba16float supported although configure threw for it")
	}
}
