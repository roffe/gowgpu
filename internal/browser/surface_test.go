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

// fakeFrameCanvas returns a canvas stand-in whose context hands out the same
// texture until the animation frame numbered expireAt, as a browser does
// until it has shown the canvas, and installs a requestAnimationFrame that
// counts frames.
func fakeFrameCanvas(t *testing.T) (canvas, context js.Value) {
	t.Helper()
	context = js.Global().Get("Function").New(`return {
		frames: 0,
		expireAt: 0,
		made: 0,
		current: null,
		configure() {},
		getCurrentTexture() {
			if (this.current === null) this.current = {width: 4, height: 4, format: "bgra8unorm", n: ++this.made}
			return this.current
		},
	}`).Invoke()
	raf := js.Global().Get("Function").New("ctx", `return (cb) => setTimeout(() => {
		ctx.frames++
		if (ctx.frames >= ctx.expireAt) ctx.current = null
		cb(ctx.frames)
	}, 0)`).Invoke(context)
	js.Global().Set("requestAnimationFrame", raf)
	t.Cleanup(func() { js.Global().Delete("requestAnimationFrame") })
	canvas = js.Global().Get("Object").New()
	canvas.Set("getContext", js.Global().Get("Function").New("ctx", "return () => ctx").Invoke(context))
	return canvas, context
}

func TestGetCurrentTextureWaitsAFrameAfterPresent(t *testing.T) {
	canvas, context := fakeFrameCanvas(t)
	s, err := NewSurface(js.Undefined(), canvas)
	if err != nil {
		t.Fatalf("NewSurface: %v", err)
	}
	if err := s.Configure(newJSObject(), 4, 4, "bgra8unorm"); err != nil {
		t.Fatalf("Configure: %v", err)
	}
	texture := func() int {
		t.Helper()
		tex, err := s.GetCurrentTexture()
		if err != nil {
			t.Fatalf("GetCurrentTexture: %v", err)
		}
		return tex.Ref().Get("n").Int()
	}

	first := texture()
	if again := texture(); again != first || context.Get("frames").Int() != 0 {
		t.Fatalf("without Present: texture %d after %d, %d frames, want the same texture at once", again, first, context.Get("frames").Int())
	}

	tex, _ := s.GetCurrentTexture()
	s.Present(tex)
	frames := context.Get("frames").Int()
	context.Set("expireAt", frames+1)
	if next := texture(); next == tex.Ref().Get("n").Int() || context.Get("frames").Int() != frames+1 {
		t.Fatalf("after Present: texture %d after %d frames, want a new one after one frame", next, context.Get("frames").Int()-frames)
	}

	tex, _ = s.GetCurrentTexture()
	s.Present(tex)
	frames = context.Get("frames").Int()
	context.Set("expireAt", 1<<30)
	if next := texture(); next != tex.Ref().Get("n").Int() || context.Get("frames").Int() != frames+1 {
		t.Fatalf("after Present on a canvas that keeps its texture: texture %d after %d frames, want the same one after one frame",
			next, context.Get("frames").Int()-frames)
	}
	if again := texture(); context.Get("frames").Int() != frames+1 {
		t.Fatalf("GetCurrentTexture without a Present in between waited: texture %d", again)
	}

	tex, _ = s.GetCurrentTexture()
	s.Present(tex)
	frames = context.Get("frames").Int()
	if _, err := AwaitPromise(js.Global().Get("Function").New("return new Promise(r => setTimeout(r, 0))").Invoke()); err != nil {
		t.Fatal(err)
	}
	if next := texture(); next != tex.Ref().Get("n").Int() || context.Get("frames").Int() != frames {
		t.Fatalf("after the event loop ran since Present: texture %d after %d frames, want the same one at once",
			next, context.Get("frames").Int()-frames)
	}

	tex, _ = s.GetCurrentTexture()
	s.Present(tex)
	frames = context.Get("frames").Int()
	var inCallback *Texture
	callback := js.FuncOf(func(js.Value, []js.Value) any {
		inCallback, _ = s.GetCurrentTexture()
		return nil
	})
	defer callback.Release()
	callback.Invoke()
	if inCallback == nil || context.Get("frames").Int() != frames {
		t.Fatalf("GetCurrentTexture in a JavaScript callback waited for a frame")
	}
}
