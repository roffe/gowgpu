//go:build js && wasm

package browser

import (
	"errors"
	"syscall/js"
	"testing"
)

func TestCatchJSReturnsTheThrownException(t *testing.T) {
	err := catchJS(func() { js.Global().Get("Uint8Array").New(-1) })
	var jsErr *JSError
	if !errors.As(err, &jsErr) || jsErr.Name != "RangeError" {
		t.Fatalf("catchJS = %v, want a RangeError", err)
	}
}

func TestCatchJSReturnsNilWhenNothingThrows(t *testing.T) {
	if err := catchJS(func() {}); err != nil {
		t.Fatalf("catchJS = %v, want nil", err)
	}
}

func TestCatchJSLetsGoPanicsThrough(t *testing.T) {
	defer func() {
		if r := recover(); r != "go panic" {
			t.Fatalf("recovered %v, want the Go panic", r)
		}
	}()
	_ = catchJS(func() { panic("go panic") })
}

func throwing(t *testing.T, method, errorType string) js.Value {
	t.Helper()
	obj := js.Global().Get("Object").New()
	obj.Set(method, js.Global().Get("Function").New("throw new "+errorType+"('thrown')"))
	return obj
}

func TestCreateViewReturnsTheException(t *testing.T) {
	tex := &Texture{ref_: throwing(t, "createView", "TypeError")}
	if _, err := tex.CreateView(js.Undefined()); err == nil {
		t.Fatal("CreateView: nil error for a throwing createView")
	}
}

func TestGetBindGroupLayoutReturnsTheException(t *testing.T) {
	pipeline := &RenderPipeline{ref_: throwing(t, "getBindGroupLayout", "RangeError")}
	if _, err := pipeline.GetBindGroupLayout(99); err == nil {
		t.Fatal("GetBindGroupLayout: nil error for a throwing getBindGroupLayout")
	}
}
