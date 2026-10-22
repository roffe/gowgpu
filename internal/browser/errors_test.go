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
