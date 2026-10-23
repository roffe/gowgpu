//go:build linux && !(js && wasm)

package egl

import "testing"

// Init loads EGL once, so an instance made while others call EGL writes
// nothing those calls read; under the race detector this fails otherwise.
func TestInitWhileEGLIsInUse(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("no EGL: %v", err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range 200 {
			GetCurrentContext()
		}
	}()
	for range 200 {
		if err := Init(); err != nil {
			t.Error(err)
			break
		}
	}
	<-done
}
