// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build integration && linux && !(js && wasm)

package gles

import (
	"runtime"
	"testing"

	"github.com/roffe/gowgpu/hal/gles/egl"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// newTestAdapterContext makes a real EGL/GL context wrapped in an
// AdapterContext, or skips when there is no EGL display to make one on.
func newTestAdapterContext(t *testing.T) *AdapterContext {
	t.Helper()
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)

	if err := egl.Init(); err != nil {
		t.Skipf("egl.Init() failed: %v", err)
	}
	config := egl.DefaultContextConfig()
	config.GLES = false
	eglCtx, err := egl.NewContext(config)
	if err != nil {
		t.Skipf("egl.NewContext() failed: %v", err)
	}
	if err := eglCtx.MakeCurrent(); err != nil {
		eglCtx.Destroy()
		t.Fatalf("MakeCurrent failed: %v", err)
	}
	glCtx := &gl.Context{}
	if err := glCtx.Load(egl.GetGLProcAddress, config.GLES); err != nil {
		eglCtx.Destroy()
		t.Fatalf("GL load failed: %v", err)
	}
	_ = egl.MakeCurrent(eglCtx.Display(), egl.NoSurface, egl.NoSurface, egl.NoContext)
	return NewAdapterContext(eglCtx, glCtx, true)
}

// TestSurfaceDoesNotDestroyAdoptedContext is the regression test for the
// shutdown panic: when a Surface had to make its own EGL context (Wayland, or
// a window kind mismatch), the Instance adopts it, so Surface.Destroy must not
// free it while the Device still uses it.
//
// Before the fix Surface.Destroy called AdapterContext.Destroy (ownsContext was
// true) and deleted the context the Device held. The Device.Destroy that came
// after locked a nil EGL context and ran GL calls on a nil *gl.Context, SIGSEGV.
func TestSurfaceDoesNotDestroyAdoptedContext(t *testing.T) {
	adapterCtx := newTestAdapterContext(t)

	inst := &Instance{}
	if prev := inst.adoptContext(adapterCtx); prev != nil {
		t.Fatalf("adoptContext returned unexpected previous context %p", prev)
	}

	// The Surface CreateSurface path B returns: it shares the adopted context.
	surf := &Surface{ctx: adapterCtx, ownsContext: false}
	device := &Device{ctx: adapterCtx, vao: 1}

	// Closing the window destroys the Surface first.
	surf.Destroy()

	// The shared context must still be alive for the Device.
	if adapterCtx.eglCtx == nil || adapterCtx.gl == nil {
		t.Fatal("Surface.Destroy freed the context the Device still uses")
	}

	// Shutdown destroys the Device next, which must not nil-deref.
	device.Destroy()
	if device.vao != 0 {
		t.Fatalf("Device.Destroy left vao set: %d", device.vao)
	}

	// The Instance is released last and frees the context.
	inst.Destroy()
	if adapterCtx.eglCtx != nil || adapterCtx.gl != nil {
		t.Fatal("Instance.Destroy did not release the adopted context")
	}
}
