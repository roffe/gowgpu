// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package gles

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
	"github.com/roffe/gowgpu/hal/gles/wgl"
)

// Backend implements hal.Backend for OpenGL ES / OpenGL 3.3+.
type Backend struct{}

// NewBackend returns a GLES backend instance.
func NewBackend() Backend { return Backend{} }

// Variant returns the backend type identifier.
func (Backend) Variant() gputypes.Backend {
	return gputypes.BackendGL
}

// CreateInstance creates a new OpenGL instance.
//
// Creates a hidden 1×1 window and initializes a GL context on it. The context
// lives for the Instance lifetime and survives any user Surface destruction.
// Follows Rust wgpu-hal/src/gles/wgl.rs Instance::init (lines 448-563).
func (Backend) CreateInstance(_ *hal.InstanceDescriptor) (hal.Instance, error) {
	if err := wgl.Init(); err != nil {
		return nil, fmt.Errorf("gles: failed to initialize WGL: %w", err)
	}

	// Create hidden 1×1 window to host the GL context (Rust: create_instance_device),
	// with its pixel format set. GL context is NOT created here — it will be
	// lazily created on the render thread's first Lock() call, avoiding
	// cross-thread WGL issues.
	hiddenWindow, err := wgl.NewHiddenWindow()
	if err != nil {
		return nil, fmt.Errorf("gles: failed to create hidden window: %w", err)
	}

	ctx := NewAdapterContext(hiddenWindow.DC())

	hal.Logger().Info("gles: instance created",
		"platform", "windows",
		"hiddenDC", fmt.Sprintf("0x%x", hiddenWindow.DC()),
	)

	return &Instance{
		ctx:          ctx,
		hiddenWindow: hiddenWindow,
	}, nil
}

// Instance implements hal.Instance for the OpenGL backend.
// Owns the hidden 1×1 window. The GL context is lazily created on the
// render thread via AdapterContext.ensureInit().
type Instance struct {
	ctx          *AdapterContext
	hiddenWindow *wgl.HiddenWindow
}

// CreateSurface creates an OpenGL surface from window handles.
// On Windows: displayHandle is ignored, windowHandle is HWND.
//
// The Surface is lightweight — it does NOT own the GL context. The context
// lives on the hidden window owned by Instance. Surface only stores the user
// HWND and a reference to the shared AdapterContext. The user window gets
// the hidden window's pixel format, unless it has it already, so that
// wglMakeCurrent can switch between hidden and user DCs during Present.
//
// Follows Rust wgpu-hal/src/gles/wgl.rs Instance::create_surface (lines 624-670).
func (i *Instance) CreateSurface(target hal.SurfaceTarget) (hal.Surface, error) {
	if err := target.RequireKind(hal.SurfaceTargetWindowsHWND); err != nil {
		return nil, fmt.Errorf("gles: %w", err)
	}
	windowHandle := target.WindowHandle
	hwnd := wgl.HWND(windowHandle)

	hdc := wgl.GetDC(hwnd)
	if hdc == 0 {
		return nil, fmt.Errorf("gles: GetDC failed for window 0x%x", windowHandle)
	}
	err := wgl.SetWindowPixelFormat(hdc, i.hiddenWindow.PixelFormat())
	wgl.ReleaseDC(hwnd, hdc)
	if err != nil {
		return nil, fmt.Errorf("gles: surface for window 0x%x: %w", windowHandle, err)
	}

	hal.Logger().Info("gles: surface created", "hwnd", fmt.Sprintf("0x%x", windowHandle))

	return &Surface{
		hwnd: hwnd,
		ctx:  i.ctx,
	}, nil
}

// EnumerateAdapters returns available OpenGL adapters.
// Triggers lazy GL context creation on the calling thread (render thread).
func (i *Instance) EnumerateAdapters(_ hal.Surface) []hal.ExposedAdapter {
	glCtx := i.ctx.Lock()
	defer i.ctx.Unlock()

	if glCtx == nil {
		hal.Logger().Error("gles: EnumerateAdapters: GL context not available")
		return nil
	}

	caps := queryAdapterCapabilities(glCtx)

	version := glCtx.GetString(gl.VERSION)
	renderer := glCtx.GetString(gl.RENDERER)
	if !caps.supported() {
		hal.Logger().Warn("gles: no adapter, the GL context is older than 3.3", "version", version, "renderer", renderer)
		return nil
	}

	return []hal.ExposedAdapter{exposeAdapter(&Adapter{
		ctx:      i.ctx,
		version:  version,
		renderer: renderer,
		caps:     caps,
	}, &caps)}
}

// Destroy releases the instance resources.
func (i *Instance) Destroy() {
	if i.ctx != nil {
		i.ctx.Destroy()
	}
	if i.hiddenWindow != nil {
		i.hiddenWindow.Destroy()
		i.hiddenWindow = nil
	}
}
