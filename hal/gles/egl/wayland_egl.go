// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

// Package egl provides EGL bindings for OpenGL context management.
//
// This file adds libwayland-egl.so bindings for creating EGL window surfaces
// on Wayland. Unlike X11 where eglCreateWindowSurface takes the raw X11
// Window handle, Wayland requires an intermediate wl_egl_window object.
//
// Enterprise reference: Rust wgpu-hal egl.rs:1390-1533 (configure),
// SDL3 SDL_waylandwindow.c:2978.
package egl

// #include "trampolines_linux.h"
import "C"

import (
	"log/slog"
	"sync"
	"unsafe"

	"github.com/roffe/gowgpu/internal/dl"
)

var (
	waylandEGLOnce  sync.Once
	waylandEGLReady bool

	waylandEGLLib dl.Lib

	symWlEGLWindowCreate  unsafe.Pointer
	symWlEGLWindowDestroy unsafe.Pointer
	symWlEGLWindowResize  unsafe.Pointer
)

// InitWaylandEGL loads libwayland-egl.so.
// Safe to call multiple times; only the first call does work.
func InitWaylandEGL() bool {
	waylandEGLOnce.Do(func() {
		var err error
		if waylandEGLLib, err = dl.Open("libwayland-egl.so.1", "libwayland-egl.so"); err != nil {
			slog.Debug("egl: libwayland-egl unavailable", "error", err)
			return
		}
		// wl_egl_window* wl_egl_window_create(wl_surface*, int width, int height)
		if symWlEGLWindowCreate, err = waylandEGLLib.Sym("wl_egl_window_create"); err != nil {
			slog.Debug("egl: libwayland-egl incomplete", "error", err)
			return
		}
		// void wl_egl_window_destroy(wl_egl_window*)
		if symWlEGLWindowDestroy, err = waylandEGLLib.Sym("wl_egl_window_destroy"); err != nil {
			slog.Debug("egl: libwayland-egl incomplete", "error", err)
			return
		}
		// void wl_egl_window_resize(wl_egl_window*, int width, int height, int dx, int dy)
		if symWlEGLWindowResize, err = waylandEGLLib.Sym("wl_egl_window_resize"); err != nil {
			slog.Debug("egl: libwayland-egl incomplete", "error", err)
			return
		}

		waylandEGLReady = true
		slog.Debug("egl: libwayland-egl loaded — Wayland EGL window support available")
	})
	return waylandEGLReady
}

// HasWaylandEGL reports whether libwayland-egl.so was loaded successfully.
func HasWaylandEGL() bool {
	return waylandEGLReady
}

// WlEGLWindowCreate creates a wl_egl_window from a wl_surface.
// Returns 0 on failure.
func WlEGLWindowCreate(wlSurface uintptr, width, height int32) uintptr {
	if !waylandEGLReady {
		return 0
	}
	var result uintptr
	result = uintptr(C.egl_u_uii(C.uintptr_t(uintptr(symWlEGLWindowCreate)), C.uintptr_t(wlSurface), C.uint32_t(width), C.uint32_t(height)))
	return result
}

// WlEGLWindowDestroy destroys a wl_egl_window.
func WlEGLWindowDestroy(eglWindow uintptr) {
	if !waylandEGLReady || eglWindow == 0 {
		return
	}
	C.egl_v_u(C.uintptr_t(uintptr(symWlEGLWindowDestroy)), C.uintptr_t(eglWindow))
}

// WlEGLWindowResize resizes a wl_egl_window.
func WlEGLWindowResize(eglWindow uintptr, width, height, dx, dy int32) {
	if !waylandEGLReady || eglWindow == 0 {
		return
	}
	C.egl_v_uiiii(C.uintptr_t(uintptr(symWlEGLWindowResize)), C.uintptr_t(eglWindow), C.uint32_t(width), C.uint32_t(height), C.uint32_t(dx), C.uint32_t(dy))
}
