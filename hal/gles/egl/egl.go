// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

// Error handling follows ADR-049 three-tier strategy:
//
//	Tier 1 (creation/submit): check the API result code
//	Tier 2 (void/hot path): infallible by GPU API contract — Vulkan §6.6, WebGPU §21.2
//	Tier 3 (platform syscalls): use errno for diagnostics (Wayland, X11, Win32)
//
// Enterprise reference: Rust wgpu-hal returns () for all draw/destroy/barrier commands.
//
// EGL calls are Tier 2: EGL errors are reported via eglGetError().
// EGLBoolean returns are checked after the call.
package egl

/*
#include "trampolines_linux.h"
#cgo noescape egl_i_
#cgo nocallback egl_i_
#cgo noescape egl_i_i
#cgo nocallback egl_i_i
#cgo noescape egl_i_u
#cgo nocallback egl_i_u
#cgo noescape egl_i_ui
#cgo nocallback egl_i_ui
#cgo noescape egl_i_upp
#cgo nocallback egl_i_upp
#cgo noescape egl_i_uppip
#cgo nocallback egl_i_uppip
#cgo noescape egl_i_uu
#cgo nocallback egl_i_uu
#cgo noescape egl_i_uuip
#cgo nocallback egl_i_uuip
#cgo noescape egl_i_uupi
#cgo nocallback egl_i_uupi
#cgo noescape egl_i_uuuu
#cgo nocallback egl_i_uuuu
#cgo noescape egl_u_
#cgo nocallback egl_u_
#cgo noescape egl_u_iup
#cgo nocallback egl_u_iup
#cgo noescape egl_u_u
#cgo nocallback egl_u_u
#cgo noescape egl_u_uii
#cgo nocallback egl_u_uii
#cgo noescape egl_u_uup
#cgo nocallback egl_u_uup
#cgo noescape egl_u_uuup
#cgo nocallback egl_u_uuup
#cgo noescape egl_P_p
#cgo nocallback egl_P_p
#cgo noescape egl_P_ui
#cgo nocallback egl_P_ui
#cgo noescape egl_v_u
#cgo nocallback egl_v_u
#cgo noescape egl_v_uiiii
#cgo nocallback egl_v_uiiii
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/roffe/gowgpu/internal/dl"
)

var (
	// eglLib is the loaded libEGL.so library.
	eglLib dl.Lib

	// EGL 1.0+ core function symbols
	symEglGetError                    unsafe.Pointer
	symEglGetDisplay                  unsafe.Pointer
	symEglInitialize                  unsafe.Pointer
	symEglTerminate                   unsafe.Pointer
	symEglQueryString                 unsafe.Pointer
	symEglChooseConfig                unsafe.Pointer
	symEglGetConfigAttrib             unsafe.Pointer
	symEglCreateWindowSurface         unsafe.Pointer
	symEglCreatePbufferSurface        unsafe.Pointer
	symEglDestroySurface              unsafe.Pointer
	symEglBindAPI                     unsafe.Pointer
	symEglSwapInterval                unsafe.Pointer
	symEglCreateContext               unsafe.Pointer
	symEglDestroyContext              unsafe.Pointer
	symEglMakeCurrent                 unsafe.Pointer
	symEglGetCurrentContext           unsafe.Pointer
	symEglGetCurrentDisplay           unsafe.Pointer
	symEglGetCurrentSurface           unsafe.Pointer
	symEglSwapBuffers                 unsafe.Pointer
	symEglGetProcAddress              unsafe.Pointer
	symEglGetPlatformDisplay          unsafe.Pointer // EGL 1.5, may be nil
	symEglCreatePlatformWindowSurface unsafe.Pointer // EGL 1.5, may be nil
	// eglSwapBuffersWithDamageKHR and EXT, as eglGetProcAddress gives
	// them: non-nil also where a display lacks the extension.
	symEglSwapBuffersWithDamageKHR unsafe.Pointer
	symEglSwapBuffersWithDamageEXT unsafe.Pointer

	// hasPlatformWindowSurface is true when eglCreatePlatformWindowSurface
	// was loaded (EGL 1.5 or EGL_EXT_platform_base).
	hasPlatformWindowSurface bool
)

var (
	initOnce sync.Once
	errInit  error
)

// Init loads the EGL library and initializes function pointers, once for
// the process: the contexts made later, and those in use, share them.
func Init() error {
	initOnce.Do(func() { errInit = load() })
	return errInit
}

func load() error {
	var err error

	if eglLib, err = dl.Open("libEGL.so.1", "libEGL.so"); err != nil {
		return fmt.Errorf("failed to load libEGL.so: %w", err)
	}

	if err := loadEGLSymbols(); err != nil {
		return err
	}

	// Probe optional EGL extensions via eglGetProcAddress (must be done
	// after the core symbols are loaded).
	loadOptionalExtensions()
	return nil
}

// loadEGLSymbols loads all required EGL function symbols.
func loadEGLSymbols() error {
	var err error

	// Core functions
	if symEglGetError, err = eglLib.Sym("eglGetError"); err != nil {
		return fmt.Errorf("eglGetError not found: %w", err)
	}
	if symEglGetDisplay, err = eglLib.Sym("eglGetDisplay"); err != nil {
		return fmt.Errorf("eglGetDisplay not found: %w", err)
	}
	if symEglInitialize, err = eglLib.Sym("eglInitialize"); err != nil {
		return fmt.Errorf("eglInitialize not found: %w", err)
	}
	if symEglTerminate, err = eglLib.Sym("eglTerminate"); err != nil {
		return fmt.Errorf("eglTerminate not found: %w", err)
	}
	if symEglQueryString, err = eglLib.Sym("eglQueryString"); err != nil {
		return fmt.Errorf("eglQueryString not found: %w", err)
	}
	if symEglChooseConfig, err = eglLib.Sym("eglChooseConfig"); err != nil {
		return fmt.Errorf("eglChooseConfig not found: %w", err)
	}
	if symEglGetConfigAttrib, err = eglLib.Sym("eglGetConfigAttrib"); err != nil {
		return fmt.Errorf("eglGetConfigAttrib not found: %w", err)
	}
	if symEglCreateWindowSurface, err = eglLib.Sym("eglCreateWindowSurface"); err != nil {
		return fmt.Errorf("eglCreateWindowSurface not found: %w", err)
	}
	if symEglCreatePbufferSurface, err = eglLib.Sym("eglCreatePbufferSurface"); err != nil {
		return fmt.Errorf("eglCreatePbufferSurface not found: %w", err)
	}
	if symEglDestroySurface, err = eglLib.Sym("eglDestroySurface"); err != nil {
		return fmt.Errorf("eglDestroySurface not found: %w", err)
	}
	if symEglBindAPI, err = eglLib.Sym("eglBindAPI"); err != nil {
		return fmt.Errorf("eglBindAPI not found: %w", err)
	}
	if symEglSwapInterval, err = eglLib.Sym("eglSwapInterval"); err != nil {
		return fmt.Errorf("eglSwapInterval not found: %w", err)
	}
	if symEglCreateContext, err = eglLib.Sym("eglCreateContext"); err != nil {
		return fmt.Errorf("eglCreateContext not found: %w", err)
	}
	if symEglDestroyContext, err = eglLib.Sym("eglDestroyContext"); err != nil {
		return fmt.Errorf("eglDestroyContext not found: %w", err)
	}
	if symEglMakeCurrent, err = eglLib.Sym("eglMakeCurrent"); err != nil {
		return fmt.Errorf("eglMakeCurrent not found: %w", err)
	}
	if symEglGetCurrentContext, err = eglLib.Sym("eglGetCurrentContext"); err != nil {
		return fmt.Errorf("eglGetCurrentContext not found: %w", err)
	}
	if symEglGetCurrentDisplay, err = eglLib.Sym("eglGetCurrentDisplay"); err != nil {
		return fmt.Errorf("eglGetCurrentDisplay not found: %w", err)
	}
	symEglGetCurrentSurface, err = eglLib.Sym("eglGetCurrentSurface")
	if err != nil {
		return fmt.Errorf("eglGetCurrentSurface not found: %w", err)
	}
	if symEglSwapBuffers, err = eglLib.Sym("eglSwapBuffers"); err != nil {
		return fmt.Errorf("eglSwapBuffers not found: %w", err)
	}
	if symEglGetProcAddress, err = eglLib.Sym("eglGetProcAddress"); err != nil {
		return fmt.Errorf("eglGetProcAddress not found: %w", err)
	}

	// EGL 1.5 optional
	symEglGetPlatformDisplay = eglLib.SymOrNil("eglGetPlatformDisplay")
	symEglCreatePlatformWindowSurface = eglLib.SymOrNil("eglCreatePlatformWindowSurface")

	return nil
}

// GetError returns the last EGL error.
func GetError() EGLInt {
	var result EGLInt
	result = EGLInt(C.egl_i_(C.uintptr_t(uintptr(symEglGetError))))
	return result
}

// Error is a failed EGL call with the error eglGetError gave for it.
type Error struct {
	Op   string // the call, such as eglMakeCurrent
	Code EGLInt
}

func (e *Error) Error() string { return fmt.Sprintf("%s failed: error 0x%x", e.Op, e.Code) }

// LastError is the Error of op, the call that failed last on this thread.
func LastError(op string) *Error { return &Error{Op: op, Code: GetError()} }

// GetDisplay returns an EGL display connection.
func GetDisplay(displayID EGLNativeDisplayType) EGLDisplay {
	var result EGLDisplay
	result = EGLDisplay(C.egl_u_u(C.uintptr_t(uintptr(symEglGetDisplay)), C.uintptr_t(displayID)))
	return result
}

// GetPlatformDisplay returns an EGL display connection for a specific platform (EGL 1.5).
// Falls back to GetDisplay if EGL 1.5 is not available.
func GetPlatformDisplay(platform EGLEnum, nativeDisplay uintptr, attribList *EGLAttrib) EGLDisplay {
	if symEglGetPlatformDisplay != nil {
		return EGLDisplay(C.egl_u_iup(C.uintptr_t(uintptr(symEglGetPlatformDisplay)), C.uint32_t(platform), C.uintptr_t(nativeDisplay), (*C.uint8_t)(unsafe.Pointer(attribList))))
	}
	// Fallback to eglGetDisplay
	return GetDisplay(EGLNativeDisplayType(nativeDisplay))
}

// Initialize initializes an EGL display connection.
func Initialize(dpy EGLDisplay, major *EGLInt, minor *EGLInt) EGLBoolean {
	// Defensive check: don't try to initialize an invalid display
	if dpy == NoDisplay {
		return False
	}

	var result EGLBoolean
	// For pointer arguments, store as uintptr and pass address of that
	result = EGLBoolean(C.egl_i_upp(C.uintptr_t(uintptr(symEglInitialize)), C.uintptr_t(dpy), (*C.uint8_t)(unsafe.Pointer(major)), (*C.uint8_t)(unsafe.Pointer(minor))))
	return result
}

// Terminate terminates an EGL display connection.
func Terminate(dpy EGLDisplay) EGLBoolean {
	var result EGLBoolean
	result = EGLBoolean(C.egl_i_u(C.uintptr_t(uintptr(symEglTerminate)), C.uintptr_t(dpy)))
	return result
}

// QueryString returns a string describing properties of the EGL client or display.
func QueryString(dpy EGLDisplay, name EGLInt) string {
	return goString(C.egl_P_ui(C.uintptr_t(uintptr(symEglQueryString)), C.uintptr_t(dpy), C.uint32_t(name)))
}

// ChooseConfig returns EGL frame buffer configurations that match specified attributes.
func ChooseConfig(dpy EGLDisplay, attribList *EGLInt, configs *EGLConfig, configSize EGLInt, numConfig *EGLInt) EGLBoolean {
	var result EGLBoolean
	result = EGLBoolean(C.egl_i_uppip(C.uintptr_t(uintptr(symEglChooseConfig)), C.uintptr_t(dpy), (*C.uint8_t)(unsafe.Pointer(attribList)), (*C.uint8_t)(unsafe.Pointer(configs)), C.uint32_t(configSize), (*C.uint8_t)(unsafe.Pointer(numConfig))))
	return result
}

// GetConfigAttrib returns information about an EGL frame buffer configuration.
func GetConfigAttrib(dpy EGLDisplay, config EGLConfig, attribute EGLInt, value *EGLInt) EGLBoolean {
	var result EGLBoolean
	result = EGLBoolean(C.egl_i_uuip(C.uintptr_t(uintptr(symEglGetConfigAttrib)), C.uintptr_t(dpy), C.uintptr_t(config), C.uint32_t(attribute), (*C.uint8_t)(unsafe.Pointer(value))))
	return result
}

// CreateWindowSurface creates a new EGL window surface.
func CreateWindowSurface(dpy EGLDisplay, config EGLConfig, win EGLNativeWindowType, attribList *EGLInt) EGLSurface {
	var result EGLSurface
	result = EGLSurface(C.egl_u_uuup(C.uintptr_t(uintptr(symEglCreateWindowSurface)), C.uintptr_t(dpy), C.uintptr_t(config), C.uintptr_t(win), (*C.uint8_t)(unsafe.Pointer(attribList))))
	return result
}

// HasPlatformWindowSurface reports whether eglCreatePlatformWindowSurface (EGL 1.5)
// is available. When true, CreatePlatformWindowSurface should be preferred over
// CreateWindowSurface on Wayland for spec-correct void* native window handling.
func HasPlatformWindowSurface() bool {
	return hasPlatformWindowSurface
}

// CreatePlatformWindowSurface creates a new EGL window surface using the EGL 1.5
// platform-aware API. Takes void* native window instead of EGLNativeWindowType.
// On Wayland: nativeWindow is wl_egl_window*. attribs ends with None.
// Rust wgpu-hal egl.rs:1485 uses this when EGL 1.5 is available.
// Falls back to CreateWindowSurface when EGL 1.5 is not available.
func CreatePlatformWindowSurface(dpy EGLDisplay, config EGLConfig, nativeWindow uintptr, attribs []EGLAttrib) EGLSurface {
	if !hasPlatformWindowSurface {
		// Fallback: cast to EGLNativeWindowType (works on x86_64 where sizes match).
		ints := make([]EGLInt, len(attribs))
		for i, a := range attribs {
			ints[i] = EGLInt(a)
		}
		return CreateWindowSurface(dpy, config, EGLNativeWindowType(nativeWindow), &ints[0])
	}
	var result EGLSurface
	result = EGLSurface(C.egl_u_uuup(C.uintptr_t(uintptr(symEglCreatePlatformWindowSurface)), C.uintptr_t(dpy), C.uintptr_t(config), C.uintptr_t(nativeWindow), (*C.uint8_t)(unsafe.Pointer(&attribs[0]))))
	return result
}

// CreatePbufferSurface creates a new EGL pixel buffer surface.
func CreatePbufferSurface(dpy EGLDisplay, config EGLConfig, attribList *EGLInt) EGLSurface {
	var result EGLSurface
	result = EGLSurface(C.egl_u_uup(C.uintptr_t(uintptr(symEglCreatePbufferSurface)), C.uintptr_t(dpy), C.uintptr_t(config), (*C.uint8_t)(unsafe.Pointer(attribList))))
	return result
}

// DestroySurface destroys an EGL surface.
func DestroySurface(dpy EGLDisplay, surface EGLSurface) EGLBoolean {
	var result EGLBoolean
	result = EGLBoolean(C.egl_i_uu(C.uintptr_t(uintptr(symEglDestroySurface)), C.uintptr_t(dpy), C.uintptr_t(surface)))
	return result
}

// BindAPI sets the current rendering API.
func BindAPI(api EGLEnum) EGLBoolean {
	var result EGLBoolean
	result = EGLBoolean(C.egl_i_i(C.uintptr_t(uintptr(symEglBindAPI)), C.uint32_t(api)))
	return result
}

// SwapInterval specifies the minimum number of video frames between buffer swaps.
func SwapInterval(dpy EGLDisplay, interval EGLInt) EGLBoolean {
	var result EGLBoolean
	result = EGLBoolean(C.egl_i_ui(C.uintptr_t(uintptr(symEglSwapInterval)), C.uintptr_t(dpy), C.uint32_t(interval)))
	return result
}

// CreateContext creates a new EGL rendering context.
func CreateContext(dpy EGLDisplay, config EGLConfig, shareContext EGLContext, attribList *EGLInt) EGLContext {
	var result EGLContext
	result = EGLContext(C.egl_u_uuup(C.uintptr_t(uintptr(symEglCreateContext)), C.uintptr_t(dpy), C.uintptr_t(config), C.uintptr_t(shareContext), (*C.uint8_t)(unsafe.Pointer(attribList))))
	return result
}

// DestroyContext destroys an EGL rendering context.
func DestroyContext(dpy EGLDisplay, ctx EGLContext) EGLBoolean {
	var result EGLBoolean
	result = EGLBoolean(C.egl_i_uu(C.uintptr_t(uintptr(symEglDestroyContext)), C.uintptr_t(dpy), C.uintptr_t(ctx)))
	return result
}

// MakeCurrent binds context to the current rendering thread and surfaces.
func MakeCurrent(dpy EGLDisplay, draw EGLSurface, read EGLSurface, ctx EGLContext) EGLBoolean {
	var result EGLBoolean
	result = EGLBoolean(C.egl_i_uuuu(C.uintptr_t(uintptr(symEglMakeCurrent)), C.uintptr_t(dpy), C.uintptr_t(draw), C.uintptr_t(read), C.uintptr_t(ctx)))
	return result
}

// GetCurrentContext returns the current EGL rendering context.
func GetCurrentContext() EGLContext {
	var result EGLContext
	result = EGLContext(C.egl_u_(C.uintptr_t(uintptr(symEglGetCurrentContext))))
	return result
}

// GetCurrentSurface returns the surface the calling thread's current
// context draws to (Draw) or reads from (Read).
func GetCurrentSurface(which EGLInt) EGLSurface {
	return EGLSurface(C.egl_u_i(C.uintptr_t(uintptr(symEglGetCurrentSurface)), C.uint32_t(uint32(which))))
}

// GetCurrentDisplay returns the current EGL display connection.
func GetCurrentDisplay() EGLDisplay {
	var result EGLDisplay
	result = EGLDisplay(C.egl_u_(C.uintptr_t(uintptr(symEglGetCurrentDisplay))))
	return result
}

// SwapBuffers posts EGL surface color buffer to a native window.
func SwapBuffers(dpy EGLDisplay, surface EGLSurface) EGLBoolean {
	var result EGLBoolean
	result = EGLBoolean(C.egl_i_uu(C.uintptr_t(uintptr(symEglSwapBuffers)), C.uintptr_t(dpy), C.uintptr_t(surface)))
	return result
}

// GetProcAddress returns the address of an EGL or client API extension function.
func GetProcAddress(procname string) unsafe.Pointer {
	cname := append([]byte(procname), 0)
	return C.egl_P_p(C.uintptr_t(uintptr(symEglGetProcAddress)), (*C.uint8_t)(unsafe.Pointer(&cname[0])))
}

// loadOptionalExtensions looks up the optional EGL entry points, after the
// core ones.
func loadOptionalExtensions() {
	symEglSwapBuffersWithDamageKHR = GetProcAddress("eglSwapBuffersWithDamageKHR")
	symEglSwapBuffersWithDamageEXT = GetProcAddress("eglSwapBuffersWithDamageEXT")

	// EGL 1.5: eglCreatePlatformWindowSurface — spec-correct void* native window.
	// Loaded from libEGL's symbol table (not via eglGetProcAddress) since it's
	// a core EGL 1.5 function. Rust wgpu-hal egl.rs:1485 uses this when available.
	hasPlatformWindowSurface = symEglCreatePlatformWindowSurface != nil
}

// damageSwap is the swap with damage of a display with the extensions
// exts, khr or ext as the display lists EGL_KHR_ or
// EGL_EXT_swap_buffers_with_damage, nil when it lists neither.
func damageSwap(exts string, khr, ext unsafe.Pointer) unsafe.Pointer {
	switch {
	case khr != nil && hasExtension(exts, "EGL_KHR_swap_buffers_with_damage"):
		return khr
	case ext != nil && hasExtension(exts, "EGL_EXT_swap_buffers_with_damage"):
		return ext
	}
	return nil
}

// swapBuffersWithDamage calls fn, an eglSwapBuffersWithDamage.
func swapBuffersWithDamage(fn unsafe.Pointer, dpy EGLDisplay, surface EGLSurface, rects *int32, nRects int32) EGLBoolean {
	return EGLBoolean(C.egl_i_uupi(C.uintptr_t(uintptr(fn)), C.uintptr_t(dpy), C.uintptr_t(surface), (*C.uint8_t)(unsafe.Pointer(rects)), C.uint32_t(nRects)))
}

// goString copies a NUL-terminated string EGL returned, "" for NULL.
func goString(cstr unsafe.Pointer) string {
	if cstr == nil {
		return ""
	}
	return C.GoString((*C.char)(cstr))
}
