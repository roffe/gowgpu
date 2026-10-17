// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

package egl

// #include "trampolines_linux.h"
import "C"

import (
	"fmt"
	"os"
	"strings"
	"unsafe"

	"github.com/roffe/gowgpu/internal/dl"
)

var (
	// x11Lib is the loaded libX11.so library.
	x11Lib dl.Lib
	// waylandClientLib is the loaded libwayland-client.so library.
	waylandClientLib dl.Lib

	// X11 function symbols
	symXOpenDisplay  unsafe.Pointer
	symXCloseDisplay unsafe.Pointer

	// Wayland function symbols
	symWlDisplayConnect    unsafe.Pointer
	symWlDisplayDisconnect unsafe.Pointer
)

// DisplayOwner manages the lifetime of a native display connection.
// It ensures the display is properly closed when the owner is destroyed.
type DisplayOwner struct {
	kind    WindowKind
	display uintptr
	lib     dl.Lib
}

// OpenX11Display opens an X11 display connection.
// Returns nil if X11 libraries are not available or display cannot be opened.
func OpenX11Display() *DisplayOwner {
	var err error

	if x11Lib, err = dl.Open("libX11.so.6", "libX11.so"); err != nil {
		return nil
	}
	// Display* XOpenDisplay(char* display_name)
	if symXOpenDisplay, err = x11Lib.Sym("XOpenDisplay"); err != nil {
		return nil
	}
	// int XCloseDisplay(Display* display)
	if symXCloseDisplay, err = x11Lib.Sym("XCloseDisplay"); err != nil {
		return nil
	}

	// Open default display (NULL = use DISPLAY environment variable)
	var display uintptr
	var displayName uintptr // NULL pointer for default display
	display = uintptr(C.egl_u_u(C.uintptr_t(uintptr(symXOpenDisplay)), C.uintptr_t(displayName)))
	if display == 0 {
		return nil
	}

	return &DisplayOwner{
		kind:    WindowKindX11,
		display: display,
		lib:     x11Lib,
	}
}

// TestWaylandDisplay tests if a Wayland display connection is available.
// Returns nil if Wayland libraries are not available or display cannot be opened.
// This function connects and immediately disconnects to test availability.
func TestWaylandDisplay() *DisplayOwner {
	var err error

	if waylandClientLib, err = dl.Open("libwayland-client.so.0", "libwayland-client.so"); err != nil {
		return nil
	}
	// wl_display* wl_display_connect(const char* name)
	if symWlDisplayConnect, err = waylandClientLib.Sym("wl_display_connect"); err != nil {
		return nil
	}
	// void wl_display_disconnect(wl_display* display)
	if symWlDisplayDisconnect, err = waylandClientLib.Sym("wl_display_disconnect"); err != nil {
		return nil
	}

	// Test connection (NULL = use WAYLAND_DISPLAY environment variable)
	var display uintptr
	var displayName uintptr // NULL pointer for default display
	display = uintptr(C.egl_u_u(C.uintptr_t(uintptr(symWlDisplayConnect)), C.uintptr_t(displayName)))
	if display == 0 {
		return nil
	}

	// Immediately disconnect - we just wanted to test availability
	C.egl_v_u(C.uintptr_t(uintptr(symWlDisplayDisconnect)), C.uintptr_t(display))

	return &DisplayOwner{
		kind:    WindowKindWayland,
		display: 0, // We don't keep the connection open
		lib:     waylandClientLib,
	}
}

// QueryClientExtensions returns EGL client extensions available without a display.
// This MUST be called with EGL_NO_DISPLAY to get client extensions.
// EGL client extensions are extensions that can be queried before display initialization.
func QueryClientExtensions() string {
	return QueryString(NoDisplay, Extensions)
}

// HasSurfacelessSupport checks if Mesa surfaceless platform is available.
func HasSurfacelessSupport() bool {
	extensions := QueryClientExtensions()
	return strings.Contains(extensions, "EGL_MESA_platform_surfaceless")
}

// DetectWindowKind detects the available window system.
// Priority order:
//  1. If no DISPLAY/WAYLAND_DISPLAY set AND surfaceless available -> Surfaceless (for CI)
//  2. Wayland (if WAYLAND_DISPLAY is set and works)
//  3. X11 (if DISPLAY is set and works)
//  4. Surfaceless (final fallback)
func DetectWindowKind() WindowKind {
	displayEnv := os.Getenv("DISPLAY")
	waylandEnv := os.Getenv("WAYLAND_DISPLAY")

	// In headless environments (no display env vars), prefer surfaceless
	// This is critical for CI environments where Xvfb may be set but doesn't work properly
	if displayEnv == "" && waylandEnv == "" {
		if HasSurfacelessSupport() {
			return WindowKindSurfaceless
		}
	}

	// Check environment variables for real displays
	if waylandEnv != "" {
		if owner := TestWaylandDisplay(); owner != nil {
			owner.Close()
			return WindowKindWayland
		}
	}

	if displayEnv != "" {
		if owner := OpenX11Display(); owner != nil {
			owner.Close()
			return WindowKindX11
		}
	}

	// Fallback to surfaceless for headless systems
	return WindowKindSurfaceless
}

// GetEGLDisplay returns an EGL display for the detected platform.
// nativeDisplay is the platform-specific native display handle:
//   - Wayland: the app's wl_display* — must NOT be 0, otherwise EGL opens a
//     second wl_display connection and wl_surface proxies will mismatch on configure.
//   - X11: the X11 Display* (0 = use DISPLAY env var via OpenX11Display).
//   - Surfaceless: ignored.
func GetEGLDisplay(nativeDisplay uintptr) (EGLDisplay, WindowKind, *DisplayOwner, error) {
	return getEGLDisplayForKind(nativeDisplay, DetectWindowKind())
}

// getEGLDisplayForKind returns an EGL display for an explicit native window
// system. It is used by typed surface targets so process-global environment
// variables cannot redirect Xlib handles into Wayland or vice versa.
func getEGLDisplayForKind(nativeDisplay uintptr, windowKind WindowKind) (EGLDisplay, WindowKind, *DisplayOwner, error) {
	switch windowKind {
	case WindowKindX11:
		if nativeDisplay != 0 {
			display := GetPlatformDisplay(PlatformX11KHR, nativeDisplay, nil)
			if display != NoDisplay {
				return display, WindowKindX11, nil, nil
			}
			display = GetDisplay(EGLNativeDisplayType(nativeDisplay))
			if display == NoDisplay {
				return NoDisplay, WindowKindUnknown, nil, fmt.Errorf("eglGetDisplay failed for supplied X11 display")
			}
			return display, WindowKindX11, nil, nil
		}
		owner := OpenX11Display()
		if owner == nil {
			return NoDisplay, WindowKindUnknown, nil, fmt.Errorf("failed to open X11 display")
		}

		// Try EGL 1.5 platform extension first
		display := GetPlatformDisplay(PlatformX11KHR, owner.display, nil)
		if display != NoDisplay {
			return display, WindowKindX11, owner, nil
		}

		// Fallback to EGL 1.4
		display = GetDisplay(EGLNativeDisplayType(owner.display))
		if display == NoDisplay {
			owner.Close()
			return NoDisplay, WindowKindUnknown, nil, fmt.Errorf("eglGetDisplay failed for X11")
		}
		return display, WindowKindX11, owner, nil

	case WindowKindWayland:
		// Wayland EGL REQUIRES the app's wl_display* — passing 0 opens a second
		// connection via wl_display_connect(NULL), causing "Proxy and queue point
		// to different wl_displays" when eglCreateWindowSurface tries to use a
		// wl_surface* from the app's display on EGL's separate display.
		//
		// If nativeDisplay==0 (Instance init, no wl_display* yet), skip Wayland
		// platform and try surfaceless instead. CreateSurface will call again
		// with the real wl_display*.
		if nativeDisplay == 0 {
			display := GetPlatformDisplay(PlatformSurfacelessMesa, 0, nil)
			if display != NoDisplay {
				return display, WindowKindSurfaceless, nil, nil
			}
			return NoDisplay, WindowKindUnknown, nil, fmt.Errorf("wayland EGL requires wl_display* (nativeDisplay=0), and surfaceless not available")
		}

		display := GetPlatformDisplay(PlatformWaylandKHR, nativeDisplay, nil)
		if display != NoDisplay {
			return display, WindowKindWayland, nil, nil
		}

		// Fallback to EGL 1.4 with same native display
		display = GetDisplay(EGLNativeDisplayType(nativeDisplay))
		if display == NoDisplay {
			return NoDisplay, WindowKindUnknown, nil, fmt.Errorf("eglGetDisplay failed for Wayland")
		}
		return display, WindowKindWayland, nil, nil

	case WindowKindSurfaceless:
		// Surfaceless rendering (headless)
		display := GetPlatformDisplay(PlatformSurfacelessMesa, 0, nil)
		if display != NoDisplay {
			return display, WindowKindSurfaceless, nil, nil
		}

		// Fallback to default display
		display = GetDisplay(DefaultDisplay)
		if display == NoDisplay {
			return NoDisplay, WindowKindUnknown, nil, fmt.Errorf("eglGetDisplay failed for surfaceless")
		}
		return display, WindowKindSurfaceless, nil, nil

	default:
		return NoDisplay, WindowKindUnknown, nil, fmt.Errorf("unknown window system")
	}
}

// Kind returns the window system type.
func (d *DisplayOwner) Kind() WindowKind {
	return d.kind
}

// Display returns the native display pointer.
func (d *DisplayOwner) Display() uintptr {
	return d.display
}

// Close closes the native display connection and unloads the library.
func (d *DisplayOwner) Close() {
	if d.display == 0 {
		return
	}

	switch d.kind {
	case WindowKindX11:
		if symXCloseDisplay != nil {
			C.egl_i_u(C.uintptr_t(uintptr(symXCloseDisplay)), C.uintptr_t(d.display))
		}
	case WindowKindWayland:
		// Wayland display is managed by the window system
		// We don't close it here
	}

	d.display = 0
}
