// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

package gles

import (
	"fmt"
	"runtime"
	"sync"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/egl"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// vendorUnknown is the placeholder vendor name used when the actual GPU vendor
// cannot be determined (e.g., no surface available during adapter enumeration).
const vendorUnknown = "Unknown"

// driverOpenGL is the AdapterInfo.Driver string for GLES backends.
const driverOpenGL = "OpenGL"

// Backend implements hal.Backend for OpenGL ES / OpenGL 3.3+ on Linux.
type Backend struct{}

// NewBackend returns a GLES backend instance.
func NewBackend() Backend { return Backend{} }

// Variant returns the backend type identifier.
func (Backend) Variant() gputypes.Backend {
	return gputypes.BackendGL
}

// CreateInstance creates a new OpenGL instance with an optional EGL context.
// Attempts to create an EGL context at instance level (Rust wgpu-hal egl.rs:846
// parity) for adapter enumeration without a surface. Uses surfaceless/pbuffer
// context — same role as Windows hidden 1×1 HWND (v0.28.6).
//
// On Wayland, this may fail (EGL needs wl_display*) — that's OK, CreateSurface
// provides the proper context later. On X11/headless, this succeeds.
func (Backend) CreateInstance(_ *hal.InstanceDescriptor) (hal.Instance, error) {
	if err := egl.Init(); err != nil {
		return nil, fmt.Errorf("gles: failed to initialize EGL: %w", err)
	}

	// Try to create instance-level EGL context (Rust wgpu-hal parity).
	// Skip on Wayland: surfaceless context would create GL objects (VAO, FBO) that
	// are invisible to the Surface's windowed context (GL objects not shared between
	// EGL contexts). Device/Queue must use the SAME context as the window surface.
	// On X11/headless, instance context IS the presentation context — safe to create.
	if egl.DetectWindowKind() == egl.WindowKindWayland {
		hal.Logger().Info("gles: skipping instance context on Wayland (surface provides context)")
		return &Instance{}, nil
	}

	config := egl.DefaultContextConfig()
	config.GLES = false
	if runtime.GOOS == "android" {
		// GLES 3.0 only. The context lives on the Instance, so what the
		// device made survives the window Android replaces on every resume.
		config.GLES, config.GLVersionMajor, config.GLVersionMinor, config.CoreProfile = true, 3, 0, false
	}
	// EGL keeps the bound API, the last error and the current context per
	// thread, so the context is made, made current and loaded on one pinned
	// thread, and released again before returning. Left current, it stays
	// bound to whichever OS thread ran this: AdapterContext.Lock on any other
	// thread fails with EGL_BAD_ACCESS and runs its GL calls without a
	// context, and Destroy from another thread only marks it for deletion.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	eglCtx, err := egl.NewContext(config)
	if err != nil {
		hal.Logger().Info("gles: instance context unavailable (expected on Wayland)", "err", err)
		return &Instance{}, nil
	}

	if err := eglCtx.MakeCurrent(); err != nil {
		eglCtx.Destroy()
		hal.Logger().Warn("gles: instance context MakeCurrent failed", "err", err)
		return &Instance{}, nil
	}

	glCtx := &gl.Context{}
	if err := glCtx.Load(egl.GetGLProcAddress, config.GLES); err != nil {
		eglCtx.Destroy()
		hal.Logger().Warn("gles: instance GL load failed", "err", err)
		return &Instance{}, nil
	}

	hal.Logger().Info("gles: instance created with EGL context",
		"version", glCtx.GetString(gl.VERSION),
		"renderer", glCtx.GetString(gl.RENDERER))

	egl.MakeCurrent(eglCtx.Display(), egl.NoSurface, egl.NoSurface, egl.NoContext)
	return &Instance{ctx: NewAdapterContext(eglCtx, glCtx, true)}, nil
}

// Instance implements hal.Instance for the OpenGL backend on Linux.
// ctx is non-nil when an EGL context is available: either made at instance
// init (X11/headless) or adopted from the first CreateSurface on Wayland,
// where no instance context could be made. The Instance owns ctx for its
// whole lifetime and destroys it last, in Destroy.
type Instance struct {
	mu  sync.Mutex
	ctx *AdapterContext
}

// context returns the instance AdapterContext. Safe to call while
// CreateSurface adopts the context it made on another goroutine.
func (i *Instance) context() *AdapterContext {
	i.mu.Lock()
	defer i.mu.Unlock()
	return i.ctx
}

// CreateSurface creates an OpenGL surface from window handles.
// On Linux: displayHandle and windowHandle are platform-specific.
// For X11: displayHandle is X11 Display*, windowHandle is Window.
// For Wayland: displayHandle is wl_display*, windowHandle is wl_surface*.
//
// When Instance has a pre-created EGL context (X11/headless), the context is
// SHARED — Surface only creates an EGL window surface for presentation. This
// matches the Windows pattern where Instance owns AdapterContext and Surface
// is lightweight (just HWND + reference to shared ctx).
//
// When Instance has no context (Wayland — no wl_display* at init), CreateSurface
// creates a new EGL context with the caller's displayHandle and hands it to the
// Instance, which owns it from then on. The Surface only shares it: the Adapter,
// Device and Queue made from the Surface use the same context and outlive the
// window, so the Surface must not free it.
func (i *Instance) CreateSurface(target hal.SurfaceTarget) (hal.Surface, error) {
	var targetWindowKind egl.WindowKind
	switch target.Kind {
	case hal.SurfaceTargetXlibWindow:
		targetWindowKind = egl.WindowKindX11
	case hal.SurfaceTargetWaylandSurface:
		targetWindowKind = egl.WindowKindWayland
	case hal.SurfaceTargetAndroidNativeWindow:
		targetWindowKind = egl.WindowKindAndroid
	default:
		return nil, fmt.Errorf("gles: %w: got %s, backend requires Xlib window, Wayland surface or Android native window", hal.ErrUnsupportedSurfaceTarget, target.Kind)
	}
	displayHandle, windowHandle := target.DisplayHandle, target.WindowHandle

	// Path A: share Instance AdapterContext (X11 — context matches window system).
	// Do NOT share if Instance context is surfaceless (headless/Wayland fallback)
	// and Surface needs a window — the EGL display won't support eglCreateWindowSurface.
	if instCtx := i.context(); instCtx != nil && instCtx.EGL() != nil && instCtx.GL() != nil && instCtx.EGL().WindowKind() == targetWindowKind {
		hal.Logger().Info("gles: surface sharing Instance AdapterContext")
		glCtx := instCtx.Lock()
		version := glCtx.GetString(gl.VERSION)
		renderer := glCtx.GetString(gl.RENDERER)
		instCtx.Unlock()
		return &Surface{
			displayHandle: displayHandle,
			windowHandle:  windowHandle,
			ctx:           instCtx,
			eglDisplay:    instCtx.EGL().Display(),
			ownsContext:   false,
			version:       version,
			renderer:      renderer,
		}, nil
	}

	// Path B: create new context (Wayland — Instance had no wl_display* at init).
	// Try desktop GL first, fall back to GLES 3.0 — Mesa Wayland EGL may only
	// expose EGL_OPENGL_ES3_BIT configs, not EGL_OPENGL_BIT. Found by @lkmavi (PR #215).
	// One pinned thread for the whole setup, and released before returning,
	// like the instance context in CreateInstance.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	config := egl.DefaultContextConfig()
	config.NativeDisplay = displayHandle
	config.WindowKind = &targetWindowKind
	eglCtx, err := egl.NewContext(config)
	if err != nil {
		config.GLES = true
		config.GLVersionMajor = 3
		config.GLVersionMinor = 0
		config.CoreProfile = false
		eglCtx, err = egl.NewContext(config)
	}
	if err != nil {
		return nil, fmt.Errorf("gles: failed to create EGL context (tried desktop GL and GLES 3.0): %w", err)
	}

	if err := eglCtx.MakeCurrent(); err != nil {
		eglCtx.Destroy()
		return nil, fmt.Errorf("gles: failed to make context current: %w", err)
	}

	glCtx := &gl.Context{}
	if err := glCtx.Load(egl.GetGLProcAddress, config.GLES); err != nil {
		eglCtx.Destroy()
		return nil, fmt.Errorf("gles: failed to load GL functions: %w", err)
	}

	version := glCtx.GetString(gl.VERSION)
	renderer := glCtx.GetString(gl.RENDERER)
	hal.Logger().Info("gles: surface created with AdapterContext",
		"version", version, "renderer", renderer, "gles", config.GLES)

	egl.MakeCurrent(eglCtx.Display(), egl.NoSurface, egl.NoSurface, egl.NoContext)

	// The Adapter, Device and Queue made from this Surface share this context
	// and outlive the Surface, so the Instance owns it, not the Surface. With
	// the Surface owning it, closing the window freed the EGL context first and
	// the Device.Destroy that followed locked a nil EGL context and ran its GL
	// calls on a nil *gl.Context, a panic on shutdown.
	adapterCtx := NewAdapterContext(eglCtx, glCtx, true)
	if prev := i.adoptContext(adapterCtx); prev != nil {
		prev.Destroy()
	}

	return &Surface{
		displayHandle: displayHandle,
		windowHandle:  windowHandle,
		ctx:           adapterCtx,
		eglDisplay:    eglCtx.Display(),
		ownsContext:   false,
		version:       version,
		renderer:      renderer,
	}, nil
}

// adoptContext makes ctx the instance AdapterContext and returns the one it
// replaces, if any, for the caller to destroy. The Instance owns the context
// from here on, so it lives as long as the Surface and the Device that share it.
func (i *Instance) adoptContext(ctx *AdapterContext) (prev *AdapterContext) {
	i.mu.Lock()
	defer i.mu.Unlock()
	prev = i.ctx
	i.ctx = ctx
	if prev == ctx {
		return nil
	}
	return prev
}

// EnumerateAdapters returns available OpenGL adapters.
// Uses surface context (preferred), instance context (X11/headless), or placeholder.
func (i *Instance) EnumerateAdapters(surfaceHint hal.Surface) []hal.ExposedAdapter {
	// Priority 1: surface provides the best context (has window handle)
	if surface, ok := surfaceHint.(*Surface); ok {
		return []hal.ExposedAdapter{surface.GetAdapterInfo()}
	}

	// Priority 2: instance-level AdapterContext (created in CreateInstance via
	// pbuffer/surfaceless, or adopted from the first CreateSurface on Wayland)
	if instCtx := i.context(); instCtx != nil && instCtx.GL() != nil {
		return []hal.ExposedAdapter{
			makeAdapterFromContext(instCtx),
		}
	}

	// Priority 3: no context available (Wayland without surface hint)
	// Return placeholder — Open() has nil guard from PR #210.
	return []hal.ExposedAdapter{placeholderExposedAdapter("OpenGL 3.3+ / ES 3.0+ (no context — use RequestAdapterWithSurface)")}
}

// placeholderExposedAdapter returns a non-openable adapter when no EGL context exists.
func placeholderExposedAdapter(driverInfo string) hal.ExposedAdapter {
	return hal.ExposedAdapter{
		Adapter: &Adapter{},
		Info: gputypes.AdapterInfo{
			Name:       "OpenGL Adapter",
			Vendor:     vendorUnknown,
			DeviceType: gputypes.DeviceTypeOther,
			Driver:     driverOpenGL,
			DriverInfo: driverInfo,
			Backend:    gputypes.BackendGL,
		},
		Capabilities: hal.Capabilities{
			Limits: gputypes.DefaultLimits(),
			AlignmentsMask: hal.Alignments{
				BufferCopyOffset: 4,
				BufferCopyPitch:  256,
			},
		},
	}
}

// makeAdapterFromContext creates an ExposedAdapter using a live AdapterContext.
func makeAdapterFromContext(ctx *AdapterContext) hal.ExposedAdapter {
	glCtx := ctx.Lock()
	defer ctx.Unlock()

	version := glCtx.GetString(gl.VERSION)
	renderer := glCtx.GetString(gl.RENDERER)
	vendor := glCtx.GetString(gl.VENDOR)

	return hal.ExposedAdapter{
		Adapter: &Adapter{
			ctx:      ctx,
			version:  version,
			renderer: renderer,
		},
		Info: gputypes.AdapterInfo{
			Name:       renderer,
			Vendor:     vendor,
			DeviceType: gputypes.DeviceTypeIntegratedGPU,
			Driver:     driverOpenGL,
			DriverInfo: version,
			Backend:    gputypes.BackendGL,
		},
		Capabilities: hal.Capabilities{
			Limits: gputypes.DefaultLimits(),
			AlignmentsMask: hal.Alignments{
				BufferCopyOffset: 4,
				BufferCopyPitch:  256,
			},
		},
	}
}

// Destroy releases the instance resources. The Instance owns its AdapterContext,
// made at init or adopted from a Surface, and is released after the Device and
// the Surface, so this is the last destroy of the shared EGL context.
func (i *Instance) Destroy() {
	i.mu.Lock()
	ctx := i.ctx
	i.ctx = nil
	i.mu.Unlock()
	if ctx != nil {
		ctx.Destroy()
	}
}
