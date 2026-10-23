// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

package egl

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

// Context wraps an EGL rendering context with its display, config, and surface.
type Context struct {
	display      EGLDisplay
	config       EGLConfig
	context      EGLContext
	pbuffer      EGLSurface
	windowKind   WindowKind
	major, minor EGLInt        // the display's EGL version
	extensions   string        // the display's extensions
	cfg          ContextConfig // what made the context, for NewShared
	shared       bool          // made by NewShared: borrows the display
	displayOwner *DisplayOwner // owns native display connection (X11); closed after eglTerminate
}

// ContextConfig holds configuration options for creating an EGL context.
type ContextConfig struct {
	// GLVersionMajor is the major OpenGL version (e.g., 3 for OpenGL 3.3).
	GLVersionMajor int
	// GLVersionMinor is the minor OpenGL version (e.g., 3 for OpenGL 3.3).
	GLVersionMinor int
	// CoreProfile requests a core profile context (vs compatibility).
	CoreProfile bool
	// Debug enables debug context with validation.
	Debug bool
	// GLES requests OpenGL ES instead of desktop OpenGL.
	GLES bool
	// Surfaceless creates a context without a surface (headless rendering).
	Surfaceless bool
	// NativeDisplay is the native display handle to use for EGL display creation.
	// On Wayland: must be the app's wl_display* — passing 0 causes EGL to open
	// a second connection, which makes wl_surface proxies mismatched on configure.
	// On X11: the X11 Display*. Zero uses the default display.
	NativeDisplay uintptr
	// WindowKind selects the native window system explicitly. Nil preserves
	// automatic environment-based detection and keeps ContextConfig's zero value
	// independent of WindowKind's numeric constants.
	WindowKind *WindowKind
}

// DefaultContextConfig returns a sensible default context configuration.
// Creates an OpenGL 3.3 core profile context.
func DefaultContextConfig() ContextConfig {
	return ContextConfig{
		GLVersionMajor: 3,
		GLVersionMinor: 3,
		CoreProfile:    true,
		Debug:          false,
		GLES:           false,
		Surfaceless:    false,
	}
}

// NewContext creates a new EGL context. An explicit WindowKind selects that
// platform directly; otherwise it detects X11, Wayland, or Surfaceless.
func NewContext(config ContextConfig) (*Context, error) {
	// Get an EGL display for the explicit or detected platform.
	// displayOwner (non-nil for X11) keeps the native display connection alive.
	windowKind := selectWindowKind(config.WindowKind, DetectWindowKind)
	display, windowKind, displayOwner, err := getEGLDisplayForKind(config.NativeDisplay, windowKind)
	if err != nil {
		return nil, fmt.Errorf("failed to get EGL display: %w", err)
	}

	// closeOwner is a helper to close the display owner on error paths.
	closeOwner := func() {
		if displayOwner != nil {
			displayOwner.Close()
		}
	}

	major, minor, err := initializeDisplay(display)
	if err != nil {
		closeOwner()
		return nil, err
	}
	fail := func(err error) (*Context, error) {
		releaseDisplay(display)
		closeOwner()
		return nil, err
	}

	// Bind OpenGL or OpenGL ES API
	api := OpenGLAPI
	if config.GLES {
		api = OpenGLESAPI
	}
	if BindAPI(api) == False {
		return fail(fmt.Errorf("eglBindAPI failed: error 0x%x", GetError()))
	}

	// Choose EGL frame buffer configuration
	eglConfig, err := chooseEGLConfig(display, config, windowKind)
	if err != nil {
		return fail(fmt.Errorf("failed to choose EGL config: %w", err))
	}

	// Create EGL context
	eglContext := createEGLContext(display, eglConfig, config, NoContext)
	if eglContext == NoContext {
		return fail(fmt.Errorf("eglCreateContext failed: error 0x%x", GetError()))
	}

	// Surfaceless context: EGL 1.5+ or EGL_KHR_surfaceless_context allows
	// MakeCurrent with EGL_NO_SURFACE. Skip pbuffer creation in that case.
	// Fallback to 1×1 pbuffer for older drivers.
	// Matches Rust wgpu-hal egl.rs:735-758.
	displayExts := QueryString(display, Extensions)
	hasSurfaceless := (major > 1 || (major == 1 && minor >= 5)) ||
		strings.Contains(displayExts, "EGL_KHR_surfaceless_context")

	var pbuffer EGLSurface
	if hasSurfaceless {
		pbuffer = NoSurface
	} else {
		pbuffer = createPbufferSurface(display, eglConfig)
		if pbuffer == NoSurface {
			err := fmt.Errorf("eglCreatePbufferSurface failed and no surfaceless support: error 0x%x", GetError())
			DestroyContext(display, eglContext)
			return fail(err)
		}
	}

	return &Context{
		display:      display,
		config:       eglConfig,
		context:      eglContext,
		cfg:          config,
		pbuffer:      pbuffer,
		windowKind:   windowKind,
		major:        major,
		minor:        minor,
		extensions:   displayExts,
		displayOwner: displayOwner,
	}, nil
}

// displayRefs counts the Contexts on each EGLDisplay this package
// initialized. EGL gives every caller in the process the same EGLDisplay for
// one platform and native display, and eglTerminate ends every context and
// surface on it, so only the last Context on a display terminates it.
var (
	displayMu   sync.Mutex
	displayRefs = map[EGLDisplay]int{}
)

// initializeDisplay initializes display, or finds it initialized, for one
// more Context.
func initializeDisplay(display EGLDisplay) (major, minor EGLInt, err error) {
	displayMu.Lock()
	defer displayMu.Unlock()
	if Initialize(display, &major, &minor) == False {
		return 0, 0, fmt.Errorf("eglInitialize failed: error 0x%x", GetError())
	}
	displayRefs[display]++
	return major, minor, nil
}

// releaseDisplay gives up a Context's hold on display, and terminates it
// when no other Context has one.
func releaseDisplay(display EGLDisplay) {
	displayMu.Lock()
	defer displayMu.Unlock()
	displayRefs[display]--
	if displayRefs[display] > 0 {
		return
	}
	delete(displayRefs, display)
	Terminate(display)
}

// HasExtension reports whether the context's display has the EGL extension.
func (c *Context) HasExtension(name string) bool {
	for _, e := range strings.Fields(c.extensions) {
		if e == name {
			return true
		}
	}
	return false
}

// SupportsGLColorspace reports whether surfaces on the display can be made
// sRGB, with EGL_GL_COLORSPACE (EGL 1.5 or EGL_KHR_gl_colorspace).
func (c *Context) SupportsGLColorspace() bool {
	return c.major > 1 || c.major == 1 && c.minor >= 5 || c.HasExtension("EGL_KHR_gl_colorspace")
}

func selectWindowKind(requested *WindowKind, detect func() WindowKind) WindowKind {
	if requested != nil {
		return *requested
	}
	return detect()
}

// chooseEGLConfig selects an appropriate EGL frame buffer configuration.
func chooseEGLConfig(display EGLDisplay, config ContextConfig, windowKind WindowKind) (EGLConfig, error) {
	// Determine renderable type
	var renderableType EGLInt
	if config.GLES {
		switch {
		case config.GLVersionMajor >= 3:
			renderableType = OpenGLES3Bit
		case config.GLVersionMajor >= 2:
			renderableType = OpenGLES2Bit
		default:
			renderableType = OpenGLESBit
		}
	} else {
		renderableType = OpenGLBit
	}

	// Tiered config selection (Rust wgpu-hal egl.rs:218-293).
	// Rust's top tier = WindowBit alone (never combined with PbufferBit).
	// Mesa Wayland EGL does NOT support PbufferBit — any tier requiring it
	// returns 0 configs. WindowBit alone = 48 configs on Mesa Wayland.
	tiers := []EGLInt{
		WindowBit | PbufferBit, // Tier 2: X11/headless (window + pbuffer)
		WindowBit,              // Tier 1: Wayland (no pbuffer support in Mesa)
		PbufferBit,             // Tier 0: surfaceless/CI fallback
	}

	// An X11 window surface needs a config whose visual matches the
	// window's, and NVIDIA rejects a 32-bit ARGB config for a default 24-bit
	// visual window (EGL_BAD_CONFIG). Without an alpha request EGL sorts the
	// smaller, 24-bit configs first. The present blit never needs window alpha.
	// ponytail: assumes a default-depth window; a transparent (32-bit visual)
	// one would need its visual ID matched against EGL_NATIVE_VISUAL_ID.
	alpha := EGLInt(8)
	if windowKind == WindowKindX11 {
		alpha = 0
	}
	baseAttribs := []EGLInt{
		RenderableType, renderableType,
		RedSize, 8,
		GreenSize, 8,
		BlueSize, 8,
		AlphaSize, alpha,
		DepthSize, 24,
		StencilSize, 8,
	}

	for _, surfaceType := range tiers {
		attribs := make([]EGLInt, 0, len(baseAttribs)+3)
		attribs = append(attribs, SurfaceType, surfaceType)
		attribs = append(attribs, baseAttribs...)
		attribs = append(attribs, None)

		var eglConfig EGLConfig
		var numConfigs EGLInt
		if ChooseConfig(display, &attribs[0], &eglConfig, 1, &numConfigs) == False {
			continue
		}
		if numConfigs > 0 {
			return eglConfig, nil
		}
	}

	return 0, fmt.Errorf("no suitable EGL configs found (tried window+pbuffer and pbuffer-only)")
}

// createEGLContext creates an EGL rendering context.
func createEGLContext(display EGLDisplay, config EGLConfig, cfg ContextConfig, share EGLContext) EGLContext {
	var attribs []EGLInt

	// Set OpenGL version
	attribs = append(attribs,
		ContextMajorVersion, EGLInt(cfg.GLVersionMajor),
		ContextMinorVersion, EGLInt(cfg.GLVersionMinor),
	)

	// Set profile (core vs compatibility) — desktop OpenGL only.
	// EGL_CONTEXT_OPENGL_PROFILE_MASK is invalid for GLES; some drivers reject it.
	if cfg.CoreProfile && !cfg.GLES {
		attribs = append(attribs,
			ContextOpenGLProfileMask, ContextOpenGLCoreProfileBit,
		)
	}

	// Enable debug context if requested
	if cfg.Debug {
		attribs = append(attribs,
			ContextFlagsKHR, ContextOpenGLDebugBitKHR,
		)
	}

	// Terminate attribute list
	attribs = append(attribs, None)

	return CreateContext(display, config, share, &attribs[0])
}

// createPbufferSurface creates a minimal pbuffer surface for the context.
func createPbufferSurface(display EGLDisplay, config EGLConfig) EGLSurface {
	attribs := []EGLInt{
		Width, 16,
		Height, 16,
		None,
	}
	return CreatePbufferSurface(display, config, &attribs[0])
}

// MakeCurrent makes this context current on the pbuffer (headless rendering).
func (c *Context) MakeCurrent() error {
	if MakeCurrent(c.display, c.pbuffer, c.pbuffer, c.context) == False {
		return fmt.Errorf("eglMakeCurrent failed: error 0x%x", GetError())
	}
	return nil
}

// MakeCurrentSurface makes this context current on a window surface (for Present).
func (c *Context) MakeCurrentSurface(surface EGLSurface) error {
	if MakeCurrent(c.display, surface, surface, c.context) == False {
		return fmt.Errorf("eglMakeCurrent(surface) failed: error 0x%x", GetError())
	}
	return nil
}

// CreateWindowSurface creates an EGL window surface for presentation.
// The surface shares this context's display and config.
func (c *Context) CreateWindowSurface(nativeWindow uintptr) (EGLSurface, error) {
	attribs := []EGLInt{None}
	surface := CreateWindowSurface(c.display, c.config, EGLNativeWindowType(nativeWindow), &attribs[0])
	if surface == NoSurface {
		return NoSurface, fmt.Errorf("eglCreateWindowSurface failed: error 0x%x", GetError())
	}
	return surface, nil
}

// Display returns the EGL display handle.
func (c *Context) Display() EGLDisplay { return c.display }

// Config returns the EGL config handle.
func (c *Context) Config() EGLConfig { return c.config }

// Destroy releases the context and its associated resources, and
// terminates the display when no other Context uses it. Order matters: EGL
// resources first, then the native display connection. Closing the native
// display (e.g. XCloseDisplay) before eglTerminate would cause EGL to access
// a freed connection.
func (c *Context) Destroy() {
	if c.context != NoContext {
		// Unbind context first
		_ = MakeCurrent(c.display, NoSurface, NoSurface, NoContext)
		DestroyContext(c.display, c.context)
		c.context = NoContext
	}
	if c.pbuffer != NoSurface {
		DestroySurface(c.display, c.pbuffer)
		c.pbuffer = NoSurface
	}
	if c.shared {
		return // the display is the sharing context's
	}
	if c.display != NoDisplay {
		releaseDisplay(c.display)
		c.display = NoDisplay
	}
	// Close native display connection AFTER eglTerminate.
	// For X11 this calls XCloseDisplay; for other platforms displayOwner is nil.
	if c.displayOwner != nil {
		c.displayOwner.Close()
		c.displayOwner = nil
	}
}

// NewShared makes a context sharing c's objects (buffers, textures,
// programs and sync objects; not VAOs or framebuffers) for a thread that
// cannot have c, with its own pbuffer when c has one. It borrows c's
// display: its Destroy releases the context and pbuffer only, and c must
// outlive it.
func (c *Context) NewShared() (*Context, error) {
	// The bound API is per thread, and this is another thread's context.
	api := OpenGLAPI
	if c.cfg.GLES {
		api = OpenGLESAPI
	}
	if BindAPI(api) == False {
		return nil, fmt.Errorf("eglBindAPI (shared) failed: error 0x%x", GetError())
	}
	ctx := createEGLContext(c.display, c.config, c.cfg, c.context)
	if ctx == NoContext {
		return nil, fmt.Errorf("eglCreateContext (shared) failed: error 0x%x", GetError())
	}
	pbuffer := NoSurface
	if c.pbuffer != NoSurface {
		if pbuffer = createPbufferSurface(c.display, c.config); pbuffer == NoSurface {
			DestroyContext(c.display, ctx)
			return nil, fmt.Errorf("eglCreatePbufferSurface (shared) failed: error 0x%x", GetError())
		}
	}
	return &Context{display: c.display, config: c.config, context: ctx, pbuffer: pbuffer, windowKind: c.windowKind,
		major: c.major, minor: c.minor, extensions: c.extensions, cfg: c.cfg, shared: true}, nil
}

// EGLContext returns the EGL context handle.
func (c *Context) EGLContext() EGLContext {
	return c.context
}

// Pbuffer returns the pbuffer surface.
func (c *Context) Pbuffer() EGLSurface {
	return c.pbuffer
}

// WindowKind returns the detected window system type.
func (c *Context) WindowKind() WindowKind {
	return c.windowKind
}

// GetGLProcAddress returns the address of an OpenGL function.
// It uses eglGetProcAddress to load both core and extension functions.
func GetGLProcAddress(name string) unsafe.Pointer {
	return GetProcAddress(name)
}
