// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

package gles

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/egl"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// Adapter implements hal.Adapter for OpenGL on Linux.
// Holds a shared *AdapterContext (owned by Instance on X11/headless, or by
// Surface on Wayland).
type Adapter struct {
	ctx           *AdapterContext
	displayHandle uintptr
	windowHandle  uintptr
	version       string
	renderer      string

	// caps holds the probed adapter capabilities (extensions, features,
	// limits, MSAA support). Populated by queryAdapterCapabilities during
	// adapter enumeration.
	caps AdapterCapabilities
}

// Open creates a logical device with the requested features and limits.
func (a *Adapter) Open(_ gputypes.Features, _ gputypes.Limits) (hal.OpenDevice, error) {
	// EnumerateAdapters(nil) path returns an adapter with nil ctx because no
	// EGL context can be created without a display/window handle. Return a
	// descriptive error instead of a nil pointer dereference at GenVertexArrays.
	if a.ctx == nil || a.ctx.GL() == nil {
		return hal.OpenDevice{}, fmt.Errorf("gles: adapter has no GL context — pass a surface hint to CreateSurface before RequestDevice")
	}

	glCtx := a.ctx.Lock()
	defer a.ctx.Unlock()

	// An sRGB attachment encodes what the shader writes, as WebGPU's sRGB
	// formats do: GLES always, desktop GL only with this enabled.
	if !glCtx.IsGLES() {
		glCtx.Enable(gl.FRAMEBUFFER_SRGB)
	}

	// VAO is created lazily in CreateCommandEncoder — ensures it's allocated
	// on the window surface (after Configure), not on the pbuffer (during Open).
	vao := uint32(0)

	// Query hardware texture unit limit for binding validation.
	var maxTexUnits int32
	glCtx.GetIntegerv(gl.MAX_TEXTURE_IMAGE_UNITS, &maxTexUnits)
	if maxTexUnits <= 0 {
		maxTexUnits = 8 // Conservative default
	}

	vendor := glCtx.GetString(gl.VENDOR)

	hal.Logger().Info("gles: device opened",
		"vendor", vendor,
		"version", a.version,
		"renderer", a.renderer,
		"maxTextureUnits", maxTexUnits,
	)

	glslVer := GLSLVersionToNaga(a.caps.GLSLVersion, a.caps.IsES)

	fence := NewFence(glCtx)
	device := &Device{
		fence:            fence,
		ctx:              a.ctx,
		displayHandle:    a.displayHandle,
		windowHandle:     a.windowHandle,
		vao:              vao,
		maxTextureUnits:  maxTexUnits,
		maxMSAA:          a.caps.MaxMSAASamples,
		formats:          newFormatCaps(&a.caps),
		depthClamp:       a.caps.Features.Contains(gputypes.FeatureDepthClipControl),
		independentBlend: a.caps.DownlevelFlags.Contains(gputypes.DownlevelFlagsIndependentBlend),
		anisotropy:       a.caps.DownlevelFlags.Contains(gputypes.DownlevelFlagsAnisotropicFiltering),
		sampleMask:       glVersionAtLeast(a.caps.GLMajor, a.caps.GLMinor, a.caps.IsES, [2]int{3, 1}, [2]int{3, 2}),
		fixedIndexRestart: glVersionAtLeast(a.caps.GLMajor, a.caps.GLMinor, a.caps.IsES, [2]int{3, 0}, [2]int{4, 3}) ||
			hasExtension(a.caps.Extensions, "GL_ARB_ES3_compatibility"),
		drawIndirect: glVersionAtLeast(a.caps.GLMajor, a.caps.GLMinor, a.caps.IsES, [2]int{3, 1}, [2]int{4, 0}) ||
			hasExtension(a.caps.Extensions, "GL_ARB_draw_indirect"),
		baseVertex:          a.caps.DownlevelFlags.Contains(gputypes.DownlevelFlagsBaseVertex),
		glslVersion:         glslVer,
		shaderBindingLayout: glslVer.SupportsExplicitLocations(),
	}

	queue := &Queue{
		ctx:   a.ctx,
		fence: fence,
	}

	return hal.OpenDevice{
		Device: device,
		Queue:  queue,
	}, nil
}

// TextureFormatCapabilities returns capabilities for a texture format.
// Uses probed GL extension and MSAA information for accurate per-format detection.
func (a *Adapter) TextureFormatCapabilities(format gputypes.TextureFormat) hal.TextureFormatCapabilities {
	return queryTextureFormatCapabilities(format, a.caps.Features, a.caps.MaxMSAASamples, a.caps.Extensions)
}

// SurfaceCapabilities returns surface capabilities. The sRGB formats need
// sRGB window surfaces on ES (EGL 1.5 or EGL_KHR_gl_colorspace), where the
// present blit decodes an sRGB swapchain, as wgpu-hal and Dawn offer them.
// Premultiplied alpha is offered on Wayland alone, whose compositor blends
// a window by the alpha of its config: an X11 window gets a config without
// alpha, and an Android window's own format decides.
func (a *Adapter) SurfaceCapabilities(_ hal.Surface) *hal.SurfaceCapabilities {
	formats := []gputypes.TextureFormat{
		gputypes.TextureFormatBGRA8Unorm,
		gputypes.TextureFormatRGBA8Unorm,
	}
	var eglCtx *egl.Context
	if a.ctx != nil {
		eglCtx = a.ctx.EGL()
	}
	if !a.caps.IsES || eglCtx != nil && eglCtx.SupportsGLColorspace() {
		formats = append(formats, gputypes.TextureFormatBGRA8UnormSrgb, gputypes.TextureFormatRGBA8UnormSrgb)
	}
	alphaModes := []gputypes.CompositeAlphaMode{hal.CompositeAlphaModeOpaque}
	if eglCtx != nil && eglCtx.WindowKind() == egl.WindowKindWayland {
		alphaModes = append(alphaModes, hal.CompositeAlphaModePremultiplied)
	}
	return &hal.SurfaceCapabilities{
		Formats: formats,
		PresentModes: []gputypes.PresentMode{
			hal.PresentModeFifo,      // VSync on
			hal.PresentModeImmediate, // VSync off (if supported)
		},
		AlphaModes: alphaModes,
	}
}

// Destroy releases the adapter.
func (a *Adapter) Destroy() {
	// Adapter doesn't own the GL context
}
