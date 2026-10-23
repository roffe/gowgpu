//go:build integration && linux && !(js && wasm)

package gles

import (
	"slices"
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/egl"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// An opaque surface on Wayland presents alpha 1 whatever alpha the frame
// has, and a premultiplied one the frame's alpha. A pbuffer with alpha
// stands in for the window.
func TestPresentOpaqueAlphaOnWayland(t *testing.T) {
	for _, tc := range []struct {
		mode gputypes.CompositeAlphaMode
		want byte
	}{
		{hal.CompositeAlphaModeOpaque, 255},
		{hal.CompositeAlphaModeAuto, 255},
		{hal.CompositeAlphaModePremultiplied, 0},
	} {
		t.Run(tc.mode.String(), func(t *testing.T) {
			ctx := newTestAdapterContext(t)
			t.Cleanup(ctx.Destroy)
			display, config := ctx.EGL().Display(), ctx.EGL().Config()
			attribs := []egl.EGLInt{egl.Width, 4, egl.Height, 4, egl.None}
			pbuffer := egl.CreatePbufferSurface(display, config, &attribs[0])
			if pbuffer == egl.NoSurface {
				t.Skipf("no pbuffer: error 0x%x", egl.GetError())
			}
			var alpha egl.EGLInt
			egl.GetConfigAttrib(display, config, egl.AlphaSize, &alpha)
			if alpha == 0 {
				egl.DestroySurface(display, pbuffer)
				t.Skip("the config has no alpha")
			}
			surf := &Surface{ctx: ctx, eglDisplay: display, eglSurface: pbuffer, isWayland: true,
				config: &hal.SurfaceConfiguration{Width: 4, Height: 4, Format: gputypes.TextureFormatRGBA8Unorm, AlphaMode: tc.mode}}
			t.Cleanup(surf.Destroy)

			glCtx := ctx.Lock()
			err := surf.reconfigureSwapchainFBOWith(glCtx, gputypes.TextureFormatRGBA8Unorm, 4, 4)
			if err == nil {
				glCtx.BindFramebuffer(gl.FRAMEBUFFER, surf.swapchainFBO)
				glCtx.ClearBufferfv(gl.COLOR, 0, &[4]float32{0.2, 0.4, 0.6, 0})
				glCtx.BindFramebuffer(gl.FRAMEBUFFER, 0)
			}
			ctx.Unlock()
			if err != nil {
				t.Fatalf("swapchain: %v", err)
			}

			if err := (&Queue{ctx: ctx}).Present(surf, nil, nil); err != nil {
				t.Fatalf("Present: %v", err)
			}

			var px [4]byte
			glCtx = ctx.LockForSurface(pbuffer)
			glCtx.BindFramebuffer(gl.READ_FRAMEBUFFER, 0)
			glCtx.ReadPixels(1, 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, unsafe.Pointer(&px[0]))
			glErr := glCtx.GetError()
			ctx.Unlock()
			if glErr != 0 {
				t.Fatalf("GL error 0x%x", glErr)
			}
			if px[0] != 51 || px[1] != 102 || px[2] != 153 || px[3] != tc.want {
				t.Fatalf("presented %v, want [51 102 153 %d]", px, tc.want)
			}
		})
	}
}

// Premultiplied alpha is offered only on Wayland, whose compositor blends a
// window by the alpha of its config.
func TestSurfaceAlphaModesOffWayland(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	if ctx.EGL().WindowKind() == egl.WindowKindWayland {
		t.Skip("the context is a Wayland one")
	}
	got := (&Adapter{ctx: ctx}).SurfaceCapabilities(nil).AlphaModes
	if want := []gputypes.CompositeAlphaMode{hal.CompositeAlphaModeOpaque}; !slices.Equal(got, want) {
		t.Fatalf("alpha modes on %s: %v, want %v", ctx.EGL().WindowKind(), got, want)
	}
}
