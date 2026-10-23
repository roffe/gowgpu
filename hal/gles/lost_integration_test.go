//go:build integration && linux && !(js && wasm)

package gles

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/egl"
)

// A context whose display is gone loses the device: Submit and the writes
// say so instead of running GL calls on no context.
func TestSubmitOnLostContextLosesDevice(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	q := &Queue{ctx: ctx}
	if _, err := q.Submit(nil); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	egl.Terminate(ctx.EGL().Display())
	if _, err := q.Submit(nil); !errors.Is(err, hal.ErrDeviceLost) {
		t.Fatalf("Submit on a terminated display: %v, want %v", err, hal.ErrDeviceLost)
	}
	if err := q.WriteBuffer(&Buffer{id: 1, target: 0x8892}, 0, []byte{1}); !errors.Is(err, hal.ErrDeviceLost) {
		t.Fatalf("WriteBuffer on a terminated display: %v, want %v", err, hal.ErrDeviceLost)
	}
}

// A window surface EGL has no more loses the surface, and Present drops it
// for the next Configure to make a new one. A pbuffer stands in for it.
func TestPresentOnDeadSurfaceLosesSurface(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	display, config := ctx.EGL().Display(), ctx.EGL().Config()
	attribs := []egl.EGLInt{egl.Width, 4, egl.Height, 4, egl.None}
	pbuffer := egl.CreatePbufferSurface(display, config, &attribs[0])
	if pbuffer == egl.NoSurface {
		t.Skipf("no pbuffer: error 0x%x", egl.GetError())
	}
	surf := &Surface{ctx: ctx, eglDisplay: display, eglSurface: pbuffer,
		config: &hal.SurfaceConfiguration{Width: 4, Height: 4, Format: gputypes.TextureFormatRGBA8Unorm}}
	t.Cleanup(surf.Destroy)
	q := &Queue{ctx: ctx}
	if err := q.Present(surf, nil, nil); err != nil {
		t.Fatalf("Present: %v", err)
	}

	egl.DestroySurface(display, pbuffer)
	if err := q.Present(surf, nil, nil); !errors.Is(err, hal.ErrSurfaceLost) {
		t.Fatalf("Present on a destroyed surface: %v, want %v", err, hal.ErrSurfaceLost)
	}
	if surf.eglSurface != egl.NoSurface {
		t.Fatal("Present kept the dead EGL surface")
	}
	if _, err := q.Submit(nil); err != nil {
		t.Fatalf("Submit after the surface was lost: %v", err)
	}
}
