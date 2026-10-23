// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

package gles

import (
	"errors"
	"fmt"
	"log/slog"
	"runtime"
	"sync"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/egl"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// AdapterContext wraps an EGL/GL context with mutex-protected MakeCurrent switching.
// Shared by Instance → Adapter → Device → Queue (X11/headless), or owned by Surface
// on Wayland (intentional context-per-surface divergence — no wl_display* at Instance init).
//
// Follows Windows AdapterContext (adapter_context.go) and Rust wgpu-hal GLES lock pattern:
//   - Lock()             → MakeCurrent to pbuffer/surfaceless (resource + command work)
//   - LockForSurface()   → MakeCurrent to window EGLSurface (present / swap)
//   - Unlock()           → UnmakeCurrent
type AdapterContext struct {
	mu     sync.Mutex
	eglCtx *egl.Context
	gl     *gl.Context
	owns   bool // true → Destroy() destroys eglCtx

	// keepCurrent (InstanceFlagsGLPinnedThread) leaves the context current
	// on the thread that bound it: the eglMakeCurrent pair around every
	// operation came to 10-19 pairs a frame under fyne (2026-10-20), and the
	// unbind's implicit flush was where frame times spiked. It needs the
	// caller's promise that one goroutine locked to its thread drives the
	// backend: an EGL context is bound to a thread, Go moves goroutines
	// between threads, and a goroutine exiting with the binding strands the
	// context, which is what a device shared across test goroutines does
	// (fyne's test driver, 2026-10-20). A lock from another thread runs on
	// helper, a context sharing the objects, which is enough for the fence
	// waits of gowgpu's map goroutine; it has no VAO or framebuffer, so
	// Submit and Present refuse to run on it. What a lock on helper changed
	// is finished before it unlocks, as the context sees another context's
	// changes only once they are complete.
	keepCurrent bool
	helper      *egl.Context
	onHelper    bool // the lock held now runs on helper
	// onMain is what the next lock on the context runs first: the deletes
	// of its VAOs and framebuffers, which helper doesn't share, that a lock
	// on helper asked for.
	onMain []func(*gl.Context)

	err error // why the lock held has no current context
}

// errNoContext is the error of a lock on a destroyed AdapterContext.
var errNoContext = errors.New("gles: the GL context is destroyed")

// lostError is the error a call reports for err, the error of a failed EGL
// call: hal.ErrDeviceLost when the context is gone, as after a power event
// (EGL_CONTEXT_LOST) or with its display terminated, and hal.ErrSurfaceLost
// when the window surface or its native window is.
func lostError(err error) error {
	var e *egl.Error
	if errors.As(err, &e) {
		switch e.Code {
		case egl.ContextLost, egl.BadContext, egl.NotInitialized, egl.BadDisplay:
			return fmt.Errorf("gles: %w: %w", hal.ErrDeviceLost, err)
		case egl.BadSurface, egl.BadNativeWindow:
			return fmt.Errorf("gles: %w: %w", hal.ErrSurfaceLost, err)
		}
	}
	return fmt.Errorf("gles: %w", err)
}

// NewAdapterContext wraps an already-created EGL context and GL function table.
// owns=true means Destroy() will call eglCtx.Destroy(); owns=false for borrowed refs.
func NewAdapterContext(eglCtx *egl.Context, glCtx *gl.Context, owns bool) *AdapterContext {
	return &AdapterContext{
		eglCtx: eglCtx,
		gl:     glCtx,
		owns:   owns,
	}
}

// Lock acquires the mutex, pins the goroutine to the current OS thread, and
// makes the GL context current on the pbuffer / surfaceless draw surface.
//
// Mirrors Windows AdapterContext.Lock() (hidden DC) and Rust AdapterContext::lock().
func (c *AdapterContext) Lock() *gl.Context {
	c.mu.Lock()
	runtime.LockOSThread()

	if c.eglCtx == nil {
		c.err = errNoContext
		slog.Error("gles: AdapterContext.Lock: nil egl context")
		return c.gl
	}
	if c.keepCurrent {
		c.bindKept(egl.NoSurface)
		return c.gl
	}
	if err := c.eglCtx.MakeCurrent(); err != nil {
		c.err = lostError(err)
		slog.Error("gles: AdapterContext.Lock MakeCurrent failed", "err", err)
	}
	return c.gl
}

// LockForSurface acquires the mutex, pins the goroutine to the current OS thread,
// and makes the GL context current on the given window EGLSurface.
//
// Mirrors Windows AdapterContext.LockForDC() and Rust lock_with_dc / egl present path.
func (c *AdapterContext) LockForSurface(surf egl.EGLSurface) *gl.Context {
	c.mu.Lock()
	runtime.LockOSThread()

	if c.eglCtx == nil {
		c.err = errNoContext
		slog.Error("gles: AdapterContext.LockForSurface: nil egl context")
		return c.gl
	}
	if c.keepCurrent {
		c.bindKept(surf)
		return c.gl
	}
	if err := c.eglCtx.MakeCurrentSurface(surf); err != nil {
		c.err = lostError(err)
		slog.Error("gles: AdapterContext.LockForSurface MakeCurrent failed",
			"err", err,
			"surface", fmt.Sprintf("0x%x", surf))
	}
	return c.gl
}

// bindKept makes the kept context current for the lock held: nothing when
// it is current on this thread already (on surf, when one is asked for;
// Lock takes whichever surface it has), a bind when no thread has it, and
// helper when another thread keeps it.
func (c *AdapterContext) bindKept(surf egl.EGLSurface) {
	c.onHelper = false
	ctx := c.eglCtx.EGLContext()
	held := egl.GetCurrentContext() == ctx
	if held && (surf == egl.NoSurface || egl.GetCurrentSurface(egl.Draw) == surf) {
		c.runOnMain()
		return
	}
	if surf == egl.NoSurface {
		surf = c.eglCtx.Pbuffer()
	}
	if egl.MakeCurrent(c.eglCtx.Display(), surf, surf, ctx) != egl.False {
		if !held {
			// This goroutine holds the context from now on, so it stays on
			// this thread for good, as the flag promised: between two calls
			// Go may otherwise move it, and the context cannot follow.
			runtime.LockOSThread()
		}
		c.runOnMain()
		return
	}
	if err := egl.LastError("eglMakeCurrent"); err.Code != egl.BadAccess {
		c.err = lostError(err)
		slog.Error("gles: AdapterContext MakeCurrent failed", "err", err, "surface", fmt.Sprintf("0x%x", surf))
		return
	}
	if c.helper == nil {
		h, err := c.eglCtx.NewShared()
		if err != nil {
			c.err = fmt.Errorf("gles: the context is current on another thread and no helper context could be made: %w", err)
			slog.Error("gles: AdapterContext: the context is current on another thread and no helper context could be made", "err", err)
			return
		}
		c.helper = h
	}
	if err := c.helper.MakeCurrent(); err != nil {
		c.err = lostError(err)
		slog.Error("gles: AdapterContext helper MakeCurrent failed", "err", err)
		return
	}
	c.onHelper = true
}

// ReleaseSurface makes the context current on its pbuffer instead of surf
// when surf is what it draws to, so that surf can be destroyed: a kept
// context (keepCurrent) stays on the window surface the last Present used,
// and destroying a current surface only marks it while the native window
// under it goes away. The lock must be held.
func (c *AdapterContext) ReleaseSurface(surf egl.EGLSurface) {
	if c.eglCtx == nil || surf == egl.NoSurface || egl.GetCurrentSurface(egl.Draw) != surf {
		return
	}
	pb := c.eglCtx.Pbuffer()
	if egl.MakeCurrent(c.eglCtx.Display(), pb, pb, c.eglCtx.EGLContext()) == egl.False {
		slog.Error("gles: AdapterContext.ReleaseSurface MakeCurrent failed", "err", fmt.Sprintf("0x%x", egl.GetError()))
	}
}

// OnHelper reports whether the lock held runs on the helper context, where
// only shared objects exist and glFinish finishes nothing of the queue's.
func (c *AdapterContext) OnHelper() bool { return c.onHelper }

// OnMain runs fn with the context current, for its VAOs and framebuffers,
// which the helper context doesn't share: now, or, when the lock held runs
// on the helper, first thing in the next lock on the context. The lock must
// be held.
func (c *AdapterContext) OnMain(fn func(*gl.Context)) {
	if c.onHelper {
		c.onMain = append(c.onMain, fn)
		return
	}
	fn(c.gl)
}

func (c *AdapterContext) runOnMain() {
	for _, fn := range c.onMain {
		fn(c.gl)
	}
	c.onMain = nil
}

// Err reports why the lock held has no current context, wrapping
// hal.ErrDeviceLost or hal.ErrSurfaceLost when that is what happened, or
// nil when the context is current: GL calls without one do nothing.
func (c *AdapterContext) Err() error { return c.err }

// Unlock unmakes the GL context current, unpins the goroutine from the OS
// thread, and releases the mutex.
//
// Guards against double-unmake: only calls eglMakeCurrent(NO_SURFACE, NO_CONTEXT)
// if a context is actually current on this thread. Matches Windows Unlock() and
// Rust WglContext::unmake_current().
func (c *AdapterContext) Unlock() {
	if c.onHelper && c.err == nil {
		c.gl.Finish()
	}
	// Check eglCtx first so Unlock is safe before egl.Init (unit tests, teardown).
	if c.eglCtx != nil && (!c.keepCurrent || c.onHelper) && egl.GetCurrentContext() != egl.NoContext {
		if egl.MakeCurrent(c.eglCtx.Display(), egl.NoSurface, egl.NoSurface, egl.NoContext) == egl.False {
			slog.Error("gles: AdapterContext.Unlock UnmakeCurrent failed",
				"err", fmt.Sprintf("0x%x", egl.GetError()))
		}
	}
	c.onHelper, c.err = false, nil
	runtime.UnlockOSThread()
	c.mu.Unlock()
}

// GL returns the GL function table.
// Safe to read without Lock; GL calls on the returned context require Lock.
func (c *AdapterContext) GL() *gl.Context {
	return c.gl
}

// EGL returns the underlying EGL context wrapper.
func (c *AdapterContext) EGL() *egl.Context {
	return c.eglCtx
}

// Destroy deletes the EGL context when this AdapterContext owns it.
// Takes the mutex so Destroy cannot race with Lock/Unlock on another goroutine.
func (c *AdapterContext) Destroy() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.helper != nil {
		c.helper.Destroy()
		c.helper = nil
	}
	if c.owns && c.eglCtx != nil {
		c.eglCtx.Destroy()
		c.eglCtx = nil
		c.gl = nil
	}
}
