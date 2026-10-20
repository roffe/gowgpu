// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

package gles

import (
	"fmt"
	"log/slog"
	"runtime"
	"sync"

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
	// Submit and Present refuse to run on it.
	keepCurrent bool
	helper      *egl.Context
	onHelper    bool // the lock held now runs on helper
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
		slog.Error("gles: AdapterContext.Lock: nil egl context")
		return c.gl
	}
	if c.keepCurrent {
		c.bindKept(egl.NoSurface)
		return c.gl
	}
	if err := c.eglCtx.MakeCurrent(); err != nil {
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
		slog.Error("gles: AdapterContext.LockForSurface: nil egl context")
		return c.gl
	}
	if c.keepCurrent {
		c.bindKept(surf)
		return c.gl
	}
	if err := c.eglCtx.MakeCurrentSurface(surf); err != nil {
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
		return
	}
	if err := egl.GetError(); err != egl.BadAccess {
		slog.Error("gles: AdapterContext MakeCurrent failed", "err", fmt.Sprintf("0x%x", err), "surface", fmt.Sprintf("0x%x", surf))
		return
	}
	if c.helper == nil {
		h, err := c.eglCtx.NewShared()
		if err != nil {
			slog.Error("gles: AdapterContext: the context is current on another thread and no helper context could be made", "err", err)
			return
		}
		c.helper = h
	}
	if err := c.helper.MakeCurrent(); err != nil {
		slog.Error("gles: AdapterContext helper MakeCurrent failed", "err", err)
		return
	}
	c.onHelper = true
}

// OnHelper reports whether the lock held runs on the helper context, where
// only shared objects exist and glFinish finishes nothing of the queue's.
func (c *AdapterContext) OnHelper() bool { return c.onHelper }

// Unlock unmakes the GL context current, unpins the goroutine from the OS
// thread, and releases the mutex.
//
// Guards against double-unmake: only calls eglMakeCurrent(NO_SURFACE, NO_CONTEXT)
// if a context is actually current on this thread. Matches Windows Unlock() and
// Rust WglContext::unmake_current().
func (c *AdapterContext) Unlock() {
	// Check eglCtx first so Unlock is safe before egl.Init (unit tests, teardown).
	if c.eglCtx != nil && (!c.keepCurrent || c.onHelper) && egl.GetCurrentContext() != egl.NoContext {
		if egl.MakeCurrent(c.eglCtx.Display(), egl.NoSurface, egl.NoSurface, egl.NoContext) == egl.False {
			slog.Error("gles: AdapterContext.Unlock UnmakeCurrent failed",
				"err", fmt.Sprintf("0x%x", egl.GetError()))
		}
	}
	c.onHelper = false
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
