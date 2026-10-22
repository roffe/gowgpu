// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build integration && linux && !(js && wasm)

package gles

import (
	"errors"
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/roffe/gowgpu/hal/gles/egl"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// With InstanceFlagsGLPinnedThread the context stays current on the thread
// that bound it, a lock there costs no eglMakeCurrent, and a lock from
// another thread runs on the helper context, which sees the shared objects
// and sync objects but takes no Submit.
func TestKeepCurrentStaysOnItsThread(t *testing.T) {
	adapterCtx := newTestAdapterContext(t)
	t.Cleanup(adapterCtx.Destroy)
	adapterCtx.keepCurrent = true

	glCtx := adapterCtx.Lock()
	buf := glCtx.GenBuffers(1)
	fence := NewFence(glCtx)
	if err := fence.Signal(1); err != nil {
		t.Fatal(err)
	}
	glCtx.Flush()
	adapterCtx.Unlock()
	if egl.GetCurrentContext() == egl.NoContext {
		t.Fatal("Unlock unbound the context")
	}
	if n := glCalls(func() { adapterCtx.Lock(); adapterCtx.Unlock() }); n > 2 {
		t.Errorf("a lock on the thread that keeps the context made %d EGL calls, want the two current-context queries at most", n)
	}

	done := make(chan string, 1)
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		g := adapterCtx.Lock()
		defer adapterCtx.Unlock()
		switch {
		case !adapterCtx.OnHelper():
			done <- "the lock from another thread did not run on the helper context"
		case egl.GetCurrentContext() == egl.NoContext:
			done <- "nothing is current on the other thread"
		case !fence.Wait(1, time.Second):
			done <- "the fence signaled on the kept context did not pass on the helper context"
		default:
			g.BindBuffer(gl.COPY_READ_BUFFER, buf)
			if err := g.GetError(); err != 0 {
				done <- "binding a buffer of the kept context on the helper context failed"
				return
			}
			g.BindBuffer(gl.COPY_READ_BUFFER, 0)
			adapterCtx.Unlock()
			_, err := (&Queue{ctx: adapterCtx}).Submit(nil)
			_, encErr := (&Device{ctx: adapterCtx}).CreateCommandEncoder(nil)
			adapterCtx.Lock() // for the deferred Unlock
			if !errors.Is(err, errOtherThread) {
				done <- fmt.Sprintf("Submit from another goroutine returned %v, want errOtherThread", err)
				return
			}
			if !errors.Is(encErr, errOtherThread) {
				done <- fmt.Sprintf("the first CreateCommandEncoder from another goroutine returned %v, want errOtherThread", encErr)
				return
			}
			done <- ""
		}
	}()
	if msg := <-done; msg != "" {
		t.Error(msg)
	}
	if egl.GetCurrentContext() != adapterCtx.eglCtx.EGLContext() {
		t.Error("the other thread's lock took the context away from this thread")
	}
	glCtx = adapterCtx.Lock()
	glCtx.DeleteBuffers(buf)
	fence.Destroy()
	adapterCtx.Unlock()
}

// Without the flag the context is bound around every lock, so any goroutine
// may lock next.
func TestBindsPerCallByDefault(t *testing.T) {
	adapterCtx := newTestAdapterContext(t)
	t.Cleanup(adapterCtx.Destroy)

	adapterCtx.Lock()
	if egl.GetCurrentContext() == egl.NoContext {
		t.Fatal("nothing is current inside the lock")
	}
	adapterCtx.Unlock()
	if egl.GetCurrentContext() != egl.NoContext {
		t.Error("the context is still current after Unlock")
	}
}
