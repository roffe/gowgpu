// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package wgl

import (
	"runtime"
	"testing"
)

// The hidden window lives on a thread of its own: made from a goroutine
// whose locked thread then exits, it survives, and Destroy from another
// thread destroys it. On the caller's thread, the window died with that
// thread, and DestroyWindow from any other one failed and leaked it.
func TestHiddenWindowOutlivesItsCaller(t *testing.T) {
	if err := Init(); err != nil {
		t.Skipf("WGL: %v", err)
	}
	type made struct {
		hw  *HiddenWindow
		err error
	}
	got := make(chan made)
	go func() {
		runtime.LockOSThread() // exits with the goroutine
		hw, err := NewHiddenWindow()
		got <- made{hw, err}
	}()
	m := <-got
	if m.err != nil {
		t.Skipf("NewHiddenWindow: %v", m.err)
	}
	isWindow, err := user32.FindProc("IsWindow")
	if err != nil {
		t.Fatal(err)
	}
	hwnd := m.hw.HWND()
	runtime.GC() // lets the runtime end the exited goroutine's thread
	if r, _, _ := isWindow.Call(uintptr(hwnd)); r == 0 {
		t.Fatal("the window died with the thread that asked for it")
	}
	m.hw.Destroy()
	if r, _, _ := isWindow.Call(uintptr(hwnd)); r != 0 {
		t.Error("the window outlived Destroy")
	}
}
