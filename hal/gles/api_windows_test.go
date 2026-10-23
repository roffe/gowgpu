//go:build windows && !(js && wasm)

package gles

import (
	"runtime"
	"strings"
	"syscall"
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/wgl"
)

var (
	user32           = syscall.NewLazyDLL("user32.dll")
	procCreateWindow = user32.NewProc("CreateWindowExW")
	procDestroyWin   = user32.NewProc("DestroyWindow")
)

// newTestWindow makes a hidden window without a pixel format, destroyed
// when the test ends. The caller's goroutine must stay on its thread.
func newTestWindow(t *testing.T) wgl.HWND {
	t.Helper()
	class, _ := syscall.UTF16PtrFromString("STATIC")
	hwnd, _, err := procCreateWindow.Call(0, uintptr(unsafe.Pointer(class)), 0, 0, 0, 0, 16, 16, 0, 0, 0, 0)
	if hwnd == 0 {
		t.Skipf("CreateWindowExW: %v", err)
	}
	t.Cleanup(func() { procDestroyWin.Call(hwnd) })
	return wgl.HWND(hwnd)
}

// A window gets the GL context's pixel format the first time, keeps it
// for a second surface, and a window with another pixel format, which
// can't be changed, gets a surface error that says so.
func TestCreateSurfaceSetsPixelFormatOnce(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	raw, err := Backend{}.CreateInstance(nil)
	if err != nil {
		t.Skipf("GL instance unavailable: %v", err)
	}
	inst := raw.(*Instance)
	defer inst.Destroy()
	want := inst.hiddenWindow.PixelFormat()

	hwnd := newTestWindow(t)
	for range 2 {
		surf, err := inst.CreateSurface(hal.SurfaceTarget{Kind: hal.SurfaceTargetWindowsHWND, WindowHandle: uintptr(hwnd)})
		if err != nil {
			t.Fatalf("CreateSurface: %v", err)
		}
		surf.Destroy()
	}
	hdc := wgl.GetDC(hwnd)
	got := wgl.GetPixelFormat(hdc)
	wgl.ReleaseDC(hwnd, hdc)
	if got != want {
		t.Fatalf("the window has pixel format %d, want the context's %d", got, want)
	}

	other := newTestWindow(t)
	hdc = wgl.GetDC(other)
	pfd := wgl.DefaultPixelFormat()
	for format := 1; format < want+16 && wgl.GetPixelFormat(hdc) == 0; format++ {
		if format != want {
			_ = wgl.SetPixelFormat(hdc, format, &pfd)
		}
	}
	got = wgl.GetPixelFormat(hdc)
	wgl.ReleaseDC(other, hdc)
	if got == 0 {
		t.Skip("no other pixel format to give a window")
	}
	_, err = inst.CreateSurface(hal.SurfaceTarget{Kind: hal.SurfaceTargetWindowsHWND, WindowHandle: uintptr(other)})
	if err == nil || !strings.Contains(err.Error(), "pixel format") {
		t.Fatalf("CreateSurface on a window of another pixel format: %v, want a pixel format error", err)
	}
}
