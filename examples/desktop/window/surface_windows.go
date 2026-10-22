package window

import (
	"unsafe"

	"github.com/go-gl/glfw/v3.4/glfw"
	wgpu "github.com/roffe/gowgpu"
)

func surfaceTarget(win *glfw.Window) wgpu.SurfaceTargetUnsafe {
	return wgpu.SurfaceTargetFromWindowsHWND(0, uintptr(unsafe.Pointer(win.GetWin32Window()))) //nolint:gosec // G103: native handle for the surface
}
