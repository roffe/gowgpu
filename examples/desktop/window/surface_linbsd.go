//go:build linux || freebsd || openbsd || netbsd

package window

import (
	"unsafe"

	"github.com/go-gl/glfw/v3.4/glfw"
	wgpu "github.com/roffe/gowgpu"
)

// surfaceTarget hands gowgpu the window GLFW made, on whichever window
// system it connected to.
func surfaceTarget(win *glfw.Window) wgpu.SurfaceTargetUnsafe {
	if glfw.GetPlatform() == glfw.PlatformWayland {
		return wgpu.SurfaceTargetFromWaylandSurface(uintptr(unsafe.Pointer(glfw.GetWaylandDisplay())), uintptr(unsafe.Pointer(win.GetWaylandWindow()))) //nolint:gosec // G103: native handles for the surface
	}
	return wgpu.SurfaceTargetFromXlibWindow(uintptr(unsafe.Pointer(glfw.GetX11Display())), uintptr(win.GetX11Window())) //nolint:gosec // G103: native handles for the surface
}
