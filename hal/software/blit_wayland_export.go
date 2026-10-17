//go:build linux && !(js && wasm)

package software

// #include <stdint.h>
import "C"

// The Wayland listeners in blit_wayland_linux.c call these. A file with
// //export may only declare C in its preamble, so they live apart from the
// calls in blit_wayland.go.

// goSwRegistryGlobal is wl_registry.global: it notes wl_shm's name for the
// bind that follows the roundtrip.
//
//export goSwRegistryGlobal
func goSwRegistryGlobal(name C.uint32_t, iface *C.char, _ C.uint32_t) {
	if iface == nil || C.GoString(iface) != "wl_shm" {
		return
	}
	pendingShmBindMu.Lock()
	pendingShmBindName = uint32(name)
	pendingShmBindMu.Unlock()
}

// goSwBufferRelease is wl_buffer.release: the compositor no longer reads the
// buffer, so it may be drawn into again (Qt6 qwaylandbuffer.cpp:30-37).
//
//export goSwBufferRelease
func goSwBufferRelease(wlBuffer C.uintptr_t) {
	bufferBusyMu.Lock()
	if buf, ok := bufferBusyMap[uintptr(wlBuffer)]; ok {
		buf.busy = false
	}
	bufferBusyMu.Unlock()
}
