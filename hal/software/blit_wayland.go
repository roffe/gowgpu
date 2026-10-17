//go:build linux && !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

// Error handling follows ADR-049 three-tier strategy:
//
//	Tier 1 (creation/submit): check the API result code
//	Tier 2 (void/hot path): infallible by GPU API contract — Vulkan §6.6, WebGPU §21.2
//	Tier 3 (platform syscalls): use errno for diagnostics (Wayland, X11, Win32)
//
// Enterprise reference: Rust wgpu-hal returns () for all draw/destroy/barrier commands.
//
// Wayland blit calls are Tier 2: wl_proxy_marshal, wl_proxy_destroy, and
// wl_display_flush are void or return status already checked inline.
// wl_display_roundtrip_queue returns -1 on error (checked via result variable).
package software

/*
#include <stdint.h>

uintptr_t swl_call_u_u(void *fn, uintptr_t a);
void swl_call_v_u(void *fn, uintptr_t a);
void swl_call_v_uu(void *fn, uintptr_t a, uintptr_t b);
int swl_call_i_u(void *fn, uintptr_t a);
int swl_call_i_uu(void *fn, uintptr_t a, uintptr_t b);
uintptr_t swl_get_registry(void *fn, uintptr_t display, const void *registry_iface);
uintptr_t swl_registry_bind(void *fn, uintptr_t registry, uint32_t name, const void *iface, const char *iface_name, uint32_t version);
uintptr_t swl_create_pool(void *fn, uintptr_t shm, const void *pool_iface, int32_t fd, int32_t size);
uintptr_t swl_create_buffer(void *fn, uintptr_t pool, const void *buffer_iface, int32_t offset, int32_t width, int32_t height, int32_t stride, uint32_t format);
void swl_marshal0(void *fn, uintptr_t proxy, uint32_t opcode);
void swl_surface_attach(void *fn, uintptr_t surface, uintptr_t buffer, int32_t x, int32_t y);
void swl_surface_damage_buffer(void *fn, uintptr_t surface, int32_t x, int32_t y, int32_t w, int32_t h);
int swl_add_registry_listener(void *fn, uintptr_t registry);
int swl_add_buffer_listener(void *fn, uintptr_t buffer);

// Everything but swl_call_i_uu (wl_display_roundtrip_queue, which runs the
// listeners and through them Go) neither calls back nor keeps a pointer.
#cgo noescape swl_call_u_u
#cgo nocallback swl_call_u_u
#cgo noescape swl_call_v_u
#cgo nocallback swl_call_v_u
#cgo noescape swl_call_v_uu
#cgo nocallback swl_call_v_uu
#cgo noescape swl_call_i_u
#cgo nocallback swl_call_i_u
#cgo noescape swl_get_registry
#cgo nocallback swl_get_registry
#cgo noescape swl_registry_bind
#cgo nocallback swl_registry_bind
#cgo noescape swl_create_pool
#cgo nocallback swl_create_pool
#cgo noescape swl_create_buffer
#cgo nocallback swl_create_buffer
#cgo noescape swl_marshal0
#cgo nocallback swl_marshal0
#cgo noescape swl_surface_attach
#cgo nocallback swl_surface_attach
#cgo noescape swl_surface_damage_buffer
#cgo nocallback swl_surface_damage_buffer
#cgo noescape swl_add_registry_listener
#cgo nocallback swl_add_registry_listener
#cgo noescape swl_add_buffer_listener
#cgo nocallback swl_add_buffer_listener
*/
import "C"

import (
	"image"
	"log/slog"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"

	"github.com/roffe/gowgpu/internal/dl"
)

// Wayland SHM presentation for the software backend.
//
// On Wayland, displayHandle is wl_display* and hwnd is wl_surface*.
// The X11 blit path (XPutImage) would crash because those are not X11 handles.
//
// This file implements the Wayland present path using wl_shm shared memory
// buffers. The pattern follows our own CSD SHM implementation in
// gogpu/internal/platform/wayland/libwayland_csd.go:
//
//  1. memfd_create + mmap → shared memory pool
//  2. wl_shm_create_pool → wl_shm_pool
//  3. wl_shm_pool::create_buffer → wl_buffer
//  4. Register wl_buffer.release listener (Qt6 pattern: qwaylandbuffer.cpp)
//  5. Copy pixels, wl_surface_attach + damage_buffer + commit + flush
//
// Triple-buffering (3 SHM buffers) with wl_buffer.release avoids writing
// into a buffer the compositor is still reading. If all buffers are busy,
// the frame is skipped (no corruption, no tearing). Qt6 uses up to 5
// buffers (qwaylandshmbackingstore.cpp); 3 is the minimum for skip-free
// presentation on slow compositors (e.g. pixman renderer).
//
// Pixel format: Software backend stores BGRA byte order. On little-endian
// (all supported Linux), BGRA bytes = uint32 0xAARRGGBB = wl_shm ARGB8888.
// No conversion needed — identical to the CSD path.

// waylandShmState holds lazily-loaded libwayland-client symbols and the
// wl_shm global object required for SHM buffer creation. Initialized once
// on first Wayland blit via waylandOnce.
var (
	waylandOnce  sync.Once
	waylandReady bool // true if Wayland SHM path is available

	wlClientLib dl.Lib

	// wl_display functions
	symWlDisplayFlush       unsafe.Pointer
	symWlDisplayCreateQueue unsafe.Pointer // wl_display_create_queue(display) -> queue*

	// wl_registry / wl_shm / wl_shm_pool / wl_surface / wl_buffer / proxy functions
	symWlProxyMarshalConstructor          unsafe.Pointer
	symWlProxyMarshalConstructorVersioned unsafe.Pointer
	symWlProxyAddListener                 unsafe.Pointer
	symWlProxyMarshal                     unsafe.Pointer
	symWlProxyDestroy                     unsafe.Pointer
	symWlProxySetQueue                    unsafe.Pointer // wl_proxy_set_queue(proxy, queue) -> void
	symWlProxyCreateWrapper               unsafe.Pointer // wl_proxy_create_wrapper(proxy) -> wrapper*
	symWlProxyWrapperDestroy              unsafe.Pointer // wl_proxy_wrapper_destroy(wrapper) -> void
	symWlDisplayRoundtripQueue            unsafe.Pointer // wl_display_roundtrip_queue(display, queue) -> int
	symWlEventQueueDestroy                unsafe.Pointer // wl_event_queue_destroy(queue) -> void

	// wl_interface pointers (loaded from libwayland-client.so data section)
	wlRegistryInterface unsafe.Pointer
	wlShmInterface      unsafe.Pointer
	wlShmPoolInterface  unsafe.Pointer
	wlBufferInterface   unsafe.Pointer
)

// waylandBlitState holds per-Surface Wayland SHM resources for triple-buffered
// presentation. Embedded in platformBlit (blit_linux.go).
//
// Triple-buffering (3 buffers) guarantees a free buffer even when the
// compositor holds two (previous + current). Qt6 uses up to 5
// (qwaylandshmbackingstore.cpp); 3 is the minimum for skip-free
// presentation on slow compositors (e.g. pixman renderer).
type waylandBlitState struct {
	isWayland bool // true when SurfaceTargetKind is Wayland
	detected  bool // true if detection has been performed

	wlShm    uintptr // bound wl_shm global (on shmQueue, 0 if not yet obtained)
	registry uintptr // wl_registry proxy (on shmQueue, for proper cleanup)

	// shmQueue is a dedicated wl_event_queue for ALL software backend Wayland
	// proxies (registry, wl_shm, wl_shm_pool, wl_buffer). Created via
	// wl_display_create_queue. All proxies inherit shmQueue through the
	// wl_proxy_create_wrapper pattern (Qt6 qwaylanddisplay.cpp:335-341):
	// wrapper = wl_proxy_create_wrapper(display)
	// wl_proxy_set_queue(wrapper, shmQueue)
	// registry = wl_display_get_registry(wrapper) → on shmQueue
	// wl_shm = wl_registry_bind(registry) → on shmQueue
	// pool/buffer inherit from wl_shm → on shmQueue
	//
	// This eliminates queue mismatch with gogpu's appQueue (ADR-041):
	// gogpu dispatches ONLY appQueue, so objects on DEFAULT queue would
	// never receive events. See BUG-SW-WAYLAND-002.
	shmQueue uintptr // 0 until first buffer created

	// Triple-buffer state: three SHM buffers, pick first non-busy.
	// This avoids writing to a buffer the compositor is still reading.
	buffers  [3]waylandShmBuffer
	frontIdx int // index of the buffer last submitted to compositor
}

// waylandShmBuffer holds one SHM buffer for Wayland presentation.
type waylandShmBuffer struct {
	fd            int     // memfd file descriptor (-1 if unused)
	data          []byte  // mmap'd pixel data
	pool          uintptr // wl_shm_pool proxy
	buffer        uintptr // wl_buffer proxy
	width         int32
	height        int32
	busy          bool // true while compositor owns this buffer (Qt6 qwaylandbuffer.cpp pattern)
	needsFullCopy bool // true after allocation — first frame must be full copy
}

// Global registry listener callback state.
// Protected by waylandOnce (only one goroutine does init).
var (
	// pendingShmBindName stores the wl_shm global name during registry roundtrip.
	pendingShmBindMu   sync.Mutex
	pendingShmBindName uint32

	// bufferBusyMap maps wl_buffer proxy address to the waylandShmBuffer owning it.
	// Protected by bufferBusyMu. Used by the release callback to clear busy flag.
	bufferBusyMu  sync.Mutex
	bufferBusyMap = map[uintptr]*waylandShmBuffer{}
)

// initWayland loads libwayland-client.so and the symbols SHM presentation calls.
func initWayland() {
	var err error
	wlClientLib, err = dl.Open("libwayland-client.so.0", "libwayland-client.so")
	if err != nil {
		slog.Debug("software: Wayland blit unavailable — could not load libwayland-client", "error", err)
		return
	}

	// Functions, and the wl_interface data symbols the constructors take.
	symbols := []struct {
		name string
		dst  *unsafe.Pointer
	}{
		{"wl_display_flush", &symWlDisplayFlush},
		{"wl_display_create_queue", &symWlDisplayCreateQueue},
		{"wl_proxy_marshal_constructor", &symWlProxyMarshalConstructor},
		{"wl_proxy_marshal_constructor_versioned", &symWlProxyMarshalConstructorVersioned},
		{"wl_proxy_add_listener", &symWlProxyAddListener},
		{"wl_proxy_marshal", &symWlProxyMarshal},
		{"wl_proxy_destroy", &symWlProxyDestroy},
		{"wl_proxy_set_queue", &symWlProxySetQueue},
		{"wl_proxy_create_wrapper", &symWlProxyCreateWrapper},
		{"wl_proxy_wrapper_destroy", &symWlProxyWrapperDestroy},
		{"wl_display_roundtrip_queue", &symWlDisplayRoundtripQueue},
		{"wl_event_queue_destroy", &symWlEventQueueDestroy},
		{"wl_registry_interface", &wlRegistryInterface},
		{"wl_shm_interface", &wlShmInterface},
		{"wl_shm_pool_interface", &wlShmPoolInterface},
		{"wl_buffer_interface", &wlBufferInterface},
	}
	for _, sym := range symbols {
		if *sym.dst, err = wlClientLib.Sym(sym.name); err != nil {
			slog.Debug("software: Wayland blit unavailable — missing symbol", "error", err)
			return
		}
	}

	waylandReady = true
}

// obtainWlShm gets the wl_shm global by creating a wl_registry, listening
// for the wl_shm global, and doing a roundtrip. Called eagerly from
// configurePlatformBlit (once per surface, on the main thread).
//
// All proxies (registry, wl_shm) are created through a display wrapper
// assigned to shmQueue — so they inherit shmQueue, NOT the default queue.
// This matches the Qt6 pattern (qwaylanddisplay.cpp:335-341):
//
//	wrapper = wl_proxy_create_wrapper(display)
//	wl_proxy_set_queue(wrapper, queue)
//	registry = wl_display_get_registry(wrapper)
//
// Without this, proxies end up on the DEFAULT queue which gogpu never
// dispatches (ADR-041), causing events to accumulate and crash.
// See BUG-SW-WAYLAND-002.
//
// Returns (wl_shm, wl_registry) — both on shmQueue. Caller stores registry
// for proper cleanup in destroyWaylandBlitState.
func obtainWlShm(display, shmQueue uintptr) (uintptr, uintptr) {
	waylandOnce.Do(initWayland)
	if !waylandReady {
		return 0, 0
	}

	wrapper := uintptr(C.swl_call_u_u(symWlProxyCreateWrapper, C.uintptr_t(display)))
	if wrapper == 0 {
		slog.Warn("software: wl_proxy_create_wrapper failed")
		return 0, 0
	}
	C.swl_call_v_uu(symWlProxySetQueue, C.uintptr_t(wrapper), C.uintptr_t(shmQueue))

	// wl_display_get_registry via WRAPPER → registry inherits shmQueue.
	registry := uintptr(C.swl_get_registry(symWlProxyMarshalConstructor, C.uintptr_t(wrapper), wlRegistryInterface))

	// Wrapper no longer needed — child proxies already inherited the queue.
	C.swl_call_v_u(symWlProxyWrapperDestroy, C.uintptr_t(wrapper))

	if registry == 0 {
		slog.Warn("software: wl_display_get_registry failed")
		return 0, 0
	}

	// Add registry listener to catch wl_shm global.
	pendingShmBindMu.Lock()
	pendingShmBindName = 0
	pendingShmBindMu.Unlock()
	C.swl_add_registry_listener(symWlProxyAddListener, C.uintptr_t(registry))

	// Roundtrip on shmQueue (not default) to receive registry events.
	// wl_display_roundtrip_queue returns -1 on error and sets errno.
	// EPIPE/ECONNRESET = compositor dead, EPROTO = protocol error.
	// Qt6 qwaylanddisplay.cpp and SDL3 SDL_waylandevents.c check errno here.
	if rt, err := C.swl_call_i_uu(symWlDisplayRoundtripQueue, C.uintptr_t(display), C.uintptr_t(shmQueue)); rt < 0 {
		slog.Warn("software: wl_display_roundtrip_queue failed (init registry)", "error", err)
	}

	pendingShmBindMu.Lock()
	shmName := pendingShmBindName
	pendingShmBindMu.Unlock()

	if shmName == 0 {
		slog.Warn("software: wl_shm not found in registry")
		C.swl_call_v_u(symWlProxyDestroy, C.uintptr_t(registry))
		return 0, 0
	}

	// wl_registry_bind(registry, name, &wl_shm_interface, 1)
	shmNameCStr := append([]byte("wl_shm"), 0)
	shm := uintptr(C.swl_registry_bind(symWlProxyMarshalConstructorVersioned, C.uintptr_t(registry), C.uint32_t(shmName), wlShmInterface, (*C.char)(unsafe.Pointer(&shmNameCStr[0])), 1))

	// Roundtrip on shmQueue to ensure bind completes.
	if rt, err := C.swl_call_i_uu(symWlDisplayRoundtripQueue, C.uintptr_t(display), C.uintptr_t(shmQueue)); rt < 0 {
		slog.Warn("software: wl_display_roundtrip_queue failed (init bind)", "error", err)
	}

	if shm == 0 {
		slog.Warn("software: wl_registry_bind for wl_shm failed")
		C.swl_call_v_u(symWlProxyDestroy, C.uintptr_t(registry))
		return 0, 0
	}
	slog.Debug("software: wl_shm bound successfully", "shm", shm, "queue", "shmQueue")
	return shm, registry
}

// createShmQueue creates a dedicated wl_event_queue for SHM buffer proxies.
// Returns 0 if Wayland is not ready or queue creation fails.
func createShmQueue(display uintptr) uintptr {
	waylandOnce.Do(initWayland)
	if !waylandReady {
		return 0
	}
	queue := uintptr(C.swl_call_u_u(symWlDisplayCreateQueue, C.uintptr_t(display)))
	if queue == 0 {
		slog.Warn("software: wl_display_create_queue failed for SHM buffers")
	}
	return queue
}

// waylandCreateShmBuffer creates a new SHM buffer for the given dimensions.
// The buffer proxy inherits shmQueue from wl_shm via the display wrapper
// pattern (BUG-SW-WAYLAND-002) — no manual wl_proxy_set_queue needed.
func waylandCreateShmBuffer(shm uintptr, width, height int32) *waylandShmBuffer {
	stride := width * 4
	size := int(stride) * int(height)

	// Create memfd.
	fd, err := unix.MemfdCreate("gogpu-sw-blit", unix.MFD_CLOEXEC)
	if err != nil {
		slog.Warn("software: Wayland memfd_create failed", "error", err)
		return nil
	}
	if err := unix.Ftruncate(fd, int64(size)); err != nil {
		_ = unix.Close(fd) // Best-effort cleanup; fd is invalid after ftruncate failure.
		slog.Warn("software: Wayland ftruncate failed", "error", err)
		return nil
	}

	// mmap the shared memory.
	data, err := unix.Mmap(fd, 0, size, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_SHARED)
	if err != nil {
		_ = unix.Close(fd) // Best-effort cleanup; mmap failed.
		slog.Warn("software: Wayland mmap failed", "error", err)
		return nil
	}

	// wl_shm_create_pool(shm, fd, size): the fd goes over the socket, the
	// pool keeps its own reference to the memory.
	pool := uintptr(C.swl_create_pool(symWlProxyMarshalConstructor, C.uintptr_t(shm), wlShmPoolInterface, C.int32_t(fd), C.int32_t(size)))
	if pool == 0 {
		_ = unix.Munmap(data)
		_ = unix.Close(fd)
		slog.Warn("software: wl_shm_create_pool failed")
		return nil
	}

	// wl_shm_pool::create_buffer(offset 0, width, height, stride, ARGB8888)
	buffer := uintptr(C.swl_create_buffer(symWlProxyMarshalConstructor, C.uintptr_t(pool), wlBufferInterface, 0, C.int32_t(width), C.int32_t(height), C.int32_t(stride), 0))
	if buffer == 0 {
		// Destroy pool: opcode 1
		destroyPool(pool)
		_ = unix.Munmap(data)
		_ = unix.Close(fd)
		slog.Warn("software: wl_shm_pool::create_buffer failed")
		return nil
	}

	// Destroy the pool immediately — the buffer keeps the fd reference alive.
	// (Same pattern as CSD code.)
	destroyPool(pool)

	buf := &waylandShmBuffer{
		fd:            fd,
		data:          data,
		pool:          0, // already destroyed
		buffer:        buffer,
		width:         width,
		height:        height,
		needsFullCopy: true, // first frame must be full copy
	}

	// Register wl_buffer.release listener (Qt6 qwaylandbuffer.cpp pattern).
	// NOTE: do NOT register in bufferBusyMap here — the caller does
	// *buf = *newBuf (value copy into wl.buffers[idx]). The map must point
	// to &wl.buffers[idx], not to this local buf. Caller registers after copy.
	C.swl_add_buffer_listener(symWlProxyAddListener, C.uintptr_t(buffer))

	// Buffer proxy inherits shmQueue from wl_shm (via pool) through the
	// display wrapper pattern (BUG-SW-WAYLAND-002). No wl_proxy_set_queue
	// needed — queue assignment is automatic via libwayland proxy_create
	// (wayland-client.c:492: proxy->queue = factory->queue).

	return buf
}

// destroyPool calls wl_shm_pool::destroy (opcode 1) then wl_proxy_destroy.
func destroyPool(pool uintptr) {
	C.swl_marshal0(symWlProxyMarshal, C.uintptr_t(pool), 1)
	C.swl_call_v_u(symWlProxyDestroy, C.uintptr_t(pool))
}

// waylandDestroyShmBuffer releases all resources associated with a SHM buffer.
func waylandDestroyShmBuffer(buf *waylandShmBuffer) {
	if buf == nil {
		return
	}
	if buf.buffer != 0 {
		// Remove from release callback map before destroying.
		bufferBusyMu.Lock()
		delete(bufferBusyMap, buf.buffer)
		bufferBusyMu.Unlock()

		C.swl_marshal0(symWlProxyMarshal, C.uintptr_t(buf.buffer), 0) // wl_buffer::destroy
		C.swl_call_v_u(symWlProxyDestroy, C.uintptr_t(buf.buffer))
		buf.buffer = 0
	}
	if buf.data != nil {
		_ = unix.Munmap(buf.data) // Best-effort cleanup on destroy.
		buf.data = nil
	}
	if buf.fd >= 0 {
		_ = unix.Close(buf.fd) // Best-effort cleanup on destroy.
		buf.fd = -1
	}
}

// waylandPresent copies pixel data into the SHM buffer and commits the surface.
// wl_shm must be obtained eagerly (in blit_linux.go detection block) before
// this method is called. If wlShm is 0, init failed and we bail out.
func (s *Surface) waylandPresent(data []byte, width, height int32) {
	wl := &s.wlState
	if wl.wlShm == 0 {
		return
	}

	// Pick a non-busy buffer (Qt6 qwaylandshmbackingstore.cpp pattern).
	backIdx := -1
	for i := 1; i <= len(wl.buffers); i++ {
		idx := (wl.frontIdx + i) % len(wl.buffers)
		if !wl.buffers[idx].busy {
			backIdx = idx
			break
		}
	}
	if backIdx < 0 {
		return
	}
	buf := &wl.buffers[backIdx]

	// Reallocate if dimensions changed.
	if buf.buffer == 0 || buf.width != width || buf.height != height {
		if buf.buffer != 0 {
			waylandDestroyShmBuffer(buf)
			*buf = waylandShmBuffer{fd: -1}
		}
		newBuf := waylandCreateShmBuffer(wl.wlShm, width, height)
		if newBuf == nil {
			return
		}
		*buf = *newBuf
		// Register AFTER value copy — buf points to wl.buffers[backIdx],
		// so release callback sets busy=false on the actual array element.
		bufferBusyMu.Lock()
		bufferBusyMap[buf.buffer] = buf
		bufferBusyMu.Unlock()
	}

	copy(buf.data[:min(len(data), len(buf.data))], data[:min(len(data), len(buf.data))])

	surface := s.hwnd
	waylandSurfaceAttach(surface, buf.buffer, 0, 0)
	waylandSurfaceDamageBuffer(surface, 0, 0, width, height)
	waylandSurfaceCommit(surface)

	buf.busy = true

	// wl_display_flush sends the commit to the compositor.
	// Returns bytes sent or -1 with errno. EAGAIN = compositor backpressure
	// (Qt6 polls POLLOUT and retries), EPIPE = compositor dead.
	if r, err := C.swl_call_i_u(symWlDisplayFlush, C.uintptr_t(s.displayHandle)); r < 0 && err != syscall.EAGAIN {
		slog.Warn("software: wl_display_flush failed", "error", err)
	}

	waylandDispatchShmQueue(s.displayHandle, wl.shmQueue)

	wl.frontIdx = backIdx
}

// waylandPresentDamage copies pixel data and commits with damage rects.
// wl_shm must be obtained eagerly (in blit_linux.go detection block) before
// this method is called. If wlShm is 0, init failed and we bail out.
func (s *Surface) waylandPresentDamage(data []byte, width, height int32, rects []image.Rectangle) {
	wl := &s.wlState
	if wl.wlShm == 0 {
		return
	}

	// Pick a non-busy buffer (same logic as waylandPresent).
	backIdx := -1
	for i := 1; i <= len(wl.buffers); i++ {
		idx := (wl.frontIdx + i) % len(wl.buffers)
		if !wl.buffers[idx].busy {
			backIdx = idx
			break
		}
	}
	if backIdx < 0 {
		return
	}
	buf := &wl.buffers[backIdx]

	if buf.buffer == 0 || buf.width != width || buf.height != height {
		if buf.buffer != 0 {
			waylandDestroyShmBuffer(buf)
			*buf = waylandShmBuffer{fd: -1}
		}
		newBuf := waylandCreateShmBuffer(wl.wlShm, width, height)
		if newBuf == nil {
			return
		}
		*buf = *newBuf
		bufferBusyMu.Lock()
		bufferBusyMap[buf.buffer] = buf
		bufferBusyMu.Unlock()
	}

	// Copy pixels: full copy for new buffers or when no damage rects provided,
	// partial row-based copy for damaged regions only.
	waylandCopyPixels(buf, data, width, height, rects)

	surface := s.hwnd

	waylandSurfaceAttach(surface, buf.buffer, 0, 0)

	// Issue damage_buffer for each rect (opcode 9, buffer coordinates).
	bounds := image.Rect(0, 0, int(width), int(height))
	for _, r := range rects {
		r = r.Intersect(bounds)
		if r.Empty() {
			continue
		}
		waylandSurfaceDamageBuffer(surface, int32(r.Min.X), int32(r.Min.Y), int32(r.Dx()), int32(r.Dy()))
	}

	waylandSurfaceCommit(surface)

	buf.busy = true

	if r, err := C.swl_call_i_u(symWlDisplayFlush, C.uintptr_t(s.displayHandle)); r < 0 && err != syscall.EAGAIN {
		slog.Warn("software: wl_display_flush failed (damage)", "error", err)
	}

	// Dispatch pending release events on the SHM queue.
	waylandDispatchShmQueue(s.displayHandle, wl.shmQueue)

	wl.frontIdx = backIdx
}

// waylandCopyPixels copies pixel data into a SHM buffer. On first use
// (needsFullCopy) or when no damage rects are provided, a full-frame copy
// is performed. Otherwise only the damaged rows are copied.
func waylandCopyPixels(buf *waylandShmBuffer, data []byte, width, height int32, rects []image.Rectangle) {
	if buf.needsFullCopy || len(rects) == 0 {
		n := int(width) * int(height) * 4
		if n > len(data) {
			n = len(data)
		}
		if n > len(buf.data) {
			n = len(buf.data)
		}
		copy(buf.data[:n], data[:n])
		buf.needsFullCopy = false
		return
	}

	// Row-based partial copy — only damaged regions.
	stride := int(width) * 4
	bounds := image.Rect(0, 0, int(width), int(height))
	for _, r := range rects {
		r = r.Intersect(bounds)
		if r.Empty() {
			continue
		}
		rowBytes := r.Dx() * 4
		for y := r.Min.Y; y < r.Max.Y; y++ {
			off := y*stride + r.Min.X*4
			end := off + rowBytes
			if end > len(data) || end > len(buf.data) {
				continue
			}
			copy(buf.data[off:end], data[off:end])
		}
	}
}

// waylandSurfaceCommit calls wl_surface_commit (opcode 6).
func waylandSurfaceCommit(surface uintptr) {
	C.swl_marshal0(symWlProxyMarshal, C.uintptr_t(surface), 6)
}

// waylandSurfaceAttach calls wl_surface_attach(surface, buffer, x, y) — opcode 1.
func waylandSurfaceAttach(surface, buffer uintptr, x, y int32) {
	C.swl_surface_attach(symWlProxyMarshal, C.uintptr_t(surface), C.uintptr_t(buffer), C.int32_t(x), C.int32_t(y))
}

// waylandSurfaceDamageBuffer calls wl_surface_damage_buffer(surface, x, y, w, h) — opcode 9.
// Preferred over deprecated wl_surface_damage (opcode 2) since wl_surface v4 (Wayland 1.10, 2016).
// Uses buffer coordinates instead of surface coordinates — correct on HiDPI.
func waylandSurfaceDamageBuffer(surface uintptr, x, y, w, h int32) {
	C.swl_surface_damage_buffer(symWlProxyMarshal, C.uintptr_t(surface), C.int32_t(x), C.int32_t(y), C.int32_t(w), C.int32_t(h))
}

// waylandDispatchShmQueue reads and dispatches events on the SHM queue.
// Uses wl_display_roundtrip_queue which does prepare_read → read_events →
// dispatch_queue — ensuring release events are actually read from the
// socket, not just dispatched from an empty queue.
//
// wl_display_dispatch_queue_pending (the previous approach) only dispatches
// already-read events. But nobody reads events into shmQueue — gogpu's
// read_events routes events to their queue, but the render thread calling
// dispatch_queue_pending may run before main thread reads. Result: empty
// queue, release callbacks never fire, all 3 buffers stuck busy=true.
func waylandDispatchShmQueue(display, queue uintptr) {
	if queue == 0 {
		return
	}
	if rt, err := C.swl_call_i_uu(symWlDisplayRoundtripQueue, C.uintptr_t(display), C.uintptr_t(queue)); rt < 0 {
		slog.Warn("software: wl_display_roundtrip_queue failed (dispatch)", "error", err)
	}
}

// destroyWaylandBlitState releases all Wayland SHM resources for a surface.
// Destruction order: buffers → wl_shm → registry → shmQueue.
// Proxies must be destroyed before the queue they're assigned to.
func (s *Surface) destroyWaylandBlitState() {
	wl := &s.wlState
	for i := range wl.buffers {
		waylandDestroyShmBuffer(&wl.buffers[i])
		wl.buffers[i] = waylandShmBuffer{fd: -1}
	}
	// Destroy wl_shm proxy (was leaked before BUG-SW-WAYLAND-002 fix).
	if wl.wlShm != 0 {
		C.swl_call_v_u(symWlProxyDestroy, C.uintptr_t(wl.wlShm))
		wl.wlShm = 0
	}
	// Destroy registry proxy (was leaked before BUG-SW-WAYLAND-002 fix).
	if wl.registry != 0 {
		C.swl_call_v_u(symWlProxyDestroy, C.uintptr_t(wl.registry))
		wl.registry = 0
	}
	// Destroy the SHM event queue LAST — all proxies assigned to it
	// must be destroyed first (wl_event_queue_destroy asserts this).
	if wl.shmQueue != 0 {
		C.swl_call_v_u(symWlEventQueueDestroy, C.uintptr_t(wl.shmQueue))
		wl.shmQueue = 0
	}
}
