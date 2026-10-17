//go:build linux && !(js && wasm)

// Error handling follows ADR-049 three-tier strategy:
//
//	Tier 1 (creation/submit): check the API result code
//	Tier 2 (void/hot path): infallible by GPU API contract — Vulkan §6.6, WebGPU §21.2
//	Tier 3 (platform syscalls): use errno for diagnostics (Wayland, X11, Win32)
//
// Enterprise reference: Rust wgpu-hal returns () for all draw/destroy/barrier commands.
//
// X11 blit calls are Tier 2: XPutImage, XFlush, XCreateGC are void or return
// status that is already handled by the caller. X11 does not use errno for
// display protocol errors.
package software

/*
#include "trampolines_x11_linux.h"
#cgo noescape sw_i_p
#cgo nocallback sw_i_p
#cgo noescape sw_u_uuuu
#cgo nocallback sw_u_uuuu
#cgo noescape sw_v_u
#cgo nocallback sw_v_u
#cgo noescape sw_v_uu
#cgo nocallback sw_v_uu
#cgo noescape sw_v_uuupiiiiii
#cgo nocallback sw_v_uuupiiiiii
*/
import "C"

import (
	"image"
	"log/slog"
	"sync"
	"unsafe"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/internal/dl"
)

// X11 constants (from X.h / Xlib.h).
const (
	zPixmap  = 2 // ZPixmap format for XImage
	lsbFirst = 0 // LSBFirst byte order (little-endian)
)

// ximage matches the C XImage struct layout on amd64 Linux (LP64).
//
// Reference: X11/Xlib.h _XImage struct.
// Layout verified against Skia RasterWindowContext_unix.cpp and SDL3
// SDL_x11framebuffer.c — both construct XImage on stack with these fields.
//
// The struct ends with a funcs sub-struct of 6 function pointers (48 bytes
// on amd64) that XInitImage fills in. We include them as opaque padding so
// XInitImage has room to write without corrupting adjacent memory.
type ximage struct {
	width          int32   // 0
	height         int32   // 4
	xoffset        int32   // 8
	format         int32   // 12
	data           uintptr // 16 (char*)
	byteOrder      int32   // 24
	bitmapUnit     int32   // 28
	bitmapBitOrder int32   // 32
	bitmapPad      int32   // 36
	depth          int32   // 40
	bytesPerLine   int32   // 44
	bitsPerPixel   int32   // 48
	_pad0          int32   // 52 (padding to align unsigned long)
	redMask        uint64  // 56 (unsigned long on LP64)
	greenMask      uint64  // 64
	blueMask       uint64  // 72
	obdata         uintptr // 80 (XPointer = char*)
	// funcs: 6 function pointers filled by XInitImage
	funcCreateImage  uintptr // 88
	funcDestroyImage uintptr // 96
	funcGetPixel     uintptr // 104
	funcPutPixel     uintptr // 112
	funcSubImage     uintptr // 120
	funcAddPixel     uintptr // 128
	// Total: 136 bytes
}

// x11State holds lazily-loaded libX11 symbols and call interfaces.
// Initialized once on first blit; subsequent blits reuse the state.
var (
	x11Once  sync.Once
	x11Ready bool // true if libX11 loaded successfully

	x11Lib dl.Lib

	symXCreateGC  unsafe.Pointer
	symXFreeGC    unsafe.Pointer
	symXPutImage  unsafe.Pointer
	symXFlush     unsafe.Pointer
	symXInitImage unsafe.Pointer

	// XCreateGC(Display*, Drawable, unsigned long valuemask, XGCValues*) -> GC
	// XFreeGC(Display*, GC) -> int
	// XPutImage(Display*, Drawable, GC, XImage*, int, int, int, int, uint, uint) -> int
	// XFlush(Display*) -> int
	// XInitImage(XImage*) -> Status (int)
)

// initX11 loads libX11.so and prepares call interfaces.
// Called once via sync.Once; sets x11Ready on success.
func initX11() {
	var err error
	if x11Lib, err = dl.Open("libX11.so.6", "libX11.so"); err != nil {
		slog.Debug("software: X11 blit unavailable — could not load libX11", "error", err)
		return
	}
	for _, sym := range []struct {
		name string
		dst  *unsafe.Pointer
	}{
		{"XCreateGC", &symXCreateGC},   // GC XCreateGC(Display*, Drawable, unsigned long valuemask, XGCValues*)
		{"XFreeGC", &symXFreeGC},       // int XFreeGC(Display*, GC)
		{"XPutImage", &symXPutImage},   // int XPutImage(Display*, Drawable, GC, XImage*, int, int, int, int, unsigned, unsigned)
		{"XFlush", &symXFlush},         // int XFlush(Display*)
		{"XInitImage", &symXInitImage}, // Status XInitImage(XImage*)
	} {
		if *sym.dst, err = x11Lib.Sym(sym.name); err != nil {
			slog.Debug("software: X11 blit unavailable", "error", err)
			return
		}
	}

	x11Ready = true
	slog.Debug("software: X11 blit initialized — XPutImage path available")
}

// platformBlit holds platform-specific blit resources for Linux.
// On X11: uses XCreateGC/XPutImage. On Wayland: uses wl_shm.
// Embedded in Surface via build tags.
type platformBlit struct {
	gc      uintptr          // X11 GC (Graphics Context), lazy-initialized on first blit
	wlState waylandBlitState // Wayland SHM state, initialized eagerly in Configure
}

// configurePlatformBlit uses the surface's typed target and eagerly obtains
// wl_shm for Wayland.
// Called from Configure() on the MAIN thread, before the event loop or render
// thread calls Present. This eliminates the race between wl_display_roundtrip
// (in obtainWlShm) and concurrent wl_display_dispatch (in gogpu event loop).
//
// Enterprise references unanimously do this during init, not first present:
//   - GLFW: wl_shm bound in glfwInit(), fatal error if missing (wl_init.c:568)
//   - SDL3: wl_shm bound in SDL_Init(SDL_INIT_VIDEO) (SDL_waylandvideo.c:1548)
//   - Qt6:  wl_shm bound in QWaylandDisplay(), before EventThread.start()
//     (qwaylanddisplay.cpp:531 — explicit comment about this race)
//   - winit: wl_shm bound in EventLoop::new(), before calloop (event_loop/mod.rs:92)
//
// See BUG-SW-WAYLAND-001: obtainWlShm lazy init → SIGSEGV on Wayland (gogpu#292).
func (s *Surface) configurePlatformBlit() {
	if s.displayHandle == 0 {
		return
	}
	if !s.wlState.detected {
		s.wlState.detected = true
		s.wlState.isWayland = s.targetKind == hal.SurfaceTargetWaylandSurface
		if s.wlState.isWayland {
			// Create shmQueue FIRST — obtainWlShm needs it for the
			// display wrapper so all proxies inherit this queue.
			s.wlState.shmQueue = createShmQueue(s.displayHandle)
			s.wlState.wlShm, s.wlState.registry = obtainWlShm(s.displayHandle, s.wlState.shmQueue)
		}
	}
}

// createPlatformFramebuffer returns nil on Linux — we use Go heap memory
// and blit via XPutImage (unlike Windows which uses kernel-allocated DIB section).
func (s *Surface) createPlatformFramebuffer(_, _ int32) []byte { return nil }

// destroyPlatformFramebuffer releases platform blit resources.
// On X11: frees the GC. On Wayland: destroys SHM buffers.
func (s *Surface) destroyPlatformFramebuffer() {
	// Clean up Wayland SHM resources if any.
	if s.wlState.isWayland {
		s.destroyWaylandBlitState()
		return
	}

	// X11 cleanup.
	if s.gc == 0 || s.displayHandle == 0 {
		return
	}

	x11Once.Do(initX11)
	if !x11Ready {
		s.gc = 0
		return
	}

	display := s.displayHandle
	gc := s.gc
	C.sw_v_uu(C.uintptr_t(uintptr(symXFreeGC)), C.uintptr_t(display), C.uintptr_t(gc))
	s.gc = 0
}

// blitFramebufferToWindow copies the framebuffer to the X11 window via XPutImage.
//
// Follows the Skia enterprise pattern (RasterWindowContext_unix.cpp):
// construct XImage on stack pointing to our BGRA pixel buffer, call XInitImage
// to fill function pointers, then XPutImage to send pixels to the X server.
//
// No pixel format conversion is needed: our framebuffer stores BGRA (the software
// rasterizer's executeFullscreenBlit does RGBA→BGRA swizzle), which matches X11's
// 32bpp ZPixmap LSBFirst format on little-endian systems. This is the same reason
// Skia and SDL3 set byte_order=LSBFirst and depth=24 with bits_per_pixel=32.
func (s *Surface) blitFramebufferToWindow(data []byte, width, height int32) {
	if s.hwnd == 0 || s.displayHandle == 0 || width <= 0 || height <= 0 || len(data) == 0 {
		return
	}

	// Wayland detected eagerly in Configure (configurePlatformBlit).
	if s.wlState.isWayland {
		s.waylandPresent(data, width, height)
		return
	}

	// Lazy-load libX11 on first blit.
	x11Once.Do(initX11)
	if !x11Ready {
		return
	}

	display := s.displayHandle
	window := s.hwnd

	// Lazy-create GC on first blit (same pattern as SDL3: XCreateGC in CreateWindowFramebuffer).
	if s.gc == 0 {
		var valuemask uintptr // 0 = no special GC values
		var values uintptr    // NULL = default values
		var gc uintptr
		gc = uintptr(C.sw_u_uuuu(C.uintptr_t(uintptr(symXCreateGC)), C.uintptr_t(display), C.uintptr_t(window), C.uintptr_t(valuemask), C.uintptr_t(values)))
		if gc == 0 {
			slog.Warn("software: XCreateGC failed — blit skipped")
			return
		}
		s.gc = gc
	}

	// Construct XImage on stack pointing to our pixel buffer.
	// Fields follow Skia RasterWindowContext_unix.cpp exactly.
	var ximg ximage
	ximg.width = width
	ximg.height = height
	ximg.format = zPixmap
	ximg.data = uintptr(unsafe.Pointer(&data[0]))
	ximg.byteOrder = lsbFirst
	ximg.bitmapUnit = 32
	ximg.bitmapBitOrder = lsbFirst
	ximg.bitmapPad = 32
	ximg.depth = 24
	ximg.bytesPerLine = 0 // XInitImage calculates this
	ximg.bitsPerPixel = 32

	// XInitImage fills the function pointer struct at the end of XImage.
	// Without this call, XPutImage would crash dereferencing nil func ptrs.
	imagePtr := &ximg
	var status int32
	status = int32(C.sw_i_p(C.uintptr_t(uintptr(symXInitImage)), (*C.uint8_t)(unsafe.Pointer(imagePtr))))
	if status == 0 {
		slog.Warn("software: XInitImage failed — blit skipped")
		return
	}

	// XPutImage: copy pixel data to the X server.
	gc := s.gc
	var srcX, srcY, dstX, dstY int32
	w := uint32(width)
	h := uint32(height)
	C.sw_v_uuupiiiiii(C.uintptr_t(uintptr(symXPutImage)), C.uintptr_t(display), C.uintptr_t(window), C.uintptr_t(gc), (*C.uint8_t)(unsafe.Pointer(imagePtr)), C.uint32_t(srcX), C.uint32_t(srcY), C.uint32_t(dstX), C.uint32_t(dstY), C.uint32_t(w), C.uint32_t(h))

	// XFlush ensures the X server processes the put request immediately.
	// Without this, pixels may not appear until the next X event.
	C.sw_v_u(C.uintptr_t(uintptr(symXFlush)), C.uintptr_t(display))
}

// blitDamageRectsToWindow copies only the specified damage regions from the
// framebuffer to the X11 window via XPutImage. Each rect is clipped to surface
// bounds before blitting. XPutImage supports sub-region coordinates natively
// via (srcX, srcY, dstX, dstY, width, height) parameters.
//
// This is the damage-aware presentation path for Linux software backend.
// Only changed regions are sent to the X server, reducing bandwidth for GUI
// apps where most of the surface is unchanged between frames.
func (s *Surface) blitDamageRectsToWindow(data []byte, width, height int32, rects []image.Rectangle) {
	if s.hwnd == 0 || s.displayHandle == 0 || width <= 0 || height <= 0 || len(data) == 0 {
		return
	}

	// Wayland detected eagerly in Configure (configurePlatformBlit).
	if s.wlState.isWayland {
		s.waylandPresentDamage(data, width, height, rects)
		return
	}

	// Lazy-load libX11 on first blit.
	x11Once.Do(initX11)
	if !x11Ready {
		return
	}

	display := s.displayHandle
	window := s.hwnd

	// Lazy-create GC on first blit.
	if s.gc == 0 {
		var valuemask uintptr
		var values uintptr
		var gc uintptr
		gc = uintptr(C.sw_u_uuuu(C.uintptr_t(uintptr(symXCreateGC)), C.uintptr_t(display), C.uintptr_t(window), C.uintptr_t(valuemask), C.uintptr_t(values)))
		if gc == 0 {
			slog.Warn("software: XCreateGC failed — damage blit skipped")
			return
		}
		s.gc = gc
	}

	// Construct XImage on stack pointing to our pixel buffer.
	// Same setup as blitFramebufferToWindow — fields follow Skia pattern.
	var img ximage
	img.width = width
	img.height = height
	img.format = zPixmap
	img.data = uintptr(unsafe.Pointer(&data[0]))
	img.byteOrder = lsbFirst
	img.bitmapUnit = 32
	img.bitmapBitOrder = lsbFirst
	img.bitmapPad = 32
	img.depth = 24
	img.bytesPerLine = 0 // XInitImage calculates this
	img.bitsPerPixel = 32

	imagePtr := &img
	var status int32
	status = int32(C.sw_i_p(C.uintptr_t(uintptr(symXInitImage)), (*C.uint8_t)(unsafe.Pointer(imagePtr))))
	if status == 0 {
		slog.Warn("software: XInitImage failed — damage blit skipped")
		return
	}

	// Surface bounds for clipping damage rects.
	bounds := image.Rect(0, 0, int(width), int(height))
	gc := s.gc

	for _, r := range rects {
		// Clip to surface bounds.
		r = r.Intersect(bounds)
		if r.Empty() {
			continue
		}

		srcX := int32(r.Min.X)
		srcY := int32(r.Min.Y)
		dstX := int32(r.Min.X)
		dstY := int32(r.Min.Y)
		w := uint32(r.Dx())
		h := uint32(r.Dy())
		C.sw_v_uuupiiiiii(C.uintptr_t(uintptr(symXPutImage)), C.uintptr_t(display), C.uintptr_t(window), C.uintptr_t(gc), (*C.uint8_t)(unsafe.Pointer(imagePtr)), C.uint32_t(srcX), C.uint32_t(srcY), C.uint32_t(dstX), C.uint32_t(dstY), C.uint32_t(w), C.uint32_t(h))
	}

	// XFlush ensures the X server processes all put requests immediately.
	C.sw_v_u(C.uintptr_t(uintptr(symXFlush)), C.uintptr_t(display))
}
