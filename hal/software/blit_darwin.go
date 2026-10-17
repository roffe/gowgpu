//go:build darwin && !(js && wasm)

// Error handling follows ADR-049 three-tier strategy:
//
//	Tier 1 (creation/submit): check the API result code
//	Tier 2 (void/hot path): infallible by GPU API contract — Vulkan §6.6, WebGPU §21.2
//	Tier 3 (platform syscalls): use errno for diagnostics (Wayland, X11, Win32)
//
// Enterprise reference: Rust wgpu-hal returns () for all draw/destroy/barrier commands.
//
// macOS blits are Tier 2: blit_darwin.c checks every object it gets back and
// skips the frame when one is nil.
package software

/*
#include <stdint.h>

int swb_init(void);
void swb_blit(uintptr_t layer, const uint8_t *data, int32_t width, int32_t height, const int32_t *rects, int32_t nrects);

#cgo noescape swb_blit
#cgo nocallback swb_blit
#cgo nocallback swb_init
*/
import "C"

import (
	"image"
	"log/slog"
	"sync"
)

// The C half resolves libobjc, CoreGraphics and Metal once per process.
var (
	blitOnce  sync.Once
	blitReady bool
)

// platformBlit holds no per-surface state on macOS: the window's layer is
// s.hwnd, and the Metal device and queue are shared.
type platformBlit struct{}

func blitInit() bool {
	blitOnce.Do(func() {
		if C.swb_init() != 0 {
			slog.Debug("software: macOS blit unavailable — libobjc or CoreGraphics missing")
			return
		}
		blitReady = true
	})
	return blitReady
}

// configurePlatformBlit is a no-op on macOS (no Wayland).
func (s *Surface) configurePlatformBlit() {}

// createPlatformFramebuffer returns nil — use Go heap memory.
func (s *Surface) createPlatformFramebuffer(_, _ int32) []byte { return nil }

// destroyPlatformFramebuffer is a no-op.
func (s *Surface) destroyPlatformFramebuffer() {}

// blitFramebufferToWindow shows the frame in the window's CALayer or
// CAMetalLayer, s.hwnd.
func (s *Surface) blitFramebufferToWindow(data []byte, width, height int32) {
	if s.hwnd == 0 || width <= 0 || height <= 0 || len(data) < int(4*width*height) || !blitInit() {
		return
	}
	C.swb_blit(C.uintptr_t(s.hwnd), (*C.uint8_t)(&data[0]), C.int32_t(width), C.int32_t(height), nil, 0)
}

// blitDamageRectsToWindow is blitFramebufferToWindow redrawing only rects.
// A CAMetalLayer always takes the whole frame.
func (s *Surface) blitDamageRectsToWindow(src []byte, w, h int32, rects []image.Rectangle) {
	if s.hwnd == 0 || w <= 0 || h <= 0 || len(rects) == 0 || len(src) < int(4*w*h) || !blitInit() {
		return
	}
	bounds := image.Rect(0, 0, int(w), int(h))
	flat := make([]C.int32_t, 0, 4*len(rects))
	for _, rect := range rects {
		if r := bounds.Intersect(rect); !r.Empty() {
			flat = append(flat, C.int32_t(r.Min.X), C.int32_t(r.Min.Y), C.int32_t(r.Dx()), C.int32_t(r.Dy()))
		}
	}
	if len(flat) == 0 {
		return
	}
	C.swb_blit(C.uintptr_t(s.hwnd), (*C.uint8_t)(&src[0]), C.int32_t(w), C.int32_t(h), &flat[0], C.int32_t(len(flat)/4))
}
