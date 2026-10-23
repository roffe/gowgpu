// Package vknoop provides do-nothing Vulkan entry points for benchmarking
// the vk wrappers without a driver.
package vknoop

/*
#include <stdint.h>
volatile uintptr_t vknoop_sink;
static void noop_draw(uint64_t cb, uint32_t a, uint32_t b, uint32_t c, uint32_t d) { vknoop_sink = cb + a + b + c + d; }
static void noop_bind(uint64_t cb, uint32_t bp, uint64_t layout, uint32_t first, uint32_t n, uintptr_t sets, uint32_t ndyn, uintptr_t dyn) {
	vknoop_sink = cb + bp + layout + first + n + *(const uint64_t *)sets + ndyn;
}
static int32_t noop_create(uint64_t dev, uintptr_t info, uintptr_t alloc, uintptr_t out) { *(uint64_t *)out = dev + 1; return 0; }
static void noop_scissor(uint64_t cb, uint32_t first, uint32_t n, uintptr_t rects) { vknoop_sink = cb + first + n + rects; }
static void noop_pipeline(uint64_t cb, uint32_t bp, uint64_t p) { vknoop_sink = cb + bp + p; }
static int32_t noop_submit(uintptr_t queue, uint32_t n, uintptr_t submits, uint64_t fence) { vknoop_sink = queue + n + submits + fence; return 0; }
static uintptr_t p_draw(void) { return (uintptr_t)&noop_draw; }
static uintptr_t p_bind(void) { return (uintptr_t)&noop_bind; }
static uintptr_t p_create(void) { return (uintptr_t)&noop_create; }
static uintptr_t p_scissor(void) { return (uintptr_t)&noop_scissor; }
static uintptr_t p_pipeline(void) { return (uintptr_t)&noop_pipeline; }
static uintptr_t p_submit(void) { return (uintptr_t)&noop_submit; }
*/
import "C"

import "unsafe"

func ptr(p uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&p)) }

func Draw() unsafe.Pointer     { return ptr(uintptr(C.p_draw())) }
func Bind() unsafe.Pointer     { return ptr(uintptr(C.p_bind())) }
func Create() unsafe.Pointer   { return ptr(uintptr(C.p_create())) }
func Scissor() unsafe.Pointer  { return ptr(uintptr(C.p_scissor())) }
func Pipeline() unsafe.Pointer { return ptr(uintptr(C.p_pipeline())) }
func Submit() unsafe.Pointer   { return ptr(uintptr(C.p_submit())) }
