//go:build !(js && wasm)

package vk

import "runtime"

// Pin pins the Go memory p points at in pins and returns p, for a pointer
// stored in memory that a Vulkan call reads: Go memory passed to C may only
// point at Go memory that is pinned, and cgocheck checks every call for it.
// It checks all of the object p points into, so that object must not hold
// other pointers to unpinned Go memory. A local the compiler keeps on the
// stack needs no pinning, as a goroutine's stack stays where it is while it
// calls C, and pinning one moves it to the heap. A nil p is returned as it
// is.
func Pin[T any](pins *runtime.Pinner, p *T) *T {
	if p != nil {
		pins.Pin(p)
	}
	return p
}
