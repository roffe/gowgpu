//go:build !cgo && !(js && wasm)

package vk

// The bindings call the Vulkan driver through cgo trampolines, so without cgo
// the build stops here with a message that says why, ahead of the "undefined:
// Commands" errors the missing cgo files cause.
const cgoRequired int = "github.com/roffe/gowgpu/hal/vulkan needs cgo: build with CGO_ENABLED=1 and a C compiler"

var _ = cgoRequired
