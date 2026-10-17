//go:build !(js && wasm)

package noop

import "github.com/roffe/wgpu/hal"

// init registers the noop backend with the HAL registry.
func init() {
	hal.RegisterBackend(Backend{})
}
