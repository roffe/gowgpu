//go:build !(js && wasm)

package software

import "github.com/roffe/wgpu/hal"

// init registers the software backend with the HAL registry.
func init() {
	hal.RegisterBackend(Backend{})
}
