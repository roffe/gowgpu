//go:build !(js && wasm)

package software

import "github.com/roffe/gowgpu/hal"

// init registers the software backend with the HAL registry.
func init() {
	hal.RegisterBackend(Backend{})
}
