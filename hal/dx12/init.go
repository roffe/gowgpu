//go:build windows && !(js && wasm)

package dx12

import "github.com/roffe/gowgpu/hal"

// init registers the DX12 backend with the HAL registry.
func init() {
	hal.RegisterBackend(Backend{})
}
