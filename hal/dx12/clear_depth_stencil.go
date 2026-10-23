//go:build windows && !arm64 && !(js && wasm)

package dx12

import "github.com/roffe/gowgpu/hal/dx12/d3d12"

// clearDepthStencil clears the planes of a depth/stencil view clearFlags
// names.
func (e *CommandEncoder) clearDepthStencil(dsv d3d12.D3D12_CPU_DESCRIPTOR_HANDLE, _ d3d12.DXGI_FORMAT, clearFlags d3d12.D3D12_CLEAR_FLAGS, depth float32, stencil uint8) {
	e.cmdList.ClearDepthStencilView(dsv, clearFlags, depth, stencil, 0, nil)
}
