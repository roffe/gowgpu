//go:build windows && arm64

package dx12

import "github.com/roffe/gowgpu/hal/dx12/d3d12"

// clearDepthStencil clears the planes of a depth/stencil view clearFlags
// names. ClearDepthStencilView takes the depth as a float argument, which
// Go's windows/arm64 syscalls put in an integer register where the callee
// reads a float one, so the clear goes through a render pass, whose clear
// values are passed by pointer.
func (e *CommandEncoder) clearDepthStencil(dsv d3d12.D3D12_CPU_DESCRIPTOR_HANDLE, format d3d12.DXGI_FORMAT, clearFlags d3d12.D3D12_CLEAR_FLAGS, depth float32, stencil uint8) {
	if err := e.clearDepthStencilByRenderPass(dsv, format, clearFlags, depth, stencil); err != nil {
		e.fail(err)
	}
}
