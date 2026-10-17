//go:build js && wasm

package wgpu

import "github.com/roffe/wgpu/internal/browser"

// ShaderModule represents a compiled shader module.
type ShaderModule struct {
	browser  *browser.ShaderModule
	released bool
}

// Release destroys the shader module.
func (m *ShaderModule) Release() {
	if m.released {
		return
	}
	m.released = true
}
