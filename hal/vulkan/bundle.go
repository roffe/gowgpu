//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

import (
	"fmt"

	"github.com/roffe/gowgpu/hal"
)

// CreateRenderBundleEncoder is not supported: the core records render
// bundles and replays them into the render pass.
func (d *Device) CreateRenderBundleEncoder(*hal.RenderBundleEncoderDescriptor) (hal.RenderBundleEncoder, error) {
	return nil, fmt.Errorf("vulkan: render bundles not supported")
}

// DestroyRenderBundle does nothing, there are no HAL render bundles.
func (d *Device) DestroyRenderBundle(hal.RenderBundle) {}
