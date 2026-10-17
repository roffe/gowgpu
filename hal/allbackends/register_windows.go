// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package allbackends

import (
	// Windows-specific HAL backend imports.

	// Vulkan backend - primary backend on Windows.
	_ "github.com/roffe/gowgpu/hal/vulkan"

	// DX12 backend - alternative high-performance backend.
	_ "github.com/roffe/gowgpu/hal/dx12"
)
