// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build ((linux && !android) || freebsd || openbsd || netbsd) && !(js && wasm)

package allbackends

import (
	// Linux-specific HAL backend imports.

	// Vulkan backend - primary backend on Linux.
	_ "github.com/roffe/gowgpu/hal/vulkan"

	// OpenGL backend (EGL) - not in the default instance, for machines
	// without Vulkan. Linux only, an empty package on the BSDs.
	_ "github.com/roffe/gowgpu/hal/gles"
)
