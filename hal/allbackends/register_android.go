//go:build android && (arm64 || amd64)

// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

package allbackends

import (
	_ "github.com/roffe/gowgpu/hal/vulkan"

	// OpenGL ES (EGL) - not in the default instance, for devices without
	// Vulkan.
	_ "github.com/roffe/gowgpu/hal/gles"
)
