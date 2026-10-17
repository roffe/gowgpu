//go:build android && (arm64 || amd64)

// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

import "github.com/roffe/wgpu/hal"

func init() {
	hal.RegisterBackend(Backend{})
}
