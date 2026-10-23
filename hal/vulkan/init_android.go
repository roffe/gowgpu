//go:build android

// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

import "github.com/roffe/gowgpu/hal"

func init() {
	hal.RegisterBackend(Backend{})
}
