// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build !(js && wasm)

package allbackends

import (
	// iOS has Metal only: there is no Vulkan loader to open.
	_ "github.com/roffe/gowgpu/hal/metal"
)
