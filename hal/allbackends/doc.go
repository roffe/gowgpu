//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

// Package allbackends imports all HAL backend implementations.
//
// Import this package for side effects to register all available backends:
//
//	import (
//		_ "github.com/roffe/gowgpu/hal/allbackends"
//	)
//
// This will register:
//   - Vulkan backend (Windows, Linux, the BSDs, macOS via MoltenVK, Android)
//   - Metal backend (macOS, iOS)
//   - DX12 backend (Windows)
//   - OpenGL backend (Windows, Linux)
//
// OpenGL is registered but outside the default instance's backends: ask for
// gputypes.BackendsGL to use it, for example when nothing else is available.
//
// After importing, use hal.GetBackend or hal.SelectBestBackend to access backends.
//
// Android registers Vulkan only.
//
// The no-op provider is not registered by this package. Import
// github.com/roffe/gowgpu/hal/noop explicitly when it is required for tests.
//
// Example usage:
//
//	import (
//		_ "github.com/roffe/gowgpu/hal/allbackends"
//		"github.com/roffe/gowgpu/core"
//	)
//
//	func main() {
//		// Instance will now enumerate real GPUs
//		instance := core.NewInstance(nil)
//		adapters := instance.EnumerateAdapters()
//		for _, a := range adapters {
//			fmt.Println(a) // Real GPU adapters
//		}
//	}
package allbackends
