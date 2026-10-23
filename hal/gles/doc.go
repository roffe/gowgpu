//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

// Package gles implements the HAL backend for OpenGL 3.3 and newer and
// OpenGL ES 3.0 and newer.
//
// It's the fallback for machines without Vulkan, Metal or DX12, so
// hal/allbackends registers it but the default instance leaves it out: ask
// for it with gputypes.BackendsGL.
//
// # Platforms
//
//   - Linux: EGL (subpackage egl), on X11, Wayland or surfaceless. Desktop
//     GL core, the first of 4.6 down to 4.3 and then 3.3 the driver makes,
//     GLES 3.0 when that's all EGL offers.
//   - Android: GLES 3.0 through EGL. The context lives on the Instance, on
//     EGL's default display, so what a device made survives the window
//     Android replaces on every resume.
//   - Windows: WGL (subpackage wgl), through syscall.
//
// There is no GL backend on the BSDs, macOS, iOS or the web. On Linux and
// Android the EGL, GL, X11 and Wayland calls go through cgo; the libraries
// are opened at run time.
//
// # Usage
//
//	import _ "github.com/roffe/gowgpu/hal/gles"
//
// Most apps get it through hal/allbackends instead. Directly, it's
// hal.GetBackend(gputypes.BackendGL).
//
// # How it works
//
// Commands are recorded while passes are encoded and replayed on the GL
// context at Queue.Submit. A surface renders into an offscreen framebuffer
// that Present blits, flipped, into the window's back buffer.
//
// Compared to Vulkan, Metal and DX12 there is one queue, no async compute,
// and texture uploads happen at once.
package gles
