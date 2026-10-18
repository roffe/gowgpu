//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

// Package vulkan implements the HAL backend for Vulkan.
//
// Calls go through the cgo trampolines in the vk subpackage, and the loader
// opens vulkan-1.dll (Windows), libvulkan.so.1 (Linux, the BSDs), libvulkan.so
// (Android) or libvulkan.dylib (macOS, MoltenVK) at run time. Nothing Vulkan
// is needed to build, and a machine without a Vulkan driver just has no
// Vulkan adapters.
//
// # Architecture
//
// The backend follows wgpu-hal:
//   - Instance: VkInstance with the surface extensions of the platform
//   - Adapter: VkPhysicalDevice, its limits, features and surface formats
//   - Device: VkDevice with its queue and memory allocator
//   - Queue: submission and synchronization
//   - Resources: buffers, textures, pipelines and bind groups
//
// Memory comes from a pooled allocator, like gpu-allocator.
//
// # Surfaces
//
//   - Windows: VK_KHR_win32_surface
//   - Linux and the BSDs: VK_KHR_xlib_surface, VK_KHR_wayland_surface
//   - macOS: VK_EXT_metal_surface through MoltenVK
//   - Android: VK_KHR_android_surface
package vulkan
