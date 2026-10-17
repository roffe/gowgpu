//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

// Package vk provides Vulkan bindings through cgo, without Vulkan headers:
// the loader library is opened at run time, every command is called through
// the pointer vkGet*ProcAddr returns, by one C trampoline per C signature
// (trampolines.h). Pointers cross as const uint8_t *, a pointee with no
// pointers as far as cgo can tell, so cgocheck never walks the Go memory a
// create info points at; noescape keeps stack pointees on the stack.
//
// # Intel Driver Compatibility
//
// Intel drivers (Iris Xe, 12th Gen+) have known quirks:
//   - vkGetInstanceProcAddr(NULL, "vkGetDeviceProcAddr") returns NULL
//   - Use SetDeviceProcAddr(instance) after vkCreateInstance
//   - vkCreateGraphicsPipelines may return VK_SUCCESS with VK_NULL_HANDLE pipeline
//
// See: https://github.com/roffe/gowgpu/issues/24
package vk

/*
#include "trampolines.h"
#cgo noescape gvk_q_hp
#cgo nocallback gvk_q_hp
#cgo noescape gvk_v_hp
#cgo nocallback gvk_v_hp
*/
import "C"

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

var (
	vulkanLib             uintptr
	vkGetInstanceProcAddr unsafe.Pointer
	vkGetDeviceProcAddr   unsafe.Pointer

	initOnce sync.Once
	errInit  error
)

// vulkanLibraryName returns platform-specific Vulkan library name.
func vulkanLibraryName() string {
	return vulkanLibraryNameFor(runtime.GOOS)
}

func vulkanLibraryNameFor(goos string) string {
	switch goos {
	case "windows":
		return "vulkan-1.dll"
	case "darwin", "ios":
		return "libvulkan.dylib" // MoltenVK
	case "android":
		return "libvulkan.so"
	default: // linux, freebsd, etc.
		return "libvulkan.so.1"
	}
}

// Init loads the Vulkan library. Safe to call multiple times - only first
// call does actual work.
func Init() error {
	initOnce.Do(func() {
		errInit = doInit()
	})
	return errInit
}

func doInit() error {
	var err error
	vulkanLib, err = openLibrary(vulkanLibraryName())
	if err != nil {
		return fmt.Errorf("failed to load Vulkan library %s: %w", vulkanLibraryName(), err)
	}
	vkGetInstanceProcAddr, err = librarySymbol(vulkanLib, "vkGetInstanceProcAddr")
	if err != nil {
		return fmt.Errorf("vkGetInstanceProcAddr not found: %w", err)
	}
	return nil
}

// procAddr calls PFN_vkVoidFunction fn(handle, name).
func procAddr(fn unsafe.Pointer, handle uintptr, name string) unsafe.Pointer {
	if fn == nil {
		return nil
	}
	cname := make([]byte, len(name)+1)
	copy(cname, name)
	r := C.gvk_q_hp(C.uintptr_t(uintptr(fn)), C.uint64_t(handle), (*C.uint8_t)(unsafe.Pointer(&cname[0])))
	return cPointer(uintptr(r))
}

// cPointer turns an address C returned (never Go memory) into a pointer.
func cPointer(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

// GetInstanceProcAddr returns function pointer for Vulkan instance function.
// Pass instance=0 for global functions (vkCreateInstance, vkEnumerateInstance*).
func GetInstanceProcAddr(instance Instance, name string) unsafe.Pointer {
	return procAddr(vkGetInstanceProcAddr, uintptr(instance), name)
}

// SetDeviceProcAddr sets the vkGetDeviceProcAddr function pointer.
// Must be called with a valid instance after vkCreateInstance.
// Some drivers (e.g., Intel) don't support loading vkGetDeviceProcAddr with instance=0.
func SetDeviceProcAddr(instance Instance) {
	if vkGetDeviceProcAddr == nil {
		vkGetDeviceProcAddr = GetInstanceProcAddr(instance, "vkGetDeviceProcAddr")
	}
}

// GetDeviceProcAddr returns function pointer for Vulkan device function.
func GetDeviceProcAddr(device Device, name string) unsafe.Pointer {
	if vkGetDeviceProcAddr == nil {
		// Try lazy load from global (may not work on all drivers)
		vkGetDeviceProcAddr = GetInstanceProcAddr(0, "vkGetDeviceProcAddr")
		if vkGetDeviceProcAddr == nil {
			return nil
		}
	}
	return procAddr(vkGetDeviceProcAddr, uintptr(device), name)
}

// DestroyDevice calls vkDestroyDevice through proc, a pointer from
// GetDeviceProcAddr, for when the device's Commands are already gone.
func DestroyDevice(proc unsafe.Pointer, device Device, allocator *AllocationCallbacks) {
	if proc == nil {
		return
	}
	C.gvk_v_hp(C.uintptr_t(uintptr(proc)), C.uint64_t(device), (*C.uint8_t)(unsafe.Pointer(allocator)))
}

// Close releases the Vulkan library.
func Close() error {
	if vulkanLib != 0 {
		err := closeLibrary(vulkanLib)
		vulkanLib = 0
		vkGetInstanceProcAddr = nil
		vkGetDeviceProcAddr = nil
		return err
	}
	return nil
}
