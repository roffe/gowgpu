// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows

//nolint:gosec // Low-level FFI diagnostic tool requires unsafe operations
package main

import (
	"fmt"
	"syscall"
	"unsafe"

	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// SyscallCreatePipeline tests using Windows syscall directly
// to completely bypass goffi and test if it's a goffi issue.
func SyscallCreatePipeline(cmds *vk.Commands, device vk.Device, createInfo *vk.GraphicsPipelineCreateInfo) (vk.Pipeline, vk.Result, error) {
	// Get the function pointer
	fnPtr := cmds.DebugFunctionPointer("vkCreateGraphicsPipelines")
	if fnPtr == nil {
		return 0, 0, fmt.Errorf("vkCreateGraphicsPipelines not loaded")
	}

	// Prepare arguments for syscall6
	// VkResult vkCreateGraphicsPipelines(
	//   VkDevice device,
	//   VkPipelineCache pipelineCache,
	//   uint32_t createInfoCount,
	//   const VkGraphicsPipelineCreateInfo* pCreateInfos,
	//   const VkAllocationCallbacks* pAllocator,
	//   VkPipeline* pPipelines
	// )

	var pipeline vk.Pipeline

	fmt.Println("    Syscall6 call:")
	fmt.Printf("      fnPtr: %p\n", fnPtr)
	fmt.Printf("      device: 0x%X\n", device)
	fmt.Printf("      createInfo: %p\n", createInfo)
	fmt.Printf("      pipeline addr: %p\n", &pipeline)

	// Call via syscall6
	r1, _, _ := syscall.SyscallN(
		uintptr(fnPtr),
		uintptr(device),
		0, // pipelineCache = VK_NULL_HANDLE
		1, // createInfoCount = 1
		uintptr(unsafe.Pointer(createInfo)),
		0, // pAllocator = NULL
		uintptr(unsafe.Pointer(&pipeline)),
	)

	result := vk.Result(int32(r1))
	fmt.Printf("      result: %d\n", result)
	fmt.Printf("      pipeline: 0x%X\n", pipeline)

	return pipeline, result, nil
}
