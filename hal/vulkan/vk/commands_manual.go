//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

// Manual Vulkan command wrappers for functions with signatures
// unsupported by the vk-gen generator.
// These are NOT overwritten by code generation.

package vk

/*
#include "trampolines.h"
#cgo noescape gvk_q_hp
#cgo nocallback gvk_q_hp
#cgo noescape gvk_r_hpq
#cgo nocallback gvk_r_hpq
#cgo noescape gvk_v_hhiihqqi
#cgo nocallback gvk_v_hhiihqqi
#cgo noescape gvk_v_hihi
#cgo nocallback gvk_v_hihi
#cgo noescape gvk_v_hipihi
#cgo nocallback gvk_v_hipihi
#cgo noescape gvk_v_hipp
#cgo nocallback gvk_v_hipp
#cgo noescape gvk_v_hippp
#cgo nocallback gvk_v_hippp
*/
import "C"

import "unsafe"

var _ = unsafe.Pointer(nil)

// CmdWriteTimestamp wraps vkCmdWriteTimestamp.
// Manual: generator cannot handle mixed handle+u32+handle+u32 signature.
func (c *Commands) CmdWriteTimestamp(commandBuffer CommandBuffer, pipelineStage PipelineStageFlagBits, queryPool QueryPool, query uint32) {
	if c.cmdWriteTimestamp == nil {
		return
	}
	C.gvk_v_hihi(C.uintptr_t(uintptr(c.cmdWriteTimestamp)), C.uint64_t(commandBuffer), C.uint32_t(pipelineStage), C.uint64_t(queryPool), C.uint32_t(query))
}

// CmdCopyQueryPoolResults wraps vkCmdCopyQueryPoolResults.
// Manual: generator cannot handle mixed handle+handle+u32+u32+handle+u64+u64+u32 signature.
func (c *Commands) CmdCopyQueryPoolResults(commandBuffer CommandBuffer, queryPool QueryPool, firstQuery, queryCount uint32, dstBuffer Buffer, dstOffset, stride uint64, flags QueryResultFlags) {
	if c.cmdCopyQueryPoolResults == nil {
		return
	}
	C.gvk_v_hhiihqqi(C.uintptr_t(uintptr(c.cmdCopyQueryPoolResults)), C.uint64_t(commandBuffer), C.uint64_t(queryPool), C.uint32_t(firstQuery), C.uint32_t(queryCount), C.uint64_t(dstBuffer), C.uint64_t(dstOffset), C.uint64_t(stride), C.uint32_t(flags))
}

// WaitSemaphores wraps vkWaitSemaphores (VK_KHR_timeline_semaphore / Vulkan 1.2).
// Manual: generator cannot handle handle+ptr+u64 signature.
func (c *Commands) WaitSemaphores(device Device, pWaitInfo *SemaphoreWaitInfo, timeout uint64) Result {
	if c.waitSemaphores == nil {
		return ErrorInitializationFailed
	}
	r := C.gvk_r_hpq(C.uintptr_t(uintptr(c.waitSemaphores)), C.uint64_t(device), (*C.uint8_t)(unsafe.Pointer(pWaitInfo)), C.uint64_t(timeout))
	return Result(r)
}

// GetBufferDeviceAddress wraps vkGetBufferDeviceAddress (Vulkan 1.2 core).
// Manual: returns VkDeviceAddress (uint64), not VkResult — unsupported by generator.
func (c *Commands) GetBufferDeviceAddress(device Device, pInfo *BufferDeviceAddressInfo) DeviceAddress {
	if c.getBufferDeviceAddress == nil {
		return 0
	}
	r := C.gvk_q_hp(C.uintptr_t(uintptr(c.getBufferDeviceAddress)), C.uint64_t(device), (*C.uint8_t)(unsafe.Pointer(pInfo)))
	return DeviceAddress(r)
}

// GetAccelerationStructureDeviceAddressKHR wraps vkGetAccelerationStructureDeviceAddressKHR.
// Manual: returns VkDeviceAddress (uint64), not VkResult — unsupported by generator.
func (c *Commands) GetAccelerationStructureDeviceAddressKHR(device Device, pInfo *AccelerationStructureDeviceAddressInfoKHR) DeviceAddress {
	if c.getAccelerationStructureDeviceAddressKHR == nil {
		return 0
	}
	r := C.gvk_q_hp(C.uintptr_t(uintptr(c.getAccelerationStructureDeviceAddressKHR)), C.uint64_t(device), (*C.uint8_t)(unsafe.Pointer(pInfo)))
	return DeviceAddress(r)
}

// CmdBuildAccelerationStructuresKHR wraps vkCmdBuildAccelerationStructuresKHR.
// Manual: generator cannot handle void(handle, u32, ptr, ptr) with ptr-to-ptr second pointer.
func (c *Commands) CmdBuildAccelerationStructuresKHR(
	commandBuffer CommandBuffer,
	infoCount uint32,
	pInfos *AccelerationStructureBuildGeometryInfoKHR,
	ppBuildRangeInfos **AccelerationStructureBuildRangeInfoKHR,
) {
	if c.cmdBuildAccelerationStructuresKHR == nil {
		return
	}
	C.gvk_v_hipp(C.uintptr_t(uintptr(c.cmdBuildAccelerationStructuresKHR)), C.uint64_t(commandBuffer), C.uint32_t(infoCount), (*C.uint8_t)(unsafe.Pointer(pInfos)), (*C.uint8_t)(unsafe.Pointer(ppBuildRangeInfos)))
}

// CmdWriteAccelerationStructuresPropertiesKHR wraps vkCmdWriteAccelerationStructuresPropertiesKHR.
// Manual: generator cannot handle void(handle, u32, ptr, u32, handle, u32).
func (c *Commands) CmdWriteAccelerationStructuresPropertiesKHR(
	commandBuffer CommandBuffer,
	accelerationStructureCount uint32,
	pAccelerationStructures *AccelerationStructureKHR,
	queryType QueryType,
	queryPool QueryPool,
	firstQuery uint32,
) {
	if c.cmdWriteAccelerationStructuresPropertiesKHR == nil {
		return
	}
	C.gvk_v_hipihi(C.uintptr_t(uintptr(c.cmdWriteAccelerationStructuresPropertiesKHR)), C.uint64_t(commandBuffer), C.uint32_t(accelerationStructureCount), (*C.uint8_t)(unsafe.Pointer(pAccelerationStructures)), C.uint32_t(queryType), C.uint64_t(queryPool), C.uint32_t(firstQuery))
}

// GetAccelerationStructureBuildSizesKHR wraps vkGetAccelerationStructureBuildSizesKHR.
// Manual: generator cannot handle void(handle, handle, ptr, ptr, ptr) with enum second arg.
func (c *Commands) GetAccelerationStructureBuildSizesKHR(
	device Device,
	buildType AccelerationStructureBuildTypeKHR,
	pBuildInfo *AccelerationStructureBuildGeometryInfoKHR,
	pMaxPrimitiveCounts *uint32,
	pSizeInfo *AccelerationStructureBuildSizesInfoKHR,
) {
	if c.getAccelerationStructureBuildSizesKHR == nil {
		return
	}
	C.gvk_v_hippp(C.uintptr_t(uintptr(c.getAccelerationStructureBuildSizesKHR)), C.uint64_t(device), C.uint32_t(buildType), (*C.uint8_t)(unsafe.Pointer(pBuildInfo)), (*C.uint8_t)(unsafe.Pointer(pMaxPrimitiveCounts)), (*C.uint8_t)(unsafe.Pointer(pSizeInfo)))
}
