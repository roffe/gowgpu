// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin && !(js && wasm)

package metal

import (
	"sync"

	"github.com/roffe/wgpu/hal"
)

var (
	initOnce sync.Once
	errInit  error
)

// Init initializes the Metal backend. libobjc, Metal and QuartzCore are
// linked, so all that is left is registering the common selectors.
func Init() error {
	initOnce.Do(func() {
		errInit = doInit()
	})
	return errInit
}

func doInit() error {
	if err := initObjCRuntime(); err != nil {
		return err
	}
	preRegisterSelectors()
	initBlockSupport()
	hal.Logger().Info("metal: framework initialized")
	return nil
}

func preRegisterSelectors() {
	// Common selectors used throughout
	commonSelectors := []string{
		"alloc", "init", "new", "retain", "release", "autorelease",
		"name", "setLabel:", "label",
		"contents", "length", "count",
		"localizedDescription", // NSError
		// MTLDevice
		"newBufferWithLength:options:",
		"newTextureWithDescriptor:",
		"newSamplerStateWithDescriptor:",
		"newCommandQueue",
		"newRenderPipelineStateWithDescriptor:error:",
		"setSupportIndirectCommandBuffers:",
		"newLibraryWithSource:options:error:",
		"newFunctionWithName:",
		"supportsFamily:",
		// MTLCommandQueue
		"commandBuffer",
		"commandBufferWithUnretainedReferences",
		// MTLCommandBuffer
		"renderCommandEncoderWithDescriptor:",
		"blitCommandEncoder",
		"computeCommandEncoder",
		"commit",
		"waitUntilCompleted",
		"waitUntilScheduled",
		"addCompletedHandler:",
		"presentDrawable:",
		// CAMetalDrawable
		"present",
		// MTLRenderCommandEncoder
		"setRenderPipelineState:",
		"setVertexBuffer:offset:atIndex:",
		"setFragmentBuffer:offset:atIndex:",
		"setVertexTexture:atIndex:",
		"setFragmentTexture:atIndex:",
		"setViewport:",
		"setScissorRect:",
		"drawPrimitives:vertexStart:vertexCount:",
		"drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
		"endEncoding",
		// MTLComputeCommandEncoder
		"setBytes:length:atIndex:",
		// CAMetalLayer
		"setDevice:",
		"setPixelFormat:",
		"setFramebufferOnly:",
		"setDrawableSize:",
		"setPresentsWithTransaction:",
		"setMaximumDrawableCount:",
		"setDisplaySyncEnabled:",
		"nextDrawable",
		// MTLSharedEvent / MTLSharedEventListener
		"newSharedEvent",
		"signaledValue",
		"setSignaledValue:",
		"encodeSignalEvent:value:",
		"notifyListener:atValue:block:",
		// MTLTexture
		"width", "height", "depth",
		"pixelFormat", "textureType",
		"replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:",
		"newTextureViewWithPixelFormat:",
		// Descriptors
		"setWidth:", "setHeight:", "setDepth:", "setArrayLength:",
		"setPixelFormat:", "setTextureType:", "setUsage:",
		"setStorageMode:", "setMipmapLevelCount:", "setSampleCount:",
		// MTLRenderPipelineDescriptor
		"setVertexFunction:", "setFragmentFunction:",
		"colorAttachments", "objectAtIndexedSubscript:",
		"setWriteMask:", "setBlendingEnabled:",
		"setSourceRGBBlendFactor:", "setDestinationRGBBlendFactor:", "setRgbBlendOperation:",
		"setSourceAlphaBlendFactor:", "setDestinationAlphaBlendFactor:", "setAlphaBlendOperation:",
	}
	for _, sel := range commonSelectors {
		RegisterSelector(sel)
	}
}

// CreateSystemDefaultDevice returns the default Metal device.
func CreateSystemDefaultDevice() ID {
	return cDefaultDevice()
}

// CopyAllDevices returns all Metal devices.
// Returns the default device alone where MTLCopyAllDevices does not exist (iOS).
func CopyAllDevices() []ID {
	nsArray := cCopyAllDevices()
	if nsArray == 0 {
		// On iOS, only default device is available
		device := CreateSystemDefaultDevice()
		if device == 0 {
			return nil
		}
		return []ID{device}
	}

	count := MsgSendUint(nsArray, Sel("count"))
	if count == 0 {
		Release(nsArray)
		return nil
	}

	devices := make([]ID, count)
	for i := uint(0); i < count; i++ {
		devices[i] = MsgSend(nsArray, Sel("objectAtIndex:"), uintptr(i))
		Retain(devices[i]) // Retain since we will release the array
	}

	Release(nsArray)
	return devices
}

// DeviceName returns the name of a Metal device.
func DeviceName(device ID) string {
	if device == 0 {
		return ""
	}
	name := MsgSend(device, Sel("name"))
	return GoString(name)
}

// DeviceSupportsFamily checks if a device supports a GPU family.
func DeviceSupportsFamily(device ID, family MTLGPUFamily) bool {
	if device == 0 {
		return false
	}
	return MsgSendBool(device, Sel("supportsFamily:"), uintptr(family))
}

// DeviceRegistryID returns the IORegistry ID of the device.
func DeviceRegistryID(device ID) uint64 {
	if device == 0 {
		return 0
	}
	return uint64(MsgSend(device, Sel("registryID")))
}

// DeviceIsLowPower returns true if the device is low-power.
func DeviceIsLowPower(device ID) bool {
	if device == 0 {
		return false
	}
	return MsgSendBool(device, Sel("isLowPower"))
}

// DeviceIsHeadless returns true if the device is headless (no display).
func DeviceIsHeadless(device ID) bool {
	if device == 0 {
		return false
	}
	return MsgSendBool(device, Sel("isHeadless"))
}

// DeviceIsRemovable returns true if the device is removable (eGPU).
func DeviceIsRemovable(device ID) bool {
	if device == 0 {
		return false
	}
	return MsgSendBool(device, Sel("isRemovable"))
}

// DeviceMaxBufferLength returns the maximum buffer length for the device.
func DeviceMaxBufferLength(device ID) uint64 {
	if device == 0 {
		return 0
	}
	return uint64(MsgSend(device, Sel("maxBufferLength")))
}

// DeviceRecommendedMaxWorkingSetSize returns the recommended max working set size.
func DeviceRecommendedMaxWorkingSetSize(device ID) uint64 {
	if device == 0 {
		return 0
	}
	return uint64(MsgSend(device, Sel("recommendedMaxWorkingSetSize")))
}
