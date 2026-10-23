//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

import (
	"fmt"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// Adapter implements hal.Adapter for Vulkan.
type Adapter struct {
	instance       *Instance
	physicalDevice vk.PhysicalDevice
	properties     vk.PhysicalDeviceProperties
	features       vk.PhysicalDeviceFeatures
}

// Open creates a logical device with the requested features and limits.
func (a *Adapter) Open(_ gputypes.Features, _ gputypes.Limits) (hal.OpenDevice, error) {
	return a.open(nil)
}

// open creates a logical device, optionally constraining it to one queue
// family. Surface-qualified adapters use the constrained path so the queue
// selected during the surface query is the same queue passed into Open.
func (a *Adapter) open(requestedQueueFamily *uint32) (hal.OpenDevice, error) {
	// Find queue families
	var queueFamilyCount uint32
	vkGetPhysicalDeviceQueueFamilyProperties(a.instance, a.physicalDevice, &queueFamilyCount, nil)

	if queueFamilyCount == 0 {
		return hal.OpenDevice{}, fmt.Errorf("vulkan: no queue families found")
	}

	queueFamilies := make([]vk.QueueFamilyProperties, queueFamilyCount)
	vkGetPhysicalDeviceQueueFamilyProperties(a.instance, a.physicalDevice, &queueFamilyCount, &queueFamilies[0])

	if queueFamilyCount < uint32(len(queueFamilies)) {
		queueFamilies = queueFamilies[:queueFamilyCount]
	}
	graphicsFamily, err := selectGraphicsQueueFamily(queueFamilies, requestedQueueFamily)
	if err != nil {
		return hal.OpenDevice{}, err
	}

	// Query supported device extensions to enable optional features.
	available := map[string]bool{}
	{
		var extCount uint32
		a.instance.cmds.EnumerateDeviceExtensionProperties(a.physicalDevice, nil, &extCount, nil)
		if extCount > 0 {
			extProps := make([]vk.ExtensionProperties, extCount)
			a.instance.cmds.EnumerateDeviceExtensionProperties(a.physicalDevice, nil, &extCount, &extProps[0])
			for i := range extProps[:extCount] {
				available[cStringToGo(extProps[i].ExtensionName[:])] = true
			}
		}
	}
	hasIncrementalPresent := available["VK_KHR_incremental_present"]

	// VK_KHR_swapchain where the device has it: one without can still compute.
	var extensions []string
	if available["VK_KHR_swapchain"] {
		extensions = append(extensions, "VK_KHR_swapchain\x00")
	}
	// Optional: VK_KHR_incremental_present for damage-aware presentation.
	// Allows chaining VkPresentRegionsKHR into VkPresentInfoKHR.PNext
	// so the compositor can skip recompositing unchanged pixels.
	if hasIncrementalPresent {
		extensions = append(extensions, "VK_KHR_incremental_present\x00")
	}
	// Optional: exporting memory, for another API such as OpenGL to fill
	// buffers the device reads (see Interop).
	if ext := externalMemoryExtension(); available[ext] {
		extensions = append(extensions, ext+"\x00")
	}
	if available["VK_KHR_portability_subset"] {
		extensions = append(extensions, "VK_KHR_portability_subset\x00")
	}
	storeOpNone := a.apiVersion() >= vkMakeVersion(1, 3, 0)
	for _, ext := range []string{"VK_KHR_load_store_op_none", "VK_EXT_load_store_op_none", "VK_QCOM_render_pass_store_ops"} {
		if !storeOpNone && available[ext] {
			extensions = append(extensions, ext+"\x00")
			storeOpNone = true
		}
	}
	extensionPtrs := make([]*byte, len(extensions))
	for i, ext := range extensions {
		extensionPtrs[i] = unsafe.StringData(ext)
	}

	// Detect timeline semaphore support (VK-IMPL-001), and on Vulkan 1.3 what
	// a video decoder sharing the device needs.
	// Query via PhysicalDeviceVulkan12Features with PNext chain on GetPhysicalDeviceFeatures2.
	// The 1.2 and 1.3 feature structs only where the device is that version:
	// newer core functionality is not to be used on an older device.
	hasTimelineSemaphore := false
	var (
		has11 = vk.PhysicalDeviceVulkan11Features{SType: vk.StructureTypePhysicalDeviceVulkan11Features}
		has12 = vk.PhysicalDeviceVulkan12Features{SType: vk.StructureTypePhysicalDeviceVulkan12Features}
		has13 = vk.PhysicalDeviceVulkan13Features{SType: vk.StructureTypePhysicalDeviceVulkan13Features}
	)
	if a.instance.cmds.HasPhysicalDeviceFeatures2() && a.apiVersion() >= vkMakeVersion(1, 2, 0) {
		features2 := vk.PhysicalDeviceFeatures2{
			SType: vk.StructureTypePhysicalDeviceFeatures2,
			PNext: (*uintptr)(unsafe.Pointer(&has12)),
		}
		if a.apiVersion() >= vkMakeVersion(1, 3, 0) {
			features2.PNext = (*uintptr)(unsafe.Pointer(&has11))
			has11.PNext = (*uintptr)(unsafe.Pointer(&has12))
			has12.PNext = (*uintptr)(unsafe.Pointer(&has13))
		}
		a.instance.cmds.GetPhysicalDeviceFeatures2(a.physicalDevice, &features2)
		hasTimelineSemaphore = usesTimelineSemaphore(a.properties.VendorID, has12.TimelineSemaphore)
	}
	video := videoDecodeSetup(a.apiVersion(), available, queueFamilies, graphicsFamily, &has11, &has12, &has13)

	// The graphics queue, and with video a compute and a video decode queue
	// gowgpu leaves to the decoder.
	queuePriority := float32(1.0)
	queueCreateInfos := []vk.DeviceQueueCreateInfo{{
		SType:            vk.StructureTypeDeviceQueueCreateInfo,
		QueueFamilyIndex: graphicsFamily,
		QueueCount:       1,
		PQueuePriorities: &queuePriority,
	}}
	if video != nil {
		for _, family := range []uint32{video.compute, video.decode} {
			info := queueCreateInfos[0]
			info.QueueFamilyIndex = family
			queueCreateInfos = append(queueCreateInfos, info)
		}
		extensions = append(extensions, video.extensions...)
		extensionPtrs = extensionPtrs[:0]
		for _, ext := range extensions {
			extensionPtrs = append(extensionPtrs, unsafe.StringData(ext))
		}
	}

	// Device create info
	deviceCreateInfo := vk.DeviceCreateInfo{
		SType:                 vk.StructureTypeDeviceCreateInfo,
		QueueCreateInfoCount:  uint32(len(queueCreateInfos)),
		PQueueCreateInfos:     &queueCreateInfos[0],
		EnabledExtensionCount: uint32(len(extensions)),
		PEnabledFeatures:      &a.features,
	}
	if len(extensionPtrs) > 0 {
		deviceCreateInfo.PpEnabledExtensionNames = &extensionPtrs[0]
	}

	// Enable timeline semaphore feature if supported.
	// Vulkan 1.2 requires explicitly enabling features via PNext chain.
	var (
		vulkan11Enable = vk.PhysicalDeviceVulkan11Features{SType: vk.StructureTypePhysicalDeviceVulkan11Features}
		vulkan12Enable = vk.PhysicalDeviceVulkan12Features{SType: vk.StructureTypePhysicalDeviceVulkan12Features}
		vulkan13Enable = vk.PhysicalDeviceVulkan13Features{SType: vk.StructureTypePhysicalDeviceVulkan13Features}
	)
	if hasTimelineSemaphore {
		vulkan12Enable.TimelineSemaphore = vk.Bool32(vk.True)
	}
	if has12.DrawIndirectCount != 0 {
		vulkan12Enable.DrawIndirectCount = vk.Bool32(vk.True)
	}
	if hasTimelineSemaphore || has12.DrawIndirectCount != 0 {
		deviceCreateInfo.PNext = (*uintptr)(unsafe.Pointer(&vulkan12Enable))
	}
	if video != nil {
		vulkan11Enable.SamplerYcbcrConversion = vk.Bool32(vk.True)
		vulkan13Enable.Synchronization2 = vk.Bool32(vk.True)
		deviceCreateInfo.PNext = (*uintptr)(unsafe.Pointer(&vulkan11Enable))
		vulkan11Enable.PNext = (*uintptr)(unsafe.Pointer(&vulkan12Enable))
		vulkan12Enable.PNext = (*uintptr)(unsafe.Pointer(&vulkan13Enable))
	}

	var device vk.Device
	result := vkCreateDevice(a.instance, a.physicalDevice, &deviceCreateInfo, nil, &device)
	if result != vk.Success {
		return hal.OpenDevice{}, fmt.Errorf("vulkan: vkCreateDevice failed: %d", result)
	}

	// Load device-level commands
	var deviceCmds vk.Commands
	if err := deviceCmds.LoadDevice(device); err != nil {
		vkDestroyDevice(device, nil)
		return hal.OpenDevice{}, fmt.Errorf("vulkan: failed to load device commands: %w", err)
	}
	if a.instance.debugEnabled {
		deviceCmds.LoadDebugUtils(&a.instance.cmds)
	}

	// Get queue handle
	var queue vk.Queue
	vkGetDeviceQueue(&deviceCmds, device, graphicsFamily, 0, &queue)

	dev := &Device{
		handle:                     device,
		physicalDevice:             a.physicalDevice,
		instance:                   a.instance,
		graphicsFamily:             graphicsFamily,
		cmds:                       &deviceCmds,
		renderPassCache:            NewRenderPassCache(device, &deviceCmds),
		descriptorAllocator:        NewDescriptorAllocator(device, &deviceCmds, DefaultDescriptorAllocatorConfig()),
		supportsMultiDrawIndirect:  a.features.MultiDrawIndirect != 0,
		supportsDrawIndirectCount:  has12.DrawIndirectCount != 0,
		maxDrawIndirectCount:       a.properties.Limits.MaxDrawIndirectCount,
		supportsIncrementalPresent: hasIncrementalPresent,
		storeOpNone:                storeOpNone,
		alignedFills:               a.properties.VendorID == 0x10DE,
		shaderOptions:              shaderOptions(a.apiVersion()),
	}
	dev.depth24, dev.depth24Stencil8 = depth24Formats(a.supportsDepthAttachment)
	dev.stencil8 = stencil8Format(a.supportsDepthAttachment, dev.depth24Stencil8)

	// Initialize synchronization fence (VK-IMPL-001 / VK-IMPL-003).
	// Prefer timeline semaphore (Vulkan 1.2+); fall back to fencePool of binary
	// VkFences on older drivers. Either way, dev.timelineFence is always set so
	// the rest of the codebase can use a single path without nil checks.
	if hasTimelineSemaphore {
		tlFence, err := initTimelineFence(dev.cmds, dev.handle)
		if err != nil {
			hal.Logger().Warn("vulkan: timeline semaphore feature reported but init failed, using binary fence pool",
				"error", err,
			)
			dev.timelineFence = initBinaryFence()
		} else {
			dev.timelineFence = tlFence
			hal.Logger().Info("vulkan: using timeline semaphore fence (VK-IMPL-001)")
		}
	} else {
		dev.timelineFence = initBinaryFence()
	}

	// Initialize memory allocator
	if err := dev.initAllocator(); err != nil {
		dev.timelineFence.destroy(dev.cmds, dev.handle)
		vkDestroyDevice(device, nil)
		return hal.OpenDevice{}, fmt.Errorf("vulkan: failed to initialize allocator: %w", err)
	}

	dev.initPipelineCache(&a.properties)

	// VK-SYNC-001: Create relay semaphores for GPU-side submission ordering.
	// This ensures consecutive vkQueueSubmit calls execute in order on the GPU,
	// which is required by the wgpu_hal Queue trait but not guaranteed by Vulkan.
	relay, err := newRelaySemaphores(dev.cmds, dev.handle)
	if err != nil {
		dev.allocator.Destroy()
		dev.timelineFence.destroy(dev.cmds, dev.handle)
		vkDestroyDevice(device, nil)
		return hal.OpenDevice{}, fmt.Errorf("vulkan: failed to create relay semaphores: %w", err)
	}

	q := &Queue{
		handle:      queue,
		device:      dev,
		familyIndex: graphicsFamily,
		relay:       relay,
	}

	// Store queue reference in device for swapchain synchronization
	dev.queue = q

	vulkan11Enable.PNext, vulkan12Enable.PNext = nil, nil
	dev.interop = &Interop{
		GetInstanceProcAddr: uintptr(vk.GetInstanceProcAddr(0, "vkGetInstanceProcAddr")),
		Instance:            a.instance.handle,
		PhysicalDevice:      a.physicalDevice,
		Device:              device,
		InstanceExtensions:  a.instance.extensions,
		DeviceExtensions:    trimNUL(extensions),
		Features:            a.features,
		Features11:          vulkan11Enable,
		Features12:          vulkan12Enable,
		Features13:          vulkan13Enable,
	}
	if video != nil {
		dev.interop.QueueFamilies = []InteropQueueFamily{
			{video.compute, 1, queueFamilies[video.compute].QueueFlags},
			{video.decode, 1, queueFamilies[video.decode].QueueFlags},
			{graphicsFamily, 1, vk.QueueFlags(vk.QueueGraphicsBit)},
		}
	}

	syncMode := "binary fence pool (VK-IMPL-003)"
	if dev.timelineFence.isTimeline {
		syncMode = "timeline semaphore (VK-IMPL-001)"
	}
	hal.Logger().Info("vulkan: device created",
		"name", cStringToGo(a.properties.DeviceName[:]),
		"queueFamily", graphicsFamily,
		"syncMode", syncMode,
	)

	return hal.OpenDevice{
		Device: dev,
		Queue:  q,
	}, nil
}

// apiVersion is the Vulkan version of a device of the adapter: the lower of
// the device's and the instance's.
func (a *Adapter) apiVersion() uint32 {
	if a.instance != nil && a.instance.apiVersion != 0 {
		return min(a.properties.ApiVersion, a.instance.apiVersion)
	}
	return a.properties.ApiVersion
}

// vendorVeriSilicon is VK_VENDOR_ID_VSI.
const vendorVeriSilicon = 0x10002

// usesTimelineSemaphore reports whether a device of vendor whose
// timelineSemaphore feature is supported gets the timeline semaphore fence.
// VeriSilicon's proprietary driver, as on NXP's i.MX 8, advertises timeline
// semaphores but crashes in vkQueueSubmit with them, wgpu-hal reports, and
// calls itself RADV, so only the vendor ID tells it.
func usesTimelineSemaphore(vendor uint32, supported vk.Bool32) bool {
	return supported != 0 && vendor != vendorVeriSilicon
}

// selectGraphicsQueueFamily preserves the exact family chosen during surface
// qualification, while keeping the ordinary headless path first-graphics.
func selectGraphicsQueueFamily(families []vk.QueueFamilyProperties, requested *uint32) (uint32, error) {
	if requested != nil {
		index := *requested
		if index >= uint32(len(families)) {
			return 0, fmt.Errorf("vulkan: requested queue family %d is unavailable", index)
		}
		family := families[index]
		if family.QueueCount == 0 {
			return 0, fmt.Errorf("vulkan: requested queue family %d has no queues", index)
		}
		if family.QueueFlags&vk.QueueFlags(vk.QueueGraphicsBit) == 0 {
			return 0, fmt.Errorf("vulkan: requested queue family %d has no graphics queue", index)
		}
		return index, nil
	}
	for index, family := range families {
		if family.QueueCount > 0 && family.QueueFlags&vk.QueueFlags(vk.QueueGraphicsBit) != 0 {
			return uint32(index), nil
		}
	}
	return 0, fmt.Errorf("vulkan: no graphics queue family found")
}

// supportsDepthAttachment reports whether the device can render depth to an
// optimally tiled image of format.
func (a *Adapter) supportsDepthAttachment(format vk.Format) bool {
	var props vk.FormatProperties
	a.instance.cmds.GetPhysicalDeviceFormatProperties(a.physicalDevice, format, &props)
	return props.OptimalTilingFeatures&vk.FormatFeatureFlags(vk.FormatFeatureDepthStencilAttachmentBit) != 0
}

// vkFormat is the Vulkan format of format on this device, see
// depth24Formats and stencil8Format.
func (a *Adapter) vkFormat(format gputypes.TextureFormat) vk.Format {
	switch format {
	case gputypes.TextureFormatDepth24Plus, gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureFormatStencil8:
		depth, depthStencil := depth24Formats(a.supportsDepthAttachment)
		switch format {
		case gputypes.TextureFormatDepth24Plus:
			return depth
		case gputypes.TextureFormatStencil8:
			return stencil8Format(a.supportsDepthAttachment, depthStencil)
		}
		return depthStencil
	}
	return textureFormatToVk(format)
}

// TextureFormatCapabilities returns capabilities for a texture format.
func (a *Adapter) TextureFormatCapabilities(format gputypes.TextureFormat) hal.TextureFormatCapabilities {
	vkFormat := a.vkFormat(format)
	if vkFormat == vk.FormatUndefined {
		return hal.TextureFormatCapabilities{}
	}

	var props vk.FormatProperties
	a.instance.cmds.GetPhysicalDeviceFormatProperties(a.physicalDevice, vkFormat, &props)

	// Use OptimalTilingFeatures for texture capabilities (most common use case)
	flags := vkFormatFeaturesToHAL(props.OptimalTilingFeatures)

	// Check multisampling support via image format properties
	// TODO: Query vkGetPhysicalDeviceImageFormatProperties for accurate multisample support
	// For now, assume common formats support multisampling if they support rendering
	if flags&hal.TextureFormatCapabilityRenderAttachment != 0 {
		flags |= hal.TextureFormatCapabilityMultisample | hal.TextureFormatCapabilityMultisampleResolve
	}

	return hal.TextureFormatCapabilities{
		Flags: flags,
	}
}

// SurfaceCapabilities returns surface capabilities.
func (a *Adapter) SurfaceCapabilities(surface hal.Surface) *hal.SurfaceCapabilities {
	vkSurface, ok := surface.(*Surface)
	if !ok || vkSurface == nil || vkSurface.handle == 0 || vkSurface.instance == nil || vkSurface.instance != a.instance {
		return nil
	}

	snapshot, err := a.querySurfaceSnapshot(vkSurface)
	if err != nil {
		return nil
	}
	return cloneSurfaceCapabilities(snapshot.public)
}

// surfaceSnapshot keeps the exact Vulkan format/color-space pairs alongside
// the public projection. The public HAL shape predates color spaces, so the
// raw pairs stay private to the Vulkan adapter and are never silently reduced
// to a fabricated format list.
type surfaceSnapshot struct {
	capabilities vk.SurfaceCapabilitiesKHR
	formats      []vk.SurfaceFormatKHR
	presentModes []vk.PresentModeKHR
	public       hal.SurfaceCapabilities
}

// qualifiedAdapter is a request-local Vulkan adapter. It carries the queue
// family proven to support both graphics and presentation for one surface and
// opens the underlying physical device with that family unchanged.
type qualifiedAdapter struct {
	base        *Adapter
	surface     *Surface
	queueFamily uint32
	snapshot    surfaceSnapshot
}

func (a *qualifiedAdapter) Open(_ gputypes.Features, _ gputypes.Limits) (hal.OpenDevice, error) {
	return a.base.open(&a.queueFamily)
}

func (a *qualifiedAdapter) TextureFormatCapabilities(format gputypes.TextureFormat) hal.TextureFormatCapabilities {
	return a.base.TextureFormatCapabilities(format)
}

func (a *qualifiedAdapter) SurfaceCapabilities(surface hal.Surface) *hal.SurfaceCapabilities {
	vkSurface, ok := surface.(*Surface)
	if !ok {
		return nil
	}
	if vkSurface == a.surface {
		return cloneSurfaceCapabilities(a.snapshot.public)
	}
	// A later surface on the same device: a second window, or the new window
	// Android hands an app that comes back to the foreground.
	return a.base.SurfaceCapabilities(surface)
}

// Destroy is intentionally a no-op. The cached physical adapter owns no
// Vulkan object, and the core instance retains ownership of the base adapter.
func (a *qualifiedAdapter) Destroy() {}

// QualifySurface selects a same-family graphics+present queue and records a
// checked surface capability snapshot without mutating the cached adapter.
func (a *Adapter) QualifySurface(surface hal.Surface) (hal.Adapter, error) {
	vkSurface, ok := surface.(*Surface)
	if !ok || vkSurface == nil || vkSurface.handle == 0 {
		return nil, fmt.Errorf("vulkan: invalid surface for adapter qualification")
	}
	if vkSurface.instance == nil || vkSurface.instance != a.instance {
		return nil, fmt.Errorf("vulkan: surface belongs to a different instance")
	}

	queueFamilies, err := a.queueFamilies()
	if err != nil {
		return nil, err
	}
	queueFamily, err := a.presentGraphicsQueueFamily(vkSurface, queueFamilies)
	if err != nil {
		return nil, err
	}

	snapshot, err := a.querySurfaceSnapshot(vkSurface)
	if err != nil {
		return nil, err
	}

	return &qualifiedAdapter{
		base:        a,
		surface:     vkSurface,
		queueFamily: queueFamily,
		snapshot:    snapshot,
	}, nil
}

func (a *Adapter) queueFamilies() ([]vk.QueueFamilyProperties, error) {
	var count uint32
	vkGetPhysicalDeviceQueueFamilyProperties(a.instance, a.physicalDevice, &count, nil)
	if count == 0 {
		return nil, fmt.Errorf("vulkan: no queue families found")
	}

	families := make([]vk.QueueFamilyProperties, count)
	vkGetPhysicalDeviceQueueFamilyProperties(a.instance, a.physicalDevice, &count, &families[0])
	if count == 0 {
		return nil, fmt.Errorf("vulkan: no queue families found")
	}
	if count < uint32(len(families)) {
		families = families[:count]
	}
	return families, nil
}

func (a *Adapter) presentGraphicsQueueFamily(surface *Surface, families []vk.QueueFamilyProperties) (uint32, error) {
	supports := make([]bool, len(families))
	for index, family := range families {
		if family.QueueCount == 0 || family.QueueFlags&vk.QueueFlags(vk.QueueGraphicsBit) == 0 {
			continue
		}

		var supported vk.Bool32
		result := a.instance.cmds.GetPhysicalDeviceSurfaceSupportKHR(
			a.physicalDevice, uint32(index), surface.handle, &supported)
		if result != vk.Success {
			return 0, fmt.Errorf("vulkan: vkGetPhysicalDeviceSurfaceSupportKHR(queueFamily=%d) failed: %d", index, result)
		}
		supports[index] = supported != 0
	}
	return selectPresentGraphicsQueueFamily(families, supports)
}

func selectPresentGraphicsQueueFamily(families []vk.QueueFamilyProperties, supports []bool) (uint32, error) {
	if len(supports) != len(families) {
		return 0, fmt.Errorf("vulkan: queue family support query is incomplete")
	}
	for index, family := range families {
		if family.QueueCount > 0 && family.QueueFlags&vk.QueueFlags(vk.QueueGraphicsBit) != 0 && supports[index] {
			return uint32(index), nil
		}
	}
	return 0, fmt.Errorf("vulkan: no graphics queue family supports presentation for surface")
}

func (a *Adapter) querySurfaceSnapshot(surface *Surface) (surfaceSnapshot, error) {
	if surface == nil || surface.handle == 0 {
		return surfaceSnapshot{}, fmt.Errorf("vulkan: invalid surface")
	}

	var capabilities vk.SurfaceCapabilitiesKHR
	result := a.instance.cmds.GetPhysicalDeviceSurfaceCapabilitiesKHR(
		a.physicalDevice, surface.handle, &capabilities)
	if result != vk.Success {
		return surfaceSnapshot{}, fmt.Errorf("vulkan: vkGetPhysicalDeviceSurfaceCapabilitiesKHR failed: %d", result)
	}

	formats, err := querySurfaceFormats(a.instance, a.physicalDevice, surface.handle)
	if err != nil {
		return surfaceSnapshot{}, err
	}

	presentModes, err := queryPresentModes(a.instance, a.physicalDevice, surface.handle)
	if err != nil {
		return surfaceSnapshot{}, err
	}
	return makeSurfaceSnapshot(capabilities, formats, presentModes)
}

func querySurfaceFormats(instance *Instance, device vk.PhysicalDevice, surface vk.SurfaceKHR) ([]vk.SurfaceFormatKHR, error) {
	return querySurfaceFormatsWith(func(count *uint32, formats *vk.SurfaceFormatKHR) vk.Result {
		return instance.cmds.GetPhysicalDeviceSurfaceFormatsKHR(device, surface, count, formats)
	})
}

func querySurfaceFormatsWith(query func(count *uint32, formats *vk.SurfaceFormatKHR) vk.Result) ([]vk.SurfaceFormatKHR, error) {
	return queryOptionalSurfaceValues("vkGetPhysicalDeviceSurfaceFormatsKHR", query)
}

func queryOptionalSurfaceValues[T any](operation string, query func(count *uint32, values *T) vk.Result) ([]T, error) {
	for attempt := 0; attempt < 2; attempt++ {
		var count uint32
		result := query(&count, nil)
		if result != vk.Success && result != vk.Incomplete {
			return nil, fmt.Errorf("vulkan: %s count failed: %d", operation, result)
		}
		if count == 0 {
			return []T{}, nil
		}

		values := make([]T, count)
		returned := count
		result = query(&returned, &values[0])
		if result != vk.Success && result != vk.Incomplete {
			return nil, fmt.Errorf("vulkan: %s failed: %d", operation, result)
		}
		if result == vk.Incomplete || returned > uint32(len(values)) {
			continue
		}
		return values[:returned], nil
	}
	return nil, fmt.Errorf("vulkan: %s returned an unstable count", operation)
}

func queryPresentModes(instance *Instance, device vk.PhysicalDevice, surface vk.SurfaceKHR) ([]vk.PresentModeKHR, error) {
	return queryPresentModesWith(func(count *uint32, modes *vk.PresentModeKHR) vk.Result {
		return instance.cmds.GetPhysicalDeviceSurfacePresentModesKHR(device, surface, count, modes)
	})
}

func queryPresentModesWith(query func(count *uint32, modes *vk.PresentModeKHR) vk.Result) ([]vk.PresentModeKHR, error) {
	return queryOptionalSurfaceValues("vkGetPhysicalDeviceSurfacePresentModesKHR", query)
}

func makeSurfaceSnapshot(capabilities vk.SurfaceCapabilitiesKHR, formats []vk.SurfaceFormatKHR, presentModes []vk.PresentModeKHR) (surfaceSnapshot, error) {
	if len(formats) == 0 {
		return surfaceSnapshot{}, fmt.Errorf("vulkan: surface returned no formats")
	}
	if len(presentModes) == 0 {
		return surfaceSnapshot{}, fmt.Errorf("vulkan: surface returned no present modes")
	}

	alphaModes := vkCompositeAlphaToHALChecked(capabilities.SupportedCompositeAlpha)
	if len(alphaModes) == 0 {
		return surfaceSnapshot{}, fmt.Errorf("vulkan: surface returned no supported composite alpha modes")
	}

	public := hal.SurfaceCapabilities{
		Formats:      make([]gputypes.TextureFormat, 0, len(formats)),
		PresentModes: make([]gputypes.PresentMode, 0, len(presentModes)),
		AlphaModes:   alphaModes,
	}
	for _, format := range formats {
		if textureFormat := textureFormatForSurfacePair(format); textureFormat != gputypes.TextureFormatUndefined {
			public.Formats = append(public.Formats, textureFormat)
		}
	}
	for _, mode := range presentModes {
		if halMode, ok := vkPresentModeToHALChecked(mode); ok {
			public.PresentModes = append(public.PresentModes, halMode)
		}
	}
	if len(public.PresentModes) == 0 {
		return surfaceSnapshot{}, fmt.Errorf("vulkan: surface returned no supported present modes")
	}
	if len(public.Formats) == 0 {
		return surfaceSnapshot{}, fmt.Errorf("vulkan: surface returned no supported formats")
	}

	return surfaceSnapshot{
		capabilities: capabilities,
		formats:      append([]vk.SurfaceFormatKHR(nil), formats...),
		presentModes: append([]vk.PresentModeKHR(nil), presentModes...),
		public:       public,
	}, nil
}

func vkPresentModeToHALChecked(mode vk.PresentModeKHR) (gputypes.PresentMode, bool) {
	switch mode {
	case vk.PresentModeImmediateKhr:
		return hal.PresentModeImmediate, true
	case vk.PresentModeMailboxKhr:
		return hal.PresentModeMailbox, true
	case vk.PresentModeFifoKhr:
		return hal.PresentModeFifo, true
	case vk.PresentModeFifoRelaxedKhr:
		return hal.PresentModeFifoRelaxed, true
	default:
		return 0, false
	}
}

func vkCompositeAlphaToHALChecked(flags vk.CompositeAlphaFlagsKHR) []gputypes.CompositeAlphaMode {
	var modes []gputypes.CompositeAlphaMode
	if vk.Flags(flags)&vk.Flags(vk.CompositeAlphaOpaqueBitKhr) != 0 {
		modes = append(modes, hal.CompositeAlphaModeOpaque)
	}
	if vk.Flags(flags)&vk.Flags(vk.CompositeAlphaPreMultipliedBitKhr) != 0 {
		modes = append(modes, hal.CompositeAlphaModePremultiplied)
	}
	if vk.Flags(flags)&vk.Flags(vk.CompositeAlphaPostMultipliedBitKhr) != 0 {
		modes = append(modes, hal.CompositeAlphaModeUnpremultiplied)
	}
	if vk.Flags(flags)&vk.Flags(vk.CompositeAlphaInheritBitKhr) != 0 {
		modes = append(modes, hal.CompositeAlphaModeInherit)
	}
	return modes
}

func cloneSurfaceCapabilities(capabilities hal.SurfaceCapabilities) *hal.SurfaceCapabilities {
	return &hal.SurfaceCapabilities{
		Formats:      append([]gputypes.TextureFormat(nil), capabilities.Formats...),
		PresentModes: append([]gputypes.PresentMode(nil), capabilities.PresentModes...),
		AlphaModes:   append([]gputypes.CompositeAlphaMode(nil), capabilities.AlphaModes...),
	}
}

// Destroy releases the adapter.
func (a *Adapter) Destroy() {
	// Adapter doesn't own resources
}

// Vulkan function wrappers using Commands methods

func vkGetPhysicalDeviceQueueFamilyProperties(i *Instance, device vk.PhysicalDevice, count *uint32, props *vk.QueueFamilyProperties) {
	i.cmds.GetPhysicalDeviceQueueFamilyProperties(device, count, props)
}

func vkCreateDevice(i *Instance, physicalDevice vk.PhysicalDevice, createInfo *vk.DeviceCreateInfo, allocator *vk.AllocationCallbacks, device *vk.Device) vk.Result {
	return i.cmds.CreateDevice(physicalDevice, createInfo, allocator, device)
}

func vkGetDeviceQueue(cmds *vk.Commands, device vk.Device, queueFamilyIndex, queueIndex uint32, queue *vk.Queue) {
	cmds.GetDeviceQueue(device, queueFamilyIndex, queueIndex, queue)
}

func vkDestroyDevice(device vk.Device, allocator *vk.AllocationCallbacks) { //nolint:unparam // allocator kept for Vulkan API signature parity
	// Get vkDestroyDevice function pointer directly since device commands
	// may not be available when destroying the device
	proc := vk.GetDeviceProcAddr(device, "vkDestroyDevice")
	vk.DestroyDevice(proc, device, allocator)
}
