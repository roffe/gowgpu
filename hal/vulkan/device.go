//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

import (
	"encoding/binary"
	"fmt"
	"sync"
	"time"
	"unsafe"

	"github.com/gogpu/naga"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/memory"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// commandAllocator holds a recycled VkCommandPool.
// Each CreateCommandEncoder gets its own dedicated pool. The encoder manages
// its own free list of VkCommandBuffers internally (VK-CMD-001).
// After GPU completion, FreeCommandBuffer recycles the pool back for reuse.
//
// Reference: Rust wgpu-hal uses the same per-encoder pool pattern — each
// CommandEncoder owns its own VkCommandPool with internal free list.
type commandAllocator struct {
	pool vk.CommandPool
}

// encoderPool reuses CommandEncoder structs across CreateCommandEncoder calls.
// Without pooling, every frame allocates a new CommandEncoder on the heap (VK-PERF-003).
var encoderPool = sync.Pool{
	New: func() any { return &CommandEncoder{} },
}

// cmdBufferResultPool reuses CommandBuffer structs returned by EndEncoding.
// Without pooling, every EndEncoding allocates a new CommandBuffer on the heap (VK-PERF-004).
var cmdBufferResultPool = sync.Pool{
	New: func() any { return &CommandBuffer{} },
}

// computePassPool reuses ComputePassEncoder structs across BeginComputePass calls.
// Without pooling, every compute pass allocates a new encoder on the heap (VK-PERF-005).
var computePassPool = sync.Pool{
	New: func() any { return &ComputePassEncoder{} },
}

// renderPassPool reuses RenderPassEncoder structs across BeginRenderPass calls.
// Without pooling, every render pass allocates a new encoder on the heap (VK-PERF-006).
var renderPassPool = sync.Pool{
	New: func() any { return &RenderPassEncoder{} },
}

// Device implements hal.Device for Vulkan.
type Device struct {
	handle                    vk.Device
	physicalDevice            vk.PhysicalDevice
	instance                  *Instance
	graphicsFamily            uint32
	allocator                 *memory.GpuAllocator
	cmds                      *vk.Commands
	supportsMultiDrawIndirect bool
	supportsDrawIndirectCount bool
	maxDrawIndirectCount      uint32
	descriptorAllocator       *DescriptorAllocator // Descriptor pool management for bind groups
	queue                     *Queue               // Primary queue (for swapchain synchronization)
	interop                   *Interop             // for a video decoder sharing the device, nil without
	renderPassCache           *RenderPassCache     // Cache for VkRenderPass and VkFramebuffer objects
	pipelineCache             vk.PipelineCache     // Driver-compiled ISA cache (#331)
	pipelineCachePath         string               // Disk path for pipeline cache persistence

	// depth24 and depth24Stencil8 are the Vulkan formats of Depth24Plus and
	// Depth24PlusStencil8 on this device, see depth24Formats, and stencil8
	// the one of Stencil8, see stencil8Format.
	depth24, depth24Stencil8, stencil8 vk.Format

	// supportsIncrementalPresent is true when VK_KHR_incremental_present
	// is enabled on this device. When true, Present can chain
	// VkPresentRegionsKHR into VkPresentInfoKHR.PNext to hint the
	// compositor about which surface regions changed (damage rects).
	supportsIncrementalPresent bool

	// storeOpNone is whether the device takes VK_ATTACHMENT_STORE_OP_NONE,
	// which a read-only depth or stencil aspect is stored with: without it
	// the aspect is stored, a write that the barriers after the pass wait for.
	storeOpNone bool

	// Timeline semaphore fence (VK-IMPL-001).
	// When available (Vulkan 1.2+), replaces both frame fences and transfer fence
	// with a single timeline semaphore. Falls back to binary fences on older drivers.
	timelineFence *deviceFence

	// Per-encoder command pool recycling (VK-POOL-001, VK-CMD-001).
	// Each CreateCommandEncoder gets a dedicated VkCommandPool.
	// The encoder manages its own command buffer free list internally.
	// After GPU completion, FreeCommandBuffer recycles the pool for reuse.
	freeAllocators []commandAllocator
	allocatorMu    sync.Mutex // protects freeAllocators

	// nonCoherentAtomSize is the minimum alignment for Invalidate/Flush
	// mapped memory range operations (from VkPhysicalDeviceLimits).
	// Vulkan spec requires offset and size passed to vkInvalidateMappedMemoryRanges
	// and vkFlushMappedMemoryRanges to be multiples of this value.
	// Queried during initAllocator. (BUG-VK-009 Fix 2)
	nonCoherentAtomSize uint64

	// timestampPeriod is the number of nanoseconds per timestamp query tick
	// (from VkPhysicalDeviceLimits.TimestampPeriod). Intel typically 1.0,
	// AMD/NVIDIA vary. Queried during initAllocator.
	timestampPeriod float32

	// mappedMemory tracks persistently mapped VkDeviceMemory objects.
	// Vulkan only allows one active vkMapMemory per VkDeviceMemory;
	// with suballocation multiple buffers share the same VkDeviceMemory,
	// so we map each VkDeviceMemory once from offset 0 and reuse it.
	mappedMemory   map[vk.DeviceMemory]uintptr
	mappedMemoryMu sync.Mutex // protects mappedMemory map (concurrent CreateBuffer)

	// debugNameBuf is a reusable buffer for null-terminating debug label strings
	// passed to vkSetDebugUtilsObjectNameEXT. Avoids heap allocation per
	// Vulkan object creation (PERF-VK-001). Not thread-safe — setObjectName
	// is only called during resource creation which is single-threaded per device.
	debugNameBuf []byte

	// configuredSurfaces contains surfaces whose live swapchains belong to this
	// device. Device teardown retires them before destroying VkDevice.
	surfaceMu          sync.Mutex
	configuredSurfaces map[*Surface]struct{}
	destroying         bool

	// live are the child objects still alive, for Destroy to sweep what
	// was never destroyed.
	live liveObjects
}

func (d *Device) registerConfiguredSurface(surface *Surface) error {
	if d == nil || surface == nil {
		return fmt.Errorf("vulkan: cannot register a nil configured surface")
	}
	d.surfaceMu.Lock()
	defer d.surfaceMu.Unlock()
	if d.destroying || d.handle == 0 {
		return hal.ErrDeviceLost
	}
	if d.configuredSurfaces == nil {
		d.configuredSurfaces = make(map[*Surface]struct{})
	}
	d.configuredSurfaces[surface] = struct{}{}
	return nil
}

func (d *Device) unregisterConfiguredSurface(surface *Surface) {
	if d == nil || surface == nil {
		return
	}
	d.surfaceMu.Lock()
	delete(d.configuredSurfaces, surface)
	d.surfaceMu.Unlock()
}

func (d *Device) beginDestroy() ([]*Surface, bool) {
	if d == nil {
		return nil, false
	}
	d.surfaceMu.Lock()
	defer d.surfaceMu.Unlock()
	if d.destroying || d.handle == 0 {
		return nil, false
	}
	d.destroying = true
	surfaces := make([]*Surface, 0, len(d.configuredSurfaces))
	for surface := range d.configuredSurfaces {
		surfaces = append(surfaces, surface)
	}
	clear(d.configuredSurfaces)
	return surfaces, true
}

// vkFormat is the Vulkan format of format on this device: the one the
// adapter chose for Depth24Plus and Depth24PlusStencil8, see depth24Formats,
// and for Stencil8, see stencil8Format. Every image, view, render pass and
// pipeline of the device takes its format from here, so they agree.
func (d *Device) vkFormat(format gputypes.TextureFormat) vk.Format {
	switch format {
	case gputypes.TextureFormatDepth24Plus:
		if d.depth24 != 0 {
			return d.depth24
		}
	case gputypes.TextureFormatDepth24PlusStencil8:
		if d.depth24Stencil8 != 0 {
			return d.depth24Stencil8
		}
	case gputypes.TextureFormatStencil8:
		if d.stencil8 != 0 {
			return d.stencil8
		}
	}
	return textureFormatToVk(format)
}

// initAllocator initializes the memory allocator for this device.
func (d *Device) initAllocator() error {
	// Get physical device memory properties
	var vkProps vk.PhysicalDeviceMemoryProperties
	d.instance.cmds.GetPhysicalDeviceMemoryProperties(d.physicalDevice, &vkProps)

	// Convert to our format
	props := memory.DeviceMemoryProperties{
		MemoryTypes: make([]memory.MemoryType, vkProps.MemoryTypeCount),
		MemoryHeaps: make([]memory.MemoryHeap, vkProps.MemoryHeapCount),
	}

	for i := uint32(0); i < vkProps.MemoryTypeCount; i++ {
		props.MemoryTypes[i] = memory.MemoryType{
			PropertyFlags: vkProps.MemoryTypes[i].PropertyFlags,
			HeapIndex:     vkProps.MemoryTypes[i].HeapIndex,
		}
	}

	for i := uint32(0); i < vkProps.MemoryHeapCount; i++ {
		props.MemoryHeaps[i] = memory.MemoryHeap{
			Size:  uint64(vkProps.MemoryHeaps[i].Size),
			Flags: vkProps.MemoryHeaps[i].Flags,
		}
	}

	// Query maxMemoryAllocationSize (Vulkan 1.1 core, BUG-VK-001).
	// This prevents SIGSEGV from vkAllocateMemory silently failing
	// when the requested size exceeds what the driver supports.
	maxAllocSize := d.queryMaxMemoryAllocationSize()

	// Query nonCoherentAtomSize from physical device limits (BUG-VK-009 Fix 2).
	// This is the minimum alignment for Invalidate/Flush mapped memory ranges.
	// Rust wgpu-hal stores this as non_coherent_map_mask = atomSize - 1 (adapter.rs:1921).
	d.queryNonCoherentAtomSize()

	// Create allocator with default config
	allocator, err := memory.NewGpuAllocator(d.handle, d.cmds, props, maxAllocSize, memory.DefaultConfig())
	if err != nil {
		return fmt.Errorf("failed to create memory allocator: %w", err)
	}

	// Register callback to clear stale entries from mappedMemory cache
	// when VkDeviceMemory is freed. Prevents use-after-free when the driver
	// recycles a handle. (BUG-VK-009 Fix 4)
	allocator.SetOnFreeCallback(func(mem vk.DeviceMemory) {
		d.mappedMemoryMu.Lock()
		delete(d.mappedMemory, mem)
		d.mappedMemoryMu.Unlock()
	})

	d.allocator = allocator

	return nil
}

// queryMaxMemoryAllocationSize queries VkPhysicalDeviceMaintenance3Properties
// to determine the maximum single VkDeviceMemory allocation size.
// Returns 0 if the query is not available (pre-Vulkan 1.1), in which case
// the allocator uses a safe fallback.
func (d *Device) queryMaxMemoryAllocationSize() uint64 {
	// vkGetPhysicalDeviceProperties2 is Vulkan 1.1 core.
	// GetPhysicalDeviceProperties2 has an internal nil-check on its function pointer
	// and returns silently if unavailable (pre-1.1 drivers).
	// We rely on the SType being properly initialized for the pNext chain.

	// Chain PhysicalDeviceMaintenance3Properties into PhysicalDeviceProperties2.
	var maint3Props vk.PhysicalDeviceMaintenance3Properties
	maint3Props.SType = vk.StructureTypePhysicalDeviceMaintenance3Properties

	props2 := vk.PhysicalDeviceProperties2{
		SType: vk.StructureTypePhysicalDeviceProperties2,
		PNext: (*uintptr)(unsafe.Pointer(&maint3Props)),
	}

	d.instance.cmds.GetPhysicalDeviceProperties2(d.physicalDevice, &props2)

	maxSize := uint64(maint3Props.MaxMemoryAllocationSize)
	if maxSize > 0 {
		hal.Logger().Info("vulkan: maxMemoryAllocationSize queried",
			"maxSize", maxSize,
			"maxSizeMB", maxSize/(1<<20),
		)
	}

	return maxSize
}

// queryNonCoherentAtomSize reads VkPhysicalDeviceLimits.NonCoherentAtomSize
// from the physical device properties. This value is the minimum alignment
// required by the Vulkan spec for vkInvalidateMappedMemoryRanges and
// vkFlushMappedMemoryRanges offset/size parameters.
//
// Guaranteed to be a power of two. Falls back to 256 (safe for all GPUs) if
// the query returns 0. Intel Iris Xe reports 1 (always aligned).
// Rust wgpu-hal stores atomSize - 1 as non_coherent_map_mask (adapter.rs:1921).
func (d *Device) queryNonCoherentAtomSize() {
	var props vk.PhysicalDeviceProperties
	d.instance.cmds.GetPhysicalDeviceProperties(d.physicalDevice, &props)

	atomSize := uint64(props.Limits.NonCoherentAtomSize)
	if atomSize == 0 {
		atomSize = 256 // safe fallback
	}
	d.nonCoherentAtomSize = atomSize

	d.timestampPeriod = props.Limits.TimestampPeriod
	if d.timestampPeriod <= 0 {
		d.timestampPeriod = 1.0
	}

	hal.Logger().Debug("vulkan: device limits queried",
		"nonCoherentAtomSize", atomSize,
		"timestampPeriod", d.timestampPeriod,
	)
}

// alignedMappedRange aligns offset down and size up to nonCoherentAtomSize
// for Invalidate/Flush mapped memory range operations. The Vulkan spec requires
// these parameters to be multiples of nonCoherentAtomSize. Matches Rust wgpu-hal
// make_memory_ranges (device.rs:233-246). (BUG-VK-009 Fix 2)
func (d *Device) alignedMappedRange(blockOffset, blockSize uint64) (offset, size vk.DeviceSize) {
	mask := d.nonCoherentAtomSize - 1 // atomSize is power of 2
	alignedOffset := blockOffset & ^mask
	alignedEnd := (blockOffset + blockSize + mask) & ^mask
	return vk.DeviceSize(alignedOffset), vk.DeviceSize(alignedEnd - alignedOffset)
}

// MaxStagingBufferSize returns the maximum safe staging buffer allocation size.
// This is min(64MB, maxMemoryAllocationSize) from the Vulkan allocator.
// Used by the staging belt to cap individual staging buffers, preventing
// vkAllocateMemory from silently failing on oversized requests (BUG-VK-001).
// Implements hal.MaxStagingBufferSizer.
func (d *Device) MaxStagingBufferSize() uint64 {
	if d.allocator == nil {
		return 0
	}
	maxAlloc := d.allocator.MaxAllocationSize()
	const defaultMax = 64 << 20 // 64 MB, matching Rust wgpu's 1 << 26
	if maxAlloc > 0 && maxAlloc < defaultMax {
		return maxAlloc
	}
	return defaultMax
}

// CreateBuffer creates a GPU buffer.
func (d *Device) CreateBuffer(desc *hal.BufferDescriptor) (hal.Buffer, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: buffer descriptor is nil in Vulkan.CreateBuffer — core validation gap")
	}

	// Every buffer can be a copy destination, as in Dawn and wgpu-core: the
	// core zeroes and clears buffers whatever their usage, and a buffer that
	// is only mapped would have no usage at all.
	vkUsage := bufferUsageToVk(desc.Usage) | vk.BufferUsageFlags(vk.BufferUsageTransferDstBit)

	// Create VkBuffer (without memory)
	createInfo := vk.BufferCreateInfo{
		SType:       vk.StructureTypeBufferCreateInfo,
		Size:        vk.DeviceSize(desc.Size),
		Usage:       vkUsage,
		SharingMode: vk.SharingModeExclusive,
	}

	var buffer vk.Buffer
	result := d.cmds.CreateBuffer(d.handle, &createInfo, nil, &buffer)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateBuffer failed: %d", result)
	}

	// Get memory requirements
	var memReqs vk.MemoryRequirements
	d.cmds.GetBufferMemoryRequirements(d.handle, buffer, &memReqs)

	// Determine usage flags for memory allocation.
	// Only MAP_READ/MAP_WRITE buffers (and MappedAtCreation) need host-visible memory.
	// CopyDst buffers stay DEVICE_LOCAL — writes go through the staging belt
	// (CopySrc staging buffer + GPU CopyBufferToBuffer). Matches Rust wgpu-hal
	// device.rs:971-988. (BUG-VK-009 Fix 1)
	memUsage := memory.UsageFastDeviceAccess
	if desc.Usage&(gputypes.BufferUsageMapRead|gputypes.BufferUsageMapWrite) != 0 || desc.MappedAtCreation {
		memUsage = memory.UsageHostAccess
		if desc.Usage&gputypes.BufferUsageMapWrite != 0 || desc.MappedAtCreation {
			memUsage |= memory.UsageUpload
		}
		if desc.Usage&gputypes.BufferUsageMapRead != 0 {
			memUsage |= memory.UsageDownload
		}
	}

	// Allocate memory
	memBlock, err := d.allocator.Alloc(memory.AllocationRequest{
		Size:           uint64(memReqs.Size),
		Alignment:      uint64(memReqs.Alignment),
		Usage:          memUsage,
		MemoryTypeBits: memReqs.MemoryTypeBits,
	})
	if err != nil {
		d.cmds.DestroyBuffer(d.handle, buffer, nil)
		return nil, fmt.Errorf("vulkan: failed to allocate buffer memory: %w", err)
	}

	// Bind memory to buffer
	result = d.cmds.BindBufferMemory(d.handle, buffer, memBlock.Memory, vk.DeviceSize(memBlock.Offset))
	if result != vk.Success {
		_ = d.allocator.Free(memBlock)
		d.cmds.DestroyBuffer(d.handle, buffer, nil)
		return nil, fmt.Errorf("vulkan: vkBindBufferMemory failed: %d", result)
	}

	// Map memory for host-visible buffers so WriteBuffer can write directly.
	if memUsage&memory.UsageHostAccess != 0 {
		if err := d.ensureMemoryMapped(memBlock); err != nil {
			_ = d.allocator.Free(memBlock)
			d.cmds.DestroyBuffer(d.handle, buffer, nil)
			return nil, err
		}
	}

	b := &Buffer{
		handle: buffer,
		memory: memBlock,
		size:   desc.Size,
		usage:  desc.Usage,
		device: d,
	}
	d.live.buffers.add(buffer)
	if desc.Label != "" {
		d.setObjectName(vk.ObjectTypeBuffer, uint64(buffer), desc.Label)
	} else {
		d.setObjectName(vk.ObjectTypeBuffer, uint64(buffer), "Buffer")
	}
	return b, nil
}

// ensureMemoryMapped maps the VkDeviceMemory backing block if not already mapped.
// Vulkan only allows one active vkMapMemory per VkDeviceMemory.
// With suballocation, multiple buffers share the same VkDeviceMemory,
// so we map from offset 0 once and compute per-buffer pointers.
func (d *Device) ensureMemoryMapped(block *memory.MemoryBlock) error {
	d.mappedMemoryMu.Lock()
	defer d.mappedMemoryMu.Unlock()

	if d.mappedMemory == nil {
		d.mappedMemory = make(map[vk.DeviceMemory]uintptr)
	}
	basePtr, alreadyMapped := d.mappedMemory[block.Memory]
	if !alreadyMapped {
		var mapped unsafe.Pointer
		result := d.cmds.MapMemory(d.handle, block.Memory, 0,
			vk.DeviceSize(vk.WholeSize), 0, &mapped)
		mappedPtr := uintptr(mapped)
		if result != vk.Success {
			return fmt.Errorf("vulkan: vkMapMemory failed: %d", result)
		}
		if mappedPtr == 0 {
			return fmt.Errorf("vulkan: vkMapMemory returned null pointer (BUG-VK-001)")
		}
		d.mappedMemory[block.Memory] = mappedPtr
		basePtr = mappedPtr
	}
	block.MappedPtr = basePtr + uintptr(block.Offset)
	block.MappedSize = block.Size // Valid mapped region = allocated block size (BUG-VK-001)
	return nil
}

// MapBuffer establishes a CPU-visible mapping for the given byte range.
//
// Vulkan maps each VkDeviceMemory block persistently at allocation time
// (ensureMemoryMapped in CreateBuffer). MapBuffer therefore just returns
// the cached pointer offset by the requested offset -- no additional
// vkMapMemory call, no syscall overhead.
//
// Coherency is reported accurately from the memory type's HOST_COHERENT bit.
// Non-coherent memory requires Invalidate before CPU reads (MAP_READ) and
// Flush after CPU writes (MAP_WRITE). Matches Rust wgpu-hal map_buffer
// which reads is_coherent from block.props() (device.rs:1072-1075).
func (d *Device) MapBuffer(buffer hal.Buffer, offset, size uint64) (hal.BufferMapping, error) {
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil || buf.memory == nil {
		return hal.BufferMapping{}, hal.ErrInvalidMapRange
	}
	if offset+size > buf.size {
		return hal.BufferMapping{}, hal.ErrInvalidMapRange
	}
	if buf.memory.MappedPtr == 0 {
		return hal.BufferMapping{}, hal.ErrInvalidMapRange
	}

	// Diagnostic logging for debugging mapped pointer issues (BUG-VK-009 Fix 6).
	hal.Logger().Debug("vulkan: MapBuffer",
		"basePtr", fmt.Sprintf("%#x", buf.memory.MappedPtr),
		"offset", offset,
		"size", size,
		"blockOffset", buf.memory.Offset,
		"blockSize", buf.memory.Size,
		"isCoherent", buf.memory.IsCoherent,
	)

	// Invalidate host caches so CPU reads see GPU writes. Only needed for
	// non-coherent memory on MAP_READ. Coherent memory is automatically
	// visible. Matches Rust wgpu-hal which only calls invalidate when
	// !is_coherent. (BUG-VK-009 Fix 5)
	if !buf.memory.IsCoherent {
		alignedOffset, alignedSize := d.alignedMappedRange(buf.memory.Offset, buf.memory.Size)
		memRange := vk.MappedMemoryRange{
			SType:  vk.StructureTypeMappedMemoryRange,
			Memory: buf.memory.Memory,
			Offset: alignedOffset,
			Size:   alignedSize,
		}
		// Check return value instead of silently ignoring (BUG-VK-009 Fix 3).
		if result := d.cmds.InvalidateMappedMemoryRanges(d.handle, 1, &memRange); result != vk.Success {
			return hal.BufferMapping{}, fmt.Errorf("vulkan: vkInvalidateMappedMemoryRanges failed: %d", result)
		}
	}

	// Go vet flags direct uintptr->Pointer conversion as "possible misuse"
	// because the verifier cannot prove that the address refers to Go-
	// managed memory. In our case the address comes from vkMapMemory and
	// therefore points at host-visible GPU memory -- converting via
	// unsafe.Add from a named base pointer keeps vet happy while
	// producing identical machine code.
	basePtr := ptrFromUintptr(buf.memory.MappedPtr)
	return hal.BufferMapping{
		Ptr:        unsafe.Add(unsafe.Pointer(basePtr), offset),
		IsCoherent: buf.memory.IsCoherent,
	}, nil
}

// UnmapBuffer is a no-op for Vulkan because VkDeviceMemory is persistently
// mapped at allocation time. However, for non-coherent memory we flush CPU
// writes back to GPU visibility so that subsequent GPU reads see the data.
// Coherent memory requires no flush. (BUG-VK-009 Fix 2/5)
func (d *Device) UnmapBuffer(buffer hal.Buffer) error {
	buf, ok := buffer.(*Buffer)
	if !ok || buf == nil || buf.memory == nil {
		return nil
	}
	// Flush CPU writes back to GPU visibility, but only for non-coherent memory.
	// Coherent memory is automatically visible. (BUG-VK-009 Fix 5)
	if !buf.memory.IsCoherent {
		alignedOffset, alignedSize := d.alignedMappedRange(buf.memory.Offset, buf.memory.Size)
		memRange := vk.MappedMemoryRange{
			SType:  vk.StructureTypeMappedMemoryRange,
			Memory: buf.memory.Memory,
			Offset: alignedOffset,
			Size:   alignedSize,
		}
		if result := d.cmds.FlushMappedMemoryRanges(d.handle, 1, &memRange); result != vk.Success {
			return fmt.Errorf("vulkan: vkFlushMappedMemoryRanges failed: %d", result)
		}
	}
	return nil
}

// DestroyBuffer destroys a GPU buffer.
func (d *Device) DestroyBuffer(buffer hal.Buffer) {
	vkBuffer, ok := buffer.(*Buffer)
	if !ok || vkBuffer == nil {
		return
	}

	if vkBuffer.handle != 0 {
		d.live.buffers.remove(vkBuffer.handle)
		d.cmds.DestroyBuffer(d.handle, vkBuffer.handle, nil)
		vkBuffer.handle = 0
	}

	if vkBuffer.memory != nil {
		// Don't unmap individually — the VkDeviceMemory mapping is shared
		// by all buffers suballocated from the same block. The mapping is
		// cleaned up when the allocator destroys the block or the device is destroyed.
		vkBuffer.memory.MappedPtr = 0
		_ = d.allocator.Free(vkBuffer.memory)
		vkBuffer.memory = nil
	}

	vkBuffer.device = nil
}

// CreateTexture creates a GPU texture.
func (d *Device) CreateTexture(desc *hal.TextureDescriptor) (hal.Texture, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: texture descriptor is nil in Vulkan.CreateTexture — core validation gap")
	}

	// Convert parameters
	vkFormat := d.vkFormat(desc.Format)
	vkUsage := textureUsageToVk(desc.Usage)
	imageType := textureDimensionToVkImageType(desc.Dimension)

	// A depth/stencil image is an attachment whatever it was made for: the
	// clear pass that zeroes it needs that (Rust wgpu-core adds
	// DEPTH_STENCIL_WRITE the same way), and textureUsageToVk maps
	// RenderAttachment to COLOR_ATTACHMENT, which it cannot have.
	if isDepthStencilFormat(desc.Format) {
		vkUsage &^= vk.ImageUsageFlags(vk.ImageUsageColorAttachmentBit)
		vkUsage |= vk.ImageUsageFlags(vk.ImageUsageDepthStencilAttachmentBit)
	}

	// Determine depth/array layers. A 1D or 2D image is one texel deep, its
	// layers are array layers.
	depth := desc.Size.DepthOrArrayLayers
	if depth == 0 || desc.Dimension != gputypes.TextureDimension3D {
		depth = 1
	}
	mipLevels := desc.MipLevelCount
	if mipLevels == 0 {
		mipLevels = 1
	}
	samples := desc.SampleCount
	if samples == 0 {
		samples = 1
	}

	// Determine array layer count
	// For 3D textures, DepthOrArrayLayers is the depth; for 1D/2D, it's the array layer count
	arrayLayers := uint32(1)
	if desc.Dimension != gputypes.TextureDimension3D {
		arrayLayers = desc.Size.DepthOrArrayLayers
		if arrayLayers == 0 {
			arrayLayers = 1
		}
	}

	// Determine image creation flags
	var imageFlags vk.ImageCreateFlags
	// A square 2D image with a multiple of 6 layers and one sample can be
	// viewed as cube maps, the only image that can be
	// (VUID-VkImageCreateInfo-imageType-00954, samples-02257).
	if desc.Dimension == gputypes.TextureDimension2D && arrayLayers%6 == 0 && samples == 1 && desc.Size.Width == desc.Size.Height {
		imageFlags |= vk.ImageCreateFlags(vk.ImageCreateCubeCompatibleBit)
	}
	// Enable mutable format if view formats are specified
	if len(desc.ViewFormats) > 0 {
		imageFlags |= vk.ImageCreateFlags(vk.ImageCreateMutableFormatBit)
	}

	// Create VkImage (without memory)
	createInfo := vk.ImageCreateInfo{
		SType:     vk.StructureTypeImageCreateInfo,
		Flags:     imageFlags,
		ImageType: imageType,
		Format:    vkFormat,
		Extent: vk.Extent3D{
			Width:  desc.Size.Width,
			Height: desc.Size.Height,
			Depth:  depth,
		},
		MipLevels:     mipLevels,
		ArrayLayers:   arrayLayers,
		Samples:       vk.SampleCountFlagBits(samples),
		Tiling:        vk.ImageTilingOptimal,
		Usage:         vkUsage,
		SharingMode:   vk.SharingModeExclusive,
		InitialLayout: vk.ImageLayoutUndefined,
	}

	var image vk.Image
	result := d.cmds.CreateImage(d.handle, &createInfo, nil, &image)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateImage failed: %d", result)
	}

	// Get memory requirements
	var memReqs vk.MemoryRequirements
	d.cmds.GetImageMemoryRequirements(d.handle, image, &memReqs)

	// Allocate memory (textures always use device-local)
	memBlock, err := d.allocator.Alloc(memory.AllocationRequest{
		Size:           uint64(memReqs.Size),
		Alignment:      uint64(memReqs.Alignment),
		Usage:          memory.UsageFastDeviceAccess,
		MemoryTypeBits: memReqs.MemoryTypeBits,
	})
	if err != nil {
		d.cmds.DestroyImage(d.handle, image, nil)
		return nil, fmt.Errorf("vulkan: failed to allocate texture memory: %w", err)
	}

	// Bind memory to image
	result = d.cmds.BindImageMemory(d.handle, image, memBlock.Memory, vk.DeviceSize(memBlock.Offset))
	if result != vk.Success {
		_ = d.allocator.Free(memBlock)
		d.cmds.DestroyImage(d.handle, image, nil)
		return nil, fmt.Errorf("vulkan: vkBindImageMemory failed: %d", result)
	}

	t := &Texture{
		handle:      image,
		memory:      memBlock,
		size:        Extent3D{Width: desc.Size.Width, Height: desc.Size.Height, Depth: depth},
		format:      desc.Format,
		usage:       desc.Usage,
		usageFlags:  vkUsage,
		mipLevels:   mipLevels,
		arrayLayers: arrayLayers,
		samples:     samples,
		dimension:   desc.Dimension,
		device:      d,
	}
	d.live.images.add(image)
	if desc.Label != "" {
		d.setObjectName(vk.ObjectTypeImage, uint64(image), desc.Label)
	} else {
		d.setObjectName(vk.ObjectTypeImage, uint64(image), "Texture")
	}
	return t, nil
}

// DestroyTexture destroys a GPU texture.
func (d *Device) DestroyTexture(texture hal.Texture) {
	vkTexture, ok := texture.(*Texture)
	if !ok || vkTexture == nil {
		return
	}

	if vkTexture.handle != 0 && !vkTexture.isExternal {
		d.live.images.remove(vkTexture.handle)
		d.cmds.DestroyImage(d.handle, vkTexture.handle, nil)
		vkTexture.handle = 0
	}

	if vkTexture.memory != nil {
		_ = d.allocator.Free(vkTexture.memory)
		vkTexture.memory = nil
	}

	vkTexture.device = nil
}

// CreateTextureView creates a view into a texture.
func (d *Device) CreateTextureView(texture hal.Texture, desc *hal.TextureViewDescriptor) (hal.TextureView, error) {
	// Extract image handle and metadata based on texture type.
	// We support both regular Texture and SwapchainTexture.
	var (
		imageHandle   vk.Image
		textureFormat gputypes.TextureFormat
		dimension     gputypes.TextureDimension
		mipLevels     uint32
		arrayLayers   uint32
		textureSize   Extent3D
	)

	switch t := texture.(type) {
	case *Texture:
		if t == nil {
			return nil, fmt.Errorf("vulkan: nil texture")
		}
		imageHandle = t.handle
		textureFormat = t.format
		dimension = t.dimension
		mipLevels = t.mipLevels
		arrayLayers = t.arrayLayers
		textureSize = t.size
	case *SwapchainTexture:
		if t == nil {
			return nil, fmt.Errorf("vulkan: nil swapchain texture")
		}
		// A surface takes no view formats, so its images are viewed in its own.
		if desc != nil && desc.Format != gputypes.TextureFormatUndefined && desc.Format != t.format {
			return nil, fmt.Errorf("vulkan: a view of a surface texture of %v cannot be %v", t.format, desc.Format)
		}
		// For swapchain textures, reuse the pre-created view from the swapchain,
		// which was named when the swapchain made it. Creating new views for
		// swapchain images can cause rendering issues. The wrapper is made
		// once per image, so a frame allocates nothing for it.
		if t.textureView == nil {
			t.textureView = &TextureView{
				handle:      t.view,
				device:      d,
				size:        t.size,
				image:       t.handle,
				isSwapchain: true,
				swapchain:   t.swapchain,
				format:      t.format,
				vkFormat:    d.vkFormat(t.format),
			}
		}
		return t.textureView, nil
	default:
		return nil, fmt.Errorf("vulkan: invalid texture type %T", texture)
	}

	// An image made for copies only has no usage a view may have, and is
	// not allowed one (VUID-VkImageViewCreateInfo-image-04441). Its users
	// get the wrapper without a VkImageView: no bind group or pass takes
	// such a texture, the usage rules it out before the HAL.
	if t, ok := texture.(*Texture); ok && t.usageFlags&viewableImageUsage == 0 {
		return &TextureView{texture: t, device: d, size: textureSize, image: imageHandle, format: t.format, vkFormat: d.vkFormat(t.format)}, nil
	}

	// Handle nil descriptor - use defaults from texture
	if desc == nil {
		desc = &hal.TextureViewDescriptor{}
	}

	// Determine format - use texture format if not specified
	format := desc.Format
	if format == gputypes.TextureFormatUndefined {
		format = textureFormat
	}

	viewDimension := desc.Dimension
	if viewDimension == gputypes.TextureViewDimensionUndefined {
		viewDimension = defaultViewDimension(dimension, arrayLayers)
	}

	// Determine mip level count
	mipLevelCount := desc.MipLevelCount
	if mipLevelCount == 0 {
		mipLevelCount = mipLevels - desc.BaseMipLevel
	}

	arrayLayerCount := desc.ArrayLayerCount
	if arrayLayerCount == 0 {
		arrayLayerCount = defaultViewLayers(viewDimension, desc.BaseArrayLayer, arrayLayers)
	}

	vkFormat := d.vkFormat(format)
	createInfo := vk.ImageViewCreateInfo{
		SType:    vk.StructureTypeImageViewCreateInfo,
		Image:    imageHandle,
		ViewType: textureViewDimensionToVk(viewDimension),
		Format:   vkFormat,
		Components: vk.ComponentMapping{
			R: vk.ComponentSwizzleIdentity,
			G: vk.ComponentSwizzleIdentity,
			B: vk.ComponentSwizzleIdentity,
			A: vk.ComponentSwizzleIdentity,
		},
		SubresourceRange: vk.ImageSubresourceRange{
			AspectMask:     textureAspectToVk(desc.Aspect, format),
			BaseMipLevel:   desc.BaseMipLevel,
			LevelCount:     mipLevelCount,
			BaseArrayLayer: desc.BaseArrayLayer,
			LayerCount:     arrayLayerCount,
		},
	}

	// An sRGB view of a storage image has the image's usage less storage,
	// which no sRGB format has (VUID-VkImageViewCreateInfo-usage-02275).
	var viewUsage vk.ImageViewUsageCreateInfo
	if t, ok := texture.(*Texture); ok && format.IsSrgb() && t.usageFlags&vk.ImageUsageFlags(vk.ImageUsageStorageBit) != 0 {
		viewUsage = vk.ImageViewUsageCreateInfo{
			SType: vk.StructureTypeImageViewUsageCreateInfo,
			Usage: t.usageFlags &^ vk.ImageUsageFlags(vk.ImageUsageStorageBit),
		}
		createInfo.PNext = (*uintptr)(unsafe.Pointer(&viewUsage))
	}

	var imageView vk.ImageView
	result := vkCreateImageView(d.cmds, d.handle, &createInfo, nil, &imageView)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateImageView failed: %d", result)
	}

	// Store texture reference and track if this is a swapchain image.
	var texRef *Texture
	var isSwapchain bool
	switch t := texture.(type) {
	case *Texture:
		texRef = t
	case *SwapchainTexture:
		isSwapchain = true
	}

	// A pass renders a view at the size of its first mip level: the
	// framebuffer may not be bigger than its attachments.
	textureSize.Width = max(textureSize.Width>>desc.BaseMipLevel, 1)
	textureSize.Height = max(textureSize.Height>>desc.BaseMipLevel, 1)
	tv := &TextureView{
		handle:      imageView,
		texture:     texRef,
		device:      d,
		size:        textureSize,
		image:       imageHandle,
		isSwapchain: isSwapchain,
		format:      format,
		vkFormat:    vkFormat,
	}
	d.live.imageViews.add(imageView)
	if desc.Label != "" {
		d.setObjectName(vk.ObjectTypeImageView, uint64(imageView), desc.Label)
	} else {
		d.setObjectName(vk.ObjectTypeImageView, uint64(imageView), "TextureView")
	}
	return tv, nil
}

// viewableImageUsage are the usages an image needs one of for a view of it.
const viewableImageUsage = vk.ImageUsageFlags(vk.ImageUsageSampledBit | vk.ImageUsageStorageBit |
	vk.ImageUsageColorAttachmentBit | vk.ImageUsageDepthStencilAttachmentBit |
	vk.ImageUsageInputAttachmentBit | vk.ImageUsageTransientAttachmentBit)

// DestroyTextureView destroys a texture view.
func (d *Device) DestroyTextureView(view hal.TextureView) {
	vkView, ok := view.(*TextureView)
	if !ok || vkView == nil {
		return
	}

	// Don't destroy swapchain views - they're owned by the swapchain, and the
	// wrapper is shared by every frame that draws to the image.
	if vkView.isSwapchain {
		return
	}

	if vkView.handle != 0 {
		// Invalidate cached framebuffers that reference this view before
		// destroying it, otherwise Vulkan validation reports the image view
		// as "in use" by the (now stale) framebuffer.
		if d.renderPassCache != nil {
			d.renderPassCache.InvalidateFramebuffer(vkView.handle)
		}
		d.live.imageViews.remove(vkView.handle)
		vkDestroyImageView(d.cmds, d.handle, vkView.handle, nil)
		vkView.handle = 0
	}

	vkView.device = nil
}

// CreateSampler creates a texture sampler.
func (d *Device) CreateSampler(desc *hal.SamplerDescriptor) (hal.Sampler, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: sampler descriptor is nil in Vulkan.CreateSampler — core validation gap")
	}

	// Determine if comparison is enabled
	compareEnable := vk.Bool32(vk.False)
	compareOp := vk.CompareOpNever
	if desc.Compare != gputypes.CompareFunctionUndefined {
		compareEnable = vk.Bool32(vk.True)
		compareOp = compareFunctionToVk(desc.Compare)
	}

	// Determine if anisotropy is enabled
	anisotropyEnable := vk.Bool32(vk.False)
	maxAnisotropy := float32(1.0)
	if desc.Anisotropy > 1 {
		anisotropyEnable = vk.Bool32(vk.True)
		maxAnisotropy = float32(desc.Anisotropy)
	}

	// LOD clamp values
	lodMinClamp := desc.LodMinClamp
	lodMaxClamp := desc.LodMaxClamp
	if lodMaxClamp == 0 {
		lodMaxClamp = vk.LodClampNone
	}

	createInfo := vk.SamplerCreateInfo{
		SType:                   vk.StructureTypeSamplerCreateInfo,
		MagFilter:               filterModeToVk(desc.MagFilter),
		MinFilter:               filterModeToVk(desc.MinFilter),
		MipmapMode:              mipmapFilterModeToVk(desc.MipmapFilter),
		AddressModeU:            addressModeToVk(desc.AddressModeU),
		AddressModeV:            addressModeToVk(desc.AddressModeV),
		AddressModeW:            addressModeToVk(desc.AddressModeW),
		MipLodBias:              0.0,
		AnisotropyEnable:        anisotropyEnable,
		MaxAnisotropy:           maxAnisotropy,
		CompareEnable:           compareEnable,
		CompareOp:               compareOp,
		MinLod:                  lodMinClamp,
		MaxLod:                  lodMaxClamp,
		BorderColor:             vk.BorderColorFloatTransparentBlack,
		UnnormalizedCoordinates: vk.Bool32(vk.False),
	}

	var sampler vk.Sampler
	result := vkCreateSampler(d.cmds, d.handle, &createInfo, nil, &sampler)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateSampler failed: %d", result)
	}

	s := &Sampler{
		handle: sampler,
		device: d,
	}
	d.live.samplers.add(sampler)
	if desc.Label != "" {
		d.setObjectName(vk.ObjectTypeSampler, uint64(sampler), desc.Label)
	} else {
		d.setObjectName(vk.ObjectTypeSampler, uint64(sampler), "Sampler")
	}
	return s, nil
}

// DestroySampler destroys a sampler.
func (d *Device) DestroySampler(sampler hal.Sampler) {
	vkSampler, ok := sampler.(*Sampler)
	if !ok || vkSampler == nil {
		return
	}

	if vkSampler.handle != 0 {
		d.live.samplers.remove(vkSampler.handle)
		vkDestroySampler(d.cmds, d.handle, vkSampler.handle, nil)
		vkSampler.handle = 0
	}

	vkSampler.device = nil
}

// CreateBindGroupLayout creates a bind group layout.
func (d *Device) CreateBindGroupLayout(desc *hal.BindGroupLayoutDescriptor) (hal.BindGroupLayout, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: bind group layout descriptor is nil in Vulkan.CreateBindGroupLayout — core validation gap")
	}

	// Convert entries to Vulkan descriptor set layout bindings and track counts
	bindings := make([]vk.DescriptorSetLayoutBinding, 0, len(desc.Entries))
	bindingTypes := make(map[uint32]vk.DescriptorType, len(desc.Entries))
	var counts DescriptorCounts

	for _, entry := range desc.Entries {
		binding := vk.DescriptorSetLayoutBinding{
			Binding:         entry.Binding,
			DescriptorCount: 1,
			StageFlags:      shaderStagesToVk(entry.Visibility),
		}

		// Determine descriptor type based on which binding is set.
		// For buffers, pass HasDynamicOffset so Vulkan gets the correct
		// dynamic descriptor type (VK_DESCRIPTOR_TYPE_*_BUFFER_DYNAMIC).
		switch {
		case entry.Buffer != nil:
			binding.DescriptorType = bufferBindingTypeToVk(entry.Buffer.Type, entry.Buffer.HasDynamicOffset)
			if entry.Buffer.HasDynamicOffset {
				if entry.Buffer.Type == gputypes.BufferBindingTypeUniform {
					counts.UniformBuffersDynamic++
				} else {
					counts.StorageBuffersDynamic++
				}
			} else {
				if entry.Buffer.Type == gputypes.BufferBindingTypeUniform {
					counts.UniformBuffers++
				} else {
					counts.StorageBuffers++
				}
			}
		case entry.Sampler != nil:
			binding.DescriptorType = vk.DescriptorTypeSampler
			counts.Samplers++
		case entry.Texture != nil:
			binding.DescriptorType = vk.DescriptorTypeSampledImage
			counts.SampledImages++
		case entry.StorageTexture != nil:
			binding.DescriptorType = vk.DescriptorTypeStorageImage
			counts.StorageImages++
		}

		bindingTypes[entry.Binding] = binding.DescriptorType
		bindings = append(bindings, binding)
	}

	createInfo := vk.DescriptorSetLayoutCreateInfo{
		SType:        vk.StructureTypeDescriptorSetLayoutCreateInfo,
		BindingCount: uint32(len(bindings)),
	}

	if len(bindings) > 0 {
		createInfo.PBindings = &bindings[0]
	}

	var layout vk.DescriptorSetLayout
	result := vkCreateDescriptorSetLayout(d.cmds, d.handle, &createInfo, nil, &layout)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateDescriptorSetLayout failed: %d", result)
	}

	bgl := &BindGroupLayout{
		handle:       layout,
		counts:       counts,
		bindingTypes: bindingTypes,
		device:       d,
	}
	d.live.setLayouts.add(layout)
	if desc.Label != "" {
		d.setObjectName(vk.ObjectTypeDescriptorSetLayout, uint64(layout), desc.Label)
	} else {
		d.setObjectName(vk.ObjectTypeDescriptorSetLayout, uint64(layout), "BindGroupLayout")
	}
	return bgl, nil
}

// DestroyBindGroupLayout destroys a bind group layout.
func (d *Device) DestroyBindGroupLayout(layout hal.BindGroupLayout) {
	vkLayout, ok := layout.(*BindGroupLayout)
	if !ok || vkLayout == nil {
		return
	}

	if vkLayout.handle != 0 {
		d.live.setLayouts.remove(vkLayout.handle)
		vkDestroyDescriptorSetLayout(d.cmds, d.handle, vkLayout.handle, nil)
		vkLayout.handle = 0
	}

	vkLayout.device = nil
}

// CreateBindGroup creates a bind group.
func (d *Device) CreateBindGroup(desc *hal.BindGroupDescriptor) (hal.BindGroup, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: bind group descriptor is nil in Vulkan.CreateBindGroup — core validation gap")
	}

	// Get the layout
	vkLayout, ok := desc.Layout.(*BindGroupLayout)
	if !ok || vkLayout == nil {
		return nil, fmt.Errorf("vulkan: invalid bind group layout")
	}

	// Initialize descriptor allocator if needed
	if d.descriptorAllocator == nil {
		d.descriptorAllocator = NewDescriptorAllocator(d.handle, d.cmds, DefaultDescriptorAllocatorConfig())
	}

	// Allocate descriptor set
	set, pool, err := d.descriptorAllocator.Allocate(vkLayout.handle, vkLayout.counts)
	if err != nil {
		return nil, fmt.Errorf("vulkan: failed to allocate descriptor set: %w", err)
	}

	// Update descriptor set with bindings
	if err := d.updateDescriptorSet(set, desc.Entries, vkLayout.bindingTypes); err != nil {
		// Free the set on error
		_ = d.descriptorAllocator.Free(pool, set)
		return nil, fmt.Errorf("vulkan: failed to update descriptor set: %w", err)
	}

	return &BindGroup{
		handle: set,
		pool:   pool,
		device: d,
	}, nil
}

// updateDescriptorSet writes resource bindings to a descriptor set.
func (d *Device) updateDescriptorSet(set vk.DescriptorSet, entries []gputypes.BindGroupEntry, bindingTypes map[uint32]vk.DescriptorType) error {
	if len(entries) == 0 {
		return nil
	}

	// Build write descriptor sets
	// Note: We need to keep the info structs alive until vkUpdateDescriptorSets returns
	writes := make([]vk.WriteDescriptorSet, 0, len(entries))
	bufferInfos := make([]vk.DescriptorBufferInfo, 0)
	imageInfos := make([]vk.DescriptorImageInfo, 0)

	for _, entry := range entries {
		write := vk.WriteDescriptorSet{
			SType:           vk.StructureTypeWriteDescriptorSet,
			DstSet:          set,
			DstBinding:      entry.Binding,
			DstArrayElement: 0,
			DescriptorCount: 1,
		}

		switch res := entry.Resource.(type) {
		case gputypes.BufferBinding:
			bufferInfo := vk.DescriptorBufferInfo{
				Buffer: vk.Buffer(res.Buffer),
				Offset: vk.DeviceSize(res.Offset),
				Range:  vk.DeviceSize(res.Size),
			}
			if res.Size == 0 {
				bufferInfo.Range = vk.DeviceSize(vk.WholeSize)
			}
			bufferInfos = append(bufferInfos, bufferInfo)
			// Use the actual descriptor type from the layout
			if dt, ok := bindingTypes[entry.Binding]; ok {
				write.DescriptorType = dt
			} else {
				write.DescriptorType = vk.DescriptorTypeUniformBuffer
			}
			write.PBufferInfo = &bufferInfos[len(bufferInfos)-1]

		case gputypes.SamplerBinding:
			imageInfo := vk.DescriptorImageInfo{
				Sampler: vk.Sampler(res.Sampler),
			}
			imageInfos = append(imageInfos, imageInfo)
			write.DescriptorType = vk.DescriptorTypeSampler
			write.PImageInfo = &imageInfos[len(imageInfos)-1]

		case gputypes.TextureViewBinding:
			// A sampled image is in the layout its barriers put a read-only
			// texture in (deriveImageLayout): SHADER_READ_ONLY_OPTIMAL for a
			// colour format, DEPTH_STENCIL_READ_ONLY_OPTIMAL for a depth or
			// stencil one, which is also where a read-only depth attachment
			// sits, so a pass can sample the depth it tests against.
			imageInfo := vk.DescriptorImageInfo{
				ImageView:   vk.ImageView(res.TextureView),
				ImageLayout: vk.ImageLayoutShaderReadOnlyOptimal,
			}
			if isDepthStencilFormat(res.Format) {
				imageInfo.ImageLayout = vk.ImageLayoutDepthStencilReadOnlyOptimal
			}
			write.DescriptorType = vk.DescriptorTypeSampledImage
			// A storage texture is a storage image, read and written in
			// GENERAL, the layout its barriers put it in (textureBarrierState).
			if dt, ok := bindingTypes[entry.Binding]; ok && dt == vk.DescriptorTypeStorageImage {
				write.DescriptorType = dt
				imageInfo.ImageLayout = vk.ImageLayoutGeneral
			}
			imageInfos = append(imageInfos, imageInfo)
			write.PImageInfo = &imageInfos[len(imageInfos)-1]

		default:
			return fmt.Errorf("unsupported binding resource type: %T", entry.Resource)
		}

		writes = append(writes, write)
	}

	if len(writes) > 0 {
		vkUpdateDescriptorSets(d.cmds, d.handle, uint32(len(writes)), &writes[0], 0, nil)
	}

	return nil
}

// DestroyBindGroup destroys a bind group.
func (d *Device) DestroyBindGroup(group hal.BindGroup) {
	vkGroup, ok := group.(*BindGroup)
	if !ok || vkGroup == nil {
		return
	}

	if vkGroup.handle != 0 && vkGroup.pool != nil && d.descriptorAllocator != nil {
		_ = d.descriptorAllocator.Free(vkGroup.pool, vkGroup.handle)
		vkGroup.handle = 0
		vkGroup.pool = nil
	}

	vkGroup.device = nil
}

// CreatePipelineLayout creates a pipeline layout.
func (d *Device) CreatePipelineLayout(desc *hal.PipelineLayoutDescriptor) (hal.PipelineLayout, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: pipeline layout descriptor is nil in Vulkan.CreatePipelineLayout — core validation gap")
	}

	// Convert bind group layouts to descriptor set layouts
	setLayouts := make([]vk.DescriptorSetLayout, 0, len(desc.BindGroupLayouts))
	for _, layout := range desc.BindGroupLayouts {
		vkLayout, ok := layout.(*BindGroupLayout)
		if !ok || vkLayout == nil {
			return nil, fmt.Errorf("vulkan: invalid bind group layout")
		}
		setLayouts = append(setLayouts, vkLayout.handle)
	}

	// Convert push constant ranges
	pushConstantRanges := make([]vk.PushConstantRange, 0, len(desc.PushConstantRanges))
	for _, pcr := range desc.PushConstantRanges {
		pushConstantRanges = append(pushConstantRanges, vk.PushConstantRange{
			StageFlags: shaderStagesToVk(pcr.Stages),
			Offset:     pcr.Range.Start,
			Size:       pcr.Range.End - pcr.Range.Start,
		})
	}

	createInfo := vk.PipelineLayoutCreateInfo{
		SType:          vk.StructureTypePipelineLayoutCreateInfo,
		SetLayoutCount: uint32(len(setLayouts)),
	}

	if len(setLayouts) > 0 {
		createInfo.PSetLayouts = &setLayouts[0]
	}

	if len(pushConstantRanges) > 0 {
		createInfo.PushConstantRangeCount = uint32(len(pushConstantRanges))
		createInfo.PPushConstantRanges = &pushConstantRanges[0]
	}

	var layout vk.PipelineLayout
	result := vkCreatePipelineLayout(d.cmds, d.handle, &createInfo, nil, &layout)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreatePipelineLayout failed: %d", result)
	}

	pl := &PipelineLayout{
		bindingLayout: bindingLayout{handle: layout, sets: setLayouts, pushConstants: pushConstantRanges},
		device:        d,
	}
	d.live.pipelineLayouts.add(layout)
	if desc.Label != "" {
		d.setObjectName(vk.ObjectTypePipelineLayout, uint64(layout), desc.Label)
	} else {
		d.setObjectName(vk.ObjectTypePipelineLayout, uint64(layout), "PipelineLayout")
	}
	return pl, nil
}

// DestroyPipelineLayout destroys a pipeline layout.
func (d *Device) DestroyPipelineLayout(layout hal.PipelineLayout) {
	vkLayout, ok := layout.(*PipelineLayout)
	if !ok || vkLayout == nil {
		return
	}

	if vkLayout.handle != 0 {
		d.live.pipelineLayouts.remove(vkLayout.handle)
		vkDestroyPipelineLayout(d.cmds, d.handle, vkLayout.handle, nil)
		vkLayout.handle = 0
	}

	vkLayout.device = nil
}

// CreateShaderModule creates a shader module.
func (d *Device) CreateShaderModule(desc *hal.ShaderModuleDescriptor) (hal.ShaderModule, error) {
	if desc == nil {
		return nil, fmt.Errorf("BUG: shader module descriptor is nil in Vulkan.CreateShaderModule — core validation gap")
	}

	var spirv []uint32

	// Compile shader source to SPIR-V.
	// WGSL compilation via naga is required for Intel Iris Xe compatibility -
	// hardcoded SPIR-V from external tools can fail silently on Intel drivers.
	switch {
	case desc.Source.WGSL != "":
		spirvBytes, err := naga.Compile(desc.Source.WGSL)
		if err != nil {
			return nil, fmt.Errorf("vulkan: naga WGSL compilation failed: %w", err)
		}
		// Convert bytes to uint32 slice
		if len(spirvBytes)%4 != 0 {
			return nil, fmt.Errorf("vulkan: naga output size not aligned to 4 bytes")
		}
		spirv = make([]uint32, len(spirvBytes)/4)
		for i := range spirv {
			spirv[i] = binary.LittleEndian.Uint32(spirvBytes[i*4:])
		}
	case len(desc.Source.SPIRV) > 0:
		spirv = desc.Source.SPIRV
	default:
		return nil, fmt.Errorf("vulkan: shader module requires WGSL or SPIR-V source")
	}

	createInfo := vk.ShaderModuleCreateInfo{
		SType:    vk.StructureTypeShaderModuleCreateInfo,
		CodeSize: uintptr(len(spirv) * 4), // Size in bytes (uint32 = 4 bytes)
		PCode:    &spirv[0],
	}

	var module vk.ShaderModule
	result := vkCreateShaderModule(d.cmds, d.handle, &createInfo, nil, &module)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateShaderModule failed: %d", result)
	}

	sourceType := "SPIR-V"
	if desc.Source.WGSL != "" {
		sourceType = "WGSL"
	}
	hal.Logger().Debug("vulkan: shader module compiled",
		"source", sourceType,
		"spirvWords", len(spirv),
	)

	sm := &ShaderModule{
		handle: module,
		device: d,
	}
	d.live.shaderModules.add(module)
	if desc.Label != "" {
		d.setObjectName(vk.ObjectTypeShaderModule, uint64(module), desc.Label)
	} else {
		d.setObjectName(vk.ObjectTypeShaderModule, uint64(module), "ShaderModule("+sourceType+")")
	}
	return sm, nil
}

// DestroyShaderModule destroys a shader module.
func (d *Device) DestroyShaderModule(module hal.ShaderModule) {
	vkModule, ok := module.(*ShaderModule)
	if !ok || vkModule == nil {
		return
	}

	if vkModule.handle != 0 {
		d.live.shaderModules.remove(vkModule.handle)
		vkDestroyShaderModule(d.cmds, d.handle, vkModule.handle, nil)
		vkModule.handle = 0
	}

	vkModule.device = nil
}

// acquireAllocator returns a command allocator (pool) for a new encoder.
// If the free list has a recycled pool, it pops one. Otherwise, it creates
// a new VkCommandPool with TRANSIENT_BIT (VK-POOL-001, VK-CMD-001).
//
// Command buffer allocation is deferred to BeginEncoding — the encoder's
// internal free list batch-allocates via vkAllocateCommandBuffers on demand.
func (d *Device) acquireAllocator() (commandAllocator, error) {
	d.allocatorMu.Lock()
	if n := len(d.freeAllocators); n > 0 {
		alloc := d.freeAllocators[n-1]
		d.freeAllocators = d.freeAllocators[:n-1]
		d.allocatorMu.Unlock()
		return alloc, nil
	}
	d.allocatorMu.Unlock()

	// Create a new dedicated pool with TRANSIENT_BIT (short-lived buffers).
	// No RESET_COMMAND_BUFFER_BIT — we use vkResetCommandPool for batch reset
	// instead of individual vkResetCommandBuffer. Mesa explicitly warns against
	// RESET_COMMAND_BUFFER_BIT: it prevents single large allocator optimization.
	createInfo := vk.CommandPoolCreateInfo{
		SType:            vk.StructureTypeCommandPoolCreateInfo,
		Flags:            vk.CommandPoolCreateFlags(vk.CommandPoolCreateTransientBit),
		QueueFamilyIndex: d.graphicsFamily,
	}

	var pool vk.CommandPool
	result := vkCreateCommandPool(d.cmds, d.handle, &createInfo, nil, &pool)
	if result != vk.Success {
		return commandAllocator{}, fmt.Errorf("vulkan: vkCreateCommandPool failed: %d", result)
	}

	d.setObjectName(vk.ObjectTypeCommandPool, uint64(pool), "CommandPool")
	d.live.commandPools.add(pool)

	return commandAllocator{pool: pool}, nil
}

// recyclePool resets a command pool and returns it to the free list for reuse.
// vkResetCommandPool returns all allocated command buffers to the "initial" state,
// allowing them to be re-allocated by the next encoder that acquires this pool.
// Without this reset, vkAllocateCommandBuffers may return CBs still in
// "executable" state, causing VUID-vkBeginCommandBuffer-commandBuffer-00049.
func (d *Device) recyclePool(pool vk.CommandPool) {
	if pool == 0 {
		return
	}
	d.cmds.ResetCommandPool(d.handle, pool, 0)
	d.allocatorMu.Lock()
	d.freeAllocators = append(d.freeAllocators, commandAllocator{pool: pool})
	d.allocatorMu.Unlock()
}

// CreateCommandEncoder creates a command encoder with its own dedicated
// VkCommandPool. Command buffers are allocated on demand from the pool's
// internal free list (VK-CMD-001). This per-encoder pool design matches
// Rust wgpu-hal and eliminates races between pool reset and buffer freeing
// that caused "Couldn't find VkCommandBuffer Object" crashes (VK-POOL-001).
// Uses sync.Pool for CommandEncoder struct reuse (VK-PERF-003).
func (d *Device) CreateCommandEncoder(desc *hal.CommandEncoderDescriptor) (hal.CommandEncoder, error) {
	alloc, err := d.acquireAllocator()
	if err != nil {
		return nil, err
	}

	// Reuse CommandEncoder from pool (VK-PERF-003).
	e := encoderPool.Get().(*CommandEncoder)
	e.device = d
	e.pool = alloc.pool
	e.active = 0
	e.label = desc.Label
	// free and discarded slices may retain capacity from previous use — clear length.
	e.free = e.free[:0]
	e.discarded = e.discarded[:0]
	return e, nil
}

// WaitIdle waits for all GPU operations to complete.
func (d *Device) WaitIdle() error {
	result := vkDeviceWaitIdle(d)
	if result != vk.Success {
		return fmt.Errorf("vulkan: vkDeviceWaitIdle failed: %d", result)
	}
	return nil
}

// ResetCommandPool resets all recycled command pools.
// Call this after ensuring all submitted command buffers have completed (e.g., after WaitIdle).
func (d *Device) ResetCommandPool() error {
	d.allocatorMu.Lock()
	defer d.allocatorMu.Unlock()

	for _, alloc := range d.freeAllocators {
		result := d.cmds.ResetCommandPool(d.handle, alloc.pool, 0)
		if result != vk.Success {
			return fmt.Errorf("vulkan: vkResetCommandPool failed: %d", result)
		}
	}
	return nil
}

// FreeCommandBuffer recycles a command buffer's pool for reuse.
// Only call this AFTER the GPU has finished using the command buffer (after fence wait).
// The pool is returned to the free list and reset on next acquireAllocator.
// This is only called for standalone-mode encoders where the CommandBuffer
// carries pool ownership. Pool-managed encoders use ResetAll instead.
func (d *Device) FreeCommandBuffer(cmdBuffer hal.CommandBuffer) {
	vkCmdBuf, ok := cmdBuffer.(*CommandBuffer)
	if !ok || vkCmdBuf.handle == 0 || vkCmdBuf.pool == 0 {
		return
	}

	// Recycle the pool back to the free list.
	// The pool contains all command buffers (active + free + discarded from
	// the encoder). Destroying the pool frees them all; resetting the pool
	// puts them all back to initial state.
	d.recyclePool(vkCmdBuf.pool)

	vkCmdBuf.handle = 0
	vkCmdBuf.pool = 0
	cmdBufferResultPool.Put(vkCmdBuf)
}

// destroyAllocators destroys all recycled command pools in the free list.
// Called during device shutdown after all GPU work is complete.
func (d *Device) destroyAllocators() {
	d.allocatorMu.Lock()
	defer d.allocatorMu.Unlock()

	for _, alloc := range d.freeAllocators {
		// Destroying a pool implicitly frees all command buffers allocated from it.
		d.live.commandPools.remove(alloc.pool)
		vkDestroyCommandPool(d.cmds, d.handle, alloc.pool, nil)
	}
	d.freeAllocators = d.freeAllocators[:0]
}

// CreateFence creates a synchronization fence.
func (d *Device) CreateFence() (hal.Fence, error) {
	createInfo := vk.FenceCreateInfo{
		SType: vk.StructureTypeFenceCreateInfo,
		Flags: 0, // Not signaled initially
	}

	var fence vk.Fence
	result := vkCreateFence(d.cmds, d.handle, &createInfo, nil, &fence)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateFence failed: %d", result)
	}

	f := &Fence{
		handle: fence,
		device: d,
	}
	d.live.fences.add(fence)
	d.setObjectName(vk.ObjectTypeFence, uint64(fence), "Fence")
	return f, nil
}

// DestroyFence destroys a fence.
func (d *Device) DestroyFence(fence hal.Fence) {
	vkFence, ok := fence.(*Fence)
	if !ok || vkFence == nil {
		return
	}

	if vkFence.handle != 0 {
		d.live.fences.remove(vkFence.handle)
		vkDestroyFence(d.cmds, d.handle, vkFence.handle, nil)
		vkFence.handle = 0
	}

	vkFence.device = nil
}

// Wait waits for a fence to reach the specified value.
// Note: Standard Vulkan fences don't support timeline values, so value is ignored.
// For timeline semantics, use VK_KHR_timeline_semaphore extension.
func (d *Device) Wait(fence hal.Fence, _ uint64, timeout time.Duration) (bool, error) {
	vkFence, ok := fence.(*Fence)
	if !ok || vkFence == nil {
		return false, fmt.Errorf("vulkan: invalid fence")
	}

	// Convert timeout to nanoseconds
	timeoutNs := uint64(timeout.Nanoseconds())
	if timeout < 0 {
		timeoutNs = ^uint64(0) // UINT64_MAX for infinite wait
	}

	result := vkWaitForFences(d.cmds, d.handle, 1, &vkFence.handle, vk.Bool32(vk.True), timeoutNs)
	switch result {
	case vk.Success:
		return true, nil
	case vk.Timeout:
		return false, nil
	case vk.ErrorDeviceLost:
		return false, hal.ErrDeviceLost
	default:
		return false, fmt.Errorf("vulkan: vkWaitForFences failed: %d", result)
	}
}

// ResetFence resets a fence to the unsignaled state.
func (d *Device) ResetFence(fence hal.Fence) error {
	vkFence, ok := fence.(*Fence)
	if !ok || vkFence == nil {
		return fmt.Errorf("vulkan: invalid fence")
	}

	result := vkResetFences(d.cmds, d.handle, 1, &vkFence.handle)
	if result != vk.Success {
		return fmt.Errorf("vulkan: vkResetFences failed: %d", result)
	}
	return nil
}

// GetFenceStatus returns true if the fence is signaled (non-blocking).
// Uses vkGetFenceStatus for efficient polling without blocking.
func (d *Device) GetFenceStatus(fence hal.Fence) (bool, error) {
	vkFence, ok := fence.(*Fence)
	if !ok || vkFence == nil {
		return false, fmt.Errorf("vulkan: invalid fence")
	}

	result := d.cmds.GetFenceStatus(d.handle, vkFence.handle)
	switch result {
	case vk.Success:
		return true, nil // Fence is signaled
	case vk.NotReady:
		return false, nil // Fence is not signaled yet
	case vk.ErrorDeviceLost:
		return false, hal.ErrDeviceLost
	default:
		return false, fmt.Errorf("vulkan: vkGetFenceStatus failed: %d", result)
	}
}

// Destroy releases the device.
func (d *Device) Destroy() {
	surfaces, ok := d.beginDestroy()
	if !ok {
		return
	}

	// One device-wide drain covers every configured swapchain. On device loss,
	// abandon child handles and let vkDestroyDevice reclaim native storage.
	drained := vkDeviceWaitIdle(d) == vk.Success
	for _, surface := range surfaces {
		surface.releaseConfiguredDevice(d, drained)
	}

	// Destroy unified fence (timeline semaphore or fencePool).
	if d.timelineFence != nil {
		d.timelineFence.destroy(d.cmds, d.handle)
		d.timelineFence = nil
	}

	// VK-SYNC-001: Destroy relay semaphores used for submission ordering.
	if d.queue != nil && d.queue.relay != nil {
		d.queue.relay.destroy(d.cmds, d.handle)
		d.queue.relay = nil
	}

	// Destroy all recycled command pools (VK-POOL-001).
	d.destroyAllocators()

	if d.descriptorAllocator != nil {
		d.descriptorAllocator.Destroy()
		d.descriptorAllocator = nil
	}

	if d.renderPassCache != nil {
		d.renderPassCache.Destroy()
		d.renderPassCache = nil
	}

	d.destroyPipelineCache()

	if drained {
		d.sweepLive()
	}

	if d.allocator != nil {
		d.allocator.Destroy()
		d.allocator = nil
	}

	if d.handle != 0 {
		vkDestroyDevice(d.handle, nil)
		d.handle = 0
	}
}

// GetRenderPassCache returns the render pass cache, creating it if needed.
func (d *Device) GetRenderPassCache() *RenderPassCache {
	if d.renderPassCache == nil {
		d.renderPassCache = NewRenderPassCache(d.handle, d.cmds)
	}
	return d.renderPassCache
}

// Vulkan function wrappers using Commands methods

func vkCreateCommandPool(cmds *vk.Commands, device vk.Device, createInfo *vk.CommandPoolCreateInfo, allocator *vk.AllocationCallbacks, pool *vk.CommandPool) vk.Result {
	return cmds.CreateCommandPool(device, createInfo, allocator, pool)
}

//nolint:unparam // Vulkan API wrapper — signature mirrors vkDestroyCommandPool spec
func vkDestroyCommandPool(cmds *vk.Commands, device vk.Device, pool vk.CommandPool, allocator *vk.AllocationCallbacks) {
	cmds.DestroyCommandPool(device, pool, allocator)
}

func vkCreateSampler(cmds *vk.Commands, device vk.Device, createInfo *vk.SamplerCreateInfo, allocator *vk.AllocationCallbacks, sampler *vk.Sampler) vk.Result {
	return cmds.CreateSampler(device, createInfo, allocator, sampler)
}

func vkDestroySampler(cmds *vk.Commands, device vk.Device, sampler vk.Sampler, allocator *vk.AllocationCallbacks) {
	cmds.DestroySampler(device, sampler, allocator)
}

func vkCreateShaderModule(cmds *vk.Commands, device vk.Device, createInfo *vk.ShaderModuleCreateInfo, allocator *vk.AllocationCallbacks, module *vk.ShaderModule) vk.Result {
	return cmds.CreateShaderModule(device, createInfo, allocator, module)
}

func vkDestroyShaderModule(cmds *vk.Commands, device vk.Device, module vk.ShaderModule, allocator *vk.AllocationCallbacks) {
	cmds.DestroyShaderModule(device, module, allocator)
}

func vkCreatePipelineLayout(cmds *vk.Commands, device vk.Device, createInfo *vk.PipelineLayoutCreateInfo, allocator *vk.AllocationCallbacks, layout *vk.PipelineLayout) vk.Result {
	return cmds.CreatePipelineLayout(device, createInfo, allocator, layout)
}

func vkDestroyPipelineLayout(cmds *vk.Commands, device vk.Device, layout vk.PipelineLayout, allocator *vk.AllocationCallbacks) {
	cmds.DestroyPipelineLayout(device, layout, allocator)
}

func vkCreateDescriptorSetLayout(cmds *vk.Commands, device vk.Device, createInfo *vk.DescriptorSetLayoutCreateInfo, allocator *vk.AllocationCallbacks, layout *vk.DescriptorSetLayout) vk.Result {
	return cmds.CreateDescriptorSetLayout(device, createInfo, allocator, layout)
}

func vkDestroyDescriptorSetLayout(cmds *vk.Commands, device vk.Device, layout vk.DescriptorSetLayout, allocator *vk.AllocationCallbacks) {
	cmds.DestroyDescriptorSetLayout(device, layout, allocator)
}

func vkCreateImageView(cmds *vk.Commands, device vk.Device, createInfo *vk.ImageViewCreateInfo, allocator *vk.AllocationCallbacks, view *vk.ImageView) vk.Result {
	return cmds.CreateImageView(device, createInfo, allocator, view)
}

func vkDestroyImageView(cmds *vk.Commands, device vk.Device, view vk.ImageView, allocator *vk.AllocationCallbacks) {
	cmds.DestroyImageView(device, view, allocator)
}

func vkCreateFence(cmds *vk.Commands, device vk.Device, createInfo *vk.FenceCreateInfo, allocator *vk.AllocationCallbacks, fence *vk.Fence) vk.Result {
	return cmds.CreateFence(device, createInfo, allocator, fence)
}

func vkDestroyFence(cmds *vk.Commands, device vk.Device, fence vk.Fence, allocator *vk.AllocationCallbacks) {
	cmds.DestroyFence(device, fence, allocator)
}

func vkWaitForFences(cmds *vk.Commands, device vk.Device, fenceCount uint32, fences *vk.Fence, waitAll vk.Bool32, timeout uint64) vk.Result {
	return cmds.WaitForFences(device, fenceCount, fences, waitAll, timeout)
}

func vkResetFences(cmds *vk.Commands, device vk.Device, fenceCount uint32, fences *vk.Fence) vk.Result {
	return cmds.ResetFences(device, fenceCount, fences)
}
