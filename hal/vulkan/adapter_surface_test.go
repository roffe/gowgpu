//go:build !(js && wasm)

package vulkan

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

func TestSelectPresentGraphicsQueueFamilyRejectsSplitQueues(t *testing.T) {
	families := []vk.QueueFamilyProperties{
		{QueueFlags: vk.QueueFlags(vk.QueueGraphicsBit), QueueCount: 1},
		{QueueFlags: vk.QueueFlags(vk.QueueTransferBit), QueueCount: 1},
	}

	if _, err := selectPresentGraphicsQueueFamily(families, []bool{false, true}); err == nil {
		t.Fatal("split graphics/present queues were accepted")
	}
}

func TestSelectPresentGraphicsQueueFamilyUsesSameFamily(t *testing.T) {
	families := []vk.QueueFamilyProperties{
		{QueueFlags: vk.QueueFlags(vk.QueueGraphicsBit), QueueCount: 1},
		{QueueFlags: vk.QueueFlags(vk.QueueGraphicsBit), QueueCount: 1},
	}

	got, err := selectPresentGraphicsQueueFamily(families, []bool{false, true})
	if err != nil {
		t.Fatalf("selectPresentGraphicsQueueFamily() error: %v", err)
	}
	if got != 1 {
		t.Fatalf("selected queue family = %d, want 1", got)
	}
}

func TestQuerySurfaceFormatsFailsClosed(t *testing.T) {
	_, err := querySurfaceFormatsWith(func(_ *uint32, _ *vk.SurfaceFormatKHR) vk.Result {
		return vk.ErrorSurfaceLostKhr
	})
	if err == nil {
		t.Fatal("failed surface format query was accepted")
	}
}

func TestQuerySurfaceFormatsRetriesIncomplete(t *testing.T) {
	countCalls := 0
	fillCalls := 0
	formats, err := querySurfaceFormatsWith(func(count *uint32, formats *vk.SurfaceFormatKHR) vk.Result {
		if formats == nil {
			countCalls++
			*count = 1
			return vk.Success
		}
		fillCalls++
		if fillCalls == 1 {
			return vk.Incomplete
		}
		*formats = vk.SurfaceFormatKHR{Format: vk.FormatR8g8b8a8Unorm, ColorSpace: vk.ColorSpaceSrgbNonlinearKhr}
		*count = 1
		return vk.Success
	})
	if err != nil {
		t.Fatalf("querySurfaceFormatsWith() error: %v", err)
	}
	if len(formats) != 1 || countCalls != 2 || fillCalls != 2 {
		t.Fatalf("query calls/results = (%d, %d, %d), want (2, 2, 1)", countCalls, fillCalls, len(formats))
	}
}

func TestMakeSurfaceSnapshotDoesNotFabricateEmptyCapabilities(t *testing.T) {
	_, err := makeSurfaceSnapshot(vk.SurfaceCapabilitiesKHR{}, nil, []vk.PresentModeKHR{vk.PresentModeFifoKhr})
	if err == nil {
		t.Fatal("empty format query produced fabricated capabilities")
	}

	_, err = makeSurfaceSnapshot(vk.SurfaceCapabilitiesKHR{}, []vk.SurfaceFormatKHR{{Format: vk.FormatR8g8b8a8Unorm}}, nil)
	if err == nil {
		t.Fatal("empty present-mode query produced fabricated capabilities")
	}
}

func TestMakeSurfaceSnapshotPreservesFormatColorSpacePairs(t *testing.T) {
	formats := []vk.SurfaceFormatKHR{
		{Format: vk.FormatR8g8b8a8Unorm, ColorSpace: vk.ColorSpaceSrgbNonlinearKhr},
		{Format: vk.FormatR8g8b8a8Unorm, ColorSpace: vk.ColorSpaceDisplayP3NonlinearExt},
	}
	snapshot, err := makeSurfaceSnapshot(
		vk.SurfaceCapabilitiesKHR{SupportedCompositeAlpha: vk.CompositeAlphaFlagsKHR(vk.CompositeAlphaOpaqueBitKhr)},
		formats,
		[]vk.PresentModeKHR{vk.PresentModeFifoKhr},
	)
	if err != nil {
		t.Fatalf("makeSurfaceSnapshot() error: %v", err)
	}
	if len(snapshot.formats) != len(formats) {
		t.Fatalf("raw format pair count = %d, want %d", len(snapshot.formats), len(formats))
	}
	for i := range formats {
		if snapshot.formats[i] != formats[i] {
			t.Errorf("raw format pair %d = %+v, want %+v", i, snapshot.formats[i], formats[i])
		}
	}
	if len(snapshot.public.Formats) != 2 {
		t.Fatalf("public format projection count = %d, want 2", len(snapshot.public.Formats))
	}
	if snapshot.public.Formats[0] != gputypes.TextureFormatRGBA8Unorm || snapshot.public.Formats[1] != gputypes.TextureFormatRGBA8Unorm {
		t.Fatalf("public formats = %v, want two RGBA8 entries", snapshot.public.Formats)
	}
}

func TestMakeSurfaceSnapshotRejectsUnknownMappings(t *testing.T) {
	_, err := makeSurfaceSnapshot(
		vk.SurfaceCapabilitiesKHR{SupportedCompositeAlpha: vk.CompositeAlphaFlagsKHR(vk.CompositeAlphaOpaqueBitKhr)},
		[]vk.SurfaceFormatKHR{{Format: vk.FormatR8g8b8a8Unorm, ColorSpace: vk.ColorSpaceSrgbNonlinearKhr}},
		[]vk.PresentModeKHR{vk.PresentModeKHR(0x7fffffff)},
	)
	if err == nil {
		t.Fatal("unknown present mode was converted into a fabricated FIFO capability")
	}

	_, err = makeSurfaceSnapshot(
		vk.SurfaceCapabilitiesKHR{},
		[]vk.SurfaceFormatKHR{{Format: vk.FormatR8g8b8a8Unorm, ColorSpace: vk.ColorSpaceSrgbNonlinearKhr}},
		[]vk.PresentModeKHR{vk.PresentModeFifoKhr},
	)
	if err == nil {
		t.Fatal("unknown composite alpha flags were converted into fabricated opaque capability")
	}
}

// VeriSilicon's driver advertises timeline semaphores and crashes with
// them, so its devices take the binary fence pool.
func TestUsesTimelineSemaphore(t *testing.T) {
	if !usesTimelineSemaphore(0x10DE, vk.True) {
		t.Error("NVIDIA with timeline semaphores does not use them")
	}
	if usesTimelineSemaphore(0x10DE, vk.False) {
		t.Error("a device without timeline semaphores uses them")
	}
	if usesTimelineSemaphore(vendorVeriSilicon, vk.True) {
		t.Error("VeriSilicon uses timeline semaphores")
	}
}

// Depth32FloatStencil8 is a feature only where the device renders to
// D32_SFLOAT_S8_UINT: Vulkan requires one of it and D24_UNORM_S8_UINT.
func TestDepth32FloatStencil8Feature(t *testing.T) {
	var features vk.PhysicalDeviceFeatures
	if featuresFromPhysicalDevice(&features, false, false).Contains(gputypes.FeatureDepth32FloatStencil8) {
		t.Error("Depth32FloatStencil8 without D32_SFLOAT_S8_UINT")
	}
	if !featuresFromPhysicalDevice(&features, false, true).Contains(gputypes.FeatureDepth32FloatStencil8) {
		t.Error("no Depth32FloatStencil8 with D32_SFLOAT_S8_UINT")
	}
}

// A loader without vkEnumerateInstanceVersion is Vulkan 1.0, which fails an
// instance asking for more, and a device works at the lower of its version
// and the instance's.
func TestAPIVersions(t *testing.T) {
	if got := instanceAPIVersion(&vk.Commands{}); got != vkMakeVersion(1, 0, 0) {
		t.Errorf("instance version on a 1.0 loader = %#x, want 1.0", got)
	}
	a := &Adapter{instance: &Instance{apiVersion: vkMakeVersion(1, 0, 0)}, properties: vk.PhysicalDeviceProperties{ApiVersion: vkMakeVersion(1, 3, 280)}}
	if got := a.apiVersion(); got != vkMakeVersion(1, 0, 0) {
		t.Errorf("1.3 device on a 1.0 instance works at %#x, want 1.0", got)
	}
	if v := shaderOptions(a.apiVersion()).Version; v.Major != 1 || v.Minor != 0 {
		t.Errorf("SPIR-V %d.%d on a 1.0 device, want 1.0", v.Major, v.Minor)
	}
}

// Limits as wgpu-hal clamps them: the 2D size every 2D use allows, the
// per-stage counts within the per-set ones shared by two stages and within
// lavapipe's maxFragmentCombinedOutputResources and maxPerStageResources,
// and binding sizes within 2^31-1.
func TestLimitsFromPropsClamps(t *testing.T) {
	props := &vk.PhysicalDeviceProperties{VendorID: 0x10005, Limits: vk.PhysicalDeviceLimits{
		MaxImageDimension2D: 16384, MaxImageDimensionCube: 8192, MaxFramebufferWidth: 16384, MaxFramebufferHeight: 4096,
		MaxPerStageDescriptorSamplers: 1 << 20, MaxPerStageDescriptorUniformBuffers: 1 << 20,
		MaxPerStageDescriptorStorageBuffers: 1 << 20, MaxPerStageDescriptorSampledImages: 1 << 20,
		MaxPerStageDescriptorStorageImages: 1 << 20,
		MaxDescriptorSetSamplers:           1 << 20, MaxDescriptorSetUniformBuffers: 1 << 20,
		MaxDescriptorSetStorageBuffers: 1 << 20, MaxDescriptorSetSampledImages: 1 << 20,
		MaxDescriptorSetStorageImages: 40,
		MaxPerStageResources:          1 << 20, MaxFragmentCombinedOutputResources: 104,
		MaxColorAttachments: 8, MaxFragmentOutputAttachments: 8,
		MaxStorageBufferRange: 1<<32 - 1, MaxUniformBufferRange: 65536,
	}}
	l := limitsFromProps(props, 1<<20)
	if l.MaxTextureDimension2D != 4096 {
		t.Errorf("MaxTextureDimension2D = %d, want the framebuffer height 4096", l.MaxTextureDimension2D)
	}
	if l.MaxStorageTexturesPerShaderStage != 20 {
		t.Errorf("MaxStorageTexturesPerShaderStage = %d, want 20, half the per-set count", l.MaxStorageTexturesPerShaderStage)
	}
	if sum := l.MaxStorageTexturesPerShaderStage + l.MaxStorageBuffersPerShaderStage + l.MaxColorAttachments; sum > 104 {
		t.Errorf("storage textures, storage buffers and colour attachments sum to %d, over 104", sum)
	}
	if l.MaxColorAttachments != 8 {
		t.Errorf("MaxColorAttachments = %d, want 8", l.MaxColorAttachments)
	}
	if l.MaxStorageBufferBindingSize != 1<<31-1 {
		t.Errorf("MaxStorageBufferBindingSize = %d, want 2^31-1", l.MaxStorageBufferBindingSize)
	}
}
