//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

import (
	"cmp"
	"fmt"
	"os"
	"runtime"
	"slices"
	"strings"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

const (
	extensionWaylandSurface = "VK_KHR_wayland_surface\x00"
	extensionXlibSurface    = "VK_KHR_xlib_surface\x00"
	portabilityEnumeration  = "VK_KHR_portability_enumeration\x00"
)

// Backend implements hal.Backend for Vulkan.
type Backend struct{}

// NewBackend returns a Vulkan backend instance.
func NewBackend() Backend { return Backend{} }

// disableRTSSLayer switches off RivaTuner's implicit Vulkan layer for this
// process on Windows, unless DISABLE_RTSS_LAYER is already set. The layer
// crashed vkCreateDevice in every gowgpu app on a Windows 11 machine with an
// RTX 5080 (RTSSVkLayer64.dll, read of address 0), and an FPS overlay has no
// business here anyway. The loader reads the variable through
// GetEnvironmentVariableW when it builds the layer list at instance creation,
// so setting it now is enough, and any value disables the layer.
func disableRTSSLayer() {
	if runtime.GOOS != "windows" {
		return
	}
	if _, set := os.LookupEnv("DISABLE_RTSS_LAYER"); !set {
		_ = os.Setenv("DISABLE_RTSS_LAYER", "1")
	}
}

// Variant returns the backend type identifier.
func (Backend) Variant() gputypes.Backend {
	return gputypes.BackendVulkan
}

// CreateInstance creates a new Vulkan instance.
func (Backend) CreateInstance(desc *hal.InstanceDescriptor) (hal.Instance, error) {
	disableRTSSLayer()
	platform, err := newPlatformInstanceState(desc)
	if err != nil {
		return nil, err
	}

	// Initialize Vulkan library
	if err := vk.Init(); err != nil {
		return nil, fmt.Errorf("vulkan: failed to initialize: %w", err)
	}

	// Create Commands and load global Vulkan functions
	cmds := vk.NewCommands()
	if err := cmds.LoadGlobal(); err != nil {
		return nil, fmt.Errorf("vulkan: failed to load global commands: %w", err)
	}

	// Prepare application info
	appName := []byte("gowgpu\x00")
	engineName := []byte("gowgpu\x00")

	appInfo := vk.ApplicationInfo{
		SType:              vk.StructureTypeApplicationInfo,
		PApplicationName:   &appName[0],
		ApplicationVersion: vkMakeVersion(1, 0, 0),
		PEngineName:        &engineName[0],
		EngineVersion:      vkMakeVersion(0, 1, 0),
		ApiVersion:         instanceAPIVersion(cmds),
	}

	// Enable VK_KHR_surface and every platform WSI extension that this
	// loader exposes. Linux can legitimately use Xlib and Wayland in the
	// same process (for example XWayland), so ambient session variables must
	// not select the instance ABI.
	availableExtensions, err := enumerateInstanceExtensions(cmds)
	if err != nil {
		return nil, fmt.Errorf("vulkan: enumerate instance extensions: %w", err)
	}
	wanted := append([]string{"VK_KHR_surface\x00"}, platformSurfaceExtensions()...)
	if runtime.GOOS == "darwin" {
		// MoltenVK is a portability driver: the loader lists it only when asked.
		wanted = append(wanted, portabilityEnumeration)
	}
	extensions := selectAvailableExtensions(wanted, availableExtensions)

	// Optional: validation layers for debug (only if available)
	var layers []string
	var validationEnabled bool
	if desc != nil && desc.Flags&gputypes.InstanceFlagsDebug != 0 {
		if isLayerAvailable(cmds, "VK_LAYER_KHRONOS_validation") {
			layers = append(layers, "VK_LAYER_KHRONOS_validation\x00")
			extensions = append(extensions, "VK_EXT_debug_utils\x00")
			validationEnabled = true
		}
		// Silently skip if validation layers not installed (Vulkan SDK not present)
	}

	// Create instance
	var pins runtime.Pinner
	defer pins.Unpin()
	createInfo := vk.InstanceCreateInfo{
		SType:                   vk.StructureTypeInstanceCreateInfo,
		PApplicationInfo:        &appInfo,
		EnabledExtensionCount:   uint32(len(extensions)),
		PpEnabledExtensionNames: cNames(&pins, extensions),
		EnabledLayerCount:       uint32(len(layers)),
		PpEnabledLayerNames:     cNames(&pins, layers),
	}

	if slices.Contains(extensions, portabilityEnumeration) {
		createInfo.Flags = vk.InstanceCreateFlags(vk.InstanceCreateEnumeratePortabilityBitKhr)
	}

	var instance vk.Instance
	result := cmds.CreateInstance(&createInfo, nil, &instance)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateInstance failed: %d", result)
	}

	// Load instance-level commands
	if err := cmds.LoadInstance(instance); err != nil {
		cmds.DestroyInstance(instance, nil)
		return nil, fmt.Errorf("vulkan: failed to load instance commands: %w", err)
	}

	// Set vkGetDeviceProcAddr for device function loading.
	// Some drivers (e.g., Intel) don't support loading it with instance=0.
	vk.SetDeviceProcAddr(instance)

	// Keep references alive
	runtime.KeepAlive(appName)
	runtime.KeepAlive(engineName)
	runtime.KeepAlive(extensions)
	runtime.KeepAlive(layers)

	inst := &Instance{
		handle:       instance,
		cmds:         *cmds,
		apiVersion:   appInfo.ApiVersion,
		debugEnabled: validationEnabled,
		platform:     platform,
		extensions:   trimNUL(extensions),
	}

	// Create debug messenger when validation layers are active.
	// This captures validation errors and logs them via Go's log package.
	if validationEnabled {
		inst.debugMessenger = createDebugMessenger(inst)
	}

	hal.Logger().Info("vulkan: instance created",
		"apiVersion", fmt.Sprintf("%d.%d.%d", vkVersionMajor(appInfo.ApiVersion), vkVersionMinor(appInfo.ApiVersion), vkVersionPatch(appInfo.ApiVersion)),
		"validation", validationEnabled,
	)

	return inst, nil
}

// instanceAPIVersion is the Vulkan version an instance asks for: 1.3, which
// a video decoder sharing the device needs, unless the loader is Vulkan 1.0,
// which fails an instance asking for more (VK_ERROR_INCOMPATIBLE_DRIVER).
// From 1.1 on a loader takes any version, and a device's is the lower of it
// and its own.
func instanceAPIVersion(cmds *vk.Commands) uint32 {
	var loader uint32
	if cmds.EnumerateInstanceVersion(&loader) != vk.Success {
		return vkMakeVersion(1, 0, 0)
	}
	return vkMakeVersion(1, 3, 0)
}

// Instance implements hal.Instance for Vulkan.
type Instance struct {
	handle         vk.Instance
	cmds           vk.Commands
	apiVersion     uint32 // asked for, see instanceAPIVersion
	debugMessenger vk.DebugUtilsMessengerEXT
	debugEnabled   bool
	platform       platformInstanceState
	extensions     []string // enabled
}

// cNames is names, each ending in a NUL, as the array of C strings a create
// info takes, the array and the names pinned in pins, or nil for no names.
func cNames(pins *runtime.Pinner, names []string) **byte {
	if len(names) == 0 {
		return nil
	}
	ptrs := make([]*byte, len(names))
	for i, name := range names {
		ptrs[i] = vk.Pin(pins, unsafe.StringData(name))
	}
	return vk.Pin(pins, &ptrs[0])
}

// trimNUL returns names without the NUL they are passed to Vulkan with.
func trimNUL(names []string) []string {
	out := make([]string, len(names))
	for i, n := range names {
		out[i] = strings.TrimSuffix(n, "\x00")
	}
	return out
}

// EnumerateAdapters returns available Vulkan adapters (physical devices).
func (i *Instance) EnumerateAdapters(surfaceHint hal.Surface) []hal.ExposedAdapter {
	// Get physical device count
	var count uint32
	i.cmds.EnumeratePhysicalDevices(i.handle, &count, nil)
	if count == 0 {
		return nil
	}

	// Get physical devices
	devices := make([]vk.PhysicalDevice, count)
	i.cmds.EnumeratePhysicalDevices(i.handle, &count, &devices[0])

	adapters := make([]hal.ExposedAdapter, 0, count)
	hal.Logger().Debug("vulkan: enumerating adapters", "count", count)
	for _, device := range devices {
		// Get device properties
		var props vk.PhysicalDeviceProperties
		i.cmds.GetPhysicalDeviceProperties(device, &props)

		// Get device features, and the Vulkan 1.2 ones the adapter offers
		// as features too: drawIndirectCount is optional and must be
		// enabled, being 1.2 is not enough.
		var features vk.PhysicalDeviceFeatures
		i.cmds.GetPhysicalDeviceFeatures(device, &features)
		var features12 vk.PhysicalDeviceVulkan12Features
		if i.cmds.HasPhysicalDeviceFeatures2() && min(props.ApiVersion, i.apiVersion) >= vkMakeVersion(1, 2, 0) {
			features12.SType = vk.StructureTypePhysicalDeviceVulkan12Features
			features2 := vk.PhysicalDeviceFeatures2{
				SType: vk.StructureTypePhysicalDeviceFeatures2,
				PNext: (*uintptr)(unsafe.Pointer(&features12)),
			}
			i.cmds.GetPhysicalDeviceFeatures2(device, &features2)
		}

		// Convert device type
		deviceType := gputypes.DeviceTypeOther
		switch props.DeviceType {
		case vk.PhysicalDeviceTypeDiscreteGpu:
			deviceType = gputypes.DeviceTypeDiscreteGPU
		case vk.PhysicalDeviceTypeIntegratedGpu:
			deviceType = gputypes.DeviceTypeIntegratedGPU
		case vk.PhysicalDeviceTypeVirtualGpu:
			deviceType = gputypes.DeviceTypeVirtualGPU
		case vk.PhysicalDeviceTypeCpu:
			deviceType = gputypes.DeviceTypeCPU
		}

		// Extract device name
		deviceName := cStringToGo(props.DeviceName[:])

		adapter := &Adapter{
			instance:       i,
			physicalDevice: device,
			properties:     props,
			features:       features,
		}

		adapterForExpose := hal.Adapter(adapter)
		if surfaceHint != nil {
			qualified, err := adapter.QualifySurface(surfaceHint)
			if err != nil {
				hal.Logger().Debug("vulkan: adapter rejected surface hint", "name", deviceName, "error", err)
				continue
			}
			adapterForExpose = qualified
		}

		hal.Logger().Info("vulkan: adapter found",
			"name", deviceName,
			"type", deviceType,
			"vendor", vendorIDToName(props.VendorID),
			"apiVersion", fmt.Sprintf("%d.%d.%d", vkVersionMajor(props.ApiVersion), vkVersionMinor(props.ApiVersion), vkVersionPatch(props.ApiVersion)),
		)

		adapters = append(adapters, hal.ExposedAdapter{
			Adapter: adapterForExpose,
			Info: gputypes.AdapterInfo{
				Name:       deviceName,
				Vendor:     vendorIDToName(props.VendorID),
				VendorID:   props.VendorID,
				DeviceID:   props.DeviceID,
				DeviceType: deviceType,
				Driver:     "Vulkan",
				DriverInfo: fmt.Sprintf("Vulkan %d.%d.%d",
					vkVersionMajor(props.ApiVersion),
					vkVersionMinor(props.ApiVersion),
					vkVersionPatch(props.ApiVersion)),
				Backend: gputypes.BackendVulkan,
			},
			Features: featuresFromPhysicalDevice(&features, features12.DrawIndirectCount != 0, adapter.supportsDepthAttachment(vk.FormatD32SfloatS8Uint)) | timestampFeature(&props),
			Capabilities: hal.Capabilities{
				Limits: limitsFromProps(&props, maxPerSetDescriptors(&i.cmds, device)),
				AlignmentsMask: hal.Alignments{
					BufferCopyOffset: 4,
					BufferCopyPitch:  256,
				},
				DownlevelCapabilities: downlevelCapabilitiesFromFeatures(&features,
					swapchainMutableFormat(min(props.ApiVersion, i.apiVersion), deviceExtensions(&i.cmds, device))),
			},
		})
	}

	return adapters
}

// Destroy releases the Vulkan instance.
func (i *Instance) Destroy() {
	if i.handle != 0 {
		// Destroy debug messenger before the instance (required by Vulkan spec).
		if i.debugMessenger != 0 {
			destroyDebugMessenger(i, i.debugMessenger)
			i.debugMessenger = 0
		}
		i.cmds.DestroyInstance(i.handle, nil)
		i.handle = 0
	}
}

// Surface implements hal.Surface for Vulkan.
type Surface struct {
	handle    vk.SurfaceKHR
	instance  *Instance
	swapchain *Swapchain
	device    *Device
}

// Configure configures the surface for presentation.
//
// Returns hal.ErrZeroArea if width or height is zero.
// This commonly happens when the window is minimized or not yet fully visible.
// Wait until the window has valid dimensions before calling Configure again.
func (s *Surface) Configure(device hal.Device, config *hal.SurfaceConfiguration) error {
	if s == nil {
		return fmt.Errorf("vulkan: surface is nil")
	}
	if config == nil {
		return fmt.Errorf("vulkan: surface configuration is nil")
	}
	// Validate dimensions first (before any side effects).
	// This matches wgpu-core behavior which returns ConfigureSurfaceError::ZeroArea.
	if config.Width == 0 || config.Height == 0 {
		return hal.ErrZeroArea
	}

	vkDevice, ok := device.(*Device)
	if !ok {
		return fmt.Errorf("vulkan: device is not a Vulkan device")
	}
	hal.Logger().Info("vulkan: surface configuring",
		"width", config.Width,
		"height", config.Height,
		"format", config.Format,
		"presentMode", config.PresentMode,
	)
	return s.createSwapchain(vkDevice, config)
}

// ActualExtent returns the dimensions selected for the swapchain. A defined
// Vulkan CurrentExtent is compositor-owned; otherwise the requested extent is
// clamped to the supported range. Returns (0, 0) if no swapchain is configured.
func (s *Surface) ActualExtent() (width, height uint32) {
	if s.swapchain == nil {
		return 0, 0
	}
	return s.swapchain.extent.Width, s.swapchain.extent.Height
}

func (s *Surface) detachSwapchainFromQueue(swapchain *Swapchain) {
	if s.device == nil || s.device.queue == nil {
		return
	}
	queue := s.device.queue
	queue.mu.Lock()
	defer queue.mu.Unlock()
	queue.release(swapchain)
}

// Unconfigure removes surface configuration.
func (s *Surface) Unconfigure(_ hal.Device) {
	if s == nil {
		return
	}
	if s.swapchain != nil {
		swapchain := s.swapchain
		if err := swapchain.destroyWithError(); err != nil {
			hal.Logger().Error("vulkan: failed to destroy swapchain during unconfigure", "error", err)
		}
		s.detachSwapchainFromQueue(swapchain)
		s.swapchain = nil
	}
	if s.device != nil {
		s.device.unregisterConfiguredSurface(s)
	}
	s.device = nil
}

// AcquireTexture acquires the next surface texture for rendering.
// Returns hal.ErrNotReady if no image is available (non-blocking mode).
func (s *Surface) AcquireTexture(_ hal.Fence) (*hal.AcquiredSurfaceTexture, error) {
	if s == nil {
		return nil, fmt.Errorf("vulkan: surface is nil")
	}
	if s.swapchain == nil {
		return nil, fmt.Errorf("vulkan: surface not configured")
	}

	// A discarded image comes back without an acquire, its acquire
	// semaphore waited for already.
	waited := s.swapchain.discarded
	texture, suboptimal, err := s.swapchain.acquireNextImage()
	if err != nil {
		return nil, err
	}

	// No image available right now - skip this frame
	if texture == nil {
		return nil, hal.ErrNotReady
	}

	// Register swapchain with queue for proper synchronization in Submit.
	// This ensures the queue waits for image acquisition before rendering
	// and signals completion before present.
	if s.device != nil && s.device.queue != nil {
		q := s.device.queue
		q.mu.Lock()
		s.swapchain.acquireUsed = waited
		q.hold(s.swapchain)
		q.mu.Unlock()
	}

	return &hal.AcquiredSurfaceTexture{
		Texture:    texture,
		Suboptimal: suboptimal,
	}, nil
}

// DiscardTexture discards a surface texture without presenting it.
func (s *Surface) DiscardTexture(_ hal.SurfaceTexture) {
	if s == nil {
		return
	}
	if s.swapchain != nil && s.swapchain.imageAcquired {
		// Vulkan has no way to give an acquired image back without
		// VK_EXT_swapchain_maintenance1, and a few kept acquired stop
		// vkAcquireNextImageKHR returning any: the next AcquireTexture
		// hands the image out again, as Dawn does. The acquire semaphore
		// is waited for by a submission of nothing when no Submit did, so
		// that it is unsignaled when its turn to be acquired with comes
		// round again.
		s.swapchain.imageAcquired = false
		s.swapchain.discarded = true
		if s.device != nil && s.device.queue != nil {
			q := s.device.queue
			q.mu.Lock()
			if err := q.consumeAcquire(s.swapchain); err != nil {
				hal.Logger().Warn("vulkan: discarded frame's acquire semaphore not consumed", "err", err)
			}
			q.release(s.swapchain)
			q.mu.Unlock()
		}
	}
}

// Destroy releases the surface.
func (s *Surface) Destroy() {
	if s == nil {
		return
	}
	if s.swapchain != nil {
		swapchain := s.swapchain
		if err := swapchain.destroyWithError(); err != nil {
			hal.Logger().Error("vulkan: failed to destroy swapchain", "error", err)
		}
		s.detachSwapchainFromQueue(swapchain)
		s.swapchain = nil
	}
	if s.device != nil {
		s.device.unregisterConfiguredSurface(s)
		s.device = nil
	}
	if s.handle != 0 && s.instance != nil {
		s.instance.cmds.DestroySurfaceKHR(s.instance.handle, s.handle, nil)
		s.handle = 0
	}
}

// releaseConfiguredDevice detaches the swapchain while VkDevice is still
// alive. Device.Destroy calls this after one device-wide idle operation.
func (s *Surface) releaseConfiguredDevice(device *Device, drained bool) {
	if s == nil || s.device != device {
		return
	}
	if s.swapchain != nil {
		if drained {
			s.swapchain.destroyAfterIdle()
		} else {
			s.swapchain.abandonDeviceResources()
		}
		s.swapchain = nil
	}
	s.device = nil
}

// Helper functions

// vkMakeVersion packs Vulkan version components (major.minor.patch).
//
//nolint:unparam,nolintlint // patch exercised in tests; unparam fires on Linux only (cross-GOOS).
func vkMakeVersion(major, minor, patch uint32) uint32 {
	return (major << 22) | (minor << 12) | patch
}

func vkVersionMajor(version uint32) uint32 {
	return version >> 22
}

func vkVersionMinor(version uint32) uint32 {
	return (version >> 12) & 0x3FF
}

func vkVersionPatch(version uint32) uint32 {
	return version & 0xFFF
}

func cStringToGo(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}

// GPU vendor names used in vendorIDToName and adapter info.
const (
	vendorAMD      = "AMD"
	vendorNVIDIA   = "NVIDIA"
	vendorIntel    = "Intel"
	vendorARM      = "ARM"
	vendorQualcomm = "Qualcomm"
	vendorImgTec   = "ImgTec"
)

func vendorIDToName(id uint32) string {
	switch id {
	case 0x1002:
		return vendorAMD
	case 0x10DE:
		return vendorNVIDIA
	case 0x8086:
		return vendorIntel
	case 0x13B5:
		return vendorARM
	case 0x5143:
		return vendorQualcomm
	case 0x1010:
		return vendorImgTec
	default:
		return fmt.Sprintf("0x%04X", id)
	}
}

// timestampFeature is FeatureTimestampQuery when every graphics and compute
// queue writes timestamps, as Dawn checks.
func timestampFeature(props *vk.PhysicalDeviceProperties) gputypes.Features {
	if props.Limits.TimestampComputeAndGraphics != 0 {
		return gputypes.Features(gputypes.FeatureTimestampQuery)
	}
	return 0
}

// featuresFromPhysicalDevice maps Vulkan physical device features to WebGPU
// features, with depth32Stencil8 whether the device renders depth and
// stencil to D32_SFLOAT_S8_UINT, which Vulkan does not require.
// Reference: wgpu-hal/src/vulkan/adapter.rs:584-829
func featuresFromPhysicalDevice(features *vk.PhysicalDeviceFeatures, drawIndirectCount, depth32Stencil8 bool) gputypes.Features {
	var result gputypes.Features

	// Texture compression features
	if features.TextureCompressionBC != 0 {
		result |= gputypes.Features(gputypes.FeatureTextureCompressionBC)
	}
	if features.TextureCompressionETC2 != 0 {
		result |= gputypes.Features(gputypes.FeatureTextureCompressionETC2)
	}
	if features.TextureCompressionASTC_LDR != 0 {
		result |= gputypes.Features(gputypes.FeatureTextureCompressionASTC)
	}

	// Draw features
	if features.DrawIndirectFirstInstance != 0 {
		result |= gputypes.Features(gputypes.FeatureIndirectFirstInstance)
	}
	if features.MultiDrawIndirect != 0 {
		result |= gputypes.Features(gputypes.FeatureMultiDrawIndirect)
	}

	// drawIndirectCount is an optional Vulkan 1.2 feature (FeatureMultiDrawIndirectCount).
	if drawIndirectCount {
		result |= gputypes.Features(gputypes.FeatureMultiDrawIndirectCount)
	}

	// Depth/clipping features
	if features.DepthClamp != 0 {
		result |= gputypes.Features(gputypes.FeatureDepthClipControl)
	}

	// Shader features
	if features.ShaderFloat64 != 0 {
		result |= gputypes.Features(gputypes.FeatureShaderFloat64)
	}

	// Query features
	if features.PipelineStatisticsQuery != 0 {
		result |= gputypes.Features(gputypes.FeaturePipelineStatisticsQuery)
	}

	if depth32Stencil8 {
		result |= gputypes.Features(gputypes.FeatureDepth32FloatStencil8)
	}

	return result
}

// downlevelCapabilitiesFromFeatures builds DownlevelCapabilities by querying
// VkPhysicalDeviceFeatures instead of assuming all flags.
// Reference: wgpu-hal/src/vulkan/adapter.rs:662-719 (PhysicalDeviceFeatures::to_wgpu)
//
// Vulkan unconditionally supports 17 downlevel flags (compute, base vertex,
// depth copies, etc.). Eight flags depend on actual physical device features
// and are set conditionally from VkPhysicalDeviceFeatures fields, and
// SurfaceViewFormats on swapchainMutableFormat, whether the device makes
// swapchains whose images are viewed in other formats.
func downlevelCapabilitiesFromFeatures(features *vk.PhysicalDeviceFeatures, swapchainMutableFormat bool) gputypes.DownlevelCapabilities {
	// Start with the 17 unconditional flags (Rust adapter.rs:684-700).
	flags := gputypes.DownlevelFlagsComputeShaders |
		gputypes.DownlevelFlagsBaseVertex |
		gputypes.DownlevelFlagsReadOnlyDepthStencil |
		gputypes.DownlevelFlagsNonPowerOfTwoMipmappedTextures |
		gputypes.DownlevelFlagsComparisonSamplers |
		gputypes.DownlevelFlagsVertexStorage |
		gputypes.DownlevelFlagsFragmentStorage |
		gputypes.DownlevelFlagsDepthTextureAndBufferCopies |
		gputypes.DownlevelFlagsBufferBindingsNot16ByteAligned |
		gputypes.DownlevelFlagsUnrestrictedIndexBuffer |
		gputypes.DownlevelFlagsIndirectExecution |
		gputypes.DownlevelFlagsViewFormats |
		gputypes.DownlevelFlagsUnrestrictedExternalTextureCopies |
		gputypes.DownlevelFlagsNonblockingQueryResolve |
		gputypes.DownlevelFlagsShaderF16InF32 |
		gputypes.DownlevelFlagsMSL21 |
		gputypes.DownlevelFlagsLinearInterpolation

	if swapchainMutableFormat {
		flags |= gputypes.DownlevelFlagsSurfaceViewFormats
	}

	// 8 conditional flags from VkPhysicalDeviceFeatures (Rust adapter.rs:702-719).
	if features.ImageCubeArray != 0 {
		flags |= gputypes.DownlevelFlagsCubeArrayTextures
	}
	if features.SamplerAnisotropy != 0 {
		flags |= gputypes.DownlevelFlagsAnisotropicFiltering
	}
	if features.FragmentStoresAndAtomics != 0 {
		flags |= gputypes.DownlevelFlagsFragmentWritableStorage
	}
	if features.SampleRateShading != 0 {
		flags |= gputypes.DownlevelFlagsMultisampledShading
	}
	if features.IndependentBlend != 0 {
		flags |= gputypes.DownlevelFlagsIndependentBlend
	}
	if features.FullDrawIndexUint32 != 0 {
		flags |= gputypes.DownlevelFlagsFullDrawIndexUint32
	}
	if features.DepthBiasClamp != 0 {
		flags |= gputypes.DownlevelFlagsDepthBiasClamp
	}
	// TEXTURE_COMPRESSION: BC || (ETC2 && ASTC) per W3C WebGPU spec.
	if features.TextureCompressionBC != 0 ||
		(features.TextureCompressionETC2 != 0 && features.TextureCompressionASTC_LDR != 0) {
		flags |= gputypes.DownlevelFlagsTextureCompression
	}

	return gputypes.DownlevelCapabilities{
		Flags:       flags,
		Limits:      gputypes.DownlevelLimits{},
		ShaderModel: gputypes.ShaderModelSm5,
	}
}

// limitsFromProps maps Vulkan physical device limits to WebGPU limits, with
// maxPerSetDescriptors from VkPhysicalDeviceMaintenance3Properties. Like
// wgpu-hal and Dawn it takes the 2D size every 2D use allows, a cube and a
// framebuffer too, and caps the per-stage counts by the per-set ones, shared
// by the two stages of a pipeline, and by what Vulkan limits their sums to.
// Reference: wgpu-hal/src/vulkan/adapter.rs to_wgpu_limits
func limitsFromProps(props *vk.PhysicalDeviceProperties, maxPerSetDescriptors uint32) gputypes.Limits {
	vkLimits := props.Limits

	// Start with default limits and override with actual hardware values
	limits := gputypes.DefaultLimits()

	// Texture dimensions
	limits.MaxTextureDimension1D = vkLimits.MaxImageDimension1D
	limits.MaxTextureDimension2D = min(vkLimits.MaxImageDimension2D, vkLimits.MaxImageDimensionCube,
		vkLimits.MaxFramebufferWidth, vkLimits.MaxFramebufferHeight)
	limits.MaxTextureDimension3D = vkLimits.MaxImageDimension3D
	limits.MaxTextureArrayLayers = vkLimits.MaxImageArrayLayers

	// Descriptor/binding limits
	const stagesPerPipeline = 2
	perStage := func(stage, set uint32) uint32 { return min(stage, set/stagesPerPipeline) }
	sampled := perStage(vkLimits.MaxPerStageDescriptorSampledImages, vkLimits.MaxDescriptorSetSampledImages)
	samplers := perStage(vkLimits.MaxPerStageDescriptorSamplers, vkLimits.MaxDescriptorSetSamplers)
	storageBuffers := perStage(vkLimits.MaxPerStageDescriptorStorageBuffers, vkLimits.MaxDescriptorSetStorageBuffers)
	storageTextures := perStage(vkLimits.MaxPerStageDescriptorStorageImages, vkLimits.MaxDescriptorSetStorageImages)
	uniforms := perStage(vkLimits.MaxPerStageDescriptorUniformBuffers, vkLimits.MaxDescriptorSetUniformBuffers)
	colorAttachments := min(vkLimits.MaxColorAttachments, vkLimits.MaxFragmentOutputAttachments, 8) // WebGPU max is 8
	// Intel, NVIDIA, AMD and ImgTec do not report
	// maxFragmentCombinedOutputResources right (gpuweb#3631).
	switch props.VendorID {
	case 0x8086, 0x10DE, 0x1002, 0x1010:
	default:
		capSum(vkLimits.MaxFragmentCombinedOutputResources, &storageTextures, &storageBuffers, &colorAttachments)
	}
	capSum(vkLimits.MaxPerStageResources, &sampled, &uniforms, &storageTextures, &storageBuffers, &colorAttachments)
	capSum(maxPerSetDescriptors/stagesPerPipeline, &sampled, &uniforms, &storageTextures, &storageBuffers, &samplers)
	limits.MaxBindGroups = min(vkLimits.MaxBoundDescriptorSets, 8) // WebGPU max is 8
	limits.MaxSampledTexturesPerShaderStage = sampled
	limits.MaxSamplersPerShaderStage = samplers
	limits.MaxStorageBuffersPerShaderStage = storageBuffers
	limits.MaxStorageTexturesPerShaderStage = storageTextures
	limits.MaxUniformBuffersPerShaderStage = uniforms
	limits.MaxColorAttachments = colorAttachments

	// Buffer limits, binding sizes within what a signed 32-bit offset in a
	// shader reaches (NVIDIA reports 4 GiB storage buffer ranges).
	const maxBindingSize = 1<<31 - 1
	limits.MaxUniformBufferBindingSize = uint64(min(vkLimits.MaxUniformBufferRange, maxBindingSize))
	limits.MaxStorageBufferBindingSize = uint64(min(vkLimits.MaxStorageBufferRange, maxBindingSize))
	limits.MinUniformBufferOffsetAlignment = uint32(vkLimits.MinUniformBufferOffsetAlignment)
	limits.MinStorageBufferOffsetAlignment = uint32(vkLimits.MinStorageBufferOffsetAlignment)

	// Vertex limits
	limits.MaxVertexAttributes = min(vkLimits.MaxVertexInputAttributes, 32) // WebGPU max is 32
	limits.MaxVertexBufferArrayStride = min(vkLimits.MaxVertexInputBindingStride, 2048)

	// Compute limits
	limits.MaxComputeWorkgroupStorageSize = vkLimits.MaxComputeSharedMemorySize
	limits.MaxComputeInvocationsPerWorkgroup = vkLimits.MaxComputeWorkGroupInvocations
	limits.MaxComputeWorkgroupSizeX = vkLimits.MaxComputeWorkGroupSize[0]
	limits.MaxComputeWorkgroupSizeY = vkLimits.MaxComputeWorkGroupSize[1]
	limits.MaxComputeWorkgroupSizeZ = vkLimits.MaxComputeWorkGroupSize[2]
	// WebGPU's limit holds for each dimension, so it is the smallest of
	// Vulkan's three, as in Rust's wgpu-hal. NVIDIA allows 2^31-1 in X but
	// 65535 in Y and Z.
	limits.MaxComputeWorkgroupsPerDimension = min(vkLimits.MaxComputeWorkGroupCount[0],
		vkLimits.MaxComputeWorkGroupCount[1], vkLimits.MaxComputeWorkGroupCount[2])

	// Push constants
	limits.MaxPushConstantSize = vkLimits.MaxPushConstantsSize

	return limits
}

// capSum lowers limits so that together they stay within sum, the smallest
// kept whole where they fit and the rest sharing what remains evenly, as
// wgpu-hal's cap_limits_to_be_under_the_sum_limit does.
func capSum(sum uint32, limits ...*uint32) {
	slices.SortFunc(limits, func(a, b *uint32) int { return cmp.Compare(*a, *b) })
	for i, limit := range limits {
		*limit = min(*limit, sum/uint32(len(limits)-i))
		sum -= *limit
	}
}

// maxPerSetDescriptors is VkPhysicalDeviceMaintenance3Properties' of device,
// or 256 where it cannot be queried, the lowest any device reports.
func maxPerSetDescriptors(cmds *vk.Commands, device vk.PhysicalDevice) uint32 {
	maint3 := vk.PhysicalDeviceMaintenance3Properties{SType: vk.StructureTypePhysicalDeviceMaintenance3Properties}
	props2 := vk.PhysicalDeviceProperties2{
		SType: vk.StructureTypePhysicalDeviceProperties2,
		PNext: (*uintptr)(unsafe.Pointer(&maint3)),
	}
	cmds.GetPhysicalDeviceProperties2(device, &props2)
	if maint3.MaxPerSetDescriptors == 0 {
		return 256
	}
	return maint3.MaxPerSetDescriptors
}

// isLayerAvailable checks if a Vulkan instance layer is available.
// Used to gracefully skip validation layers when Vulkan SDK is not installed.
func isLayerAvailable(cmds *vk.Commands, layerName string) bool {
	// Get layer count
	var count uint32
	cmds.EnumerateInstanceLayerProperties(&count, nil)
	if count == 0 {
		return false
	}

	// Get layer properties
	layers := make([]vk.LayerProperties, count)
	cmds.EnumerateInstanceLayerProperties(&count, &layers[0])

	// Check if requested layer is available
	for i := range layers {
		name := cStringToGo(layers[i].LayerName[:])
		if name == layerName {
			return true
		}
	}
	return false
}

func enumerateInstanceExtensions(cmds *vk.Commands) (map[string]struct{}, error) {
	for range 3 {
		var count uint32
		result := cmds.EnumerateInstanceExtensionProperties(nil, &count, nil)
		if result != vk.Success && result != vk.Incomplete {
			return nil, fmt.Errorf("vkEnumerateInstanceExtensionProperties(count) failed: %d", result)
		}
		if count == 0 {
			return map[string]struct{}{}, nil
		}

		properties := make([]vk.ExtensionProperties, count)
		result = cmds.EnumerateInstanceExtensionProperties(nil, &count, &properties[0])
		if result == vk.Incomplete {
			continue
		}
		if result != vk.Success {
			return nil, fmt.Errorf("vkEnumerateInstanceExtensionProperties(list) failed: %d", result)
		}
		available := make(map[string]struct{}, count)
		for index := range count {
			available[cStringToGo(properties[index].ExtensionName[:])] = struct{}{}
		}
		return available, nil
	}
	return nil, fmt.Errorf("vkEnumerateInstanceExtensionProperties remained incomplete")
}

func selectAvailableExtensions(candidates []string, available map[string]struct{}) []string {
	selected := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		name := strings.TrimSuffix(candidate, "\x00")
		if _, ok := available[name]; ok {
			selected = append(selected, candidate)
		}
	}
	return selected
}
