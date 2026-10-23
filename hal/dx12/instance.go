// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

// Package dx12 provides the DirectX 12 HAL backend for Windows.
//
// This backend requires Windows 10 (1809+) or Windows 11 and a DirectX 12
// capable GPU. It uses pure Go with syscall for all COM interactions,
// requiring no CGO.
//
// # Instance Creation
//
// The Instance manages DXGI factory creation, debug layer initialization,
// and adapter enumeration. It prefers discrete GPUs when enumerating adapters.
//
// # Debug Layer
//
// When InstanceFlagsDebug is set, the D3D12 debug layer is enabled via
// D3D12GetDebugInterface. This provides detailed validation messages but
// significantly impacts performance. Only use in development.
package dx12

import (
	"fmt"
	"runtime"
	"slices"
	"unsafe"

	"golang.org/x/sys/windows"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
	"github.com/roffe/gowgpu/hal/dx12/dxgi"
)

// Backend implements hal.Backend for DirectX 12.
type Backend struct{}

// NewBackend returns a DX12 backend instance.
func NewBackend() Backend { return Backend{} }

// Variant returns the backend type identifier.
func (Backend) Variant() gputypes.Backend {
	return gputypes.BackendDX12
}

// CreateInstance creates a new DirectX 12 instance.
func (Backend) CreateInstance(desc *hal.InstanceDescriptor) (hal.Instance, error) {
	if unsafe.Sizeof(uintptr(0)) < 8 {
		return nil, fmt.Errorf("dx12: needs a 64-bit build: its COM calls pass 64-bit values in one uintptr")
	}
	instance := &Instance{}

	// Load DXGI library first
	dxgiLib, err := dxgi.LoadDXGI()
	if err != nil {
		return nil, fmt.Errorf("dx12: failed to load dxgi.dll: %w", err)
	}
	instance.dxgiLib = dxgiLib

	// Determine factory creation flags
	var factoryFlags uint32
	debugRequested := desc != nil && desc.Flags&gputypes.InstanceFlagsDebug != 0
	if debugRequested {
		factoryFlags |= dxgi.DXGI_CREATE_FACTORY_DEBUG
		instance.flags = desc.Flags
	}

	// Create DXGI factory (with graceful debug fallback)
	factory, err := dxgiLib.CreateFactory4(factoryFlags)
	if err != nil && debugRequested {
		// Debug layer not installed (requires Windows "Graphics Tools" optional feature).
		// Fall back to non-debug factory.
		hal.Logger().Warn("dx12: debug layer not available, falling back to non-debug mode", "err", err)
		factoryFlags = 0
		instance.flags = 0
		factory, err = dxgiLib.CreateFactory4(factoryFlags)
	}
	if err != nil {
		return nil, fmt.Errorf("dx12: failed to create DXGI factory: %w", err)
	}
	instance.factory = factory
	instance.factory6 = factory.Factory6()

	// Load D3D12 library
	d3d12Lib, err := d3d12.LoadD3D12()
	if err != nil {
		instance.Destroy()
		return nil, fmt.Errorf("dx12: failed to load d3d12.dll: %w", err)
	}
	instance.d3d12Lib = d3d12Lib

	// Enable debug layer if requested (and factory was created with debug flags)
	if instance.flags&gputypes.InstanceFlagsDebug != 0 {
		if err := instance.enableDebugLayer(); err != nil {
			hal.Logger().Warn("dx12: debug layer enable failed (non-fatal)", "err", err)
		}
	}

	// Check for tearing support (DXGI 1.5 feature)
	instance.checkTearingSupport()

	// Set a finalizer to ensure cleanup
	runtime.SetFinalizer(instance, (*Instance).Destroy)

	hal.Logger().Info("dx12: instance created",
		"tearing", instance.allowTearing,
	)

	return instance, nil
}

// Instance implements hal.Instance for DirectX 12.
type Instance struct {
	factory      *dxgi.IDXGIFactory4
	factory6     *dxgi.IDXGIFactory6 // nil before Windows 10 1803
	d3d12Lib     *d3d12.D3D12Lib
	dxgiLib      *dxgi.DXGILib
	debugLayer   *d3d12.ID3D12Debug
	allowTearing bool
	flags        gputypes.InstanceFlags
}

// enableDebugLayer enables the D3D12 debug layer for validation.
func (i *Instance) enableDebugLayer() error {
	debug, err := i.d3d12Lib.GetDebugInterface()
	if err != nil {
		return fmt.Errorf("dx12: D3D12GetDebugInterface failed: %w", err)
	}

	debug.EnableDebugLayer()
	i.debugLayer = debug

	// Try to get ID3D12Debug1 for GPU-based validation
	debug1, err := i.d3d12Lib.GetDebugInterface1()
	if err == nil {
		// GPU-based validation catches resource state, descriptor, and shader errors
		// at the cost of significant performance. Enable with InstanceFlagsValidation.
		enableGBV := i.flags&gputypes.InstanceFlagsValidation != 0
		debug1.SetEnableGPUBasedValidation(enableGBV)
		if enableGBV {
			hal.Logger().Info("dx12: GPU-based validation enabled (slow)")
		}
		debug1.Release()
	}

	// Enable DRED (Device Removed Extended Data) for TDR diagnostics.
	// DRED must be configured BEFORE device creation. When a TDR occurs,
	// DRED provides auto-breadcrumbs showing which GPU command was executing
	// and page fault information showing which allocation was accessed.
	dredSettings, err := i.d3d12Lib.GetDREDSettings()
	if err == nil {
		dredSettings.SetAutoBreadcrumbsEnablement(d3d12.D3D12DREDEnablementForcedOn)
		dredSettings.SetPageFaultEnablement(d3d12.D3D12DREDEnablementForcedOn)
		dredSettings.Release()
		hal.Logger().Info("dx12: DRED enabled (auto breadcrumbs + page fault tracking)")
	} else {
		hal.Logger().Debug("dx12: DRED not available (requires Windows 10 1903+)", "err", err)
	}

	return nil
}

// checkTearingSupport checks if variable refresh rate is supported.
func (i *Instance) checkTearingSupport() {
	if i.factory6 == nil {
		return
	}
	var allowTearing int32
	err := i.factory6.CheckFeatureSupport(
		dxgi.DXGI_FEATURE_PRESENT_ALLOW_TEARING,
		unsafe.Pointer(&allowTearing),
		4, // sizeof(BOOL)
	)
	if err == nil && allowTearing != 0 {
		i.allowTearing = true
	}
}

// CreateSurface creates a rendering surface from platform handles.
// displayHandle is not used on Windows (can be 0).
// windowHandle must be a valid HWND.
func (i *Instance) CreateSurface(target hal.SurfaceTarget) (hal.Surface, error) {
	if err := target.RequireKind(hal.SurfaceTargetWindowsHWND); err != nil {
		return nil, fmt.Errorf("dx12: %w", err)
	}
	if target.WindowHandle == 0 {
		return nil, fmt.Errorf("dx12: windowHandle (HWND) is required")
	}

	return &Surface{
		instance: i,
		hwnd:     target.WindowHandle,
	}, nil
}

// EnumerateAdapters enumerates available physical GPUs.
// Adapters are sorted by preference: discrete GPUs first, then integrated,
// then others. Software adapters (WARP) only come in when no hardware adapter
// works, as a CPU adapter: a VM or remote session without a GPU still renders,
// and nobody else pays for creating a WARP device to probe it.
func (i *Instance) EnumerateAdapters(surfaceHint hal.Surface) []hal.ExposedAdapter {
	if i.factory6 == nil {
		return i.enumerateAdaptersLegacy(surfaceHint)
	}
	var adapters []hal.ExposedAdapter
	var software []*Adapter

	// Enumerate adapters by GPU preference (high performance first)
	for idx := uint32(0); ; idx++ {
		raw, err := i.factory6.EnumAdapterByGpuPreference(
			idx, dxgi.DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
		if err != nil {
			// No more adapters or factory doesn't support EnumAdapterByGpuPreference
			if idx == 0 {
				// Fall back to legacy enumeration
				adapters = i.enumerateAdaptersLegacy(surfaceHint)
			}
			break
		}

		// Get adapter description
		desc, err := raw.GetDesc1()
		if err != nil {
			raw.Release()
			continue
		}

		if withdrawnAdapter(&desc) {
			raw.Release()
			continue
		}
		adapter := &Adapter{
			raw:      raw,
			desc:     desc,
			instance: i,
		}
		if desc.Flags&dxgi.DXGI_ADAPTER_FLAG_SOFTWARE != 0 {
			software = append(software, adapter)
			continue
		}

		// Probe adapter capabilities by creating a temporary device
		if err := adapter.probeCapabilities(); err != nil || !bindingTierSupported(&desc, adapter.capabilities.ResourceBindingTier) {
			raw.Release()
			continue
		}

		exposed := adapter.toExposedAdapter()

		hal.Logger().Info("dx12: adapter found",
			"name", exposed.Info.Name,
			"type", exposed.Info.DeviceType,
			"vendorID", fmt.Sprintf("0x%04X", exposed.Info.VendorID),
		)

		adapters = append(adapters, exposed)
	}

	for _, adapter := range software {
		if len(adapters) > 0 || adapter.probeCapabilities() != nil || !bindingTierSupported(&adapter.desc, adapter.capabilities.ResourceBindingTier) {
			adapter.raw.Release()
			continue
		}
		exposed := adapter.toExposedAdapter()
		hal.Logger().Info("dx12: no hardware adapter, using WARP", "name", exposed.Info.Name)
		adapters = append(adapters, exposed)
	}

	return adapters
}

// enumerateAdaptersLegacy uses the legacy IDXGIFactory1 enumeration method.
// This is a fallback for older Windows versions or when EnumAdapterByGpuPreference fails.
func (i *Instance) enumerateAdaptersLegacy(surfaceHint hal.Surface) []hal.ExposedAdapter {
	_ = surfaceHint // Surface hint not used in DX12; all adapters support all surfaces

	var adapters []hal.ExposedAdapter //nolint:prealloc // Size unknown until enumeration
	var discreteAdapters []hal.ExposedAdapter
	var integratedAdapters []hal.ExposedAdapter
	var otherAdapters []hal.ExposedAdapter
	var software []*AdapterLegacy

	for idx := uint32(0); ; idx++ {
		raw, err := i.factory.EnumAdapters1(idx)
		if err != nil {
			break
		}

		desc, err := raw.GetDesc1()
		if err != nil {
			raw.Release()
			continue
		}

		if withdrawnAdapter(&desc) {
			raw.Release()
			continue
		}
		adapter := &AdapterLegacy{
			raw:      raw,
			desc:     desc,
			instance: i,
		}
		if desc.Flags&dxgi.DXGI_ADAPTER_FLAG_SOFTWARE != 0 {
			software = append(software, adapter) // WARP, see EnumerateAdapters
			continue
		}

		if err := adapter.probeCapabilities(); err != nil || !bindingTierSupported(&desc, adapter.capabilities.ResourceBindingTier) {
			raw.Release()
			continue
		}

		exposed := adapter.toExposedAdapter()

		hal.Logger().Info("dx12: adapter found (legacy)",
			"name", exposed.Info.Name,
			"type", exposed.Info.DeviceType,
			"vendorID", fmt.Sprintf("0x%04X", exposed.Info.VendorID),
		)

		// Sort by device type for proper preference ordering
		switch exposed.Info.DeviceType {
		case gputypes.DeviceTypeDiscreteGPU:
			discreteAdapters = append(discreteAdapters, exposed)
		case gputypes.DeviceTypeIntegratedGPU:
			integratedAdapters = append(integratedAdapters, exposed)
		default:
			otherAdapters = append(otherAdapters, exposed)
		}
	}

	// Combine in preference order: discrete > integrated > other
	adapters = append(adapters, discreteAdapters...)
	adapters = append(adapters, integratedAdapters...)
	adapters = append(adapters, otherAdapters...)

	for _, adapter := range software {
		if len(adapters) > 0 || adapter.probeCapabilities() != nil || !bindingTierSupported(&adapter.desc, adapter.capabilities.ResourceBindingTier) {
			adapter.raw.Release()
			continue
		}
		exposed := adapter.toExposedAdapter()
		hal.Logger().Info("dx12: no hardware adapter, using WARP (legacy)", "name", exposed.Info.Name)
		adapters = append(adapters, exposed)
	}

	return adapters
}

// haswellDeviceIDs are the Intel Haswell GPUs, which Intel withdrew D3D12
// from for a security flaw. Rust's wgpu-hal leaves them out, as their
// devices are lost some time after they are made. The list is
// https://dgpu-docs.intel.com/devices/hardware-table.html.
var haswellDeviceIDs = []uint32{
	0x0422, 0x0426, 0x042A, 0x042B, 0x042E, 0x0C22, 0x0C26, 0x0C2A, 0x0C2B, 0x0C2E, 0x0A22,
	0x0A2A, 0x0A2B, 0x0D2A, 0x0D2B, 0x0D2E, 0x0A26, 0x0A2E, 0x0D22, 0x0D26, 0x0412, 0x0416,
	0x0D12, 0x041A, 0x041B, 0x0C12, 0x0C16, 0x0C1A, 0x0C1B, 0x0C1E, 0x0A12, 0x0A1A, 0x0A1B,
	0x0D16, 0x0D1A, 0x0D1B, 0x0D1E, 0x041E, 0x0A16, 0x0A1E, 0x0402, 0x0406, 0x040A, 0x040B,
	0x040E, 0x0C02, 0x0C06, 0x0C0A, 0x0C0B, 0x0C0E, 0x0A02, 0x0A06, 0x0A0A, 0x0A0B, 0x0A0E,
	0x0D02, 0x0D06, 0x0D0A, 0x0D0B, 0x0D0E,
}

// withdrawnAdapter reports an adapter D3D12 is not to be used on.
func withdrawnAdapter(desc *dxgi.DXGI_ADAPTER_DESC1) bool {
	return desc.VendorID == 0x8086 && slices.Contains(haswellDeviceIDs, desc.DeviceID)
}

// bindingTierSupported reports whether an adapter has resource binding
// tier 2: every root signature with a sampler binds a table of 2048
// samplers, more than tier 1's 16, as Rust's wgpu-hal requires it too.
func bindingTierSupported(desc *dxgi.DXGI_ADAPTER_DESC1, tier uint32) bool {
	if tier >= 2 {
		return true
	}
	hal.Logger().Info("dx12: adapter has resource binding tier 1, which this backend can't use",
		"name", utf16ToString(desc.Description[:]))
	return false
}

// Destroy releases the DirectX 12 instance and all associated resources.
func (i *Instance) Destroy() {
	if i == nil {
		return
	}

	// Clear finalizer to prevent double-free
	runtime.SetFinalizer(i, nil)

	if i.debugLayer != nil {
		i.debugLayer.Release()
		i.debugLayer = nil
	}

	if i.factory6 != nil {
		i.factory6.Release()
		i.factory6 = nil
	}
	if i.factory != nil {
		i.factory.Release()
		i.factory = nil
	}

	// Libraries are unloaded when their handles go out of scope
	// via their own finalizers
	i.d3d12Lib = nil
	i.dxgiLib = nil
}

// AllowsTearing returns whether variable refresh rate (tearing) is supported.
func (i *Instance) AllowsTearing() bool {
	return i.allowTearing
}

// Surface implements hal.Surface for DirectX 12.
type Surface struct {
	instance *Instance
	hwnd     uintptr
	device   *Device

	// Swapchain state
	swapchain                  *dxgi.IDXGISwapChain4
	backBuffers                []backBuffer
	width                      uint32
	height                     uint32
	format                     dxgi.DXGI_FORMAT
	halFormat                  gputypes.TextureFormat
	presentMode                gputypes.PresentMode
	swapchainFlags             uint32
	allowTearing               bool
	frameLatencyWaitableObject uintptr // HANDLE from GetFrameLatencyWaitableObject

	// damagePresent is true when the swapchain was created with
	// DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL (instead of FLIP_DISCARD).
	// Only in this mode can Present1 with dirty rects be used.
	damagePresent bool

	// dcomp holds the DirectComposition visual tree state when using
	// per-pixel alpha (CreateSwapChainForComposition path). nil for
	// the standard HWND path.
	dcomp *dcompState
}

// Configure configures the surface for presentation.
func (s *Surface) Configure(device hal.Device, config *hal.SurfaceConfiguration) error {
	if config.Width == 0 || config.Height == 0 {
		return hal.ErrZeroArea
	}

	// Type assert to DX12 Device
	dx12Device, ok := device.(*Device)
	if !ok {
		return fmt.Errorf("dx12: device is not a DX12 device")
	}

	// If we already have a swapchain with the same device, resize it —
	// UNLESS the alpha mode changed (Opaque↔Premultiplied requires a
	// different swap chain creation path: HWND vs DirectComposition).
	if s.swapchain != nil && s.device == dx12Device {
		needsDComp := config.AlphaMode == hal.CompositeAlphaModePremultiplied
		hasDComp := s.dcomp != nil
		if needsDComp == hasDComp {
			return s.resizeSwapchain(config)
		}
	}

	// Destroy old swapchain if switching devices
	s.Unconfigure(device)

	// Create new swapchain (implementation in surface.go)
	return s.createSwapchain(dx12Device, config)
}

// Unconfigure removes surface configuration.
func (s *Surface) Unconfigure(_ hal.Device) {
	// Wait for GPU to finish using the swapchain
	if s.device != nil {
		_ = s.device.waitForGPU()
	}

	// Release back buffer resources
	s.releaseBackBuffers()

	// Close frame latency waitable object handle
	if s.frameLatencyWaitableObject != 0 {
		_ = windows.CloseHandle(windows.Handle(s.frameLatencyWaitableObject))
		s.frameLatencyWaitableObject = 0
	}

	// Release swapchain
	if s.swapchain != nil {
		s.swapchain.Release()
		s.swapchain = nil
	}

	// Release DirectComposition state (must be after swapchain release —
	// the visual holds a reference to the swap chain via SetContent).
	if s.dcomp != nil {
		s.dcomp.release()
		s.dcomp = nil
	}

	s.device = nil
	s.width = 0
	s.height = 0
}

// AcquireTexture acquires the next surface texture for rendering.
// It first waits for the GPU to finish the old frame that occupied the
// current slot and recycles its allocators. This deferred wait (moved
// from Present) allows the CPU to overlap work with GPU execution.
func (s *Surface) AcquireTexture(_ hal.Fence) (*hal.AcquiredSurfaceTexture, error) {
	if s.swapchain == nil {
		return nil, fmt.Errorf("dx12: surface not configured")
	}

	if s.device != nil {
		if err := s.device.recycleFrameSlot(); err != nil {
			return nil, fmt.Errorf("dx12: recycle frame slot failed: %w", err)
		}
	}

	// Wait on the frame latency waitable object for proper frame pacing.
	if s.frameLatencyWaitableObject != 0 {
		result, err := windows.WaitForSingleObject(
			windows.Handle(s.frameLatencyWaitableObject),
			1000, // 1 second timeout (same as Rust wgpu)
		)
		if err != nil {
			return nil, fmt.Errorf("dx12: WaitForSingleObject on frame latency waitable failed: %w", err)
		}
		// A removed device never releases a frame: every acquire would
		// wait out the timeout.
		if result == uint32(windows.WAIT_TIMEOUT) && s.device != nil {
			if reason := s.device.raw.GetDeviceRemovedReason(); reason != nil {
				return nil, fmt.Errorf("dx12: acquire: %w", deviceLost(reason))
			}
		}
	}

	// Get current back buffer index
	index := s.swapchain.GetCurrentBackBufferIndex()

	if int(index) >= len(s.backBuffers) {
		return nil, fmt.Errorf("dx12: back buffer index %d out of range", index)
	}

	bb := &s.backBuffers[index]

	// Create surface texture wrapper
	surfaceTexture := &SurfaceTexture{
		surface:    s,
		index:      index,
		resource:   bb.resource,
		stateOwner: bb.texture,
		rtvHandle:  bb.rtvHandle,
		format:     s.halFormat,
		width:      s.width,
		height:     s.height,
		// Note: Suboptimal detection requires DXGI_STATUS_OCCLUDED/DXGI_ERROR_DEVICE_REMOVED checks.
		suboptimal: false,
	}

	return &hal.AcquiredSurfaceTexture{
		Texture:    surfaceTexture,
		Suboptimal: surfaceTexture.suboptimal,
	}, nil
}

// DiscardTexture discards a surface texture without presenting it.
func (s *Surface) DiscardTexture(_ hal.SurfaceTexture) {
	// For DX12 with flip model swapchains, discarding is a no-op.
	// The texture is simply not presented, and the swapchain will
	// continue to use the existing frame.
}

// ActualExtent returns the configured swapchain dimensions.
// DX12 DXGI does not clamp the extent, so these always match the requested values.
// Returns (0, 0) if the surface is not configured.
func (s *Surface) ActualExtent() (width, height uint32) {
	return s.width, s.height
}

// Destroy releases the surface.
func (s *Surface) Destroy() {
	s.Unconfigure(nil)
}

// Compile-time interface assertions.
var (
	_ hal.Backend  = Backend{}
	_ hal.Instance = (*Instance)(nil)
	_ hal.Surface  = (*Surface)(nil)
)
