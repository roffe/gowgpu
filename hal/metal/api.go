// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin && !(js && wasm)

package metal

import (
	"fmt"
	"math"
	"runtime"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// Backend implements hal.Backend for Metal.
type Backend struct{}

// NewBackend returns a Metal backend instance.
func NewBackend() Backend { return Backend{} }

// Variant returns the backend type identifier.
func (Backend) Variant() gputypes.Backend {
	return gputypes.BackendMetal
}

// CreateInstance creates a new Metal instance.
func (Backend) CreateInstance(desc *hal.InstanceDescriptor) (hal.Instance, error) {
	if err := Init(); err != nil {
		return nil, fmt.Errorf("metal: failed to initialize: %w", err)
	}
	hal.Logger().Info("metal: instance created")
	return &Instance{}, nil
}

// Instance implements hal.Instance for Metal.
type Instance struct{}

// CreateSurface creates a rendering surface from a CAMetalLayer target.
func (i *Instance) CreateSurface(target hal.SurfaceTarget) (hal.Surface, error) {
	if err := target.RequireKind(hal.SurfaceTargetMetalLayer); err != nil {
		return nil, fmt.Errorf("metal: %w", err)
	}
	layer := ID(target.WindowHandle)
	if layer == 0 {
		return nil, fmt.Errorf("metal: window handle is nil")
	}
	Retain(layer)
	return &Surface{layer: layer}, nil
}

// EnumerateAdapters returns available Metal adapters (devices).
func (i *Instance) EnumerateAdapters(surfaceHint hal.Surface) []hal.ExposedAdapter {
	devices := CopyAllDevices()
	if len(devices) == 0 {
		return nil
	}

	hal.Logger().Debug("metal: enumerating adapters", "count", len(devices))

	adapters := make([]hal.ExposedAdapter, 0, len(devices))
	for _, device := range devices {
		deviceName := DeviceName(device)

		// Determine device type
		var deviceType gputypes.DeviceType
		switch {
		case DeviceIsHeadless(device):
			deviceType = gputypes.DeviceTypeOther
		case DeviceIsRemovable(device), !DeviceIsLowPower(device):
			deviceType = gputypes.DeviceTypeDiscreteGPU
		default:
			deviceType = gputypes.DeviceTypeIntegratedGPU
		}

		// Build features
		// ponytail: no TimestampQuery until query sets get counter sample
		// buffers.
		var features gputypes.Features
		features.Insert(gputypes.FeatureDepthClipControl)
		// The arguments of an indirect draw have a base instance, which the
		// families that draw from a base vertex and instance honour.
		if DeviceSupportsFamily(device, MTLGPUFamilyApple3) || DeviceSupportsFamily(device, MTLGPUFamilyMac2) {
			features.Insert(gputypes.FeatureIndirectFirstInstance)
		}
		// ponytail: no TextureCompressionBC, textureFormatToMTL maps no BC
		// format and Metal aborts on pixel format 0. Map them (MTLPixelFormat
		// 130-153) and check supportsBCTextureCompression to offer it.

		adapter := &Adapter{
			instance:              i,
			raw:                   device,
			formatDepth24Stencil8: DeviceSupportsDepth24Stencil8(device),
		}

		maxBuf := DeviceMaxBufferLength(device)
		family := gpuFamilyLimits(device)
		sampledTextures := family.textureEntries * 3 / 4
		interStage := family.fragmentInputs - 4
		if DeviceSupportsFamily(device, MTLGPUFamilyApple1) {
			interStage = min(family.fragmentInputs, family.fragmentInputComponents/4)
		}
		// naga's bounds checks and `_buffer_sizes` take a binding's size as a
		// u32.
		storageBinding := min(maxBuf, math.MaxUint32&^3)

		hal.Logger().Info("metal: adapter found",
			"name", deviceName,
			"type", deviceType,
			"lowPower", DeviceIsLowPower(device),
			"removable", DeviceIsRemovable(device),
			"headless", DeviceIsHeadless(device),
			"maxBuffer", maxBuf,
			"depth24Stencil8", adapter.formatDepth24Stencil8,
		)

		adapters = append(adapters, hal.ExposedAdapter{
			Adapter: adapter,
			Info: gputypes.AdapterInfo{
				Name:       deviceName,
				Vendor:     "Apple",
				VendorID:   0x106b, // Apple Inc.
				DeviceID:   uint32(DeviceRegistryID(device) & 0xFFFFFFFF),
				DeviceType: deviceType,
				Driver:     "Metal",
				DriverInfo: "Metal API",
				Backend:    gputypes.BackendMetal,
			},
			Features: features,
			Capabilities: hal.Capabilities{
				Limits: gputypes.Limits{
					MaxTextureDimension1D:                     family.textureSize,
					MaxTextureDimension2D:                     family.textureSize,
					MaxTextureDimension3D:                     2048,
					MaxTextureArrayLayers:                     2048,
					MaxBindGroups:                             4,
					MaxBindGroupsPlusVertexBuffers:            24,
					MaxBindingsPerBindGroup:                   1000,
					MaxDynamicUniformBuffersPerPipelineLayout: 12,
					MaxDynamicStorageBuffersPerPipelineLayout: 4,
					MaxSampledTexturesPerShaderStage:          sampledTextures,
					MaxSamplersPerShaderStage:                 16,
					MaxStorageBuffersPerShaderStage:           8,
					MaxStorageTexturesPerShaderStage:          family.textureEntries - sampledTextures,
					MaxUniformBuffersPerShaderStage:           12,
					MaxUniformBufferBindingSize:               maxBuf,
					MaxStorageBufferBindingSize:               storageBinding,
					MinUniformBufferOffsetAlignment:           256,
					MinStorageBufferOffsetAlignment:           256,
					MaxVertexBuffers:                          vertexBuffersLimit,
					MaxBufferSize:                             maxBuf,
					MaxVertexAttributes:                       31,
					MaxVertexBufferArrayStride:                2048,

					MaxInterStageShaderVariables:      interStage,
					MaxColorAttachments:               family.colorTargets,
					MaxColorAttachmentBytesPerSample:  family.renderTargetBytes,
					MaxComputeWorkgroupStorageSize:    family.threadgroupMemory,
					MaxComputeInvocationsPerWorkgroup: family.threadsPerThreadgroup,
					MaxComputeWorkgroupSizeX:          family.threadsPerThreadgroup,
					MaxComputeWorkgroupSizeY:          family.threadsPerThreadgroup,
					MaxComputeWorkgroupSizeZ:          family.threadsPerThreadgroup,
					MaxComputeWorkgroupsPerDimension:  65535,
					MaxNonSamplerBindings:             1000000,
				},
				AlignmentsMask: hal.Alignments{
					BufferCopyOffset: 4,
					BufferCopyPitch:  256,
				},
				// DefaultDownlevelCapabilities (all 27 flags) is correct for Metal.
				// Rust wgpu-hal Metal conditionally sets 8 flags from feature sets
				// (adapter.rs:1337-1371), but ALL of those checks pass on our minimum
				// targets (macOS 15.0+ / iOS 18.0+):
				//   FRAGMENT_WRITABLE_STORAGE — macOS 10.12+ (available!(macos=10.12))
				//   CUBE_ARRAY_TEXTURES       — macOS_GPUFamily1_v1 (macOS 10.11+)
				//   COMPARISON_SAMPLERS        — macOS_GPUFamily1_v1 (macOS 10.11+)
				//   INDIRECT_EXECUTION         — macOS_GPUFamily1_v1 (macOS 10.11+)
				//   BASE_VERTEX               — same as INDIRECT_EXECUTION
				//   ANISOTROPIC_FILTERING      — always true in Rust
				//   MSL2_1                    — MSL 2.1 requires macOS 10.14+
				//   TEXTURE_COMPRESSION        — macOS always has BC; iOS has EAC+ASTC
				DownlevelCapabilities: gputypes.DefaultDownlevelCapabilities(),
			},
		})
	}

	return adapters
}

// familyLimits are the limits that differ between GPU families: the
// entries of a function's texture argument table, threads and memory of a
// threadgroup, a fragment function's inputs and input components, the 1D
// and 2D texture size, color render targets and the bytes a sample of
// them all may take.
type familyLimits struct {
	textureEntries, threadsPerThreadgroup, threadgroupMemory uint32
	fragmentInputs, fragmentInputComponents                  uint32
	textureSize, colorTargets, renderTargetBytes             uint32
}

// familyTable is the Metal feature set tables as Dawn has them, by family,
// the first one a device supports picked: on macOS Mac2, which Apple silicon
// Macs support too.
var familyTable = []struct {
	family MTLGPUFamily
	familyLimits
}{
	{MTLGPUFamilyMac2, familyLimits{128, 1024, 32768, 32, 124, 16384, 8, 128}},
	{MTLGPUFamilyApple9, familyLimits{128, 1024, 32768, 124, 124, 16384, 8, 64}},
	{MTLGPUFamilyApple8, familyLimits{128, 1024, 32768, 124, 124, 16384, 8, 64}},
	{MTLGPUFamilyApple7, familyLimits{128, 1024, 32768, 124, 124, 16384, 8, 64}},
	{MTLGPUFamilyApple6, familyLimits{128, 1024, 32768, 124, 124, 16384, 8, 64}},
	{MTLGPUFamilyApple5, familyLimits{96, 1024, 32768, 124, 124, 16384, 8, 64}},
	{MTLGPUFamilyApple4, familyLimits{96, 1024, 32768, 124, 124, 16384, 8, 64}},
	{MTLGPUFamilyApple3, familyLimits{31, 512, 16384, 60, 60, 16384, 8, 32}},
	{MTLGPUFamilyApple2, familyLimits{31, 512, 16352, 60, 60, 8192, 8, 32}},
	{MTLGPUFamilyApple1, familyLimits{31, 512, 16352, 60, 60, 8192, 4, 16}},
}

// gpuFamilyLimits returns the familyLimits of device.
//
// Reference: Dawn native/metal/PhysicalDeviceMTL.mm InitializeSupportedLimitsImpl.
func gpuFamilyLimits(device ID) familyLimits {
	for _, f := range familyTable {
		if f.family == MTLGPUFamilyMac2 && runtime.GOOS != "darwin" {
			continue
		}
		if DeviceSupportsFamily(device, f.family) {
			return f.familyLimits
		}
	}
	return familyTable[0].familyLimits
}

// Destroy releases the instance.
func (i *Instance) Destroy() {
	// Nothing to release
}
