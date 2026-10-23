// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"runtime"
	"sync"
	"unsafe"

	"github.com/gogpu/naga/hlsl"
	"github.com/gogpu/naga/ir"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// -----------------------------------------------------------------------------
// ShaderModule Implementation
// -----------------------------------------------------------------------------

// ShaderModule implements hal.ShaderModule for DirectX 12. WGSL is lowered
// to naga IR when the module is made and compiled for each pipeline, with
// the pipeline's layout (see Device.stageBytecode). Pre-compiled bytecode
// is kept by entry point name.
type ShaderModule struct {
	wgslSource  string
	irModule    *ir.Module
	entryPoints map[string][]byte
	device      *Device
}

// Destroy releases the shader module resources.
func (m *ShaderModule) Destroy() {
	m.wgslSource = ""
	m.irModule = nil
	m.entryPoints = nil
	m.device = nil
}

// EntryPointBytecode returns the pre-compiled bytecode of the given entry
// point, nil for a WGSL module.
func (m *ShaderModule) EntryPointBytecode(name string) []byte {
	if m.entryPoints == nil {
		return nil
	}
	return m.entryPoints[name]
}

// IsDeferred returns true if this module contains WGSL, which each
// pipeline compiles with its own layout.
func (m *ShaderModule) IsDeferred() bool {
	return m.irModule != nil
}

// entryPoint returns the module's WGSL entry point named name.
func (m *ShaderModule) entryPoint(name string) (*ir.EntryPoint, bool) {
	if m.irModule == nil {
		return nil, false
	}
	for i := range m.irModule.EntryPoints {
		if m.irModule.EntryPoints[i].Name == name {
			return &m.irModule.EntryPoints[i], true
		}
	}
	return nil, false
}

// -----------------------------------------------------------------------------
// BindGroupLayout Implementation
// -----------------------------------------------------------------------------

// BindGroupLayoutEntry stores binding information for root signature creation.
type BindGroupLayoutEntry struct {
	Binding    uint32
	Type       BindingType
	Visibility gputypes.ShaderStages
	Count      uint32 // For arrays

	// HasDynamicOffset is set on a buffer bound with a dynamic offset. A
	// uniform buffer gets a root CBV instead of a descriptor in the table, a
	// storage buffer keeps its descriptor and gets its offset in root
	// constants that naga adds to every address, as in Rust's wgpu-hal.
	HasDynamicOffset bool
}

// dynamicUniform reports whether e is a uniform buffer with a root CBV.
func (e BindGroupLayoutEntry) dynamicUniform() bool {
	return e.HasDynamicOffset && e.Type == BindingTypeUniformBuffer
}

// BindingType describes the type of resource binding.
type BindingType uint8

const (
	BindingTypeUniformBuffer BindingType = iota
	BindingTypeStorageBuffer
	BindingTypeReadOnlyStorageBuffer
	BindingTypeSampler
	BindingTypeComparisonSampler
	BindingTypeSampledTexture
	BindingTypeStorageTexture
)

// BindGroupLayout implements hal.BindGroupLayout for DirectX 12.
type BindGroupLayout struct {
	entries []BindGroupLayoutEntry // in binding order, the order of the dynamic offsets
	device  *Device
}

// Destroy releases the bind group layout resources.
func (l *BindGroupLayout) Destroy() {
	l.entries = nil
	l.device = nil
}

// Entries returns the bind group layout entries.
func (l *BindGroupLayout) Entries() []BindGroupLayoutEntry {
	return l.entries
}

// -----------------------------------------------------------------------------
// PipelineLayout Implementation
// -----------------------------------------------------------------------------

// rootParamMapping stores actual root parameter indices for a single bind group.
// Values are -1 when the group has no descriptors of that type.
type rootParamMapping struct {
	cbvSrvUavIndex int // root param index for CBV/SRV/UAV table, or -1

	// From root param dynamicIndex on: a root CBV for each of the group's
	// dynamicUniforms uniform buffers with a dynamic offset, in binding
	// order, then, when dynamicStorage > 0, root constants with the offsets
	// of its storage buffers with one.
	dynamicIndex    int
	dynamicUniforms int
	dynamicStorage  int
}

// PipelineLayout implements hal.PipelineLayout for DirectX 12.
// It wraps an ID3D12RootSignature and stores naga HLSL options for deferred
// shader compilation, matching Rust wgpu-hal architecture.
type PipelineLayout struct {
	rootSignature     *d3d12.ID3D12RootSignature
	rootSignatureHash [32]byte // SHA-256 of serialized root signature blob (PSO cache key)
	bindGroupLayouts  []*BindGroupLayout
	groupMappings     []rootParamMapping // actual root param indices per bind group
	samplerRootIndex  int                // root param index for global sampler heap table, or -1
	nagaOptions       *hlsl.Options      // HLSL compile options with proper BindingMap

	// specialConstantsIndex is the root param index of the constants
	// naga reads first_vertex, first_instance and num_workgroups from.
	specialConstantsIndex int

	// indirect are the command signatures that set those constants
	// from the indirect records, made on first use.
	indirect     indirectSignatures
	indirectOnce sync.Once
	indirectErr  error

	device *Device
}

// indirectSignatures are command signatures of one root signature for
// indirect records with the special constants ahead of the arguments.
type indirectSignatures struct {
	draw, drawIndexed, dispatch *d3d12.ID3D12CommandSignature
}

// Destroy releases the pipeline layout resources.
func (l *PipelineLayout) Destroy() {
	for _, sig := range []*d3d12.ID3D12CommandSignature{l.indirect.draw, l.indirect.drawIndexed, l.indirect.dispatch} {
		if sig != nil {
			sig.Release()
		}
	}
	l.indirect = indirectSignatures{}
	if l.rootSignature != nil {
		l.rootSignature.Release()
		l.rootSignature = nil
	}
	l.bindGroupLayouts = nil
	l.device = nil
}

// RootSignature returns the underlying D3D12 root signature.
func (l *PipelineLayout) RootSignature() *d3d12.ID3D12RootSignature {
	return l.rootSignature
}

// -----------------------------------------------------------------------------
// BindGroup Implementation (Updated)
// -----------------------------------------------------------------------------

// BindGroup implements hal.BindGroup for DirectX 12.
// Uses the Rust wgpu-hal sampler heap pattern: samplers are not copied into
// a per-group sampler table. Instead, a sampler index buffer (StructuredBuffer<uint>)
// is created containing the global sampler pool indices. The shader reads
// nagaSamplerHeap[indexBuffer[binding]] to access the sampler.
type BindGroup struct {
	layout        *BindGroupLayout
	gpuDescHandle d3d12.D3D12_GPU_DESCRIPTOR_HANDLE // GPU handle for CBV/SRV/UAV table
	device        *Device

	// Tracked allocation indices for descriptor recycling
	viewHeapIndex uint32
	viewCount     uint32

	// Sampler index buffer (StructuredBuffer<uint> containing sampler pool indices).
	// This GPU buffer is allocated per bind group if the group has any samplers.
	samplerIndexBuffer *d3d12.ID3D12Resource

	// dynamicBuffers are the buffers bound with a dynamic offset, in binding
	// order like the offsets SetBindGroup gets.
	dynamicBuffers []dynamicBuffer

	// storageBuffers holds references to storage buffers bound in this group.
	// Populated at CreateBindGroup time by matching layout entries with type
	// BindingTypeStorageBuffer against the corresponding buffer bindings.
	// Used by ComputePassEncoder to track which buffers transition to
	// UNORDERED_ACCESS after Dispatch() (BUG-DX12-012 fix).
	storageBuffers         []*Buffer
	readOnlyStorageBuffers []*Buffer
	uniformBuffers         []*Buffer
	sampledTextures        []*TextureView
	storageTextures        []*TextureView
}

// Destroy releases the bind group resources and recycles descriptor heap slots.
func (g *BindGroup) Destroy() {
	if g.device != nil {
		if g.viewCount > 0 {
			g.device.viewHeap.Free(g.viewHeapIndex, g.viewCount)
		}
	}
	if g.samplerIndexBuffer != nil {
		g.samplerIndexBuffer.Release()
		g.samplerIndexBuffer = nil
	}
	g.layout = nil
	g.device = nil
}

// GPUDescriptorHandle returns the GPU descriptor handle for CBV/SRV/UAV.
func (g *BindGroup) GPUDescriptorHandle() d3d12.D3D12_GPU_DESCRIPTOR_HANDLE {
	return g.gpuDescHandle
}

// dynamicBuffer is a buffer bound with a dynamic offset: a uniform buffer
// with the address its root CBV gets before the offset, or a storage buffer,
// whose offset goes in the group's root constants.
type dynamicBuffer struct {
	uniform bool
	gpuVA   uint64
}

// -----------------------------------------------------------------------------
// Pipeline Creation Helpers
// -----------------------------------------------------------------------------

// pipelineLayoutResult holds the output of root signature creation.
type pipelineLayoutResult struct {
	rootSignature         *d3d12.ID3D12RootSignature
	rootSignatureHash     [32]byte
	groupMappings         []rootParamMapping
	samplerRootIndex      int
	specialConstantsIndex int
	nagaOptions           *hlsl.Options
}

// createRootSignatureFromLayouts creates a D3D12 root signature from bind group layouts.
// Implements the Rust wgpu-hal architecture:
//   - Monotonic per-type register counters (bind_cbv, bind_srv, bind_uav)
//   - Sampler bindings mapped to space=255 with index_within_group as register
//   - Per-group sampler index buffer SRV in the CBV/SRV/UAV table
//   - Global sampler heap root parameter (2x2048 sampler ranges)
//   - Root CBVs for dynamic uniform buffers and root constants for the
//     offsets of dynamic storage buffers after each group's table
//   - Full naga HLSL options with BindingMap and SamplerBufferBindingMap
func (d *Device) createRootSignatureFromLayouts(layouts []hal.BindGroupLayout) (*pipelineLayoutResult, error) {
	var rootParams []d3d12.D3D12_ROOT_PARAMETER
	var allRanges []d3d12.D3D12_DESCRIPTOR_RANGE // flat slice to prevent reallocation
	var groupMappings []rootParamMapping

	// Monotonic per-type register counters (matches Rust wgpu-hal).
	var bindCBV, bindSRV, bindUAV uint32

	// naga HLSL binding maps.
	bindingMap := make(map[hlsl.ResourceBinding]hlsl.BindTarget)
	samplerBufferBindingMap := make(map[uint32]hlsl.BindTarget)
	dynamicStorageOffsets := make(map[uint32]hlsl.OffsetsBindTarget)

	// Track whether any bind group has samplers.
	samplerInAnyGroup := false

	// Pre-count total descriptor ranges to avoid reallocation.
	totalRanges := 0
	for _, layout := range layouts {
		bgLayout, ok := layout.(*BindGroupLayout)
		if !ok {
			continue
		}
		samplerInGroup := false
		for _, entry := range bgLayout.entries {
			switch entry.Type {
			case BindingTypeSampler, BindingTypeComparisonSampler:
				samplerInGroup = true
			default:
				totalRanges++
			}
		}
		if samplerInGroup {
			totalRanges++ // sampler index buffer SRV
			samplerInAnyGroup = true
		}
	}
	if samplerInAnyGroup {
		totalRanges += 2 // global sampler heap ranges (standard + comparison)
	}
	allRanges = make([]d3d12.D3D12_DESCRIPTOR_RANGE, 0, totalRanges)

	for groupIdx, layout := range layouts {
		bgLayout, ok := layout.(*BindGroupLayout)
		if !ok {
			return nil, fmt.Errorf("dx12: invalid bind group layout type at index %d", groupIdx)
		}

		mapping := rootParamMapping{cbvSrvUavIndex: -1}

		// Skip empty layouts
		if len(bgLayout.entries) == 0 {
			groupMappings = append(groupMappings, mapping)
			continue
		}

		// Build CBV/SRV/UAV descriptor ranges for this group.
		rangeBase := len(allRanges)

		for _, entry := range bgLayout.entries {
			rangeType, isSampler := bindingTypeToD3D12DescriptorRangeType(entry.Type)
			if isSampler || entry.dynamicUniform() {
				continue // Samplers handled separately below, dynamic uniform buffers get root CBVs
			}

			// Assign register using per-type monotonic counter.
			var reg *uint32
			switch rangeType {
			case d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_CBV:
				reg = &bindCBV
			case d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_SRV:
				reg = &bindSRV
			case d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_UAV:
				reg = &bindUAV
			default:
				reg = &bindSRV
			}

			target := hlsl.BindTarget{
				Space:    0,
				Register: *reg,
			}
			if entry.HasDynamicOffset {
				// A storage buffer: naga adds its offset, the group's root
				// constant at this index, to every address.
				index := uint32(mapping.dynamicStorage)
				target.DynamicStorageBufferOffsetsIndex = &index
				mapping.dynamicStorage++
			}
			bindingMap[hlsl.ResourceBinding{
				Group:   uint32(groupIdx),
				Binding: entry.Binding,
			}] = target

			allRanges = append(allRanges, d3d12.D3D12_DESCRIPTOR_RANGE{
				RangeType:                         rangeType,
				NumDescriptors:                    1,
				BaseShaderRegister:                *reg,
				RegisterSpace:                     0,
				OffsetInDescriptorsFromTableStart: 0xFFFFFFFF, // D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND
			})
			*reg++
		}

		// Handle samplers: assign to BindingMap with space=255 and
		// index_within_group as register. Add sampler index buffer SRV.
		var samplerIdxInGroup uint32
		for _, entry := range bgLayout.entries {
			if entry.Type != BindingTypeSampler && entry.Type != BindingTypeComparisonSampler {
				continue
			}
			bindingMap[hlsl.ResourceBinding{
				Group:   uint32(groupIdx),
				Binding: entry.Binding,
			}] = hlsl.BindTarget{
				Space:    255,
				Register: samplerIdxInGroup,
			}
			samplerIdxInGroup++
		}

		if samplerIdxInGroup > 0 {
			// Add sampler index buffer SRV to the CBV/SRV/UAV table.
			// This is the StructuredBuffer<uint> that maps sampler binding index
			// to the global sampler heap slot.
			samplerBufferBindingMap[uint32(groupIdx)] = hlsl.BindTarget{
				Space:    0,
				Register: bindSRV,
			}
			allRanges = append(allRanges, d3d12.D3D12_DESCRIPTOR_RANGE{
				RangeType:                         d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
				NumDescriptors:                    1,
				BaseShaderRegister:                bindSRV,
				RegisterSpace:                     0,
				OffsetInDescriptorsFromTableStart: 0xFFFFFFFF,
			})
			bindSRV++
		}

		// Create root parameter for CBV/SRV/UAV table (includes sampler index buffer SRV).
		if len(allRanges) > rangeBase {
			mapping.cbvSrvUavIndex = len(rootParams)

			param := d3d12.D3D12_ROOT_PARAMETER{
				ParameterType:    d3d12.D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
				ShaderVisibility: d3d12.D3D12_SHADER_VISIBILITY_ALL,
			}

			rangeSlice := allRanges[rangeBase:]
			table := (*d3d12.D3D12_ROOT_DESCRIPTOR_TABLE)(unsafe.Pointer(&param.Union[0]))
			table.NumDescriptorRanges = uint32(len(rangeSlice))
			table.DescriptorRanges = &rangeSlice[0]

			rootParams = append(rootParams, param)
		}

		// A root CBV for each uniform buffer with a dynamic offset, which
		// SetBindGroup points at the binding plus the offset, then root
		// constants with the offsets of the dynamic storage buffers.
		mapping.dynamicIndex = len(rootParams)
		for _, entry := range bgLayout.entries {
			if !entry.dynamicUniform() {
				continue
			}
			// Nothing bounds checks a root CBV, so naga clamps the indexing.
			bindingMap[hlsl.ResourceBinding{
				Group:   uint32(groupIdx),
				Binding: entry.Binding,
			}] = hlsl.BindTarget{
				Space:            0,
				Register:         bindCBV,
				RestrictIndexing: true,
			}
			param := d3d12.D3D12_ROOT_PARAMETER{
				ParameterType:    d3d12.D3D12_ROOT_PARAMETER_TYPE_CBV,
				ShaderVisibility: d3d12.D3D12_SHADER_VISIBILITY_ALL,
			}
			*(*d3d12.D3D12_ROOT_DESCRIPTOR)(unsafe.Pointer(&param.Union[0])) = d3d12.D3D12_ROOT_DESCRIPTOR{ShaderRegister: bindCBV}
			rootParams = append(rootParams, param)
			mapping.dynamicUniforms++
			bindCBV++
		}
		if mapping.dynamicStorage > 0 {
			dynamicStorageOffsets[uint32(groupIdx)] = hlsl.OffsetsBindTarget{
				Space:    0,
				Register: bindCBV,
				Size:     uint32(mapping.dynamicStorage),
			}
			param := d3d12.D3D12_ROOT_PARAMETER{
				ParameterType:    d3d12.D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
				ShaderVisibility: d3d12.D3D12_SHADER_VISIBILITY_ALL,
			}
			*(*d3d12.D3D12_ROOT_CONSTANTS)(unsafe.Pointer(&param.Union[0])) = d3d12.D3D12_ROOT_CONSTANTS{
				ShaderRegister: bindCBV,
				Num32BitValues: uint32(mapping.dynamicStorage),
			}
			rootParams = append(rootParams, param)
			bindCBV++
		}

		groupMappings = append(groupMappings, mapping)
	}

	// Global sampler heap root parameter (matches Rust wgpu-hal).
	// Two ranges in the same descriptor table: standard samplers at s0-s2047
	// and comparison samplers at s2048-s4095, both in space 0.
	samplerRootIndex := -1
	if samplerInAnyGroup {
		samplerRootIndex = len(rootParams)

		rangeBase := len(allRanges)
		// Standard samplers (registers 0-2047) and comparison samplers (registers 2048-4095).
		allRanges = append(allRanges,
			d3d12.D3D12_DESCRIPTOR_RANGE{
				RangeType:                         d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
				NumDescriptors:                    2048,
				BaseShaderRegister:                0,
				RegisterSpace:                     0,
				OffsetInDescriptorsFromTableStart: 0,
			},
			d3d12.D3D12_DESCRIPTOR_RANGE{
				RangeType:                         d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER,
				NumDescriptors:                    2048,
				BaseShaderRegister:                2048,
				RegisterSpace:                     0,
				OffsetInDescriptorsFromTableStart: 0,
			},
		)

		samplerRanges := allRanges[rangeBase:]
		param := d3d12.D3D12_ROOT_PARAMETER{
			ParameterType:    d3d12.D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
			ShaderVisibility: d3d12.D3D12_SHADER_VISIBILITY_ALL,
		}
		table := (*d3d12.D3D12_ROOT_DESCRIPTOR_TABLE)(unsafe.Pointer(&param.Union[0]))
		table.NumDescriptorRanges = uint32(len(samplerRanges))
		table.DescriptorRanges = &samplerRanges[0]

		rootParams = append(rootParams, param)
	}

	// Root constants naga reads first_vertex and first_instance, or the
	// workgroup counts, from, at the next CBV register, as in Rust's
	// wgpu-hal: SV_VertexID, SV_InstanceID and SV_GroupID hold none of them.
	specialConstantsIndex := len(rootParams)
	specialConstants := hlsl.BindTarget{Space: 0, Register: bindCBV}
	specialParam := d3d12.D3D12_ROOT_PARAMETER{
		ParameterType:    d3d12.D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
		ShaderVisibility: d3d12.D3D12_SHADER_VISIBILITY_ALL,
	}
	*(*d3d12.D3D12_ROOT_CONSTANTS)(unsafe.Pointer(&specialParam.Union[0])) = d3d12.D3D12_ROOT_CONSTANTS{
		ShaderRegister: bindCBV,
		Num32BitValues: 3,
	}
	rootParams = append(rootParams, specialParam)
	bindCBV++

	// Build root signature description
	desc := d3d12.D3D12_ROOT_SIGNATURE_DESC{
		Flags: d3d12.D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT,
	}

	if len(rootParams) > 0 {
		desc.NumParameters = uint32(len(rootParams))
		desc.Parameters = &rootParams[0]
	}

	// Serialize root signature
	blob, errorBlob, err := d.instance.d3d12Lib.SerializeRootSignature(&desc, d3d12.D3D_ROOT_SIGNATURE_VERSION_1_0)
	// The ranges are only reachable through the root parameters' unions,
	// which the GC does not see as pointers.
	runtime.KeepAlive(allRanges)
	runtime.KeepAlive(rootParams)
	if err != nil {
		if errorBlob != nil {
			errorBlob.Release()
		}
		return nil, fmt.Errorf("dx12: failed to serialize root signature: %w", err)
	}
	defer blob.Release()

	rootSigHash := sha256.Sum256(unsafe.Slice((*byte)(blob.GetBufferPointer()), blob.GetBufferSize()))

	// Check if device is already lost before attempting to create root signature.
	if reason := d.raw.GetDeviceRemovedReason(); reason != nil {
		d.logDREDBreadcrumbs()
		return nil, fmt.Errorf("dx12: device already removed before CreateRootSignature: %w", deviceLost(reason))
	}

	// Create root signature
	rootSig, err := d.raw.CreateRootSignature(0, blob.GetBufferPointer(), blob.GetBufferSize())
	if err != nil {
		if reason := d.raw.GetDeviceRemovedReason(); reason != nil {
			d.logDREDBreadcrumbs()
			return nil, fmt.Errorf("dx12: failed to create root signature (device removed: %w): %w", deviceLost(reason), err)
		}
		return nil, fmt.Errorf("dx12: failed to create root signature: %w", err)
	}

	// Build naga HLSL options for deferred shader compilation.
	nagaOpts := hlsl.DefaultOptions()
	nagaOpts.BindingMap = bindingMap
	nagaOpts.FakeMissingBindings = false
	nagaOpts.SamplerBufferBindingMap = samplerBufferBindingMap
	nagaOpts.DynamicStorageBufferOffsetsTargets = dynamicStorageOffsets
	nagaOpts.SamplerHeapTargets = hlsl.SamplerHeapBindTargets{
		StandardSamplers:   hlsl.BindTarget{Space: 0, Register: 0},
		ComparisonSamplers: hlsl.BindTarget{Space: 0, Register: 2048},
	}
	nagaOpts.SpecialConstantsBinding = &specialConstants

	return &pipelineLayoutResult{
		rootSignature:         rootSig,
		rootSignatureHash:     rootSigHash,
		groupMappings:         groupMappings,
		samplerRootIndex:      samplerRootIndex,
		specialConstantsIndex: specialConstantsIndex,
		nagaOptions:           nagaOpts,
	}, nil
}

// indirectSignatures returns the layout's command signatures that set the
// special constants from the indirect records, as Rust's wgpu-hal makes
// them for a layout: the constants, two of them for a draw, the workgroup
// counts for a dispatch, ahead of the arguments of each record.
func (l *PipelineLayout) indirectSignatures() (indirectSignatures, error) {
	l.indirectOnce.Do(func() {
		create := func(argType d3d12.D3D12_INDIRECT_ARGUMENT_TYPE, constants, argBytes uint32) *d3d12.ID3D12CommandSignature {
			if l.indirectErr != nil {
				return nil
			}
			args := [2]d3d12.D3D12_INDIRECT_ARGUMENT_DESC{{Type: d3d12.D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT}, {Type: argType}}
			binary.LittleEndian.PutUint32(args[0].Union[0:], uint32(l.specialConstantsIndex))
			binary.LittleEndian.PutUint32(args[0].Union[4:], 0)
			binary.LittleEndian.PutUint32(args[0].Union[8:], constants)
			sig, err := l.device.raw.CreateCommandSignature(&d3d12.D3D12_COMMAND_SIGNATURE_DESC{
				ByteStride:       constants*4 + argBytes,
				NumArgumentDescs: 2,
				ArgumentDescs:    &args[0],
			}, l.rootSignature)
			runtime.KeepAlive(&args)
			l.indirectErr = err
			return sig
		}
		l.indirect.draw = create(d3d12.D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, 2, drawRecordSize)
		l.indirect.drawIndexed = create(d3d12.D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED, 2, drawIndexedRecordSize)
		l.indirect.dispatch = create(d3d12.D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, 3, dispatchRecordSize)
	})
	return l.indirect, l.indirectErr
}

// Bytes in an indirect draw, indexed draw and dispatch record.
const (
	drawRecordSize        = 16
	drawIndexedRecordSize = 20
	dispatchRecordSize    = 12
)

// readsSpecialConstants reports whether the entry point reads a builtin
// naga takes from the special constants: vertex_index and instance_index
// in a vertex stage, num_workgroups in a compute stage.
func readsSpecialConstants(module *ShaderModule, entryPoint string) bool {
	ep, ok := module.entryPoint(entryPoint)
	if !ok {
		return false
	}
	special := func(b *ir.Binding) bool {
		if b == nil {
			return false
		}
		builtin, ok := (*b).(ir.BuiltinBinding)
		if !ok {
			return false
		}
		switch builtin.Builtin {
		case ir.BuiltinVertexIndex, ir.BuiltinInstanceIndex, ir.BuiltinNumWorkGroups:
			return true
		}
		return false
	}
	for _, arg := range ep.Function.Arguments {
		if special(arg.Binding) {
			return true
		}
		if int(arg.Type) < len(module.irModule.Types) {
			if st, ok := module.irModule.Types[arg.Type].Inner.(ir.StructType); ok {
				for _, m := range st.Members {
					if special(m.Binding) {
						return true
					}
				}
			}
		}
	}
	return false
}

// bindingTypeToD3D12DescriptorRangeType converts binding type to D3D12 descriptor range type.
// Returns the range type and whether it's a sampler.
func bindingTypeToD3D12DescriptorRangeType(t BindingType) (d3d12.D3D12_DESCRIPTOR_RANGE_TYPE, bool) {
	switch t {
	case BindingTypeUniformBuffer:
		return d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_CBV, false
	case BindingTypeStorageBuffer:
		return d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_UAV, false
	case BindingTypeReadOnlyStorageBuffer:
		return d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_SRV, false
	case BindingTypeSampler, BindingTypeComparisonSampler:
		return d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, true
	case BindingTypeSampledTexture:
		return d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_SRV, false
	case BindingTypeStorageTexture:
		return d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_UAV, false
	default:
		return d3d12.D3D12_DESCRIPTOR_RANGE_TYPE_SRV, false
	}
}

// -----------------------------------------------------------------------------
// RenderPipeline Creation
// -----------------------------------------------------------------------------

// buildGraphicsPipelineStateDesc builds a D3D12_GRAPHICS_PIPELINE_STATE_DESC from a render pipeline descriptor.
// Note: semanticNames is passed to keep the byte slices alive during PSO creation (D3D12 reads the pointers).
func (d *Device) buildGraphicsPipelineStateDesc(
	desc *hal.RenderPipelineDescriptor,
	inputElements []d3d12.D3D12_INPUT_ELEMENT_DESC,
	_ [][]byte, // Keep semantic name strings alive (prevents GC during PSO creation)
	vs, ps []byte,
) (*d3d12.D3D12_GRAPHICS_PIPELINE_STATE_DESC, error) {
	psoDesc := &d3d12.D3D12_GRAPHICS_PIPELINE_STATE_DESC{}

	// Set root signature from pipeline layout.
	// DX12 requires a valid root signature for every PSO. If no layout is
	// provided (shader has no resource bindings), use the device's shared
	// empty root signature. This prevents GPU hangs on drivers that don't
	// gracefully handle nil root signatures (observed as DPC_WATCHDOG_VIOLATION).
	if desc.Layout != nil {
		pipelineLayout, ok := desc.Layout.(*PipelineLayout)
		if !ok {
			return nil, fmt.Errorf("dx12: invalid pipeline layout type")
		}
		psoDesc.RootSignature = pipelineLayout.rootSignature
	} else {
		emptyRS, err := d.getOrCreateEmptyRootSignature()
		if err != nil {
			return nil, fmt.Errorf("dx12: failed to get empty root signature: %w", err)
		}
		psoDesc.RootSignature = emptyRS
	}

	if len(vs) > 0 {
		psoDesc.VS = d3d12.D3D12_SHADER_BYTECODE{
			ShaderBytecode: unsafe.Pointer(&vs[0]),
			BytecodeLength: uintptr(len(vs)),
		}
	}
	if len(ps) > 0 {
		psoDesc.PS = d3d12.D3D12_SHADER_BYTECODE{
			ShaderBytecode: unsafe.Pointer(&ps[0]),
			BytecodeLength: uintptr(len(ps)),
		}
	}

	// Input layout
	if len(inputElements) > 0 {
		psoDesc.InputLayout = d3d12.D3D12_INPUT_LAYOUT_DESC{
			InputElementDescs: &inputElements[0],
			NumElements:       uint32(len(inputElements)),
		}
	}

	// Primitive topology type
	psoDesc.PrimitiveTopologyType = primitiveTopologyTypeToD3D12(desc.Primitive.Topology)

	// Index buffer strip cut value for strip topologies
	if desc.Primitive.StripIndexFormat != nil {
		switch *desc.Primitive.StripIndexFormat {
		case gputypes.IndexFormatUint16:
			psoDesc.IBStripCutValue = d3d12.D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF
		case gputypes.IndexFormatUint32:
			psoDesc.IBStripCutValue = d3d12.D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF
		}
	}

	// Rasterizer state
	psoDesc.RasterizerState = d3d12.D3D12_RASTERIZER_DESC{
		FillMode:              d3d12.D3D12_FILL_MODE_SOLID,
		CullMode:              cullModeToD3D12(desc.Primitive.CullMode),
		FrontCounterClockwise: frontFaceToD3D12(desc.Primitive.FrontFace),
		DepthBias:             0,
		DepthBiasClamp:        0,
		SlopeScaledDepthBias:  0,
		DepthClipEnable:       1, // TRUE - enable depth clipping by default
		MultisampleEnable:     boolToInt32(desc.Multisample.Count > 1),
		AntialiasedLineEnable: 0,
		ForcedSampleCount:     0,
		ConservativeRaster:    d3d12.D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
	}

	// Handle unclipped depth
	if desc.Primitive.UnclippedDepth {
		psoDesc.RasterizerState.DepthClipEnable = 0 // FALSE
	}

	// Depth/stencil state
	if desc.DepthStencil != nil {
		psoDesc.DepthStencilState = buildDepthStencilDesc(desc.DepthStencil)
		psoDesc.DSVFormat = textureFormatToD3D12(desc.DepthStencil.Format)

		// Apply depth bias from depth/stencil state
		psoDesc.RasterizerState.DepthBias = desc.DepthStencil.DepthBias
		psoDesc.RasterizerState.DepthBiasClamp = desc.DepthStencil.DepthBiasClamp
		psoDesc.RasterizerState.SlopeScaledDepthBias = desc.DepthStencil.DepthBiasSlopeScale
	}

	// Blend state — IndependentBlendEnable must be TRUE so that D3D12 uses
	// the per-target blend descriptors in RenderTarget[0..N] instead of
	// replicating RenderTarget[0] across all targets. This is required for
	// correct MRT (multiple render targets) behavior.
	// Matches Rust wgpu-hal: IndependentBlendEnable: true.into() (dx12/device.rs:1980).
	psoDesc.BlendState = d3d12.D3D12_BLEND_DESC{
		AlphaToCoverageEnable:  boolToInt32(desc.Multisample.AlphaToCoverageEnabled),
		IndependentBlendEnable: 1, // TRUE: each render target has its own blend state
	}

	// Color targets
	if desc.Fragment != nil {
		psoDesc.NumRenderTargets = uint32(len(desc.Fragment.Targets))
		for i, target := range desc.Fragment.Targets {
			if i >= 8 {
				break // D3D12 supports max 8 render targets
			}
			psoDesc.RTVFormats[i] = textureFormatToD3D12(target.Format)
			psoDesc.BlendState.RenderTarget[i] = buildRenderTargetBlendDesc(&target)
		}
	}

	// Multisample state
	psoDesc.SampleDesc = d3d12.DXGI_SAMPLE_DESC{
		Count:   desc.Multisample.Count,
		Quality: 0,
	}
	if psoDesc.SampleDesc.Count == 0 {
		psoDesc.SampleDesc.Count = 1
	}
	psoDesc.SampleMask = uint32(desc.Multisample.Mask & 0xFFFFFFFF)
	if psoDesc.SampleMask == 0 {
		psoDesc.SampleMask = 0xFFFFFFFF
	}

	return psoDesc, nil
}

// overrideBytecode returns the bytecode in the file overrideEnv names, if
// it names one, in place of bytecode.
func overrideBytecode(bytecode []byte, overrideEnv string) ([]byte, error) {
	path := os.Getenv(overrideEnv)
	if path == "" {
		return bytecode, nil
	}
	override, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s %q: %w", overrideEnv, path, err)
	}
	fmt.Fprintf(os.Stderr, "[dx12] shader overridden from %s (%d bytes)\n", path, len(override))
	return override, nil
}

// buildDepthStencilDesc builds a D3D12_DEPTH_STENCIL_DESC from a depth/stencil state.
func buildDepthStencilDesc(ds *hal.DepthStencilState) d3d12.D3D12_DEPTH_STENCIL_DESC {
	desc := d3d12.D3D12_DEPTH_STENCIL_DESC{
		DepthEnable:      1, // TRUE - enable depth testing
		DepthWriteMask:   d3d12.D3D12_DEPTH_WRITE_MASK_ZERO,
		DepthFunc:        compareFunctionToD3D12(ds.DepthCompare),
		StencilEnable:    0,
		StencilReadMask:  uint8(ds.StencilReadMask),
		StencilWriteMask: uint8(ds.StencilWriteMask),
	}

	// Depth write
	if ds.DepthWriteEnabled {
		desc.DepthWriteMask = d3d12.D3D12_DEPTH_WRITE_MASK_ALL
	}

	// Disable depth test if compare is Always and no write
	if ds.DepthCompare == gputypes.CompareFunctionAlways && !ds.DepthWriteEnabled {
		desc.DepthEnable = 0
	}

	// Stencil operations
	if stencilEnabled(ds) {
		desc.StencilEnable = 1
		desc.FrontFace = stencilFaceToD3D12(ds.StencilFront)
		desc.BackFace = stencilFaceToD3D12(ds.StencilBack)
	}

	return desc
}

// stencilEnabled reports whether the stencil test does anything, as Rust's
// wgpu-types StencilState::is_enabled does: a face that is not the ignore
// state, and a mask to read or write through.
func stencilEnabled(ds *hal.DepthStencilState) bool {
	if stencilFaceIgnored(ds.StencilFront) && stencilFaceIgnored(ds.StencilBack) {
		return false
	}
	return ds.StencilReadMask != 0 || ds.StencilWriteMask != 0
}

// stencilFaceIgnored reports a face that passes every fragment and keeps
// the stencil value. Undefined values are WebGPU's defaults, Always and Keep.
func stencilFaceIgnored(face hal.StencilFaceState) bool {
	keeps := func(op gputypes.StencilOperation) bool {
		return op == gputypes.StencilOperationUndefined || op == hal.StencilOperationKeep
	}
	return (face.Compare == gputypes.CompareFunctionUndefined || face.Compare == gputypes.CompareFunctionAlways) &&
		keeps(face.FailOp) && keeps(face.DepthFailOp) && keeps(face.PassOp)
}

func stencilFaceToD3D12(face hal.StencilFaceState) d3d12.D3D12_DEPTH_STENCILOP_DESC {
	compare := face.Compare
	if compare == gputypes.CompareFunctionUndefined {
		compare = gputypes.CompareFunctionAlways
	}
	return d3d12.D3D12_DEPTH_STENCILOP_DESC{
		StencilFailOp:      stencilOpToD3D12(face.FailOp),
		StencilDepthFailOp: stencilOpToD3D12(face.DepthFailOp),
		StencilPassOp:      stencilOpToD3D12(face.PassOp),
		StencilFunc:        compareFunctionToD3D12(compare),
	}
}

// buildRenderTargetBlendDesc builds a D3D12_RENDER_TARGET_BLEND_DESC from a color target state.
func buildRenderTargetBlendDesc(target *gputypes.ColorTargetState) d3d12.D3D12_RENDER_TARGET_BLEND_DESC {
	desc := d3d12.D3D12_RENDER_TARGET_BLEND_DESC{
		BlendEnable:           0,
		LogicOpEnable:         0,
		SrcBlend:              d3d12.D3D12_BLEND_ONE,
		DestBlend:             d3d12.D3D12_BLEND_ZERO,
		BlendOp:               d3d12.D3D12_BLEND_OP_ADD,
		SrcBlendAlpha:         d3d12.D3D12_BLEND_ONE,
		DestBlendAlpha:        d3d12.D3D12_BLEND_ZERO,
		BlendOpAlpha:          d3d12.D3D12_BLEND_OP_ADD,
		LogicOp:               d3d12.D3D12_LOGIC_OP_NOOP,
		RenderTargetWriteMask: colorWriteMaskToD3D12(target.WriteMask),
	}

	if target.Blend != nil {
		desc.BlendEnable = 1
		desc.SrcBlend = blendFactorToD3D12(target.Blend.Color.SrcFactor, false)
		desc.DestBlend = blendFactorToD3D12(target.Blend.Color.DstFactor, false)
		desc.BlendOp = blendOperationToD3D12(target.Blend.Color.Operation)
		desc.SrcBlendAlpha = blendFactorToD3D12(target.Blend.Alpha.SrcFactor, true)
		desc.DestBlendAlpha = blendFactorToD3D12(target.Blend.Alpha.DstFactor, true)
		desc.BlendOpAlpha = blendOperationToD3D12(target.Blend.Alpha.Operation)
	}

	return desc
}

// buildInputLayout builds input element descriptors from vertex buffer layouts.
func buildInputLayout(buffers []gputypes.VertexBufferLayout) ([]d3d12.D3D12_INPUT_ELEMENT_DESC, [][]byte) {
	var elements []d3d12.D3D12_INPUT_ELEMENT_DESC
	var semanticNames [][]byte // Keep strings alive

	// Create semantic names like "TEXCOORD0", "TEXCOORD1", etc.
	// D3D12 uses semantic name + index for vertex attributes

	for slotIdx, buffer := range buffers {
		for _, attr := range buffer.Attributes {
			// Create semantic name "LOC" with semantic index = shader location.
			// Must match naga HLSL output which uses LOC{N} for @location(N).
			semanticName := []byte("LOC\x00")
			semanticNames = append(semanticNames, semanticName)

			element := d3d12.D3D12_INPUT_ELEMENT_DESC{
				SemanticName:         &semanticNames[len(semanticNames)-1][0],
				SemanticIndex:        attr.ShaderLocation,
				Format:               vertexFormatToD3D12(attr.Format),
				InputSlot:            uint32(slotIdx),
				AlignedByteOffset:    uint32(attr.Offset),
				InputSlotClass:       inputStepModeToD3D12(buffer.StepMode),
				InstanceDataStepRate: 0,
			}

			if buffer.StepMode == gputypes.VertexStepModeInstance {
				element.InstanceDataStepRate = 1
			}

			elements = append(elements, element)
		}
	}

	return elements, semanticNames
}

// boolToInt32 converts a bool to int32 (for D3D12 BOOL).
func boolToInt32(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

// -----------------------------------------------------------------------------
// RenderPipeline Implementation
// -----------------------------------------------------------------------------

// RenderPipeline implements hal.RenderPipeline for DirectX 12.
type RenderPipeline struct {
	pso              *d3d12.ID3D12PipelineState
	rootSignature    *d3d12.ID3D12RootSignature // Reference, not owned
	groupMappings    []rootParamMapping         // bind group → root param index mapping
	samplerRootIndex int                        // root param index for global sampler heap, or -1
	topology         d3d12.D3D_PRIMITIVE_TOPOLOGY
	vertexStrides    []uint32 // Strides per vertex buffer slot

	// specialConstants is the pipeline's layout when its vertex stage
	// reads vertex_index or instance_index, which then take the special
	// constants the draws set.
	specialConstants *PipelineLayout
}

// Destroy releases the render pipeline resources.
func (p *RenderPipeline) Destroy() {
	if p.pso != nil {
		p.pso.Release()
		p.pso = nil
	}
	// Don't release rootSignature - it's owned by PipelineLayout
	p.rootSignature = nil
	p.vertexStrides = nil
}

// PSO returns the underlying D3D12 pipeline state object.
func (p *RenderPipeline) PSO() *d3d12.ID3D12PipelineState {
	return p.pso
}

// RootSignature returns the root signature.
func (p *RenderPipeline) RootSignature() *d3d12.ID3D12RootSignature {
	return p.rootSignature
}

// Topology returns the primitive topology.
func (p *RenderPipeline) Topology() d3d12.D3D_PRIMITIVE_TOPOLOGY {
	return p.topology
}

// VertexStrides returns the vertex buffer strides.
func (p *RenderPipeline) VertexStrides() []uint32 {
	return p.vertexStrides
}

// -----------------------------------------------------------------------------
// ComputePipeline Implementation
// -----------------------------------------------------------------------------

// ComputePipeline implements hal.ComputePipeline for DirectX 12.
type ComputePipeline struct {
	pso              *d3d12.ID3D12PipelineState
	rootSignature    *d3d12.ID3D12RootSignature // Reference, not owned
	groupMappings    []rootParamMapping         // bind group → root param index mapping
	samplerRootIndex int                        // root param index for global sampler heap, or -1

	// specialConstants is the pipeline's layout when its compute stage
	// reads num_workgroups, which then takes the special constants the
	// dispatches set.
	specialConstants *PipelineLayout
}

// Destroy releases the compute pipeline resources.
func (p *ComputePipeline) Destroy() {
	if p.pso != nil {
		p.pso.Release()
		p.pso = nil
	}
	// Don't release rootSignature - it's owned by PipelineLayout
	p.rootSignature = nil
}

// PSO returns the underlying D3D12 pipeline state object.
func (p *ComputePipeline) PSO() *d3d12.ID3D12PipelineState {
	return p.pso
}

// RootSignature returns the root signature.
func (p *ComputePipeline) RootSignature() *d3d12.ID3D12RootSignature {
	return p.rootSignature
}

// -----------------------------------------------------------------------------
// Compile-time interface assertions
// -----------------------------------------------------------------------------

var (
	_ hal.ShaderModule    = (*ShaderModule)(nil)
	_ hal.BindGroupLayout = (*BindGroupLayout)(nil)
	_ hal.PipelineLayout  = (*PipelineLayout)(nil)
	_ hal.BindGroup       = (*BindGroup)(nil)
	_ hal.RenderPipeline  = (*RenderPipeline)(nil)
	_ hal.ComputePipeline = (*ComputePipeline)(nil)
)
