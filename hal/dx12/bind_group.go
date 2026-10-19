// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// viewBinding pairs a CBV/SRV/UAV bind group entry with the layout entry that
// describes it. The layout entry type decides which descriptor kind is written.
type viewBinding struct {
	layout BindGroupLayoutEntry
	entry  gputypes.BindGroupEntry
}

// orderBindGroupEntries matches bind group entries to their layout entries and
// returns them in layout order, split into CBV/SRV/UAV entries and global
// sampler pool slots. A uniform buffer with a dynamic offset is in neither,
// it has a root CBV instead of a descriptor in the table.
//
// Layout order is required: createRootSignatureFromLayouts emits one descriptor
// range per non-sampler layout entry (D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND) and
// numbers sampler index-buffer slots in layout order, so descriptor i of the
// bind group's table must be the resource of the i-th non-sampler layout entry.
// WebGPU lets the caller list the entries in any order (fyne lists a shader's
// textures from a map), and Rust wgpu-core sorts both lists by binding before
// calling the HAL, so the DX12 HAL here never relies on the caller's order.
//
// An entry whose resource does not match its layout entry is refused here
// rather than written as the wrong descriptor kind.
func orderBindGroupEntries(layout *BindGroupLayout, entries []gputypes.BindGroupEntry) ([]viewBinding, []uint32, error) {
	views := make([]viewBinding, 0, len(layout.entries))
	var samplerPoolIndices []uint32

	for _, le := range layout.entries {
		entry, ok := findBindGroupEntry(entries, le.Binding)
		if !ok {
			return nil, nil, fmt.Errorf("dx12: bind group has no entry for layout binding %d", le.Binding)
		}

		switch le.Type {
		case BindingTypeSampler, BindingTypeComparisonSampler:
			res, ok := entry.Resource.(gputypes.SamplerBinding)
			if !ok || res.Sampler == 0 {
				return nil, nil, fmt.Errorf("dx12: binding %d: layout expects a sampler, got %T", le.Binding, entry.Resource)
			}
			sampler := fromHandle[Sampler](res.Sampler)
			samplerPoolIndices = append(samplerPoolIndices, sampler.samplerPoolSlot)

		case BindingTypeUniformBuffer, BindingTypeStorageBuffer, BindingTypeReadOnlyStorageBuffer:
			res, ok := entry.Resource.(gputypes.BufferBinding)
			if !ok || res.Buffer == 0 {
				return nil, nil, fmt.Errorf("dx12: binding %d: layout expects a buffer, got %T", le.Binding, entry.Resource)
			}
			if !le.dynamicUniform() {
				views = append(views, viewBinding{layout: le, entry: entry})
			}

		default: // BindingTypeSampledTexture, BindingTypeStorageTexture
			res, ok := entry.Resource.(gputypes.TextureViewBinding)
			if !ok || res.TextureView == 0 {
				return nil, nil, fmt.Errorf("dx12: binding %d: layout expects a texture view, got %T", le.Binding, entry.Resource)
			}
			views = append(views, viewBinding{layout: le, entry: entry})
		}
	}

	return views, samplerPoolIndices, nil
}

func findBindGroupEntry(entries []gputypes.BindGroupEntry, binding uint32) (gputypes.BindGroupEntry, bool) {
	for _, e := range entries {
		if e.Binding == binding {
			return e, true
		}
	}
	return gputypes.BindGroupEntry{}, false
}

// bufferBindingSize returns the bound byte size; Size 0 means "rest of buffer".
func bufferBindingSize(bufferSize uint64, res gputypes.BufferBinding) uint64 {
	if res.Size != 0 {
		return res.Size
	}
	if res.Offset >= bufferSize {
		return 0
	}
	return bufferSize - res.Offset
}

// constantBufferViewSize rounds a CBV size up to the 256-byte
// D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT. Uniform buffers are
// allocated with the same rounding in CreateBuffer, so the view stays inside
// the resource.
func constantBufferViewSize(size uint64) uint32 {
	return uint32((size + 255) &^ 255)
}

// rawBufferViewRange converts a byte range into the 32-bit element range used
// by raw (ByteAddressBuffer) SRVs and UAVs: FirstElement = offset/4,
// NumElements = size/4. WebGPU requires storage binding offsets to be
// 256-aligned and sizes to be multiples of 4, so no bytes are dropped.
func rawBufferViewRange(offset, size uint64) (firstElement uint64, numElements uint32) {
	return offset / 4, uint32(size / 4)
}

// writeBufferViewDescriptor writes the descriptor for one buffer binding at
// dest, with the view kind taken from the layout entry type. This is what
// the root signature range for the slot expects
// (bindingTypeToD3D12DescriptorRangeType) and what naga's HLSL declares:
//   - uniform buffer           -> CBV            (cbuffer, b#)
//   - read-only storage buffer -> raw buffer SRV (ByteAddressBuffer, t#)
//   - storage buffer           -> raw buffer UAV (RWByteAddressBuffer, u#)
//
// Same as the BindingType::Buffer arm of Rust wgpu-hal's create_bind_group.
func (d *Device) writeBufferViewDescriptor(le BindGroupLayoutEntry, buf *Buffer, res gputypes.BufferBinding, dest d3d12.D3D12_CPU_DESCRIPTOR_HANDLE) error {
	if le.HasDynamicOffset {
		// A storage buffer the shader adds its dynamic offset to: the view
		// runs to the end of the buffer, as in Rust's wgpu-hal.
		// ponytail: so arrayLength() on it counts from the bind group's
		// offset to the end of the buffer, whatever the binding size and
		// the dynamic offset, as in Rust. Fixing it takes the binding size
		// in the root constants and naga reading it from there.
		res.Size = 0
	}
	size := bufferBindingSize(buf.size, res)

	switch le.Type {
	case BindingTypeUniformBuffer:
		d.raw.CreateConstantBufferView(&d3d12.D3D12_CONSTANT_BUFFER_VIEW_DESC{
			BufferLocation: buf.gpuVA + res.Offset,
			SizeInBytes:    constantBufferViewSize(size),
		}, dest)

	case BindingTypeReadOnlyStorageBuffer:
		first, num := rawBufferViewRange(res.Offset, size)
		srvDesc := d3d12.D3D12_SHADER_RESOURCE_VIEW_DESC{
			Format:                  d3d12.DXGI_FORMAT_R32_TYPELESS,
			Shader4ComponentMapping: d3d12.D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
		}
		srvDesc.SetBuffer(first, num, 0, d3d12.D3D12_BUFFER_SRV_FLAG_RAW)
		d.raw.CreateShaderResourceView(buf.raw, &srvDesc, dest)

	case BindingTypeStorageBuffer:
		first, num := rawBufferViewRange(res.Offset, size)
		uavDesc := d3d12.D3D12_UNORDERED_ACCESS_VIEW_DESC{
			Format: d3d12.DXGI_FORMAT_R32_TYPELESS,
		}
		uavDesc.SetBuffer(first, num, 0, 0, d3d12.D3D12_BUFFER_UAV_FLAG_RAW)
		d.raw.CreateUnorderedAccessView(buf.raw, nil, &uavDesc, dest)

	default:
		return fmt.Errorf("dx12: binding %d: buffer bound to non-buffer layout entry (type %d)", le.Binding, le.Type)
	}
	return nil
}
