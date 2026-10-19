//go:build windows && !(js && wasm)

package d3d12

import (
	"encoding/binary"
	"testing"
	"unsafe"
)

func TestIndirectArgumentDescABI(t *testing.T) {
	if got := unsafe.Sizeof(D3D12_INDIRECT_ARGUMENT_DESC{}); got != 16 {
		t.Fatalf("D3D12_INDIRECT_ARGUMENT_DESC size = %d, want 16", got)
	}
	if got := unsafe.Offsetof(D3D12_INDIRECT_ARGUMENT_DESC{}.Union); got != 4 {
		t.Fatalf("D3D12_INDIRECT_ARGUMENT_DESC.Union offset = %d, want 4", got)
	}
}

// The driver reads the whole C struct, and walks arrays of them by its size,
// so a Go struct of another size has it reading the wrong bytes. Sizes and
// offsets are those of the C structs on 64-bit Windows.
func TestStructSizesMatchC(t *testing.T) {
	for _, c := range []struct {
		name      string
		got, want uintptr
	}{
		{"D3D12_UNORDERED_ACCESS_VIEW_DESC", unsafe.Sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC{}), 40},
		{"D3D12_UNORDERED_ACCESS_VIEW_DESC.Union", unsafe.Offsetof(D3D12_UNORDERED_ACCESS_VIEW_DESC{}.Union), 8},
		{"D3D12_SHADER_RESOURCE_VIEW_DESC", unsafe.Sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC{}), 40},
		{"D3D12_SHADER_RESOURCE_VIEW_DESC.Union", unsafe.Offsetof(D3D12_SHADER_RESOURCE_VIEW_DESC{}.Union), 16},
		{"D3D12_RENDER_TARGET_VIEW_DESC", unsafe.Sizeof(D3D12_RENDER_TARGET_VIEW_DESC{}), 24},
		{"D3D12_DEPTH_STENCIL_VIEW_DESC", unsafe.Sizeof(D3D12_DEPTH_STENCIL_VIEW_DESC{}), 24},
		{"D3D12_RAYTRACING_GEOMETRY_DESC", unsafe.Sizeof(D3D12_RAYTRACING_GEOMETRY_DESC{}), 56},
		{"D3D12_AUTO_BREADCRUMB_NODE1", unsafe.Sizeof(D3D12AutoBreadcrumbNode1{}), 96},
		{"D3D12_AUTO_BREADCRUMB_NODE1.BreadcrumbCount", unsafe.Offsetof(D3D12AutoBreadcrumbNode1{}.BreadcrumbCount), 48},
		{"D3D12_DRED_ALLOCATION_NODE1", unsafe.Sizeof(D3D12DREDAllocationNode1{}), 40},
		{"D3D12_RENDER_PASS_ENDING_ACCESS", unsafe.Sizeof(D3D12_RENDER_PASS_ENDING_ACCESS{}), 56},
		{"D3D12_RENDER_PASS_DEPTH_STENCIL_DESC", unsafe.Sizeof(D3D12_RENDER_PASS_DEPTH_STENCIL_DESC{}), 168},
	} {
		if c.got != c.want {
			t.Errorf("%s size = %d, want %d", c.name, c.got, c.want)
		}
	}
}

// d3d12BufferUAV mirrors D3D12_BUFFER_UAV (x64 layout: 8+4+4+8+4, padded to 32).
type d3d12BufferUAV struct {
	FirstElement         uint64
	NumElements          uint32
	StructureByteStride  uint32
	CounterOffsetInBytes uint64
	Flags                uint32
}

// The buffer SRV and UAV setters write the D3D12_BUFFER_SRV and
// D3D12_BUFFER_UAV union members at the offsets d3d12.h gives them. The UAV
// union used to be 16 bytes, too small for D3D12_BUFFER_UAV, so a buffer UAV
// had the driver reading past the Go value.
func TestBufferViewSetters(t *testing.T) {
	var uav D3D12_UNORDERED_ACCESS_VIEW_DESC
	if size := unsafe.Sizeof(d3d12BufferUAV{}); size > uintptr(len(uav.Union)) {
		t.Fatalf("D3D12_BUFFER_UAV (%d bytes) does not fit the UAV union (%d bytes)", size, len(uav.Union))
	}
	uav.SetBuffer(0x1_0000_0002, 77, 0, 0x30, D3D12_BUFFER_UAV_FLAG_RAW)
	if uav.ViewDimension != D3D12_UAV_DIMENSION_BUFFER {
		t.Errorf("UAV ViewDimension = %d, want BUFFER", uav.ViewDimension)
	}
	le := binary.LittleEndian
	if got := le.Uint64(uav.Union[0:8]); got != 0x1_0000_0002 {
		t.Errorf("UAV FirstElement = %#x", got)
	}
	if got := le.Uint32(uav.Union[8:12]); got != 77 {
		t.Errorf("UAV NumElements = %d", got)
	}
	if got := le.Uint32(uav.Union[12:16]); got != 0 {
		t.Errorf("UAV StructureByteStride = %d", got)
	}
	if got := le.Uint64(uav.Union[16:24]); got != 0x30 {
		t.Errorf("UAV CounterOffsetInBytes = %#x", got)
	}
	if got := le.Uint32(uav.Union[24:28]); got != uint32(D3D12_BUFFER_UAV_FLAG_RAW) {
		t.Errorf("UAV Flags = %d", got)
	}

	var srv D3D12_SHADER_RESOURCE_VIEW_DESC
	srv.SetBuffer(5, 9, 16, D3D12_BUFFER_SRV_FLAG_RAW)
	if srv.ViewDimension != D3D12_SRV_DIMENSION_BUFFER {
		t.Errorf("SRV ViewDimension = %d, want BUFFER", srv.ViewDimension)
	}
	if le.Uint64(srv.Union[0:8]) != 5 || le.Uint32(srv.Union[8:12]) != 9 ||
		le.Uint32(srv.Union[12:16]) != 16 || le.Uint32(srv.Union[16:20]) != uint32(D3D12_BUFFER_SRV_FLAG_RAW) {
		t.Errorf("D3D12_BUFFER_SRV bytes = %v", srv.Union[:20])
	}
}
