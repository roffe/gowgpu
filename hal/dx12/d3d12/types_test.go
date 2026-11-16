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
// offsets are those of the C structs on 64-bit and 32-bit Windows, as the
// mingw-w64 headers lay them out.
func TestStructSizesMatchC(t *testing.T) {
	for _, c := range []struct {
		name              string
		got, want, want32 uintptr
	}{
		{"D3D12_UNORDERED_ACCESS_VIEW_DESC", unsafe.Sizeof(D3D12_UNORDERED_ACCESS_VIEW_DESC{}), 40, 40},
		{"D3D12_UNORDERED_ACCESS_VIEW_DESC.Union", unsafe.Offsetof(D3D12_UNORDERED_ACCESS_VIEW_DESC{}.Union), 8, 8},
		{"D3D12_SHADER_RESOURCE_VIEW_DESC", unsafe.Sizeof(D3D12_SHADER_RESOURCE_VIEW_DESC{}), 40, 40},
		{"D3D12_SHADER_RESOURCE_VIEW_DESC.Union", unsafe.Offsetof(D3D12_SHADER_RESOURCE_VIEW_DESC{}.Union), 16, 16},
		{"D3D12_RENDER_TARGET_VIEW_DESC", unsafe.Sizeof(D3D12_RENDER_TARGET_VIEW_DESC{}), 24, 24},
		{"D3D12_DEPTH_STENCIL_VIEW_DESC", unsafe.Sizeof(D3D12_DEPTH_STENCIL_VIEW_DESC{}), 24, 24},
		{"D3D12_RAYTRACING_GEOMETRY_DESC", unsafe.Sizeof(D3D12_RAYTRACING_GEOMETRY_DESC{}), 56, 56},
		{"D3D12_AUTO_BREADCRUMB_NODE1", unsafe.Sizeof(D3D12AutoBreadcrumbNode1{}), 96, 48},
		{"D3D12_AUTO_BREADCRUMB_NODE1.BreadcrumbCount", unsafe.Offsetof(D3D12AutoBreadcrumbNode1{}.BreadcrumbCount), 48, 24},
		{"D3D12_DRED_ALLOCATION_NODE1", unsafe.Sizeof(D3D12DREDAllocationNode1{}), 40, 20},
		{"D3D12_RENDER_PASS_ENDING_ACCESS", unsafe.Sizeof(D3D12_RENDER_PASS_ENDING_ACCESS{}), 56, 32},
		{"D3D12_RENDER_PASS_ENDING_ACCESS.Union", unsafe.Offsetof(D3D12_RENDER_PASS_ENDING_ACCESS{}.Union), 8, 4},
		{"D3D12_RENDER_PASS_RENDER_TARGET_DESC", unsafe.Sizeof(D3D12_RENDER_PASS_RENDER_TARGET_DESC{}), 88, 60},
		{"D3D12_RENDER_PASS_DEPTH_STENCIL_DESC", unsafe.Sizeof(D3D12_RENDER_PASS_DEPTH_STENCIL_DESC{}), 168, 116},
		{"D3D12_RESOURCE_DESC", unsafe.Sizeof(D3D12_RESOURCE_DESC{}), 56, 56},
		{"D3D12_RESOURCE_DESC.Alignment", unsafe.Offsetof(D3D12_RESOURCE_DESC{}.Alignment), 8, 8},
		{"D3D12_RESOURCE_DESC.Flags", unsafe.Offsetof(D3D12_RESOURCE_DESC{}.Flags), 48, 48},
		{"D3D12_HEAP_DESC", unsafe.Sizeof(D3D12_HEAP_DESC{}), 48, 48},
		{"D3D12_HEAP_DESC.Alignment", unsafe.Offsetof(D3D12_HEAP_DESC{}.Alignment), 32, 32},
		{"D3D12_CONSTANT_BUFFER_VIEW_DESC", unsafe.Sizeof(D3D12_CONSTANT_BUFFER_VIEW_DESC{}), 16, 16},
		{"D3D12_PLACED_SUBRESOURCE_FOOTPRINT", unsafe.Sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT{}), 32, 32},
		{"D3D12_RESOURCE_BARRIER", unsafe.Sizeof(D3D12_RESOURCE_BARRIER{}), 32, 24},
		{"D3D12_ROOT_PARAMETER", unsafe.Sizeof(D3D12_ROOT_PARAMETER{}), 32, 20},
		{"D3D12_ROOT_PARAMETER.ShaderVisibility", unsafe.Offsetof(D3D12_ROOT_PARAMETER{}.ShaderVisibility), 24, 16},
		{"D3D12_TEXTURE_COPY_LOCATION", unsafe.Sizeof(D3D12_TEXTURE_COPY_LOCATION{}), 48, 40},
		{"D3D12_TEXTURE_COPY_LOCATION.Union", unsafe.Offsetof(D3D12_TEXTURE_COPY_LOCATION{}.Union), 16, 8},
		{"D3D12_WRITEBUFFERIMMEDIATE_PARAMETER", unsafe.Sizeof(D3D12_WRITEBUFFERIMMEDIATE_PARAMETER{}), 16, 16},
	} {
		want := c.want
		if ptrSize == 4 {
			want = c.want32
		}
		if c.got != want {
			t.Errorf("%s = %d, want %d", c.name, c.got, want)
		}
	}
}

func TestSplit64(t *testing.T) {
	const v = 0x1122_3344_5566_7788
	if ptrSize == 8 {
		if got := join64(lo(v), 0xdead); got != v {
			t.Errorf("join64 = %#x, want %#x", got, uint64(v))
		}
		return
	}
	if lo(v) != 0x5566_7788 || hi(v) != 0x1122_3344 {
		t.Errorf("lo, hi = %#x, %#x", lo(v), hi(v))
	}
	if got := join64(lo(v), hi(v)); got != v {
		t.Errorf("join64 = %#x, want %#x", got, uint64(v))
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
