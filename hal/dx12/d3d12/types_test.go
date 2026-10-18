//go:build windows && !(js && wasm)

package d3d12

import (
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
		{"D3D12_RENDER_TARGET_VIEW_DESC", unsafe.Sizeof(D3D12_RENDER_TARGET_VIEW_DESC{}), 24},
		{"D3D12_DEPTH_STENCIL_VIEW_DESC", unsafe.Sizeof(D3D12_DEPTH_STENCIL_VIEW_DESC{}), 24},
		{"D3D12_RAYTRACING_GEOMETRY_DESC", unsafe.Sizeof(D3D12_RAYTRACING_GEOMETRY_DESC{}), 56},
		{"D3D12_AUTO_BREADCRUMB_NODE1", unsafe.Sizeof(D3D12AutoBreadcrumbNode1{}), 96},
		{"D3D12_AUTO_BREADCRUMB_NODE1.BreadcrumbCount", unsafe.Offsetof(D3D12AutoBreadcrumbNode1{}.BreadcrumbCount), 48},
		{"D3D12_DRED_ALLOCATION_NODE1", unsafe.Sizeof(D3D12DREDAllocationNode1{}), 40},
	} {
		if c.got != c.want {
			t.Errorf("%s size = %d, want %d", c.name, c.got, c.want)
		}
	}
}
