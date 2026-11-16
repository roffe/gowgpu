//go:build windows && !(js && wasm)

package dxgi

import (
	"testing"
	"unsafe"
)

// Sizes and offsets of the C structs on 64-bit and 32-bit Windows, as the
// mingw-w64 headers lay them out. The adapter memory sizes are SIZE_T.
func TestStructSizesMatchC(t *testing.T) {
	for _, c := range []struct {
		name              string
		got, want, want32 uintptr
	}{
		{"DXGI_ADAPTER_DESC1", unsafe.Sizeof(DXGI_ADAPTER_DESC1{}), 312, 296},
		{"DXGI_ADAPTER_DESC1.AdapterLuid", unsafe.Offsetof(DXGI_ADAPTER_DESC1{}.AdapterLuid), 296, 284},
		{"DXGI_FRAME_STATISTICS", unsafe.Sizeof(DXGI_FRAME_STATISTICS{}), 32, 32},
		{"DXGI_FRAME_STATISTICS.SyncQPCTime", unsafe.Offsetof(DXGI_FRAME_STATISTICS{}.SyncQPCTime), 16, 16},
	} {
		want := c.want
		if unsafe.Sizeof(uintptr(0)) == 4 {
			want = c.want32
		}
		if c.got != want {
			t.Errorf("%s = %d, want %d", c.name, c.got, want)
		}
	}
}
