//go:build windows && !(js && wasm)

package dxgi

import (
	"errors"
	"syscall"
	"testing"

	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

func TestPresentTakesStatusCodesAsSuccess(t *testing.T) {
	for _, tc := range []struct {
		name string
		hr   d3d12.HRESULTError
		ok   bool
	}{
		{"S_OK", d3d12.S_OK, true},
		{"DXGI_STATUS_OCCLUDED", DXGI_STATUS_OCCLUDED, true},
		{"DXGI_STATUS_MODE_CHANGE_IN_PROGRESS", DXGI_STATUS_MODE_CHANGE_IN_PROGRESS, true},
		{"DXGI_ERROR_DEVICE_REMOVED", DXGI_ERROR_DEVICE_REMOVED, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			hr := uintptr(uint32(tc.hr))
			sc := &IDXGISwapChain4{vtbl: &idxgiSwapChain4Vtbl{
				Present:  syscall.NewCallback(func(_, _, _ uintptr) uintptr { return hr }),
				Present1: syscall.NewCallback(func(_, _, _, _ uintptr) uintptr { return hr }),
			}}
			for name, err := range map[string]error{
				"Present":  sc.Present(0, 0),
				"Present1": sc.Present1(0, 0, &DXGI_PRESENT_PARAMETERS{}),
			} {
				if tc.ok && err != nil {
					t.Errorf("%s = %v, want nil", name, err)
				}
				if !tc.ok && !errors.Is(err, tc.hr) {
					t.Errorf("%s = %v, want %v", name, err, tc.hr)
				}
			}
		})
	}
}
