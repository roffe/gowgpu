//go:build windows && !(js && wasm)

package dx12

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
	"github.com/roffe/gowgpu/hal/dx12/dxgi"
)

func TestRemovedDeviceErrorsAreDeviceLost(t *testing.T) {
	for _, hr := range []d3d12.HRESULTError{dxgi.DXGI_ERROR_DEVICE_REMOVED, dxgi.DXGI_ERROR_DEVICE_HUNG, dxgi.DXGI_ERROR_DEVICE_RESET, dxgi.DXGI_ERROR_DRIVER_INTERNAL_ERROR} {
		err := removedDeviceLost(hr)
		if !errors.Is(err, hal.ErrDeviceLost) || !errors.Is(err, hr) {
			t.Errorf("%v: %v is not hal.ErrDeviceLost wrapping the HRESULT", hr, err)
		}
	}
	if err := removedDeviceLost(d3d12.E_INVALIDARG); errors.Is(err, hal.ErrDeviceLost) {
		t.Errorf("E_INVALIDARG became %v", err)
	}
	if err := deviceLost(dxgi.DXGI_ERROR_INVALID_CALL); !errors.Is(err, hal.ErrDeviceLost) {
		t.Errorf("a removal reason became %v, not hal.ErrDeviceLost", err)
	}
}
