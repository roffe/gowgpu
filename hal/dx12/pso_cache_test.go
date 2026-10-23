//go:build windows && !(js && wasm)

package dx12

import (
	"fmt"
	"testing"

	"github.com/roffe/gowgpu/hal/dx12/d3d12"
	"github.com/roffe/gowgpu/hal/dx12/dxgi"
)

func TestStaleCachedPSOErrorsRetryWithoutTheBlob(t *testing.T) {
	for _, hr := range []d3d12.HRESULTError{d3d12.E_INVALIDARG, d3d12.D3D12_ERROR_DRIVER_VERSION_MISMATCH, d3d12.D3D12_ERROR_ADAPTER_NOT_FOUND} {
		if !isInvalidCachedPSOError(fmt.Errorf("create: %w", hr)) {
			t.Errorf("%v does not drop the cached blob", hr)
		}
	}
	if isInvalidCachedPSOError(d3d12.E_OUTOFMEMORY) {
		t.Error("E_OUTOFMEMORY drops the cached blob")
	}
}

func TestPSOCacheKeyHasTheDriverVersionNotTheLUID(t *testing.T) {
	desc := dxgi.DXGI_ADAPTER_DESC1{VendorID: 0x10DE, DeviceID: 0x2702, SubSysID: 1, Revision: 0xA1, AdapterLuid: dxgi.LUID{LowPart: 1}}
	key := psoCacheKey(&desc, 32)
	if psoCacheKey(&desc, 33) == key {
		t.Error("another driver version keeps the key")
	}
	desc.AdapterLuid.LowPart = 2
	if psoCacheKey(&desc, 32) != key {
		t.Error("another LUID, which a reboot gives, changes the key")
	}
}
