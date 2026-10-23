//go:build windows && !(js && wasm)

package dx12

import (
	"math"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
	"github.com/roffe/gowgpu/hal/dx12/dxgi"
)

func TestFeaturesLeaveOutShaderF16(t *testing.T) {
	a := &Adapter{capabilities: AdapterCapabilities{FeatureLevel: d3d12.D3D_FEATURE_LEVEL_12_1, ShaderModel: d3d12.D3D_SHADER_MODEL_6_7}}
	if a.Features().Contains(gputypes.FeatureShaderF16) {
		t.Fatal("ShaderF16 advertised, but shaders are compiled for shader model 5.1 or 6.0, which have no 16-bit floats")
	}
}

func TestAdaptersThisBackendCanNotUse(t *testing.T) {
	haswell := dxgi.DXGI_ADAPTER_DESC1{VendorID: 0x8086, DeviceID: 0x0416}
	if !withdrawnAdapter(&haswell) {
		t.Error("an Intel Haswell GPU is kept")
	}
	nvidia := dxgi.DXGI_ADAPTER_DESC1{VendorID: 0x10DE, DeviceID: 0x0416}
	if withdrawnAdapter(&nvidia) {
		t.Error("another vendor's device of a Haswell id is left out")
	}
	if bindingTierSupported(&nvidia, 1) {
		t.Error("a resource binding tier 1 adapter is kept")
	}
	if !bindingTierSupported(&nvidia, 2) || !bindingTierSupported(&nvidia, 3) {
		t.Error("a resource binding tier 2 or 3 adapter is left out")
	}
}

func TestStorageLimitsFitTheUAVsOfTheBindingTier(t *testing.T) {
	for _, tc := range []struct {
		tier, want uint32
	}{{2, 16}, {3, 64}} {
		limits := (&Adapter{capabilities: AdapterCapabilities{ResourceBindingTier: tc.tier}}).limits()
		if limits.MaxStorageBuffersPerShaderStage != tc.want || limits.MaxStorageTexturesPerShaderStage != tc.want {
			t.Errorf("tier %d: storage buffers %d, storage textures %d per stage, want %d each",
				tc.tier, limits.MaxStorageBuffersPerShaderStage, limits.MaxStorageTexturesPerShaderStage, tc.want)
		}
	}
}

func TestMaxBufferSizeFitsAVertexBufferView(t *testing.T) {
	if got := (&Adapter{}).limits().MaxBufferSize; got > math.MaxUint32 {
		t.Fatalf("MaxBufferSize = %d, more than the 32-bit size of a vertex or index buffer view", got)
	}
}
