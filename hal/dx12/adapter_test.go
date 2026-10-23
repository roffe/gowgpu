//go:build windows && !(js && wasm)

package dx12

import (
	"math"
	"slices"
	"testing"
	"unicode/utf16"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
	"github.com/roffe/gowgpu/hal/dx12/dxgi"
)

func TestFeaturesLeaveOutShaderF16(t *testing.T) {
	a := &Adapter{capabilities: AdapterCapabilities{FeatureLevel: d3d12.D3D_FEATURE_LEVEL_12_1, ShaderModel: d3d12.D3D_SHADER_MODEL_6_7}}
	if a.Features().Contains(gputypes.FeatureShaderF16) {
		t.Fatal("ShaderF16 advertised, but shaders are compiled for shader model 5.1 or 6.0, which have no 16-bit floats")
	}
}

func TestFeaturesHaveIndirectFirstInstance(t *testing.T) {
	for name, features := range map[string]gputypes.Features{
		"Adapter":       (&Adapter{}).Features(),
		"AdapterLegacy": (&AdapterLegacy{}).Features(),
	} {
		if !features.Contains(gputypes.FeatureIndirectFirstInstance) {
			t.Errorf("%s leaves out IndirectFirstInstance, which ExecuteIndirect draws have", name)
		}
	}
}

func TestAdaptersThisBackendCanNotUse(t *testing.T) {
	haswell := dxgi.DXGI_ADAPTER_DESC1{VendorID: 0x8086, DeviceID: 0x0416}
	if !skippedAdapter(&haswell) {
		t.Error("an Intel Haswell GPU is kept")
	}
	nvidia := dxgi.DXGI_ADAPTER_DESC1{VendorID: 0x10DE, DeviceID: 0x0416}
	if skippedAdapter(&nvidia) {
		t.Error("another vendor's device of a Haswell id is left out")
	}
	basic := dxgi.DXGI_ADAPTER_DESC1{VendorID: 0x1414, DeviceID: 0x8C}
	copy(basic.Description[:], utf16.Encode([]rune("Microsoft Basic Render Driver")))
	if !skippedAdapter(&basic) {
		t.Error("the Microsoft Basic Render Driver without the software flag is kept")
	}
	basic.Flags = dxgi.DXGI_ADAPTER_FLAG_SOFTWARE
	if skippedAdapter(&basic) {
		t.Error("WARP with the software flag is left out")
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

// TestAdaptersWithoutFactory6 enumerates adapters as on DXGI before
// Windows 10 1803, which has no IDXGIFactory6, and checks the same adapters
// come out.
func TestAdaptersWithoutFactory6(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping GPU test in short mode")
	}
	raw, err := Backend{}.CreateInstance(&hal.InstanceDescriptor{Backends: gputypes.BackendsDX12})
	if err != nil {
		t.Skipf("DX12 unavailable: %v", err)
	}
	defer raw.Destroy()
	inst := raw.(*Instance)
	names := func() []string {
		var names []string
		for _, a := range inst.EnumerateAdapters(nil) {
			names = append(names, a.Info.Name)
		}
		slices.Sort(names)
		return names
	}
	want := names()
	if len(want) == 0 {
		t.Skip("no DX12 adapters")
	}
	inst.factory6.Release()
	inst.factory6 = nil
	if got := names(); !slices.Equal(got, want) {
		t.Fatalf("adapters without IDXGIFactory6 = %q, want %q", got, want)
	}
}
