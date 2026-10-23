//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// A sampler asking for more anisotropy than the platform has gets what it
// has: Vulkan takes up to maxSamplerAnisotropy, Metal up to 16.
func TestSamplerAnisotropyIsClamped(t *testing.T) {
	device, h := newValidatedDevice(t)
	for _, anisotropy := range []uint16{16, 32, 64, 1000} {
		sampler, err := device.CreateSampler(&SamplerDescriptor{
			MagFilter: gputypes.FilterModeLinear, MinFilter: gputypes.FilterModeLinear,
			MipmapFilter: gputypes.FilterModeLinear, LodMaxClamp: 32, Anisotropy: anisotropy,
		})
		if err != nil {
			t.Fatalf("CreateSampler(anisotropy %d): %v", anisotropy, err)
		}
		sampler.Release()
	}
	assertNoValidation(t, h)
}
