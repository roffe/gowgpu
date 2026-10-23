//go:build darwin && !(js && wasm)

package metal

import (
	"math"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestAdapterLimitsFitTheGPUFamily checks the limits each adapter offers
// against what its GPU family has, and against WebGPU's defaults, which
// families from Apple4 and Mac2 on reach.
func TestAdapterLimitsFitTheGPUFamily(t *testing.T) {
	if err := Init(); err != nil {
		t.Fatalf("Init failed: %v", err)
	}
	instance := &Instance{}
	adapters := instance.EnumerateAdapters(nil)
	if len(adapters) == 0 {
		t.Skip("no Metal adapters available")
	}
	want := gputypes.DefaultLimits()
	for _, exposed := range adapters {
		adapter := exposed.Adapter.(*Adapter)
		l := exposed.Capabilities.Limits
		family := gpuFamilyLimits(adapter.raw)
		t.Logf("%s: %+v", exposed.Info.Name, family)
		if textures := l.MaxSampledTexturesPerShaderStage + l.MaxStorageTexturesPerShaderStage; textures > family.textureEntries {
			t.Errorf("%s: %d textures a stage, the texture argument table has %d", exposed.Info.Name, textures, family.textureEntries)
		}
		if l.MaxStorageBufferBindingSize > math.MaxUint32 {
			t.Errorf("%s: storage bindings of %d bytes, past what a u32 `_buffer_sizes` field holds", exposed.Info.Name, l.MaxStorageBufferBindingSize)
		}
		if !DeviceSupportsFamily(adapter.raw, MTLGPUFamilyApple4) && !DeviceSupportsFamily(adapter.raw, MTLGPUFamilyMac2) {
			adapter.Destroy()
			continue
		}
		for _, c := range []struct {
			name      string
			got, want uint32
		}{
			{"MaxTextureDimension2D", l.MaxTextureDimension2D, want.MaxTextureDimension2D},
			{"MaxSampledTexturesPerShaderStage", l.MaxSampledTexturesPerShaderStage, want.MaxSampledTexturesPerShaderStage},
			{"MaxStorageTexturesPerShaderStage", l.MaxStorageTexturesPerShaderStage, want.MaxStorageTexturesPerShaderStage},
			{"MaxInterStageShaderVariables", l.MaxInterStageShaderVariables, want.MaxInterStageShaderVariables},
			{"MaxColorAttachments", l.MaxColorAttachments, want.MaxColorAttachments},
			{"MaxColorAttachmentBytesPerSample", l.MaxColorAttachmentBytesPerSample, want.MaxColorAttachmentBytesPerSample},
			{"MaxComputeWorkgroupStorageSize", l.MaxComputeWorkgroupStorageSize, want.MaxComputeWorkgroupStorageSize},
			{"MaxComputeInvocationsPerWorkgroup", l.MaxComputeInvocationsPerWorkgroup, want.MaxComputeInvocationsPerWorkgroup},
			{"MaxVertexBuffers", l.MaxVertexBuffers, want.MaxVertexBuffers},
		} {
			if c.got < c.want {
				t.Errorf("%s: %s %d, below WebGPU's default %d", exposed.Info.Name, c.name, c.got, c.want)
			}
		}
		adapter.Destroy()
	}
}
