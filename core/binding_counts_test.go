//go:build !(js && wasm)

package core

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func uniformEntries(n int, visibility gputypes.ShaderStages, dynamic bool) []gputypes.BindGroupLayoutEntry {
	entries := make([]gputypes.BindGroupLayoutEntry, n)
	for i := range entries {
		entries[i] = gputypes.BindGroupLayoutEntry{Binding: uint32(i), Visibility: visibility,
			Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform, HasDynamicOffset: dynamic}}
	}
	return entries
}

func TestBindGroupLayoutLimits(t *testing.T) {
	limits := gputypes.DefaultLimits()
	for name, entries := range map[string][]gputypes.BindGroupLayoutEntry{
		"binding past the limit": {{Binding: limits.MaxBindingsPerBindGroup, Visibility: gputypes.ShaderStageFragment,
			Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}}},
		"too many dynamic uniform buffers":    uniformEntries(int(limits.MaxDynamicUniformBuffersPerPipelineLayout)+1, gputypes.ShaderStageVertex, true),
		"too many uniform buffers in a stage": uniformEntries(int(limits.MaxUniformBuffersPerShaderStage)+1, gputypes.ShaderStageFragment, false),
	} {
		if err := ValidateBindGroupLayoutDescriptor(&hal.BindGroupLayoutDescriptor{Entries: entries}, limits); err == nil {
			t.Errorf("%s: no error", name)
		}
	}
	// Each stage counts what it sees: as many uniform buffers as the limit
	// in each of two stages is fine.
	both := append(uniformEntries(int(limits.MaxUniformBuffersPerShaderStage), gputypes.ShaderStageVertex, false),
		uniformEntries(int(limits.MaxUniformBuffersPerShaderStage), gputypes.ShaderStageFragment, false)...)
	for i := range both {
		both[i].Binding = uint32(i)
	}
	if err := ValidateBindGroupLayoutDescriptor(&hal.BindGroupLayoutDescriptor{Entries: both}, limits); err != nil {
		t.Errorf("the limit in each of two stages: %v", err)
	}
}

func TestPipelineLayoutBindingsCountAcrossGroups(t *testing.T) {
	limits := gputypes.DefaultLimits()
	half := int(limits.MaxDynamicUniformBuffersPerPipelineLayout)/2 + 1
	dynamic := uniformEntries(half, gputypes.ShaderStageVertex, true)
	if err := ValidatePipelineLayoutBindings("one", [][]gputypes.BindGroupLayoutEntry{dynamic}, limits); err != nil {
		t.Fatalf("one group: %v", err)
	}
	if err := ValidatePipelineLayoutBindings("two", [][]gputypes.BindGroupLayoutEntry{dynamic, dynamic}, limits); err == nil {
		t.Error("two groups over the dynamic uniform buffer limit together: no error")
	}
	perStage := uniformEntries(int(limits.MaxUniformBuffersPerShaderStage)/2+1, gputypes.ShaderStageCompute, false)
	if err := ValidatePipelineLayoutBindings("stage", [][]gputypes.BindGroupLayoutEntry{perStage, perStage}, limits); err == nil {
		t.Error("two groups over the per-stage uniform buffer limit together: no error")
	}
}

func TestTextureFormatSupport(t *testing.T) {
	limits := gputypes.DefaultLimits()
	desc := func(format gputypes.TextureFormat, usage gputypes.TextureUsage, samples uint32) *hal.TextureDescriptor {
		return &hal.TextureDescriptor{Size: hal.Extent3D{Width: 16, Height: 16, DepthOrArrayLayers: 1}, MipLevelCount: 1,
			SampleCount: samples, Dimension: gputypes.TextureDimension2D, Format: format, Usage: usage}
	}
	var cte *CreateTextureError
	if err := ValidateTextureDescriptor(desc(gputypes.TextureFormatRGBA8UnormSrgb, gputypes.TextureUsageStorageBinding, 1), limits, 0); !errors.As(err, &cte) || cte.Kind != CreateTextureErrorInvalidUsage {
		t.Errorf("sRGB storage texture: %v, want CreateTextureErrorInvalidUsage", err)
	}
	if err := ValidateTextureDescriptor(desc(gputypes.TextureFormatRG32Float, gputypes.TextureUsageRenderAttachment, 4), limits, 0); !errors.As(err, &cte) || cte.Kind != CreateTextureErrorInvalidSampleCount {
		t.Errorf("multisampled RG32Float: %v, want CreateTextureErrorInvalidSampleCount", err)
	}
	if err := ValidateTextureDescriptor(desc(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureUsageRenderAttachment, 4), limits, 0); err != nil {
		t.Errorf("multisampled RGBA8Unorm: %v", err)
	}
	if err := ValidateTextureDescriptor(desc(gputypes.TextureFormatRGBA8UnormSrgb, gputypes.TextureUsageStorageBinding, 1), limits,
		gputypes.Features(gputypes.FeatureTextureAdapterSpecificFormatFeatures)); err != nil {
		t.Errorf("sRGB storage texture with adapter-specific format features: %v", err)
	}
}
