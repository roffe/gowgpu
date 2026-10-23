//go:build !(js && wasm)

package core

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// float32Bindings checks a texture of format bound at binding 0 of
// sample type st, with a filtering sampler at binding 1.
func float32Bindings(format gputypes.TextureFormat, st gputypes.TextureSampleType, features gputypes.Features) error {
	layout := map[uint32]*gputypes.BindGroupLayoutEntry{
		0: {Binding: 0, Texture: &gputypes.TextureBindingLayout{SampleType: st}},
		1: {Binding: 1, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
	}
	samplers := []BindGroupSamplerInfo{{Binding: 1, Desc: &hal.SamplerDescriptor{MagFilter: gputypes.FilterModeLinear}}}
	textures := []BindGroupTextureInfo{{Binding: 0, Format: format, Usage: gputypes.TextureUsageTextureBinding, SampleCount: 1}}
	return validateBindGroupResources("float32", layout, nil, samplers, textures, features)
}

func TestFloat32TextureAtAFilterableBindingNeedsTheFeature(t *testing.T) {
	t.Parallel()
	if err := float32Bindings(gputypes.TextureFormatR32Float, gputypes.TextureSampleTypeFloat, 0); !IsFeatureError(err) {
		t.Fatalf("expected feature error, got %T: %v", err, err)
	}
	features := gputypes.Features(gputypes.FeatureFloat32Filterable)
	if err := float32Bindings(gputypes.TextureFormatR32Float, gputypes.TextureSampleTypeFloat, features); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

// A bind group doesn't pair its textures with its samplers: a 32-bit float
// texture at an unfilterable binding next to a filtering sampler is fine,
// the pipelines that sample them together are checked when they are made.
func TestFloat32TextureAtAnUnfilterableBindingNeedsNoFeature(t *testing.T) {
	t.Parallel()
	if err := float32Bindings(gputypes.TextureFormatR32Float, gputypes.TextureSampleTypeUnfilterableFloat, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestFilterableFormatNeedsNoFeature(t *testing.T) {
	t.Parallel()
	if err := float32Bindings(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureSampleTypeFloat, 0); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestSamplerUsesFiltering(t *testing.T) {
	t.Parallel()
	if samplerUsesFiltering(nil) {
		t.Fatal("nil sampler should not use filtering")
	}
	if !samplerUsesFiltering(&hal.SamplerDescriptor{MinFilter: gputypes.FilterModeLinear}) {
		t.Fatal("linear min filter should count as filtering")
	}
}

func TestIsUnfilterable32BitFloatFormat(t *testing.T) {
	t.Parallel()
	if !isUnfilterable32BitFloatFormat(gputypes.TextureFormatRG32Float) {
		t.Fatal("RG32Float should be unfilterable")
	}
	if isUnfilterable32BitFloatFormat(gputypes.TextureFormatRGBA8Unorm) {
		t.Fatal("RGBA8Unorm should not be unfilterable 32-bit float")
	}
}

func TestValidateSPIRVShaderFeatures_Float64_NoFeature(t *testing.T) {
	t.Parallel()
	spirv := buildSPIRVWithCapabilities(spirvCapFloat64, spirvCapShader)
	err := validateSPIRVShaderFeatures(spirv, gputypes.Features(0))
	if !IsFeatureError(err) {
		t.Fatalf("expected feature error, got %T: %v", err, err)
	}
}

func TestValidateSPIRVShaderFeatures_Float64_WithFeature(t *testing.T) {
	t.Parallel()
	spirv := buildSPIRVWithCapabilities(spirvCapFloat64, spirvCapShader)
	features := gputypes.Features(gputypes.FeatureShaderFloat64)
	if err := validateSPIRVShaderFeatures(spirv, features); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

const spirvCapShader = 1

func buildSPIRVWithCapabilities(caps ...uint32) []uint32 {
	words := make([]uint32, 0, 5+2*len(caps))
	words = append(words,
		spirvMagic,
		0x00010400, // SPIR-V 1.4
		0,
		1,
		0,
	)
	for _, capID := range caps {
		words = append(words, (2<<16)|spirvOpCodeCap, capID)
	}
	return words
}
