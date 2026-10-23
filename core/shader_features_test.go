//go:build !(js && wasm)

package core

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func TestValidateShaderModuleFeatures_EmptySource(t *testing.T) {
	t.Parallel()
	err := validateShaderModuleFeatures(&hal.ShaderModuleDescriptor{}, gputypes.Features(0))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestValidateShaderModuleFeatures_SPIRVPath(t *testing.T) {
	t.Parallel()
	spirv := buildSPIRVWithCapabilities(spirvCapFloat64, spirvCapShader)
	desc := &hal.ShaderModuleDescriptor{
		Source: hal.ShaderSource{SPIRV: spirv},
	}
	err := validateShaderModuleFeatures(desc, gputypes.Features(0))
	if !IsFeatureError(err) {
		t.Fatalf("expected feature error, got %T: %v", err, err)
	}
}

func TestValidateShaderSubgroupFeatures(t *testing.T) {
	t.Parallel()
	ops := gputypes.Features(gputypes.FeatureSubgroupOperations)
	for _, tc := range []struct {
		name     string
		source   string
		features gputypes.Features
		wantErr  bool
	}{
		{"barrier without the features", `@compute @workgroup_size(1) fn main() { subgroupBarrier(); }`, ops, true},
		{"operation with the feature", `@compute @workgroup_size(1) fn main() { let x = subgroupAdd(1u); }`, ops, false},
		{"operation without the feature", `@compute @workgroup_size(1) fn main() { let x = subgroupAdd(1u); }`, 0, true},
		{"built-in without the feature", `@compute @workgroup_size(1) fn main(@builtin(subgroup_size) n: u32) { }`, 0, true},
		{"the word in a comment and a name", `// subgroup sizes vary
@compute @workgroup_size(1) fn main() { let subgroupish = 1u; }`, 0, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateWGSLShaderFeatures(tc.source, tc.features)
			switch {
			case tc.wantErr && !IsFeatureError(err):
				t.Fatalf("expected feature error, got %T: %v", err, err)
			case !tc.wantErr && err != nil:
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestNagaCapabilitiesForFeatures(t *testing.T) {
	t.Parallel()
	features := gputypes.Features(gputypes.FeatureShaderF16 | gputypes.FeatureShaderFloat64)
	caps := nagaCapabilitiesForFeatures(features)
	if caps == 0 {
		t.Fatal("expected non-zero capabilities")
	}
}
