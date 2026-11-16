//go:build !(js && wasm)

package core

import (
	"fmt"

	"github.com/gogpu/naga/valid"
	"github.com/gogpu/naga/wgsl"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func validateShaderModuleFeatures(desc *hal.ShaderModuleDescriptor, features gputypes.Features) error {
	if desc.Source.WGSL != "" {
		return validateWGSLShaderFeatures(desc.Source.WGSL, features)
	}
	if len(desc.Source.SPIRV) > 0 {
		return validateSPIRVShaderFeatures(desc.Source.SPIRV, features)
	}
	return nil
}

func validateWGSLShaderFeatures(source string, features gputypes.Features) error {
	module, _, err := wgsl.Parse(source, wgsl.ParseOptions{})
	if err != nil {
		return nil //nolint:nilerr // parse errors are reported by HAL shader compilation
	}
	caps := nagaCapabilitiesForFeatures(features)
	if missing := valid.MissingCapabilities(module, caps); missing != 0 {
		return missingFeatureError(missing, features)
	}
	if _, err := valid.Validate(module, valid.Options{Capabilities: caps}); err != nil {
		return fmt.Errorf("shader module: %w", err)
	}
	return nil
}

// featureCapabilities are the naga capabilities a gputypes feature enables.
// The rest are allowed here and left to the backend, which refuses what it
// cannot express.
const featureCapabilities = valid.CapShaderFloat16 | valid.CapFloat64 | valid.CapShaderInt64 |
	valid.CapSubgroup | valid.CapSubgroupBarrier

func nagaCapabilitiesForFeatures(features gputypes.Features) valid.Capabilities {
	caps := valid.CapAll &^ featureCapabilities
	if features.Contains(gputypes.FeatureShaderF16) {
		caps |= valid.CapShaderFloat16
	}
	if features.Contains(gputypes.FeatureShaderFloat64) {
		caps |= valid.CapFloat64 | valid.CapShaderInt64
	}
	if features.Contains(gputypes.FeatureSubgroupOperations) {
		caps |= valid.CapSubgroup
	}
	if features.Contains(gputypes.FeatureSubgroupBarrier) {
		caps |= valid.CapSubgroupBarrier
	}
	return caps
}

// missingFeatureError names the feature behind a capability the module needs.
func missingFeatureError(missing valid.Capabilities, features gputypes.Features) error {
	switch {
	case missing&valid.CapSubgroupBarrier != 0:
		return RequireFeature(features, gputypes.FeatureSubgroupBarrier, "CreateShaderModule")
	case missing&valid.CapSubgroup != 0:
		return RequireFeature(features, gputypes.FeatureSubgroupOperations, "CreateShaderModule")
	case missing&valid.CapShaderFloat16 != 0:
		return RequireFeature(features, gputypes.FeatureShaderF16, "CreateShaderModule")
	default:
		return RequireFeature(features, gputypes.FeatureShaderFloat64, "CreateShaderModule")
	}
}
