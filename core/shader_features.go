//go:build !(js && wasm)

package core

import (
	"fmt"
	"strings"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/naga"
	nagair "github.com/roffe/naga/ir"
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
	ast, err := naga.Parse(source)
	if err != nil {
		return nil //nolint:nilerr // parse errors are reported by HAL shader compilation
	}
	module, err := naga.Lower(ast)
	if err != nil {
		return nil //nolint:nilerr // lowering errors are reported by HAL shader compilation
	}
	if err := validateShaderSubgroupFeatures(module, features); err != nil {
		return err
	}

	caps := nagaCapabilitiesForFeatures(features)
	if capErrors, err := nagair.ValidateWithCapabilities(module, caps); err != nil {
		return fmt.Errorf("shader module: capability validation failed: %w", err)
	} else if len(capErrors) > 0 {
		return mapShaderCapabilityError(capErrors[0], features)
	}

	return nil
}

func nagaCapabilitiesForFeatures(features gputypes.Features) nagair.Capabilities {
	var caps nagair.Capabilities
	if features.Contains(gputypes.FeatureShaderF16) {
		caps |= nagair.CapShaderFloat16
	}
	if features.Contains(gputypes.FeatureShaderFloat64) {
		caps |= nagair.CapFloat64 | nagair.CapShaderInt64
	}
	return caps
}

func mapShaderCapabilityError(ve nagair.ValidationError, features gputypes.Features) error {
	msg := ve.Message
	switch {
	case strings.Contains(msg, "SHADER_FLOAT16") || strings.Contains(msg, "f16"):
		return RequireFeature(features, gputypes.FeatureShaderF16, "CreateShaderModule")
	case strings.Contains(msg, "FLOAT64") || strings.Contains(msg, "f64"):
		return RequireFeature(features, gputypes.FeatureShaderFloat64, "CreateShaderModule")
	case strings.Contains(msg, "SHADER_INT64") || strings.Contains(msg, "i64") || strings.Contains(msg, "u64"):
		return RequireFeature(features, gputypes.FeatureShaderFloat64, "CreateShaderModule")
	default:
		return fmt.Errorf("shader module: %s", msg)
	}
}

// validateShaderSubgroupFeatures requires FeatureSubgroupOperations for a
// module that uses subgroup operations or built-ins, and that and
// FeatureSubgroupBarrier for one with a subgroup barrier.
//
// Reference: naga valid/function.rs and interface.rs (Capabilities::SUBGROUP
// and SUBGROUP_BARRIER)
func validateShaderSubgroupFeatures(mod *nagair.Module, features gputypes.Features) error {
	ops, barrier := subgroupUse(mod)
	if barrier {
		if err := RequireFeature(features, gputypes.FeatureSubgroupBarrier, "CreateShaderModule"); err != nil {
			return err
		}
	}
	if ops || barrier {
		return RequireFeature(features, gputypes.FeatureSubgroupOperations, "CreateShaderModule")
	}
	return nil
}

// subgroupUse reports whether mod uses subgroup operations or built-ins, and
// whether it has a subgroup barrier.
func subgroupUse(mod *nagair.Module) (ops, barrier bool) {
	var visit func(nagair.Block)
	visit = func(block nagair.Block) {
		for _, st := range block {
			switch st := st.Kind.(type) {
			case nagair.StmtSubgroupBallot, nagair.StmtSubgroupCollectiveOperation, nagair.StmtSubgroupGather:
				ops = true
			case nagair.StmtBarrier:
				barrier = barrier || st.Flags&nagair.BarrierSubGroup != 0
			case nagair.StmtBlock:
				visit(st.Block)
			case nagair.StmtIf:
				visit(st.Accept)
				visit(st.Reject)
			case nagair.StmtSwitch:
				for _, c := range st.Cases {
					visit(c.Body)
				}
			case nagair.StmtLoop:
				visit(st.Body)
				visit(st.Continuing)
			}
		}
	}
	builtin := func(b *nagair.Binding) bool {
		if b == nil {
			return false
		}
		bb, ok := (*b).(nagair.BuiltinBinding)
		if !ok {
			return false
		}
		switch bb.Builtin {
		case nagair.BuiltinNumSubgroups, nagair.BuiltinSubgroupID, nagair.BuiltinSubgroupSize, nagair.BuiltinSubgroupInvocationID:
			return true
		}
		return false
	}
	for i := range mod.Functions {
		visit(mod.Functions[i].Body)
	}
	for i := range mod.EntryPoints {
		fn := &mod.EntryPoints[i].Function
		visit(fn.Body)
		for _, a := range fn.Arguments {
			ops = ops || builtin(a.Binding)
			if st, ok := mod.Types[a.Type].Inner.(nagair.StructType); ok {
				for _, m := range st.Members {
					ops = ops || builtin(m.Binding)
				}
			}
		}
	}
	return ops, barrier
}
