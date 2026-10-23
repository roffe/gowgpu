//go:build !(js && wasm)

package core

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
)

// shaderStages are the stages the per-stage limits count bindings for, in
// the order of bindingCounts' arrays, and stageNames their names.
var (
	shaderStages = [3]gputypes.ShaderStages{gputypes.ShaderStageVertex, gputypes.ShaderStageFragment, gputypes.ShaderStageCompute}
	stageNames   = [3]string{"vertex", "fragment", "compute"}
)

// bindingCounts counts bind group layout entries by kind for each shader
// stage that sees them, and the buffers with a dynamic offset: what the
// per-stage and per-pipeline-layout limits cap, for one bind group layout
// or all of a pipeline layout's.
//
// Reference: wgpu-core binding_model.rs BindingTypeMaxCountValidator
type bindingCounts struct {
	samplers, sampledTextures, storageBuffers, storageTextures, uniformBuffers [3]uint32
	dynamicUniformBuffers, dynamicStorageBuffers                               uint32
}

// add counts entries in.
func (c *bindingCounts) add(entries []gputypes.BindGroupLayoutEntry) {
	for i := range entries {
		e := &entries[i]
		var perStage *[3]uint32
		switch {
		case e.Buffer != nil && e.Buffer.Type == gputypes.BufferBindingTypeUniform:
			perStage = &c.uniformBuffers
			if e.Buffer.HasDynamicOffset {
				c.dynamicUniformBuffers++
			}
		case e.Buffer != nil:
			perStage = &c.storageBuffers
			if e.Buffer.HasDynamicOffset {
				c.dynamicStorageBuffers++
			}
		case e.Sampler != nil:
			perStage = &c.samplers
		case e.Texture != nil:
			perStage = &c.sampledTextures
		case e.StorageTexture != nil:
			perStage = &c.storageTextures
		default:
			continue
		}
		for s, stage := range shaderStages {
			if e.Visibility&stage != 0 {
				perStage[s]++
			}
		}
	}
}

// validate checks the counts against limits. A limit of 0 is not set.
func (c *bindingCounts) validate(limits gputypes.Limits) error {
	check := func(what string, n, limit uint32) error {
		if limit > 0 && n > limit {
			return fmt.Errorf("%d %s, the limit is %d", n, what, limit)
		}
		return nil
	}
	for s := range shaderStages {
		for _, perStage := range []struct {
			what  string
			n     uint32
			limit uint32
		}{
			{"samplers", c.samplers[s], limits.MaxSamplersPerShaderStage},
			{"sampled textures", c.sampledTextures[s], limits.MaxSampledTexturesPerShaderStage},
			{"storage buffers", c.storageBuffers[s], limits.MaxStorageBuffersPerShaderStage},
			{"storage textures", c.storageTextures[s], limits.MaxStorageTexturesPerShaderStage},
			{"uniform buffers", c.uniformBuffers[s], limits.MaxUniformBuffersPerShaderStage},
		} {
			if err := check(perStage.what+" in the "+stageNames[s]+" stage", perStage.n, perStage.limit); err != nil {
				return err
			}
		}
	}
	if err := check("uniform buffers with dynamic offsets", c.dynamicUniformBuffers, limits.MaxDynamicUniformBuffersPerPipelineLayout); err != nil {
		return err
	}
	return check("storage buffers with dynamic offsets", c.dynamicStorageBuffers, limits.MaxDynamicStorageBuffersPerPipelineLayout)
}

// ValidatePipelineLayoutBindings checks the bind group layouts of a pipeline
// layout, given by their entries, against the per-stage limits and the
// limits on dynamic buffers of a pipeline layout, counting all of them.
//
// Reference: wgpu-core device/resource.rs create_pipeline_layout
// (count_validator.merge)
func ValidatePipelineLayoutBindings(label string, groups [][]gputypes.BindGroupLayoutEntry, limits gputypes.Limits) error {
	var counts bindingCounts
	for _, entries := range groups {
		counts.add(entries)
	}
	if err := counts.validate(limits); err != nil {
		return fmt.Errorf("pipeline layout %q: %w", label, err)
	}
	return nil
}
