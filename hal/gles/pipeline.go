// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"fmt"
	"time"

	"github.com/gogpu/naga/glsl"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// createRenderPipeline compiles and links a render pipeline. The context
// must be current.
func (d *Device) createRenderPipeline(glCtx *gl.Context, desc *RenderPipelineDescriptor) (*RenderPipeline, error) {
	start := time.Now()
	perTarget := desc.Fragment != nil && !sameColorTargets(desc.Fragment.Targets)
	if perTarget && !d.independentBlend {
		return nil, fmt.Errorf("gles: color targets with different blending or write masks need OpenGL 4.0, ES 3.2 or GL_*_draw_buffers_indexed")
	}
	// Handle nil layout (auto-layout for shaders without bindings).
	var layout *PipelineLayout
	if desc.Layout != nil {
		var ok bool
		layout, ok = desc.Layout.(*PipelineLayout)
		if !ok {
			return nil, fmt.Errorf("gles: invalid pipeline layout type")
		}
	} else {
		layout = &PipelineLayout{}
	}

	vertexModule, ok := desc.Vertex.Module.(*ShaderModule)
	if !ok {
		return nil, fmt.Errorf("gles: invalid vertex shader module type")
	}

	// Compile WGSL → GLSL for vertex stage.
	vertexGLSL, vertexTranslationInfo, err := compileWGSLToGLSL(d.glslVersion, vertexModule.source, desc.Vertex.EntryPoint, layout.bindingMap)
	if err != nil {
		return nil, fmt.Errorf("gles: vertex shader: %w", err)
	}

	vertexID := glCtx.CreateShader(gl.VERTEX_SHADER)
	glCtx.ShaderSource(vertexID, vertexGLSL)
	glCtx.CompileShader(vertexID)

	var status int32
	glCtx.GetShaderiv(vertexID, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		log := glCtx.GetShaderInfoLog(vertexID)
		glCtx.DeleteShader(vertexID)
		return nil, fmt.Errorf("gles: vertex shader compilation failed: %s", log)
	}
	if infoLog := glCtx.GetShaderInfoLog(vertexID); infoLog != "" {
		hal.Logger().Debug("gles: vertex shader compile info", "info", infoLog)
	}

	// Compile fragment shader
	var fragmentID uint32
	var fragmentTranslationInfo glsl.TranslationInfo
	if desc.Fragment != nil {
		var err error
		fragmentID, fragmentTranslationInfo, err = compileFragmentShader(glCtx, d.glslVersion, desc.Fragment, layout.bindingMap)
		if err != nil {
			glCtx.DeleteShader(vertexID)
			return nil, err
		}
	}

	// Link program
	programID := glCtx.CreateProgram()
	glCtx.AttachShader(programID, vertexID)
	if fragmentID != 0 {
		glCtx.AttachShader(programID, fragmentID)
	}
	glCtx.LinkProgram(programID)

	glCtx.GetProgramiv(programID, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		log := glCtx.GetProgramInfoLog(programID)
		glCtx.DeleteShader(vertexID)
		if fragmentID != 0 {
			glCtx.DeleteShader(fragmentID)
		}
		glCtx.DeleteProgram(programID)
		return nil, fmt.Errorf("gles: program linking failed: %s", log)
	}
	if infoLog := glCtx.GetProgramInfoLog(programID); infoLog != "" {
		hal.Logger().Debug("gles: program link info", "info", infoLog)
	}

	// On GL < 4.2 (GLSL < 420), layout(binding=N) is unavailable so naga
	// omits it. We must assign bindings at runtime after linking, following
	// the Rust wgpu-hal pattern (device.rs:438-461).
	if !d.shaderBindingLayout {
		if err := assignBindingsAfterLink(glCtx, programID, layout, vertexTranslationInfo, fragmentTranslationInfo); err != nil {
			glCtx.DeleteShader(vertexID)
			if fragmentID != 0 {
				glCtx.DeleteShader(fragmentID)
			}
			glCtx.DeleteProgram(programID)
			return nil, err
		}
	}

	// Shaders can be deleted after linking
	glCtx.DeleteShader(vertexID)
	if fragmentID != 0 {
		glCtx.DeleteShader(fragmentID)
	}

	hal.Logger().Debug("gles: render pipeline created",
		"programID", programID,
		"vertexEntry", desc.Vertex.EntryPoint,
		"elapsed", time.Since(start),
	)

	// Extract per-target blend/write-mask configuration for MRT.
	// Matches Rust wgpu-hal GLES device.rs:1559-1570: stores ALL color targets
	// so that SetPipeline can apply per-draw-buffer blend state.
	var colorTargets []ColorTargetDesc
	if desc.Fragment != nil {
		colorTargets = make([]ColorTargetDesc, len(desc.Fragment.Targets))
		for i, ct := range desc.Fragment.Targets {
			colorTargets[i] = ColorTargetDesc{
				Blend:     ct.Blend,
				WriteMask: ct.WriteMask,
			}
		}
	}

	pipeline := &RenderPipeline{
		programID:         programID,
		layout:            layout,
		glCtx:             glCtx,
		primitiveTopology: desc.Primitive.Topology,
		cullMode:          desc.Primitive.CullMode,
		frontFace:         desc.Primitive.FrontFace,
		depthStencil:      desc.DepthStencil,
		depthClamp:        d.depthClamp,
		unclippedDepth:    desc.Primitive.UnclippedDepth,
		multisample:       desc.Multisample,
		colorTargets:      colorTargets,
		perTargetBlend:    perTarget,
		vertexBuffers:     desc.Vertex.Buffers,
	}

	pipeline.samplerBindMap = samplerBindMapFor(layout, vertexTranslationInfo, fragmentTranslationInfo)
	// naga writes instance_index as gl_InstanceID plus this uniform, and
	// gl_InstanceID starts at 0 even in a base instance draw.
	pipeline.firstInstanceLoc = glCtx.GetUniformLocation(programID, "naga_vs_first_instance")

	return pipeline, nil
}

// createComputePipeline compiles and links a compute pipeline. The context
// must be current.
//
// TODO(compute-constants): Apply desc.Compute.Constants via naga's
// pipeline_constants::process_overrides before GLSL emission. Rust wgpu-hal
// GLES calls naga::back::pipeline_constants::process_overrides() in
// create_shader (gles/device.rs:226).
//
// TODO(zero-init-workgroup): Pass desc.Compute.ZeroInitializeWorkgroupMemory
// to naga GLSL options. Rust wgpu-hal sets naga_options.zero_initialize_workgroup_memory
// per-stage (gles/device.rs:268).
func (d *Device) createComputePipeline(glCtx *gl.Context, desc *ComputePipelineDescriptor) (*ComputePipeline, error) {
	start := time.Now()
	layout, ok := desc.Layout.(*PipelineLayout)
	if !ok {
		return nil, fmt.Errorf("gles: invalid pipeline layout type")
	}

	computeModule, ok := desc.Compute.Module.(*ShaderModule)
	if !ok {
		return nil, fmt.Errorf("gles: invalid compute shader module type")
	}

	// Compile WGSL → GLSL for compute stage.
	computeGLSL, computeTranslationInfo, err := compileWGSLToGLSL(d.glslVersion, computeModule.source, desc.Compute.EntryPoint, layout.bindingMap)
	if err != nil {
		return nil, fmt.Errorf("gles: compute shader: %w", err)
	}

	computeID := glCtx.CreateShader(gl.COMPUTE_SHADER)
	glCtx.ShaderSource(computeID, computeGLSL)
	glCtx.CompileShader(computeID)

	var status int32
	glCtx.GetShaderiv(computeID, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		log := glCtx.GetShaderInfoLog(computeID)
		glCtx.DeleteShader(computeID)
		return nil, fmt.Errorf("gles: compute shader compilation failed: %s", log)
	}
	if infoLog := glCtx.GetShaderInfoLog(computeID); infoLog != "" {
		hal.Logger().Debug("gles: compute shader compile info", "info", infoLog)
	}

	// Link program
	programID := glCtx.CreateProgram()
	glCtx.AttachShader(programID, computeID)
	glCtx.LinkProgram(programID)

	glCtx.GetProgramiv(programID, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		log := glCtx.GetProgramInfoLog(programID)
		glCtx.DeleteShader(computeID)
		glCtx.DeleteProgram(programID)
		return nil, fmt.Errorf("gles: compute program linking failed: %s", log)
	}
	if infoLog := glCtx.GetProgramInfoLog(programID); infoLog != "" {
		hal.Logger().Debug("gles: compute program link info", "info", infoLog)
	}

	// Runtime binding fallback for GL < 4.2 (see CreateRenderPipeline).
	if !d.shaderBindingLayout {
		if err := assignBindingsAfterLink(glCtx, programID, layout, computeTranslationInfo); err != nil {
			glCtx.DeleteShader(computeID)
			glCtx.DeleteProgram(programID)
			return nil, err
		}
	}

	glCtx.DeleteShader(computeID)

	hal.Logger().Debug("gles: compute pipeline created",
		"programID", programID,
		"entryPoint", desc.Compute.EntryPoint,
		"elapsed", time.Since(start),
	)

	return &ComputePipeline{
		programID:      programID,
		layout:         layout,
		glCtx:          glCtx,
		samplerBindMap: samplerBindMapFor(layout, computeTranslationInfo),
	}, nil
}

// compileFragmentShader compiles a fragment shader from WGSL source via GLSL.
func compileFragmentShader(glCtx *gl.Context, version glsl.Version, frag *hal.FragmentState, bindingMap map[glsl.BindingMapKey]uint8) (uint32, glsl.TranslationInfo, error) {
	fragmentModule, ok := frag.Module.(*ShaderModule)
	if !ok {
		return 0, glsl.TranslationInfo{}, fmt.Errorf("gles: invalid fragment shader module type")
	}

	fragmentGLSL, translationInfo, err := compileWGSLToGLSL(version, fragmentModule.source, frag.EntryPoint, bindingMap)
	if err != nil {
		return 0, glsl.TranslationInfo{}, fmt.Errorf("gles: fragment shader: %w", err)
	}

	fragmentID := glCtx.CreateShader(gl.FRAGMENT_SHADER)
	glCtx.ShaderSource(fragmentID, fragmentGLSL)
	glCtx.CompileShader(fragmentID)

	var status int32
	glCtx.GetShaderiv(fragmentID, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		log := glCtx.GetShaderInfoLog(fragmentID)
		glCtx.DeleteShader(fragmentID)
		return 0, glsl.TranslationInfo{}, fmt.Errorf("gles: fragment shader compilation failed: %s", log)
	}
	if infoLog := glCtx.GetShaderInfoLog(fragmentID); infoLog != "" {
		hal.Logger().Debug("gles: fragment shader compile info", "info", infoLog)
	}

	return fragmentID, translationInfo, nil
}

// sameColorTargets reports whether every target blends and masks like the
// first.
func sameColorTargets(targets []gputypes.ColorTargetState) bool {
	for _, t := range targets[min(1, len(targets)):] {
		if t.WriteMask != targets[0].WriteMask || (t.Blend == nil) != (targets[0].Blend == nil) ||
			t.Blend != nil && *t.Blend != *targets[0].Blend {
			return false
		}
	}
	return true
}
