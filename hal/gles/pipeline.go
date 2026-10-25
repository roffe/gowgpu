// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"fmt"
	"time"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
	"github.com/roffe/naga/glsl"
	"github.com/roffe/naga/ir"
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
	vertexGLSL, vertexTranslationInfo, err := compileWGSLToGLSL(d.glslVersion, vertexModule.source, ir.StageVertex, desc.Vertex.EntryPoint, nil, layout.bindingMap, false)
	if err != nil {
		return nil, fmt.Errorf("gles: vertex shader: %w", err)
	}

	vertexID, err := compileGLSL(glCtx, gl.VERTEX_SHADER, "vertex", vertexGLSL)
	if err != nil {
		return nil, err
	}

	var fragmentID uint32
	var fragmentTranslationInfo glsl.TranslationInfo
	if desc.Fragment != nil {
		fragmentID, fragmentTranslationInfo, err = compileFragmentShader(glCtx, d.glslVersion, desc.Fragment, layout.bindingMap)
	} else {
		// ES links no program without a fragment shader. wgpu-hal gives
		// a vertex-only pipeline an empty one on every context.
		fragmentID, err = compileGLSL(glCtx, gl.FRAGMENT_SHADER, "fragment", "#version "+d.glslVersion.String()+"\nvoid main(void) {}\n")
	}
	if err != nil {
		glCtx.DeleteShader(vertexID)
		return nil, err
	}

	// Link program
	programID := glCtx.CreateProgram()
	glCtx.AttachShader(programID, vertexID)
	glCtx.AttachShader(programID, fragmentID)
	glCtx.LinkProgram(programID)

	var status int32
	glCtx.GetProgramiv(programID, gl.LINK_STATUS, &status)
	if status == gl.FALSE {
		log := glCtx.GetProgramInfoLog(programID)
		glCtx.DeleteShader(vertexID)
		glCtx.DeleteShader(fragmentID)
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
			glCtx.DeleteShader(fragmentID)
			glCtx.DeleteProgram(programID)
			return nil, err
		}
	}

	// Shaders can be deleted after linking
	glCtx.DeleteShader(vertexID)
	glCtx.DeleteShader(fragmentID)

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
		sampleMask:        sampleMaskFor(d.sampleMask, desc.Multisample.Mask),
		attribs:           attribLocations(desc.Vertex.Buffers),
		colorTargets:      colorTargets,
		perTargetBlend:    perTarget,
		vertexBuffers:     desc.Vertex.Buffers,
	}

	pipeline.samplerBindMap = samplerBindMapFor(layout, vertexTranslationInfo, fragmentTranslationInfo)
	// naga writes instance_index as gl_InstanceID plus this uniform, and
	// gl_InstanceID starts at 0 even in a base instance draw.
	pipeline.firstInstanceLoc = glCtx.GetUniformLocation(programID, "naga_vs_first_instance")
	pipeline.fragDepthRangeLoc = glCtx.GetUniformLocation(programID, glsl.FragDepthRangeUniform)
	pipeline.clampsFragDepth = pipeline.fragDepthRangeLoc >= 0

	return pipeline, nil
}

// createComputePipeline compiles and links a compute pipeline. The context
// must be current.
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
	computeGLSL, computeTranslationInfo, err := compileWGSLToGLSL(d.glslVersion, computeModule.source, ir.StageCompute, desc.Compute.EntryPoint,
		desc.Compute.Constants, layout.bindingMap, !desc.Compute.ZeroInitializeWorkgroupMemory)
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

// attribLocations is the set of shader locations buffers feed.
func attribLocations(buffers []gputypes.VertexBufferLayout) (m uint32) {
	for _, b := range buffers {
		for _, a := range b.Attributes {
			if a.ShaderLocation < 32 {
				m |= 1 << a.ShaderLocation
			}
		}
	}
	return m
}

// sampleMaskFor is the GL sample mask of a pipeline's mask, all samples for
// 0 as on the other backends, and 0 on a context without sample masks: ES
// 3.0 has none, and GL_SAMPLE_MASK is no enum there.
func sampleMaskFor(contextHasMasks bool, mask uint64) uint32 {
	switch {
	case !contextHasMasks:
		return 0
	case uint32(mask) == 0:
		return 0xFFFFFFFF
	}
	return uint32(mask)
}

// compileFragmentShader compiles a fragment shader from WGSL source via GLSL.
func compileFragmentShader(glCtx *gl.Context, version glsl.Version, frag *hal.FragmentState, bindingMap map[glsl.BindingMapKey]uint8) (uint32, glsl.TranslationInfo, error) {
	fragmentModule, ok := frag.Module.(*ShaderModule)
	if !ok {
		return 0, glsl.TranslationInfo{}, fmt.Errorf("gles: invalid fragment shader module type")
	}

	fragmentGLSL, translationInfo, err := compileWGSLToGLSL(version, fragmentModule.source, ir.StageFragment, frag.EntryPoint, nil, bindingMap, false)
	if err != nil {
		return 0, glsl.TranslationInfo{}, fmt.Errorf("gles: fragment shader: %w", err)
	}

	fragmentID, err := compileGLSL(glCtx, gl.FRAGMENT_SHADER, "fragment", fragmentGLSL)
	return fragmentID, translationInfo, err
}

// compileGLSL compiles a shader of the stage given from GLSL source.
func compileGLSL(glCtx *gl.Context, stage uint32, name, source string) (uint32, error) {
	id := glCtx.CreateShader(stage)
	glCtx.ShaderSource(id, source)
	glCtx.CompileShader(id)
	var status int32
	glCtx.GetShaderiv(id, gl.COMPILE_STATUS, &status)
	if status == gl.FALSE {
		log := glCtx.GetShaderInfoLog(id)
		glCtx.DeleteShader(id)
		return 0, fmt.Errorf("gles: %s shader compilation failed: %s", name, log)
	}
	if infoLog := glCtx.GetShaderInfoLog(id); infoLog != "" {
		hal.Logger().Debug("gles: shader compile info", "stage", name, "info", infoLog)
	}
	return id, nil
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
