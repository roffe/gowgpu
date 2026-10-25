// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/naga/glsl"
	"github.com/roffe/naga/ir"
)

// Image loads read zero out of bounds where GLSL has textureQueryLevels,
// 4.30: an out-of-bounds texelFetch is undefined in GL, and WebGPU wants
// zero or a texel of the texture.
func TestImageLoadsAreBoundsChecked(t *testing.T) {
	src := hal.ShaderSource{WGSL: `
@group(0) @binding(0) var tex: texture_2d<f32>;
@fragment fn fs_main(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
	return textureLoad(tex, vec2<i32>(p.xy), 0);
}`}
	for _, tc := range []struct {
		v       glsl.Version
		checked bool
	}{
		{glsl.Version{Major: 4, Minor: 50}, true},
		{glsl.Version{Major: 4, Minor: 30}, true},
		{glsl.Version{Major: 3, Minor: 30}, false},
		{glsl.VersionES300, false},
	} {
		code, _, err := compileWGSLToGLSL(tc.v, src, ir.StageFragment, "fs_main", nil, nil, false)
		if err != nil {
			t.Fatalf("GLSL %s: %v", tc.v, err)
		}
		if got := strings.Contains(code, "textureQueryLevels"); got != tc.checked {
			t.Errorf("GLSL %s: load checked %v, want %v:\n%s", tc.v, got, tc.checked, code)
		}
	}
}

// Indices that may be out of bounds are clamped to the last element at
// every GLSL version: GL leaves such an access undefined, and WGSL keeps it
// inside the array.
func TestIndicesAreClamped(t *testing.T) {
	src := hal.ShaderSource{WGSL: `
struct U { a: array<vec4<f32>, 4>, i: i32 }
@group(0) @binding(0) var<uniform> u: U;
var<private> p: array<f32, 5>;
@fragment fn fs_main() -> @location(0) vec4<f32> {
	return u.a[u.i] * p[u.i];
}`}
	for _, v := range []glsl.Version{{Major: 4, Minor: 50}, {Major: 3, Minor: 30}, glsl.VersionES300} {
		code, _, err := compileWGSLToGLSL(v, src, ir.StageFragment, "fs_main", nil, nil, false)
		if err != nil {
			t.Fatalf("GLSL %s: %v", v, err)
		}
		for _, clamp := range []string{", 3u)]", ", 4u)]"} {
			if !strings.Contains(code, clamp) {
				t.Errorf("GLSL %s: no index clamped with %q:\n%s", v, clamp, code)
			}
		}
	}
}

// A module's overrides take the pipeline's constants in the GLSL of a
// stage, and their defaults where the pipeline gives none: naga writes no
// GLSL for an override itself.
func TestOverridesTakePipelineConstants(t *testing.T) {
	src := hal.ShaderSource{WGSL: `
override answer: u32 = 1u;
override required: u32;
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(1) fn cs_main() {
	data[0] = answer;
	data[1] = required;
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
	return vec4<f32>(f32(answer));
}`}
	v := glsl.Version{Major: 4, Minor: 50}
	code, _, err := compileWGSLToGLSL(v, src, ir.StageCompute, "cs_main", map[string]float64{"answer": 42, "required": 7}, nil, false)
	if err != nil {
		t.Fatalf("compute: %v", err)
	}
	for _, want := range []string{"answer = 42u;", "required = 7u;"} {
		if !strings.Contains(code, want) {
			t.Errorf("compute: no %q:\n%s", want, code)
		}
	}
	if _, _, err := compileWGSLToGLSL(v, src, ir.StageCompute, "cs_main", map[string]float64{"answer": 42}, nil, false); err == nil {
		t.Error("compute without a value for required compiled")
	}
	code, _, err = compileWGSLToGLSL(v, src, ir.StageFragment, "fs_main", nil, nil, false)
	if err != nil {
		t.Fatalf("fragment: %v", err)
	}
	if !strings.Contains(code, "answer = 1u;") || strings.Contains(code, "required") {
		t.Errorf("fragment: want answer at its default and no required:\n%s", code)
	}
}
