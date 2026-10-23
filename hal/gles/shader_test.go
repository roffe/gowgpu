// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/naga/glsl"
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
		code, _, err := compileWGSLToGLSL(tc.v, src, "fs_main", nil)
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
		code, _, err := compileWGSLToGLSL(v, src, "fs_main", nil)
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
