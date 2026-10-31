// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"strings"
	"testing"

	"github.com/gogpu/naga"
	"github.com/gogpu/naga/dxil"
	"github.com/gogpu/naga/hlsl"
)

// The DXIL options take everything of the layout's HLSL options that
// changes the bytecode, and the DXIL shader cache keys on all of it: a vertex
// stage compiled for a fragment stage reading other locations, or with other
// special constants, dynamic offsets or pipeline constants, is another
// shader.
func TestDXILOptionsFollowTheHLSLOnes(t *testing.T) {
	index := uint32(1)
	got := hlslToDXILBindingMap(map[hlsl.ResourceBinding]hlsl.BindTarget{
		{Group: 0, Binding: 2}: {Space: 1, Register: 3, RestrictIndexing: true, DynamicStorageBufferOffsetsIndex: &index},
	})[dxil.BindingLocation{Group: 0, Binding: 2}]
	if got.Space != 1 || got.Register != 3 || !got.RestrictIndexing || got.DynamicStorageBufferOffsetsIndex == nil || *got.DynamicStorageBufferOffsetsIndex != 1 {
		t.Errorf("bind target %+v, want space 1, register 3, restricted indexing and offset index 1", got)
	}

	fragment := func(src string) *dxil.FragmentEntryPoint {
		t.Helper()
		ast, err := naga.Parse(src)
		if err != nil {
			t.Fatal(err)
		}
		m, err := naga.LowerWithSource(ast, src)
		if err != nil {
			t.Fatal(err)
		}
		return &dxil.FragmentEntryPoint{Module: m, Function: &m.EntryPoints[0].Function}
	}
	base := dxil.DefaultOptions()
	keys := map[string]string{}
	addKey := func(name, key string) {
		for other, k := range keys {
			if k == key {
				t.Errorf("%s has the cache key of %s", name, other)
			}
		}
		keys[name] = key
	}
	add := func(name string, opts dxil.Options) {
		addKey(name, dxilCacheSource("wgsl", "vs_main", nil, opts))
	}
	add("default", base)
	withFragment := base
	withFragment.FragmentEntryPoint = fragment(`@fragment fn fs(@location(1) v: f32) -> @location(0) vec4<f32> { return vec4(v); }`)
	add("fragment reading location 1", withFragment)
	withFragment.FragmentEntryPoint = fragment(`@fragment fn fs(@location(0) v: f32) -> @location(0) vec4<f32> { return vec4(v); }`)
	add("fragment reading location 0", withFragment)
	special := base
	special.SpecialConstantsBinding = &dxil.BindTarget{Register: 4}
	add("special constants", special)
	offsets := base
	offsets.DynamicStorageBufferOffsetsTargets = map[uint32]dxil.OffsetsBindTarget{0: {Register: 5, Size: 1}}
	add("dynamic offsets", offsets)
	restricted := base
	restricted.BindingMap = dxil.BindingMap{{Group: 0, Binding: 0}: {RestrictIndexing: true}}
	add("restricted indexing", restricted)
	addKey("constant 42", dxilCacheSource("wgsl", "vs_main", map[string]float64{"answer": 42}, base))
	addKey("constant 7", dxilCacheSource("wgsl", "vs_main", map[string]float64{"answer": 7}, base))
}

// A stage is compiled with its module's overrides replaced by the
// pipeline's constants, or by their defaults where it has none, as render
// stages always have.
func TestStagesTakeTheirOverrides(t *testing.T) {
	const src = `
override level: f32 = 0.5;
@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(level); }
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(level); }
`
	ast, err := naga.Parse(src)
	if err != nil {
		t.Fatal(err)
	}
	module, err := naga.LowerWithSource(ast, src)
	if err != nil {
		t.Fatal(err)
	}
	sm := &ShaderModule{irModule: module}
	for _, tc := range []struct {
		entry     string
		constants map[string]float64
		want      string
	}{
		{"vs_main", nil, "0.5"},
		{"fs_main", nil, "0.5"},
		{"fs_main", map[string]float64{"level": 0.25}, "0.25"},
	} {
		ep, _ := sm.entryPoint(tc.entry)
		processed, processedEP, err := withOverrides(module, ep, tc.constants)
		if err != nil {
			t.Fatalf("%s with %v: %v", tc.entry, tc.constants, err)
		}
		if processedEP.Name != tc.entry {
			t.Fatalf("%s with %v: entry point %q", tc.entry, tc.constants, processedEP.Name)
		}
		opts := hlsl.DefaultOptions()
		opts.EntryPoint = tc.entry
		source, _, err := hlsl.Compile(processed, opts)
		if err != nil {
			t.Fatalf("%s with %v: hlsl.Compile: %v", tc.entry, tc.constants, err)
		}
		if !strings.Contains(source, tc.want) {
			t.Errorf("%s with %v: the HLSL has no %s:\n%s", tc.entry, tc.constants, tc.want, source)
		}
	}
	ep, _ := sm.entryPoint("fs_main")
	if _, _, err := withOverrides(module, ep, map[string]float64{"missing": 1}); err == nil {
		t.Error("a constant that names no override compiled")
	}
}
