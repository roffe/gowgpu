// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/naga"
	"github.com/roffe/naga/dxil"
	"github.com/roffe/naga/hlsl"
)

// The DXIL options take everything of the layout's HLSL options that
// changes the bytecode, and the DXIL shader cache keys on all of it: a vertex
// stage compiled for a fragment stage reading other locations, or with other
// special constants or dynamic offsets, is another shader.
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
	add := func(name string, opts dxil.Options) {
		key := dxilCacheSource("wgsl", "vs_main", opts)
		for other, k := range keys {
			if k == key {
				t.Errorf("%s has the cache key of %s", name, other)
			}
		}
		keys[name] = key
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
}
