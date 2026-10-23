// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

// Package mslmap builds the naga MSL binding map for the Metal backend.
//
// Metal binds a resource at an index of its kind: [[buffer(N)]],
// [[texture(N)]] or [[sampler(N)]]. The encoder binds a bind group's
// resources at the indices the pipeline layout gives them, so the MSL must
// use the layout's numbering too, not one made from the globals a module
// happens to declare: a layout may hold entries a shader does not use.
//
// naga translates WGSL arrayLength() into arithmetic on an extra entry point
// argument named `_buffer_sizes`, which gets a [[buffer(N)]] attribute only
// through msl.Options.PerEntryPointMap[ep].SizesBuffer.
//
// The package has no build tags and does not call Objective-C, so its tests run
// on any CI runner.
//
// Reference: Rust wgpu-hal metal/device.rs create_pipeline_layout (the
// per-stage BindingMap and sizes_buffer) and load_shader.
package mslmap

import (
	"cmp"
	"fmt"
	"slices"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/naga/ir"
	"github.com/roffe/naga/msl"
)

// maxBufferSlot is the largest buffer slot naga can encode, because
// msl.BindTarget stores the slot in a uint8. Real Metal devices allow only 31
// buffers, so a value above this limit means the caller made a mistake.
const maxBufferSlot = 255

const (
	kindBuffer = iota
	kindTexture
	kindSampler
	kindNone
)

// Resources numbers the resources of a pipeline layout whose bind group i has
// the entries groups[i], the way the encoder binds them: group after group,
// a group's entries by ascending binding, and buffers, textures and samplers
// each counted from zero. Storage textures are textures.
func Resources(groups [][]gputypes.BindGroupLayoutEntry) (map[ir.ResourceBinding]msl.BindTarget, error) {
	resources := make(map[ir.ResourceBinding]msl.BindTarget)
	var next [kindNone]int
	for group, entries := range groups {
		sorted := slices.SortedFunc(slices.Values(entries), func(a, b gputypes.BindGroupLayoutEntry) int {
			return cmp.Compare(a.Binding, b.Binding)
		})
		for _, e := range sorted {
			kind := entryKind(e)
			if kind == kindNone {
				continue
			}
			if next[kind] > maxBufferSlot {
				return nil, fmt.Errorf("mslmap: more than %d resources of one kind", maxBufferSlot+1)
			}
			slot := uint8(next[kind])
			next[kind]++
			var target msl.BindTarget
			switch kind {
			case kindBuffer:
				target.Buffer = &slot
			case kindTexture:
				target.Texture = &slot
			case kindSampler:
				target.Sampler = &msl.BindSamplerTarget{Slot: slot}
			}
			resources[ir.ResourceBinding{Group: uint32(group), Binding: e.Binding}] = target
		}
	}
	return resources, nil
}

func entryKind(e gputypes.BindGroupLayoutEntry) int {
	switch {
	case e.Buffer != nil:
		return kindBuffer
	case e.Texture != nil, e.StorageTexture != nil:
		return kindTexture
	case e.Sampler != nil:
		return kindSampler
	}
	return kindNone
}

func globalKind(module *ir.Module, global *ir.GlobalVariable) int {
	if int(global.Type) < len(module.Types) {
		switch module.Types[global.Type].Inner.(type) {
		case ir.SamplerType:
			return kindSampler
		case ir.ImageType:
			return kindTexture
		}
	}
	return kindBuffer
}

func targetKind(t msl.BindTarget) int {
	switch {
	case t.Buffer != nil:
		return kindBuffer
	case t.Texture != nil:
		return kindTexture
	case t.Sampler != nil:
		return kindSampler
	}
	return kindNone
}

// Options returns the MSL options that compile the entry point name of stage
// in module against a pipeline layout numbered by Resources, with naga's
// `_buffer_sizes` argument at buffer slot sizesSlot. Every resource the entry
// point uses must be in the layout, as a resource of the same kind.
//
// sizesSlot is the first buffer slot after the layout's buffers, as in Rust
// wgpu-hal.
func Options(module *ir.Module, stage ir.ShaderStage, name string, resources map[ir.ResourceBinding]msl.BindTarget, sizesSlot int) (msl.Options, error) {
	opts := msl.DefaultOptions()
	if module == nil {
		return opts, fmt.Errorf("mslmap: nil IR module")
	}
	if sizesSlot < 0 || sizesSlot > maxBufferSlot {
		return opts, fmt.Errorf("mslmap: buffer sizes slot %d out of range 0..%d", sizesSlot, maxBufferSlot)
	}
	used, err := entryPointGlobals(module, stage, name)
	if err != nil {
		return opts, err
	}
	for h, use := range used {
		global := &module.GlobalVariables[h]
		if !use || global.Binding == nil {
			continue
		}
		rb := *global.Binding
		target, ok := resources[rb]
		if !ok {
			return opts, fmt.Errorf("mslmap: %s at @group(%d) @binding(%d) is not in the pipeline layout", global.Name, rb.Group, rb.Binding)
		}
		if targetKind(target) != globalKind(module, global) {
			return opts, fmt.Errorf("mslmap: the pipeline layout has another kind of resource at @group(%d) @binding(%d), where %s is", rb.Group, rb.Binding, global.Name)
		}
	}

	slot := uint8(sizesSlot)
	opts.PerEntryPointMap = map[string]msl.EntryPointResources{
		name: {Resources: resources, SizesBuffer: &slot},
	}
	return opts, nil
}

// SizesBindings returns the bindings whose byte sizes the entry point name of
// stage in module reads from its `_buffer_sizes` argument, in field order, or
// nil if it has no such argument. naga gives the argument one field for each
// global of the module that holds a runtime-sized array, and gives it only to
// an entry point that uses one of them.
func SizesBindings(module *ir.Module, stage ir.ShaderStage, name string) ([]ir.ResourceBinding, error) {
	used, err := entryPointGlobals(module, stage, name)
	if err != nil {
		return nil, err
	}
	for h, use := range used {
		if use && typeNeedsArrayLength(module, module.GlobalVariables[h].Type) {
			return RuntimeArrayBindings(module)
		}
	}
	return nil, nil
}

// entryPointGlobals reports, by global variable handle, the globals the entry
// point name of stage uses, those of the functions it calls included. That
// is how naga picks the resources an entry point takes.
func entryPointGlobals(module *ir.Module, stage ir.ShaderStage, name string) ([]bool, error) {
	if module == nil {
		return nil, fmt.Errorf("mslmap: nil IR module")
	}
	i := slices.IndexFunc(module.EntryPoints, func(ep ir.EntryPoint) bool { return ep.Name == name && ep.Stage == stage })
	if i < 0 {
		return nil, fmt.Errorf("mslmap: the shader has no entry point %q for the stage", name)
	}
	used := make([]bool, len(module.GlobalVariables))
	walked := make([]bool, len(module.Functions))
	var walk func(fn *ir.Function)
	walk = func(fn *ir.Function) {
		for _, e := range fn.Expressions {
			if g, ok := e.Kind.(ir.ExprGlobalVariable); ok && int(g.Variable) < len(used) {
				used[g.Variable] = true
			}
		}
		walkCalls(fn.Body, func(call ir.StmtCall) {
			if f := int(call.Function); f < len(walked) && !walked[f] {
				walked[f] = true
				walk(&module.Functions[f])
			}
		})
	}
	walk(&module.EntryPoints[i].Function)
	return used, nil
}

func walkCalls(block ir.Block, visit func(ir.StmtCall)) {
	for _, s := range block {
		switch s := s.Kind.(type) {
		case ir.StmtCall:
			visit(s)
		case ir.StmtBlock:
			walkCalls(s.Block, visit)
		case ir.StmtIf:
			walkCalls(s.Accept, visit)
			walkCalls(s.Reject, visit)
		case ir.StmtSwitch:
			for _, c := range s.Cases {
				walkCalls(c.Body, visit)
			}
		case ir.StmtLoop:
			walkCalls(s.Body, visit)
			walkCalls(s.Continuing, visit)
		}
	}
}

// RuntimeArrayBindings returns the bindings of the global variables whose type
// holds a runtime-sized array. The order is the IR declaration order.
//
// naga builds one uint32 field per such global, in this same order. Each field
// must hold the byte size of the buffer that is bound to it.
//
// Reference: naga v0.18.0 msl/internal/codegen/writer.go:786-797
// (scanBufferSizeGlobals).
func RuntimeArrayBindings(module *ir.Module) ([]ir.ResourceBinding, error) {
	if module == nil {
		return nil, nil
	}
	var out []ir.ResourceBinding
	for i, global := range module.GlobalVariables {
		if !typeNeedsArrayLength(module, global.Type) {
			continue
		}
		if global.Binding == nil {
			// Such a global has a field in the sizes struct, but no
			// size that the encoder can find. Skipping it would
			// shift all later fields. WGSL cannot declare this
			// today, so report an error instead of writing the
			// sizes to the wrong fields.
			return nil, fmt.Errorf("mslmap: global %d has a runtime-sized array but no binding", i)
		}
		out = append(out, *global.Binding)
	}
	return out, nil
}

// typeNeedsArrayLength reports whether the type is a runtime-sized array, or a
// struct whose last member ends in one.
//
// Reference: naga v0.18.0 msl/internal/codegen/writer.go:770-784
// (needsArrayLength).
func typeNeedsArrayLength(module *ir.Module, handle ir.TypeHandle) bool {
	if int(handle) >= len(module.Types) {
		return false
	}
	switch inner := module.Types[handle].Inner.(type) {
	case ir.ArrayType:
		return inner.Size.Constant == nil
	case ir.StructType:
		if len(inner.Members) == 0 {
			return false
		}
		return typeNeedsArrayLength(module, inner.Members[len(inner.Members)-1].Type)
	default:
		return false
	}
}

// VertexBuffers returns the vertex buffer mappings of naga's vertex pulling
// transform for a render pipeline's vertex buffers, buffer i at buffer slot
// top-i, where the encoder binds it. A buffer the pipeline doesn't read has
// no mapping. A vertex stage that pulls its vertices reads the bound size of
// each mapped buffer from `_buffer_sizes`, after the fields RuntimeArrayBindings
// lists, in mapping order.
//
// Reference: Rust wgpu-hal metal/device.rs create_render_pipeline.
func VertexBuffers(buffers []gputypes.VertexBufferLayout, top uint32) ([]msl.VertexBufferMapping, error) {
	var mappings []msl.VertexBufferMapping
	for i, buf := range buffers {
		if buf.StepMode == gputypes.VertexStepModeVertexBufferNotUsed || len(buf.Attributes) == 0 {
			continue
		}
		m := msl.VertexBufferMapping{ID: top - uint32(i), Stride: uint32(buf.ArrayStride), StepMode: msl.VertexStepModeByVertex}
		switch {
		case buf.ArrayStride == 0:
			m.StepMode = msl.VertexStepModeConstant
		case buf.StepMode == gputypes.VertexStepModeInstance:
			m.StepMode = msl.VertexStepModeByInstance
		}
		for _, a := range buf.Attributes {
			format, ok := vertexFormats[a.Format]
			if !ok {
				return nil, fmt.Errorf("mslmap: vertex format %v", a.Format)
			}
			m.Attributes = append(m.Attributes, msl.AttributeMapping{ShaderLocation: a.ShaderLocation, Offset: uint32(a.Offset), Format: format})
			if buf.ArrayStride == 0 {
				m.Stride = max(m.Stride, uint32(a.Offset+a.Format.Size()))
			}
		}
		mappings = append(mappings, m)
	}
	return mappings, nil
}

var vertexFormats = map[gputypes.VertexFormat]msl.VertexFormat{
	gputypes.VertexFormatUint8x2:      msl.VertexFormatUint8x2,
	gputypes.VertexFormatUint8x4:      msl.VertexFormatUint8x4,
	gputypes.VertexFormatSint8x2:      msl.VertexFormatSint8x2,
	gputypes.VertexFormatSint8x4:      msl.VertexFormatSint8x4,
	gputypes.VertexFormatUnorm8x2:     msl.VertexFormatUnorm8x2,
	gputypes.VertexFormatUnorm8x4:     msl.VertexFormatUnorm8x4,
	gputypes.VertexFormatSnorm8x2:     msl.VertexFormatSnorm8x2,
	gputypes.VertexFormatSnorm8x4:     msl.VertexFormatSnorm8x4,
	gputypes.VertexFormatUint16x2:     msl.VertexFormatUint16x2,
	gputypes.VertexFormatUint16x4:     msl.VertexFormatUint16x4,
	gputypes.VertexFormatSint16x2:     msl.VertexFormatSint16x2,
	gputypes.VertexFormatSint16x4:     msl.VertexFormatSint16x4,
	gputypes.VertexFormatUnorm16x2:    msl.VertexFormatUnorm16x2,
	gputypes.VertexFormatUnorm16x4:    msl.VertexFormatUnorm16x4,
	gputypes.VertexFormatSnorm16x2:    msl.VertexFormatSnorm16x2,
	gputypes.VertexFormatSnorm16x4:    msl.VertexFormatSnorm16x4,
	gputypes.VertexFormatFloat16x2:    msl.VertexFormatFloat16x2,
	gputypes.VertexFormatFloat16x4:    msl.VertexFormatFloat16x4,
	gputypes.VertexFormatFloat32:      msl.VertexFormatFloat32,
	gputypes.VertexFormatFloat32x2:    msl.VertexFormatFloat32x2,
	gputypes.VertexFormatFloat32x3:    msl.VertexFormatFloat32x3,
	gputypes.VertexFormatFloat32x4:    msl.VertexFormatFloat32x4,
	gputypes.VertexFormatUint32:       msl.VertexFormatUint32,
	gputypes.VertexFormatUint32x2:     msl.VertexFormatUint32x2,
	gputypes.VertexFormatUint32x3:     msl.VertexFormatUint32x3,
	gputypes.VertexFormatUint32x4:     msl.VertexFormatUint32x4,
	gputypes.VertexFormatSint32:       msl.VertexFormatSint32,
	gputypes.VertexFormatSint32x2:     msl.VertexFormatSint32x2,
	gputypes.VertexFormatSint32x3:     msl.VertexFormatSint32x3,
	gputypes.VertexFormatSint32x4:     msl.VertexFormatSint32x4,
	gputypes.VertexFormatUnorm1010102: msl.VertexFormatUnorm10_10_10_2,
}
