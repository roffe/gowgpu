//go:build !(js && wasm)

package wgpu

import (
	"fmt"
	"slices"

	"github.com/gogpu/naga/ir"
	"github.com/roffe/gowgpu/gputypes"
)

// A pipeline's shaders have to fit its layout, its vertex buffers and its
// targets: every resource an entry point uses is in the layout, visible to
// the stage and of the kind the shader declares, every vertex input has an
// attribute of its scalar kind, and every color target the fragment shader
// writes has a format of the kind it writes. They are checked like
// wgpu-core's check_stage does, from the naga IR CreateShaderModule keeps; a
// module without one (SPIR-V) is not checked.

// checkStageBindings checks the resources the entry point of a stage uses
// against layouts, the bind group layouts of the pipeline layout.
//
// Reference: wgpu-core validation.rs Interface::check_stage and
// Resource::check_binding_use
func checkStageBindings(module *ShaderModule, entryPoint string, stage ir.ShaderStage, visibility gputypes.ShaderStages, layouts []*BindGroupLayout) error {
	mod, ep := entryPointOf(module, entryPoint, stage)
	if ep == nil {
		return nil
	}
	uses := globalUses{used: make([]bool, len(mod.GlobalVariables)), sampled: make([]bool, len(mod.GlobalVariables))}
	uses.walk(mod, &ep.Function, nil)
	// entryOf returns the layout entry of a global and whether its layout is
	// one derived from the shaders.
	entryOf := func(gv *ir.GlobalVariable) (*gputypes.BindGroupLayoutEntry, bool) {
		rb := *gv.Binding
		if int(rb.Group) >= len(layouts) || layouts[rb.Group] == nil {
			return nil, false
		}
		entries := layouts[rb.Group].entries
		if i := slices.IndexFunc(entries, func(e gputypes.BindGroupLayoutEntry) bool { return e.Binding == rb.Binding }); i >= 0 {
			return &entries[i], layouts[rb.Group].auto
		}
		return nil, false
	}
	name := func(gv *ir.GlobalVariable) string {
		return fmt.Sprintf("%s (@group(%d) @binding(%d))", gv.Name, gv.Binding.Group, gv.Binding.Binding)
	}
	for h, used := range uses.used {
		gv := &mod.GlobalVariables[h]
		if !used || gv.Binding == nil {
			continue
		}
		entry, _ := entryOf(gv)
		where := name(gv)
		switch {
		case entry == nil:
			return fmt.Errorf("%s is not in the pipeline layout", where)
		case entry.Visibility&visibility == 0:
			return fmt.Errorf("%s is not visible to the stage in the pipeline layout", where)
		}
		shader, err := autoEntry(mod, gv, uses.sampled[h])
		if err != nil {
			continue // binding arrays and external textures: nothing to compare
		}
		if reason := bindingMismatch(entry, &shader); reason != "" {
			return fmt.Errorf("%s: %s", where, reason)
		}
	}
	// A filtering sampler can't sample a texture whose layout entry says it
	// doesn't filter. Layouts derived from the shaders agree by making.
	//
	// Reference: wgpu-core validation.rs check_stage (sampling_pairs,
	// FilteringError)
	for _, pair := range uses.pairs {
		tex, smp := &mod.GlobalVariables[pair[0]], &mod.GlobalVariables[pair[1]]
		if tex.Binding == nil || smp.Binding == nil {
			continue
		}
		texEntry, auto := entryOf(tex)
		smpEntry, _ := entryOf(smp)
		if auto || texEntry == nil || smpEntry == nil || texEntry.Texture == nil || smpEntry.Sampler == nil {
			continue
		}
		if t := smpEntry.Sampler.Type; t != gputypes.SamplerBindingTypeFiltering && t != gputypes.SamplerBindingTypeUndefined {
			continue
		}
		switch st := texEntry.Texture.SampleType; st {
		case gputypes.TextureSampleTypeUnfilterableFloat, gputypes.TextureSampleTypeSint, gputypes.TextureSampleTypeUint:
			return fmt.Errorf("%s is sampled with the filtering sampler %s, though its sample type %v doesn't filter", name(tex), name(smp), st)
		}
	}
	return nil
}

// bindingMismatch returns how the layout entry for a resource differs from
// what the shader declares, or "".
func bindingMismatch(layout, shader *gputypes.BindGroupLayoutEntry) string {
	switch {
	case shader.Buffer != nil:
		if layout.Buffer == nil {
			return "the shader has a buffer, the layout no buffer"
		}
		if layout.Buffer.Type != shader.Buffer.Type {
			return fmt.Sprintf("the shader has a %v buffer, the layout a %v one", shader.Buffer.Type, layout.Buffer.Type)
		}
	case shader.Sampler != nil:
		if layout.Sampler == nil {
			return "the shader has a sampler, the layout no sampler"
		}
		if (layout.Sampler.Type == gputypes.SamplerBindingTypeComparison) != (shader.Sampler.Type == gputypes.SamplerBindingTypeComparison) {
			return "the shader and the layout disagree on a comparison sampler"
		}
	case shader.Texture != nil:
		if layout.Texture == nil {
			return "the shader has a texture, the layout no texture"
		}
		l, s := layout.Texture, shader.Texture
		viewDim := l.ViewDimension
		if viewDim == gputypes.TextureViewDimensionUndefined {
			viewDim = gputypes.TextureViewDimension2D
		}
		switch {
		case viewDim != s.ViewDimension:
			return fmt.Sprintf("the shader has a %v texture, the layout a %v one", s.ViewDimension, viewDim)
		case l.Multisampled != s.Multisampled:
			return "the shader and the layout disagree on a multisampled texture"
		case sampleClass(l.SampleType) != sampleClass(s.SampleType):
			return fmt.Sprintf("the shader samples %v, the layout has sample type %v", s.SampleType, l.SampleType)
		}
	case shader.StorageTexture != nil:
		if layout.StorageTexture == nil {
			return "the shader has a storage texture, the layout no storage texture"
		}
		l, s := layout.StorageTexture, shader.StorageTexture
		viewDim := l.ViewDimension
		if viewDim == gputypes.TextureViewDimensionUndefined {
			viewDim = gputypes.TextureViewDimension2D
		}
		switch {
		case l.Format != s.Format:
			return fmt.Sprintf("the shader has storage format %v, the layout %v", s.Format, l.Format)
		case viewDim != s.ViewDimension:
			return fmt.Sprintf("the shader has a %v storage texture, the layout a %v one", s.ViewDimension, viewDim)
		case l.Access != s.Access &&
			(s.Access != gputypes.StorageTextureAccessWriteOnly || l.Access != gputypes.StorageTextureAccessReadWrite):
			return fmt.Sprintf("the shader has storage access %v, the layout %v", s.Access, l.Access)
		}
	case shader.AccelerationStructure != nil:
		if layout.AccelerationStructure == nil {
			return "the shader has an acceleration structure, the layout none"
		}
	}
	return ""
}

// sampleClass folds the float sample types into one: a texture_2d<f32>
// takes a filterable or an unfilterable float binding.
func sampleClass(t gputypes.TextureSampleType) gputypes.TextureSampleType {
	if t == gputypes.TextureSampleTypeUnfilterableFloat || t == gputypes.TextureSampleTypeUndefined {
		return gputypes.TextureSampleTypeFloat
	}
	return t
}

// checkVertexInputs checks that every input of the vertex entry point has
// an attribute in buffers of the same scalar kind.
//
// Reference: wgpu-core validation.rs check_stage (InputError::Missing,
// InputError::WrongType)
func checkVertexInputs(module *ShaderModule, entryPoint string, buffers []gputypes.VertexBufferLayout) error {
	mod, ep := entryPointOf(module, entryPoint, ir.StageVertex)
	if ep == nil {
		return nil
	}
	for _, in := range locations(mod, ep.Function.Arguments) {
		var attr *gputypes.VertexAttribute
		for i := range buffers {
			for j := range buffers[i].Attributes {
				if buffers[i].Attributes[j].ShaderLocation == in.location {
					attr = &buffers[i].Attributes[j]
				}
			}
		}
		if attr == nil {
			return fmt.Errorf("vertex input @location(%d) has no vertex attribute", in.location)
		}
		if kind := vertexFormatKind(attr.Format); kind != in.kind {
			return fmt.Errorf("vertex input @location(%d) is %s, its attribute's format %v %s",
				in.location, scalarKindName(in.kind), attr.Format, scalarKindName(kind))
		}
	}
	return nil
}

// checkFragmentOutputs checks that every color target the fragment entry
// point writes has a format of the scalar kind it writes.
func checkFragmentOutputs(module *ShaderModule, entryPoint string, targets []gputypes.ColorTargetState) error {
	mod, ep := entryPointOf(module, entryPoint, ir.StageFragment)
	if ep == nil || ep.Function.Result == nil {
		return nil
	}
	result := []ir.FunctionArgument{{Type: ep.Function.Result.Type, Binding: ep.Function.Result.Binding}}
	for _, out := range locations(mod, result) {
		if int(out.location) >= len(targets) || targets[out.location].Format == gputypes.TextureFormatUndefined {
			continue
		}
		f := targets[out.location].Format
		if kind := textureFormatKind(f); kind != out.kind {
			return fmt.Errorf("fragment output @location(%d) is %s, color target %d's format %v %s",
				out.location, scalarKindName(out.kind), out.location, f, scalarKindName(kind))
		}
	}
	return nil
}

// entryPointOf returns the IR of module and its entry point name for stage,
// or nil without an IR or such an entry point.
func entryPointOf(module *ShaderModule, name string, stage ir.ShaderStage) (*ir.Module, *ir.EntryPoint) {
	if module == nil || module.irModule == nil {
		return nil, nil
	}
	mod := module.irModule
	i := slices.IndexFunc(mod.EntryPoints, func(ep ir.EntryPoint) bool { return ep.Name == name && ep.Stage == stage })
	if i < 0 {
		return nil, nil
	}
	return mod, &mod.EntryPoints[i]
}

// location is a user-defined shader input or output: its location and the
// scalar kind of its type.
type location struct {
	location uint32
	kind     ir.ScalarKind
}

// locations returns the @location bindings of args, those of the members of
// a struct argument included.
func locations(mod *ir.Module, args []ir.FunctionArgument) []location {
	var out []location
	add := func(b *ir.Binding, t ir.TypeHandle) {
		if b == nil {
			if st, ok := mod.Types[t].Inner.(ir.StructType); ok {
				for _, m := range st.Members {
					if m.Binding != nil {
						if loc, ok := (*m.Binding).(ir.LocationBinding); ok {
							out = append(out, location{loc.Location, scalarKind(mod, m.Type)})
						}
					}
				}
			}
			return
		}
		if loc, ok := (*b).(ir.LocationBinding); ok {
			out = append(out, location{loc.Location, scalarKind(mod, t)})
		}
	}
	for _, a := range args {
		add(a.Binding, a.Type)
	}
	return out
}

// scalarKind is the scalar kind of a scalar or vector type.
func scalarKind(mod *ir.Module, t ir.TypeHandle) ir.ScalarKind {
	switch inner := mod.Types[t].Inner.(type) {
	case ir.ScalarType:
		return inner.Kind
	case ir.VectorType:
		return inner.Scalar.Kind
	}
	return ir.ScalarFloat
}

// vertexFormatKind is the scalar kind a vertex format reads as.
func vertexFormatKind(f gputypes.VertexFormat) ir.ScalarKind {
	switch f {
	case gputypes.VertexFormatUint8x2, gputypes.VertexFormatUint8x4, gputypes.VertexFormatUint16x2,
		gputypes.VertexFormatUint16x4, gputypes.VertexFormatUint32, gputypes.VertexFormatUint32x2,
		gputypes.VertexFormatUint32x3, gputypes.VertexFormatUint32x4:
		return ir.ScalarUint
	case gputypes.VertexFormatSint8x2, gputypes.VertexFormatSint8x4, gputypes.VertexFormatSint16x2,
		gputypes.VertexFormatSint16x4, gputypes.VertexFormatSint32, gputypes.VertexFormatSint32x2,
		gputypes.VertexFormatSint32x3, gputypes.VertexFormatSint32x4:
		return ir.ScalarSint
	}
	return ir.ScalarFloat
}

// textureFormatKind is the scalar kind a color format is written as.
func textureFormatKind(f gputypes.TextureFormat) ir.ScalarKind {
	switch f {
	case gputypes.TextureFormatR8Uint, gputypes.TextureFormatR16Uint, gputypes.TextureFormatRG8Uint,
		gputypes.TextureFormatR32Uint, gputypes.TextureFormatRG16Uint, gputypes.TextureFormatRGBA8Uint,
		gputypes.TextureFormatRGB10A2Uint, gputypes.TextureFormatRG32Uint, gputypes.TextureFormatRGBA16Uint,
		gputypes.TextureFormatRGBA32Uint:
		return ir.ScalarUint
	case gputypes.TextureFormatR8Sint, gputypes.TextureFormatR16Sint, gputypes.TextureFormatRG8Sint,
		gputypes.TextureFormatR32Sint, gputypes.TextureFormatRG16Sint, gputypes.TextureFormatRGBA8Sint,
		gputypes.TextureFormatRG32Sint, gputypes.TextureFormatRGBA16Sint, gputypes.TextureFormatRGBA32Sint:
		return ir.ScalarSint
	}
	return ir.ScalarFloat
}

func scalarKindName(k ir.ScalarKind) string {
	switch k {
	case ir.ScalarSint:
		return "sint"
	case ir.ScalarUint:
		return "uint"
	case ir.ScalarBool:
		return "bool"
	}
	return "float"
}
