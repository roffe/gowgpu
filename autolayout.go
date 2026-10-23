//go:build !(js && wasm)

package wgpu

import (
	"cmp"
	"errors"
	"fmt"
	"slices"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/naga/ir"
)

// A pipeline created without a Layout gets WebGPU's default pipeline layout:
// a bind group layout entry for every resource its entry points use, visible
// to the stages that use it, typed by the WGSL declaration. It is derived
// like wgpu-core does it (validation.rs: check_stage with a derived layout,
// derive_binding_type), from the naga IR CreateShaderModule keeps.

// autoStage is a shader stage of a pipeline that gets an automatic layout.
type autoStage struct {
	module     *ShaderModule
	entryPoint string
	stage      ir.ShaderStage
	visibility gputypes.ShaderStages
}

// createAutoLayout derives the pipeline layout of stages and creates it. Its
// bind group layouts are marked auto: they belong to the pipeline, which
// releases them with the layout (releaseAuto).
//
// ponytail: WebGPU ties these layouts to their pipeline (exclusivePipeline).
// Here they are compatible by entries like any other, so a bind group made
// from one also works with another pipeline whose layout has the same
// entries, where WebGPU refuses it.
func (d *Device) createAutoLayout(label string, stages ...autoStage) (*PipelineLayout, error) {
	groups, err := autoLayoutEntries(stages)
	if err != nil {
		return nil, fmt.Errorf("wgpu: automatic layout of pipeline %q: %w", label, err)
	}
	bgls := make([]*BindGroupLayout, 0, len(groups))
	fail := func(err error) (*PipelineLayout, error) {
		for _, l := range bgls {
			l.release()
		}
		return nil, err
	}
	for _, entries := range groups {
		bgl, err := d.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Label: label, Entries: entries})
		if err != nil {
			return fail(err)
		}
		bgl.auto = true
		bgls = append(bgls, bgl)
	}
	layout, err := d.CreatePipelineLayout(&PipelineLayoutDescriptor{Label: label, BindGroupLayouts: bgls})
	if err != nil {
		return fail(err)
	}
	return layout, nil
}

// releaseAuto releases a layout from createAutoLayout and its bind group
// layouts. Nil is a pipeline with a layout of its own: nothing to release.
func (l *PipelineLayout) releaseAuto() {
	if l == nil {
		return
	}
	l.Release()
	for _, bgl := range l.bindGroupLayouts {
		bgl.release()
	}
}

// autoLayoutEntries returns the entries of the bind group layouts, one list
// per group up to the last group used, sorted by binding. A group below it
// that no stage uses gets no entries.
func autoLayoutEntries(stages []autoStage) ([][]gputypes.BindGroupLayoutEntry, error) {
	var groups [][]gputypes.BindGroupLayoutEntry
	for _, s := range stages {
		mod := s.module.irModule
		if mod == nil {
			return nil, errors.New("a shader without WGSL can't have one, pass a Layout")
		}
		ep := slices.IndexFunc(mod.EntryPoints, func(ep ir.EntryPoint) bool {
			return ep.Name == s.entryPoint && ep.Stage == s.stage
		})
		if ep < 0 {
			return nil, fmt.Errorf("the shader has no entry point %q for the stage", s.entryPoint)
		}
		uses := globalUses{used: make([]bool, len(mod.GlobalVariables)), sampled: make([]bool, len(mod.GlobalVariables))}
		uses.walk(mod, &mod.EntryPoints[ep].Function, nil)

		for h, used := range uses.used {
			gv := &mod.GlobalVariables[h]
			if !used || gv.Binding == nil {
				continue
			}
			rb := *gv.Binding
			if rb.Group >= MaxBindGroups {
				return nil, fmt.Errorf("%s: @group(%d), there are only %d", gv.Name, rb.Group, MaxBindGroups)
			}
			entry, err := autoEntry(mod, gv, uses.sampled[h])
			if err != nil {
				return nil, fmt.Errorf("%s: %w", gv.Name, err)
			}
			entry.Visibility = s.visibility
			for len(groups) <= int(rb.Group) {
				groups = append(groups, nil)
			}
			group := groups[rb.Group]
			i := slices.IndexFunc(group, func(e gputypes.BindGroupLayoutEntry) bool { return e.Binding == rb.Binding })
			if i < 0 {
				groups[rb.Group] = append(group, entry)
			} else if !mergeAutoEntry(&group[i], entry) {
				return nil, fmt.Errorf("@group(%d) @binding(%d) is declared twice, differently", rb.Group, rb.Binding)
			}
		}
	}
	for _, group := range groups {
		slices.SortFunc(group, func(a, b gputypes.BindGroupLayoutEntry) int { return cmp.Compare(a.Binding, b.Binding) })
	}
	return groups, nil
}

// mergeAutoEntry folds e, a binding prev already has from another stage,
// into prev the way WebGPU merges them: the visibility is both stages', a
// buffer the larger minimum size, a float texture filterable if one stage
// samples it. Any other difference is a conflict, and false.
//
// ponytail: WebGPU also makes a write-only storage texture read-write when
// another stage reads and writes it; that is a conflict here.
func mergeAutoEntry(prev *gputypes.BindGroupLayoutEntry, e gputypes.BindGroupLayoutEntry) bool {
	if prev.Buffer != nil && e.Buffer != nil {
		prev.Buffer.MinBindingSize = max(prev.Buffer.MinBindingSize, e.Buffer.MinBindingSize)
		e.Buffer.MinBindingSize = prev.Buffer.MinBindingSize
	}
	if prev.Texture != nil && e.Texture != nil && isFloatSampleType(prev.Texture.SampleType) && isFloatSampleType(e.Texture.SampleType) {
		if e.Texture.SampleType == gputypes.TextureSampleTypeFloat {
			prev.Texture.SampleType = gputypes.TextureSampleTypeFloat
		}
		e.Texture.SampleType = prev.Texture.SampleType
	}
	prev.Visibility |= e.Visibility
	e.Visibility = prev.Visibility
	return bindGroupLayoutEntriesEqual(prev, &e)
}

func isFloatSampleType(t gputypes.TextureSampleType) bool {
	return t == gputypes.TextureSampleTypeFloat || t == gputypes.TextureSampleTypeUnfilterableFloat
}

// autoEntry is the layout entry for gv, as wgpu-core's derive_binding_type
// makes it.
func autoEntry(mod *ir.Module, gv *ir.GlobalVariable, sampled bool) (gputypes.BindGroupLayoutEntry, error) {
	e := gputypes.BindGroupLayoutEntry{Binding: gv.Binding.Binding}
	switch t := mod.Types[gv.Type].Inner.(type) {
	case ir.SamplerType:
		e.Sampler = &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}
		if t.Comparison {
			e.Sampler.Type = gputypes.SamplerBindingTypeComparison
		}
	case ir.ImageType:
		if err := autoTextureEntry(&e, t, sampled); err != nil {
			return e, err
		}
	case ir.AccelerationStructureType:
		e.AccelerationStructure = &gputypes.AccelerationStructureBindingLayout{}
	case ir.BindingArrayType:
		return e, errors.New("binding arrays need a Layout")
	default:
		e.Buffer = &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform, MinBindingSize: uint64(ir.TypeSize(mod, gv.Type))}
		switch {
		case gv.Space == ir.SpaceStorage && gv.Access == ir.StorageRead:
			e.Buffer.Type = gputypes.BufferBindingTypeReadOnlyStorage
		case gv.Space == ir.SpaceStorage:
			e.Buffer.Type = gputypes.BufferBindingTypeStorage
		case gv.Space != ir.SpaceUniform:
			return e, fmt.Errorf("a buffer in address space %d", gv.Space)
		}
	}
	return e, nil
}

// autoTextureEntry fills in e for a texture of type t. A f32 texture is
// filterable when the stage samples it with a sampler, unfilterable-float
// when it only loads or queries it.
func autoTextureEntry(e *gputypes.BindGroupLayoutEntry, t ir.ImageType, sampled bool) error {
	dim := autoViewDimension(t)
	switch t.Class {
	case ir.ImageClassSampled:
		e.Texture = &gputypes.TextureBindingLayout{ViewDimension: dim, Multisampled: t.Multisampled}
		switch t.SampledKind {
		case ir.ScalarFloat:
			e.Texture.SampleType = gputypes.TextureSampleTypeUnfilterableFloat
			if sampled {
				e.Texture.SampleType = gputypes.TextureSampleTypeFloat
			}
		case ir.ScalarSint:
			e.Texture.SampleType = gputypes.TextureSampleTypeSint
		case ir.ScalarUint:
			e.Texture.SampleType = gputypes.TextureSampleTypeUint
		default:
			return fmt.Errorf("a texture of scalar kind %d", t.SampledKind)
		}
	case ir.ImageClassDepth:
		e.Texture = &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeDepth, ViewDimension: dim, Multisampled: t.Multisampled}
	case ir.ImageClassStorage:
		var format gputypes.TextureFormat
		if int(t.StorageFormat) < len(storageTextureFormats) {
			format = storageTextureFormats[t.StorageFormat]
		}
		if format == gputypes.TextureFormatUndefined {
			return fmt.Errorf("storage texture format %d has no texture format", t.StorageFormat)
		}
		access := gputypes.StorageTextureAccessReadWrite
		switch t.StorageAccess {
		case ir.StorageAccessRead:
			access = gputypes.StorageTextureAccessReadOnly
		case ir.StorageAccessWrite:
			access = gputypes.StorageTextureAccessWriteOnly
		}
		e.StorageTexture = &gputypes.StorageTextureBindingLayout{Access: access, Format: format, ViewDimension: dim}
	default:
		return errors.New("external textures need a Layout")
	}
	return nil
}

func autoViewDimension(t ir.ImageType) gputypes.TextureViewDimension {
	switch {
	case t.Dim == ir.Dim1D:
		return gputypes.TextureViewDimension1D
	case t.Dim == ir.Dim3D:
		return gputypes.TextureViewDimension3D
	case t.Dim == ir.DimCube && t.Arrayed:
		return gputypes.TextureViewDimensionCubeArray
	case t.Dim == ir.DimCube:
		return gputypes.TextureViewDimensionCube
	case t.Arrayed:
		return gputypes.TextureViewDimension2DArray
	}
	return gputypes.TextureViewDimension2D
}

// storageTextureFormats maps naga's storage texture formats to texture
// formats. The zero value, Undefined, is a format WebGPU has no texture for.
var storageTextureFormats = [...]gputypes.TextureFormat{
	ir.StorageFormatR8Unorm:       gputypes.TextureFormatR8Unorm,
	ir.StorageFormatR8Snorm:       gputypes.TextureFormatR8Snorm,
	ir.StorageFormatR8Uint:        gputypes.TextureFormatR8Uint,
	ir.StorageFormatR8Sint:        gputypes.TextureFormatR8Sint,
	ir.StorageFormatR16Uint:       gputypes.TextureFormatR16Uint,
	ir.StorageFormatR16Sint:       gputypes.TextureFormatR16Sint,
	ir.StorageFormatR16Float:      gputypes.TextureFormatR16Float,
	ir.StorageFormatRg8Unorm:      gputypes.TextureFormatRG8Unorm,
	ir.StorageFormatRg8Snorm:      gputypes.TextureFormatRG8Snorm,
	ir.StorageFormatRg8Uint:       gputypes.TextureFormatRG8Uint,
	ir.StorageFormatRg8Sint:       gputypes.TextureFormatRG8Sint,
	ir.StorageFormatR32Uint:       gputypes.TextureFormatR32Uint,
	ir.StorageFormatR32Sint:       gputypes.TextureFormatR32Sint,
	ir.StorageFormatR32Float:      gputypes.TextureFormatR32Float,
	ir.StorageFormatRg16Uint:      gputypes.TextureFormatRG16Uint,
	ir.StorageFormatRg16Sint:      gputypes.TextureFormatRG16Sint,
	ir.StorageFormatRg16Float:     gputypes.TextureFormatRG16Float,
	ir.StorageFormatRgba8Unorm:    gputypes.TextureFormatRGBA8Unorm,
	ir.StorageFormatRgba8Snorm:    gputypes.TextureFormatRGBA8Snorm,
	ir.StorageFormatRgba8Uint:     gputypes.TextureFormatRGBA8Uint,
	ir.StorageFormatRgba8Sint:     gputypes.TextureFormatRGBA8Sint,
	ir.StorageFormatBgra8Unorm:    gputypes.TextureFormatBGRA8Unorm,
	ir.StorageFormatRgb10a2Uint:   gputypes.TextureFormatRGB10A2Uint,
	ir.StorageFormatRgb10a2Unorm:  gputypes.TextureFormatRGB10A2Unorm,
	ir.StorageFormatRg11b10Ufloat: gputypes.TextureFormatRG11B10Ufloat,
	ir.StorageFormatRg32Uint:      gputypes.TextureFormatRG32Uint,
	ir.StorageFormatRg32Sint:      gputypes.TextureFormatRG32Sint,
	ir.StorageFormatRg32Float:     gputypes.TextureFormatRG32Float,
	ir.StorageFormatRgba16Uint:    gputypes.TextureFormatRGBA16Uint,
	ir.StorageFormatRgba16Sint:    gputypes.TextureFormatRGBA16Sint,
	ir.StorageFormatRgba16Float:   gputypes.TextureFormatRGBA16Float,
	ir.StorageFormatRgba32Uint:    gputypes.TextureFormatRGBA32Uint,
	ir.StorageFormatRgba32Sint:    gputypes.TextureFormatRGBA32Sint,
	ir.StorageFormatRgba32Float:   gputypes.TextureFormatRGBA32Float,
	ir.StorageFormatR16Unorm:      gputypes.TextureFormatR16Unorm,
	ir.StorageFormatR16Snorm:      gputypes.TextureFormatR16Snorm,
	ir.StorageFormatRg16Unorm:     gputypes.TextureFormatRG16Unorm,
	ir.StorageFormatRg16Snorm:     gputypes.TextureFormatRG16Snorm,
	ir.StorageFormatRgba16Unorm:   gputypes.TextureFormatRGBA16Unorm,
	ir.StorageFormatRgba16Snorm:   gputypes.TextureFormatRGBA16Snorm,
}

// globalUses is, by global variable handle, which globals an entry point
// uses, and which textures it samples with a sampler, the functions it calls
// included. A global counts as used when a function refers to it, which is
// how naga's GLSL and MSL backends pick an entry point's resources.
type globalUses struct {
	used, sampled []bool
	// pairs are the textures sampled with each sampler, as globals.
	pairs [][2]int
}

// walk adds fn's uses. args holds the global the caller passed for each of
// fn's arguments, or -1, so a texture passed to a function that samples it
// counts as sampled.
func (u *globalUses) walk(mod *ir.Module, fn *ir.Function, args []int) {
	global := func(h ir.ExpressionHandle) int {
		switch e := fn.Expressions[h].Kind.(type) {
		case ir.ExprGlobalVariable:
			return int(e.Variable)
		case ir.ExprFunctionArgument:
			if int(e.Index) < len(args) {
				return args[e.Index]
			}
		}
		return -1
	}
	for _, e := range fn.Expressions {
		switch e := e.Kind.(type) {
		case ir.ExprGlobalVariable:
			u.used[e.Variable] = true
		case ir.ExprImageSample:
			if g := global(e.Image); g >= 0 {
				u.sampled[g] = true
				if s := global(e.Sampler); s >= 0 {
					u.pairs = append(u.pairs, [2]int{g, s})
				}
			}
		}
	}
	walkCalls(fn.Body, func(call ir.StmtCall) {
		callArgs := make([]int, len(call.Arguments))
		for i, a := range call.Arguments {
			callArgs[i] = global(a)
		}
		u.walk(mod, &mod.Functions[call.Function], callArgs)
	})
}

// walkCalls calls visit for every function call in block.
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
