//go:build !(js && wasm)

package core

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
)

// validateBindGroupResources checks that each entry binds the kind of
// resource its layout entry declares, and that a texture or sampler fits
// it: a sampled texture made with TextureBinding, viewed with the layout's
// dimension, of a format and aspect the sample type takes, multisampled
// when the layout says so; a storage texture made with StorageBinding,
// viewed in the layout's format and dimension at one mip level; a sampler
// that compares when the layout's does and does not filter when the layout
// says non-filtering.
//
// Reference: wgpu-core device/resource.rs create_bind_group
// (WrongBindingType, create_sampler_binding, create_texture_binding)
func validateBindGroupResources(
	label string,
	layoutByBinding map[uint32]*gputypes.BindGroupLayoutEntry,
	bufferInfos []BindGroupBufferInfo,
	samplerInfos []BindGroupSamplerInfo,
	textureInfos []BindGroupTextureInfo,
	features gputypes.Features,
) error {
	kinds := make(map[uint32]string, len(layoutByBinding))
	for _, b := range bufferInfos {
		kinds[b.Binding] = "a buffer"
	}
	for _, s := range samplerInfos {
		kinds[s.Binding] = "a sampler"
	}
	for _, t := range textureInfos {
		kinds[t.Binding] = "a texture view"
	}
	for binding, le := range layoutByBinding {
		want := ""
		switch {
		case le.Buffer != nil:
			want = "a buffer"
		case le.Sampler != nil:
			want = "a sampler"
		case le.Texture != nil, le.StorageTexture != nil:
			want = "a texture view"
		default:
			continue
		}
		if got, ok := kinds[binding]; ok && got != want {
			return wrongResource(label, binding, fmt.Sprintf("%s where the layout has %s", got, want))
		}
	}
	for _, s := range samplerInfos {
		le := layoutByBinding[s.Binding]
		if le == nil || le.Sampler == nil || s.Desc == nil {
			continue
		}
		comparison := s.Desc.Compare != gputypes.CompareFunctionUndefined
		switch {
		case (le.Sampler.Type == gputypes.SamplerBindingTypeComparison) != comparison:
			return wrongResource(label, s.Binding, "the sampler and the layout disagree on comparison")
		case le.Sampler.Type == gputypes.SamplerBindingTypeNonFiltering && samplerUsesFiltering(s.Desc):
			return wrongResource(label, s.Binding, "a filtering sampler where the layout has a non-filtering one")
		}
	}
	for i := range textureInfos {
		t := &textureInfos[i]
		le := layoutByBinding[t.Binding]
		if le == nil {
			continue
		}
		if reason := textureBindingMismatch(le, t); reason != "" {
			return wrongResource(label, t.Binding, reason)
		}
		if le.Texture != nil && filterableFloat(le.Texture.SampleType) && isUnfilterable32BitFloatFormat(t.viewFormat()) {
			if err := RequireFeature(features, gputypes.FeatureFloat32Filterable, "CreateBindGroup"); err != nil {
				return err
			}
		}
	}
	return nil
}

// filterableFloat reports whether st is the filterable float sample type,
// the default.
func filterableFloat(st gputypes.TextureSampleType) bool {
	return st == gputypes.TextureSampleTypeFloat || st == gputypes.TextureSampleTypeUndefined
}

// viewFormat is the format of the view, the texture's unless it says.
func (t *BindGroupTextureInfo) viewFormat() gputypes.TextureFormat {
	if t.ViewFormat == gputypes.TextureFormatUndefined {
		return t.Format
	}
	return t.ViewFormat
}

func wrongResource(label string, binding uint32, reason string) error {
	return &CreateBindGroupError{Kind: CreateBindGroupErrorWrongResource, Label: label, Binding: binding, Reason: reason}
}

// textureBindingMismatch returns how the texture view t does not fit the
// layout entry le, or "".
func textureBindingMismatch(le *gputypes.BindGroupLayoutEntry, t *BindGroupTextureInfo) string {
	viewFormat := t.viewFormat()
	dimension := func(d gputypes.TextureViewDimension) gputypes.TextureViewDimension {
		if d == gputypes.TextureViewDimensionUndefined {
			return gputypes.TextureViewDimension2D
		}
		return d
	}
	switch {
	case le.Texture != nil:
		switch {
		case t.Usage&gputypes.TextureUsageTextureBinding == 0:
			return "the texture was not made with TextureBinding"
		case t.ViewDimension != gputypes.TextureViewDimensionUndefined && t.ViewDimension != dimension(le.Texture.ViewDimension):
			return fmt.Sprintf("a %v view where the layout has %v", t.ViewDimension, dimension(le.Texture.ViewDimension))
		case le.Texture.Multisampled != (t.SampleCount > 1):
			return "the texture and the layout disagree on multisampling"
		}
		return sampleTypeMismatch(le.Texture.SampleType, viewFormat, t.Aspect)
	case le.StorageTexture != nil:
		switch {
		case t.Usage&gputypes.TextureUsageStorageBinding == 0:
			return "the texture was not made with StorageBinding"
		case viewFormat != le.StorageTexture.Format:
			return fmt.Sprintf("a %v view where the layout has %v", viewFormat, le.StorageTexture.Format)
		case t.ViewDimension != gputypes.TextureViewDimensionUndefined && t.ViewDimension != dimension(le.StorageTexture.ViewDimension):
			return fmt.Sprintf("a %v view where the layout has %v", t.ViewDimension, dimension(le.StorageTexture.ViewDimension))
		case t.MipLevelCount > 1:
			return "a storage texture view of more than one mip level"
		}
	}
	return ""
}

// sampleTypeMismatch returns how a view of format f and aspect a can't be
// bound at a texture binding of sample type st, or "". A depth aspect binds
// as depth or unfilterable float, a stencil aspect as uint, and a combined
// depth/stencil format only one aspect at a time; a float format binds as
// filterable or unfilterable float, a 32-bit one as filterable only with
// Float32Filterable, which the caller checks.
func sampleTypeMismatch(st gputypes.TextureSampleType, f gputypes.TextureFormat, a gputypes.TextureAspect) string {
	if st == gputypes.TextureSampleTypeUndefined {
		st = gputypes.TextureSampleTypeFloat
	}
	format, ok := AspectFormat(f, a)
	switch {
	case !ok:
		return fmt.Sprintf("format %v has no aspect %v", f, a)
	case format.HasDepth() && format.HasStencil():
		return fmt.Sprintf("a view of both aspects of %v, a binding takes one", f)
	case format.HasDepth():
		if st != gputypes.TextureSampleTypeDepth && st != gputypes.TextureSampleTypeUnfilterableFloat {
			return fmt.Sprintf("depth format %v where the layout has sample type %v", format, st)
		}
		return ""
	case format.HasStencil():
		if st != gputypes.TextureSampleTypeUint {
			return fmt.Sprintf("stencil format %v where the layout has sample type %v", format, st)
		}
		return ""
	}
	var kind gputypes.TextureSampleType
	switch formatScalarKind(format) {
	case scalarUint:
		kind = gputypes.TextureSampleTypeUint
	case scalarSint:
		kind = gputypes.TextureSampleTypeSint
	default:
		kind = gputypes.TextureSampleTypeFloat
	}
	switch {
	case kind == st:
	case kind == gputypes.TextureSampleTypeFloat && st == gputypes.TextureSampleTypeUnfilterableFloat:
	default:
		return fmt.Sprintf("format %v where the layout has sample type %v", format, st)
	}
	return ""
}

type scalarKind int

const (
	scalarFloat scalarKind = iota
	scalarUint
	scalarSint
)

// formatScalarKind is the scalar kind a color format reads as.
func formatScalarKind(f gputypes.TextureFormat) scalarKind {
	switch f {
	case gputypes.TextureFormatR8Uint, gputypes.TextureFormatR16Uint, gputypes.TextureFormatRG8Uint,
		gputypes.TextureFormatR32Uint, gputypes.TextureFormatRG16Uint, gputypes.TextureFormatRGBA8Uint,
		gputypes.TextureFormatRGB10A2Uint, gputypes.TextureFormatRG32Uint, gputypes.TextureFormatRGBA16Uint,
		gputypes.TextureFormatRGBA32Uint:
		return scalarUint
	case gputypes.TextureFormatR8Sint, gputypes.TextureFormatR16Sint, gputypes.TextureFormatRG8Sint,
		gputypes.TextureFormatR32Sint, gputypes.TextureFormatRG16Sint, gputypes.TextureFormatRGBA8Sint,
		gputypes.TextureFormatRG32Sint, gputypes.TextureFormatRGBA16Sint, gputypes.TextureFormatRGBA32Sint:
		return scalarSint
	}
	return scalarFloat
}
