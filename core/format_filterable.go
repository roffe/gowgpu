//go:build !(js && wasm)

package core

import (
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// isUnfilterable32BitFloatFormat reports 32-bit float formats that require
// FeatureFloat32Filterable for linear/mipmap filtering (WebGPU spec).
func isUnfilterable32BitFloatFormat(format gputypes.TextureFormat) bool {
	switch format {
	case gputypes.TextureFormatR32Float,
		gputypes.TextureFormatRG32Float,
		gputypes.TextureFormatRGBA32Float:
		return true
	default:
		return false
	}
}

// samplerUsesFiltering reports whether a sampler applies anything other than
// nearest-neighbor filtering (mag, min, or mipmap).
func samplerUsesFiltering(desc *hal.SamplerDescriptor) bool {
	if desc == nil {
		return false
	}
	return desc.MagFilter == gputypes.FilterModeLinear ||
		desc.MinFilter == gputypes.FilterModeLinear ||
		desc.MipmapFilter == gputypes.FilterModeLinear
}

// BindGroupSamplerInfo carries sampler metadata for bind-group validation.
type BindGroupSamplerInfo struct {
	Binding uint32
	Desc    *hal.SamplerDescriptor
}

// BindGroupTextureInfo carries texture metadata for bind-group validation:
// the texture's format, usage and sample count, and the view's resolved
// format, dimension, aspect and mip count.
type BindGroupTextureInfo struct {
	Binding       uint32
	Format        gputypes.TextureFormat
	Usage         gputypes.TextureUsage
	SampleCount   uint32
	ViewFormat    gputypes.TextureFormat
	ViewDimension gputypes.TextureViewDimension
	Aspect        gputypes.TextureAspect
	MipLevelCount uint32
}
