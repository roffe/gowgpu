//go:build !(js && wasm)

package core

import "github.com/roffe/gowgpu/gputypes"

// FormatFeatures is what WebGPU guarantees textures of a format can do.
type FormatFeatures struct {
	// Usages are the usages a texture of the format may be made with.
	Usages gputypes.TextureUsage
	// Multisample is set when a texture of the format may have 4 samples,
	// Resolve when a multisampled color attachment of it may be resolved.
	Multisample, Resolve bool
	// Blendable is set when a render target of the format may blend.
	Blendable bool
	// StorageReadWrite is set when a storage texture of the format may be
	// read-write, besides read-only and write-only.
	StorageReadWrite bool
}

// GuaranteedFormatFeatures returns what WebGPU guarantees textures of
// format f can do on a device with features.
//
// Reference: wgpu-types TextureFormat::guaranteed_format_features
func GuaranteedFormatFeatures(f gputypes.TextureFormat, features gputypes.Features) FormatFeatures {
	const (
		basic      = gputypes.TextureUsageCopySrc | gputypes.TextureUsageCopyDst | gputypes.TextureUsageTextureBinding
		attachment = basic | gputypes.TextureUsageRenderAttachment
		storage    = basic | gputypes.TextureUsageStorageBinding
		all        = attachment | storage
	)
	var ff FormatFeatures
	switch f {
	case gputypes.TextureFormatR8Unorm, gputypes.TextureFormatR16Float, gputypes.TextureFormatRG8Unorm,
		gputypes.TextureFormatRG16Float, gputypes.TextureFormatRGBA8UnormSrgb, gputypes.TextureFormatBGRA8UnormSrgb,
		gputypes.TextureFormatRGB10A2Unorm:
		ff = FormatFeatures{Usages: attachment, Multisample: true, Resolve: true}
	case gputypes.TextureFormatR8Uint, gputypes.TextureFormatR8Sint, gputypes.TextureFormatR16Uint,
		gputypes.TextureFormatR16Sint, gputypes.TextureFormatRG8Uint, gputypes.TextureFormatRG8Sint,
		gputypes.TextureFormatRG16Uint, gputypes.TextureFormatRG16Sint, gputypes.TextureFormatRGB10A2Uint,
		gputypes.TextureFormatStencil8, gputypes.TextureFormatDepth16Unorm, gputypes.TextureFormatDepth24Plus,
		gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureFormatDepth32Float,
		gputypes.TextureFormatDepth32FloatStencil8:
		ff = FormatFeatures{Usages: attachment, Multisample: true}
	case gputypes.TextureFormatR32Uint, gputypes.TextureFormatR32Sint:
		ff = FormatFeatures{Usages: all, StorageReadWrite: true}
	case gputypes.TextureFormatR32Float:
		ff = FormatFeatures{Usages: all, Multisample: true, StorageReadWrite: true}
	case gputypes.TextureFormatRGBA8Unorm, gputypes.TextureFormatRGBA16Float:
		ff = FormatFeatures{Usages: all, Multisample: true, Resolve: true}
	case gputypes.TextureFormatRGBA8Uint, gputypes.TextureFormatRGBA8Sint, gputypes.TextureFormatRGBA16Uint,
		gputypes.TextureFormatRGBA16Sint:
		ff = FormatFeatures{Usages: all, Multisample: true}
	case gputypes.TextureFormatRG32Uint, gputypes.TextureFormatRG32Sint, gputypes.TextureFormatRG32Float,
		gputypes.TextureFormatRGBA32Uint, gputypes.TextureFormatRGBA32Sint, gputypes.TextureFormatRGBA32Float:
		ff = FormatFeatures{Usages: all}
	case gputypes.TextureFormatRGBA8Snorm:
		ff = FormatFeatures{Usages: storage}
	case gputypes.TextureFormatR16Unorm, gputypes.TextureFormatR16Snorm, gputypes.TextureFormatRG16Unorm,
		gputypes.TextureFormatRG16Snorm, gputypes.TextureFormatRGBA16Unorm, gputypes.TextureFormatRGBA16Snorm:
		ff = FormatFeatures{Usages: storage, Multisample: true}
	case gputypes.TextureFormatBGRA8Unorm:
		ff = FormatFeatures{Usages: attachment, Multisample: true, Resolve: true}
		if features.Contains(gputypes.FeatureBGRA8UnormStorage) {
			ff.Usages |= gputypes.TextureUsageStorageBinding
		}
	case gputypes.TextureFormatRG11B10Ufloat:
		ff = FormatFeatures{Usages: basic, Multisample: true}
		if features.Contains(gputypes.FeatureRG11B10UfloatRenderable) {
			ff.Usages, ff.Resolve = attachment, true
		}
	case gputypes.TextureFormatUndefined:
	default: // snorm 8-bit, RGB9E5 and the compressed formats
		ff = FormatFeatures{Usages: basic}
	}
	ff.Blendable = ff.Usages&gputypes.TextureUsageRenderAttachment != 0 && blendable(f)
	return ff
}

// blendable reports whether f is a filterable float format, the color
// formats a render target blends in.
func blendable(f gputypes.TextureFormat) bool {
	switch f {
	case gputypes.TextureFormatR8Unorm, gputypes.TextureFormatRG8Unorm, gputypes.TextureFormatRGBA8Unorm,
		gputypes.TextureFormatRGBA8UnormSrgb, gputypes.TextureFormatBGRA8Unorm, gputypes.TextureFormatBGRA8UnormSrgb,
		gputypes.TextureFormatR16Float, gputypes.TextureFormatRG16Float, gputypes.TextureFormatRGBA16Float,
		gputypes.TextureFormatRGB10A2Unorm, gputypes.TextureFormatRG11B10Ufloat:
		return true
	}
	return false
}
