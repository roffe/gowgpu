//go:build !(js && wasm)

package core

import (
	"errors"
	"fmt"
	"slices"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// ErrInvalidTextureView is what ResolveTextureViewDescriptor wraps for a
// descriptor WebGPU rejects.
var ErrInvalidTextureView = errors.New("invalid texture view")

// AspectFormat returns the format of one aspect of a texture format, as
// wgpu-types' aspect_specific_format does: Depth24Plus and Stencil8 for the
// aspects of Depth24PlusStencil8, Depth32Float and Stencil8 for those of
// Depth32FloatStencil8, and the format itself for TextureAspectAll and for
// the one aspect a depth or stencil format has. ok is false for an aspect the
// format does not have.
func AspectFormat(f gputypes.TextureFormat, aspect gputypes.TextureAspect) (gputypes.TextureFormat, bool) {
	switch aspect {
	case gputypes.TextureAspectAll, gputypes.TextureAspectUndefined:
		return f, true
	case gputypes.TextureAspectDepthOnly:
		switch f {
		case gputypes.TextureFormatDepth24PlusStencil8:
			return gputypes.TextureFormatDepth24Plus, true
		case gputypes.TextureFormatDepth32FloatStencil8:
			return gputypes.TextureFormatDepth32Float, true
		}
		return f, f.HasDepth()
	case gputypes.TextureAspectStencilOnly:
		if f.HasStencil() {
			return gputypes.TextureFormatStencil8, true
		}
	}
	return gputypes.TextureFormatUndefined, false
}

// ValidateViewFormats checks the view formats of a texture of format f:
// each is f or f's sRGB or linear twin, the formats WebGPU lets a view
// reinterpret a texture as.
func ValidateViewFormats(f gputypes.TextureFormat, viewFormats []gputypes.TextureFormat) error {
	for _, vf := range viewFormats {
		if vf != f && !srgbTwins(f, vf) {
			return fmt.Errorf("%w: view format %v of a %v texture differs in more than being sRGB", ErrInvalidTextureView, vf, f)
		}
	}
	return nil
}

// srgbTwins reports whether a and b differ only in being sRGB. Every sRGB
// format follows its linear twin in the enum.
func srgbTwins(a, b gputypes.TextureFormat) bool {
	return b.IsSrgb() && a == b-1 || a.IsSrgb() && b == a-1
}

// arrayLayerCount is the number of array layers of a texture: a 3D texture
// has one, its depth is slices of it.
func arrayLayerCount(dim gputypes.TextureDimension, size gputypes.Extent3D) uint32 {
	if dim == gputypes.TextureDimension3D {
		return 1
	}
	return max(size.DepthOrArrayLayers, 1)
}

// ResolveTextureViewDescriptor fills in the defaults of a view of tex, whose
// texture was made with viewFormats, the way WebGPU resolves them, checks
// the result, and returns the descriptor for the HAL: every field set, and a
// view of one aspect of a combined depth/stencil format in that combined
// format, with the aspect saying which. desc nil is the default view.
//
// Reference: wgpu-core resource.rs Texture::create_view_inner, WebGPU
// "resolving GPUTextureViewDescriptor defaults" and createView
func ResolveTextureViewDescriptor(tex *Texture, viewFormats []gputypes.TextureFormat, desc *hal.TextureViewDescriptor) (hal.TextureViewDescriptor, error) {
	var r hal.TextureViewDescriptor
	if desc != nil {
		r = *desc
	}
	invalid := func(format string, args ...any) (hal.TextureViewDescriptor, error) {
		return hal.TextureViewDescriptor{}, fmt.Errorf("%w %q: %s", ErrInvalidTextureView, r.Label, fmt.Sprintf(format, args...))
	}

	texFormat, dim, size := tex.Format(), tex.Dimension(), tex.Size()
	layers, mips := arrayLayerCount(dim, size), max(tex.MipLevelCount(), 1)
	if r.Aspect == gputypes.TextureAspectUndefined {
		r.Aspect = gputypes.TextureAspectAll
	}
	aspectFormat, ok := AspectFormat(texFormat, r.Aspect)
	if !ok {
		return invalid("format %v has no aspect %v", texFormat, r.Aspect)
	}
	if r.Format == gputypes.TextureFormatUndefined {
		r.Format = aspectFormat
	}
	if r.Dimension == gputypes.TextureViewDimensionUndefined {
		switch {
		case dim == gputypes.TextureDimension1D:
			r.Dimension = gputypes.TextureViewDimension1D
		case dim == gputypes.TextureDimension3D:
			r.Dimension = gputypes.TextureViewDimension3D
		case layers == 1:
			r.Dimension = gputypes.TextureViewDimension2D
		default:
			r.Dimension = gputypes.TextureViewDimension2DArray
		}
	}
	if r.MipLevelCount == 0 && r.BaseMipLevel < mips {
		r.MipLevelCount = mips - r.BaseMipLevel
	}
	if r.ArrayLayerCount == 0 {
		switch r.Dimension {
		case gputypes.TextureViewDimensionCube:
			r.ArrayLayerCount = 6
		case gputypes.TextureViewDimension2DArray, gputypes.TextureViewDimensionCubeArray:
			if r.BaseArrayLayer < layers {
				r.ArrayLayerCount = layers - r.BaseArrayLayer
			}
		default:
			r.ArrayLayerCount = 1
		}
	}

	switch {
	case r.Aspect == gputypes.TextureAspectAll && r.Format != texFormat && !slices.Contains(viewFormats, r.Format):
		return invalid("format %v is neither the texture's %v nor one of its view formats", r.Format, texFormat)
	case r.Aspect != gputypes.TextureAspectAll && r.Format != aspectFormat:
		return invalid("format %v is not %v, the format of aspect %v of %v", r.Format, aspectFormat, r.Aspect, texFormat)
	case tex.SampleCount() > 1 && r.Dimension != gputypes.TextureViewDimension2D:
		return invalid("a view of a multisampled texture is 2D, not %v", r.Dimension)
	case viewTextureDimension(r.Dimension) != dim:
		return invalid("dimension %v does not view a %v texture", r.Dimension, dim)
	}
	switch r.Dimension {
	case gputypes.TextureViewDimension1D, gputypes.TextureViewDimension2D, gputypes.TextureViewDimension3D:
		if r.ArrayLayerCount != 1 {
			return invalid("a %v view has 1 array layer, not %d", r.Dimension, r.ArrayLayerCount)
		}
	case gputypes.TextureViewDimensionCube:
		if r.ArrayLayerCount != 6 {
			return invalid("a cube view has 6 array layers, not %d", r.ArrayLayerCount)
		}
	case gputypes.TextureViewDimensionCubeArray:
		if r.ArrayLayerCount%6 != 0 {
			return invalid("a cube array view has a multiple of 6 array layers, not %d", r.ArrayLayerCount)
		}
	}
	if (r.Dimension == gputypes.TextureViewDimensionCube || r.Dimension == gputypes.TextureViewDimensionCubeArray) && size.Width != size.Height {
		return invalid("a cube view is of a square texture, not %dx%d", size.Width, size.Height)
	}
	if r.MipLevelCount == 0 || uint64(r.BaseMipLevel)+uint64(r.MipLevelCount) > uint64(mips) {
		return invalid("mip levels %d to %d of a texture with %d", r.BaseMipLevel, uint64(r.BaseMipLevel)+uint64(r.MipLevelCount), mips)
	}
	if r.ArrayLayerCount == 0 || uint64(r.BaseArrayLayer)+uint64(r.ArrayLayerCount) > uint64(layers) {
		return invalid("array layers %d to %d of a texture with %d", r.BaseArrayLayer, uint64(r.BaseArrayLayer)+uint64(r.ArrayLayerCount), layers)
	}

	if r.Aspect != gputypes.TextureAspectAll && texFormat.HasDepth() && texFormat.HasStencil() {
		r.Format = texFormat
	}
	return r, nil
}

// viewTextureDimension is the dimension of the textures a view dimension
// views.
func viewTextureDimension(d gputypes.TextureViewDimension) gputypes.TextureDimension {
	switch d {
	case gputypes.TextureViewDimension1D:
		return gputypes.TextureDimension1D
	case gputypes.TextureViewDimension3D:
		return gputypes.TextureDimension3D
	}
	return gputypes.TextureDimension2D
}
