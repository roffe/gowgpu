//go:build !(js && wasm)

package core

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

func TestValidateRenderPassAttachments(t *testing.T) {
	color := func(format gputypes.TextureFormat, samples uint32) *TextureView {
		return testView(testTexture(format, 64, 64, samples))
	}
	ca := func(v *TextureView) RenderPassColorAttachment {
		return RenderPassColorAttachment{View: v, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}
	}
	depth := func(format gputypes.TextureFormat) *RenderPassDepthStencilAttachment {
		ds := &RenderPassDepthStencilAttachment{View: color(format, 1)}
		if format.HasDepth() {
			ds.DepthLoadOp, ds.DepthStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
		}
		if format.HasStencil() {
			ds.StencilLoadOp, ds.StencilStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
		}
		return ds
	}
	withView := func(change func(*TextureView)) *TextureView {
		v := color(gputypes.TextureFormatRGBA8Unorm, 1)
		change(v)
		return v
	}

	for _, tc := range []struct {
		name  string
		desc  RenderPassDescriptor
		valid bool
	}{
		{name: "color", valid: true, desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{ca(color(gputypes.TextureFormatRGBA8Unorm, 1))}}},
		{name: "depth only", valid: true, desc: RenderPassDescriptor{DepthStencilAttachment: depth(gputypes.TextureFormatDepth24PlusStencil8)}},
		{name: "read-only depth without ops", valid: true, desc: RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{
			View: color(gputypes.TextureFormatDepth32Float, 1), DepthReadOnly: true,
		}}},
		{name: "resolve", valid: true, desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: color(gputypes.TextureFormatRGBA8Unorm, 4), ResolveTarget: color(gputypes.TextureFormatRGBA8Unorm, 1),
			LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpDiscard,
		}}}},

		{name: "no attachment", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{}}}},
		{name: "not a render attachment", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{ca(withView(func(v *TextureView) {
			v.Parent.usage = gputypes.TextureUsageTextureBinding
		}))}}},
		{name: "two mips", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{ca(withView(func(v *TextureView) { v.MipLevelCount = 2 }))}}},
		{name: "two layers", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{ca(withView(func(v *TextureView) { v.ArrayLayerCount = 2 }))}}},
		{name: "cube view", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{ca(withView(func(v *TextureView) {
			v.Dimension = gputypes.TextureViewDimensionCube
		}))}}},
		{name: "unrenderable color format", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{ca(color(gputypes.TextureFormatRGBA8Snorm, 1))}}},
		{name: "depth format as color", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{ca(color(gputypes.TextureFormatDepth32Float, 1))}}},
		{name: "color format as depth", desc: RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{
			View: color(gputypes.TextureFormatRGBA8Unorm, 1), DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore,
		}}},
		{name: "no load op", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: color(gputypes.TextureFormatRGBA8Unorm, 1), StoreOp: gputypes.StoreOpStore,
		}}}},
		{name: "no depth ops", desc: RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{View: color(gputypes.TextureFormatDepth16Unorm, 1)}}},
		{name: "ops for read-only depth", desc: RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{
			View: color(gputypes.TextureFormatDepth32Float, 1), DepthReadOnly: true,
			DepthLoadOp: gputypes.LoadOpLoad, DepthStoreOp: gputypes.StoreOpStore,
		}}},
		{name: "stencil ops without stencil", desc: RenderPassDescriptor{DepthStencilAttachment: func() *RenderPassDepthStencilAttachment {
			ds := depth(gputypes.TextureFormatDepth32Float)
			ds.StencilLoadOp, ds.StencilStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
			return ds
		}()}},
		{name: "depth clear value out of range", desc: RenderPassDescriptor{DepthStencilAttachment: func() *RenderPassDepthStencilAttachment {
			ds := depth(gputypes.TextureFormatDepth32Float)
			ds.DepthClearValue = 7
			return ds
		}()}},
		{name: "resolve of a single-sampled attachment", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: color(gputypes.TextureFormatRGBA8Unorm, 1), ResolveTarget: color(gputypes.TextureFormatRGBA8Unorm, 1),
			LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
		}}}},
		{name: "multisampled resolve target", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: color(gputypes.TextureFormatRGBA8Unorm, 4), ResolveTarget: color(gputypes.TextureFormatRGBA8Unorm, 4),
			LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
		}}}},
		{name: "resolve into another format", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: color(gputypes.TextureFormatRGBA8Unorm, 4), ResolveTarget: color(gputypes.TextureFormatBGRA8Unorm, 1),
			LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
		}}}},
		{name: "resolve into another size", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: color(gputypes.TextureFormatRGBA8Unorm, 4), ResolveTarget: testView(testTexture(gputypes.TextureFormatRGBA8Unorm, 32, 32, 1)),
			LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
		}}}},
		{name: "resolve of an integer format", desc: RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: color(gputypes.TextureFormatRGBA8Uint, 4), ResolveTarget: color(gputypes.TextureFormatRGBA8Uint, 1),
			LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
		}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ValidateRenderPassDescriptor(&tc.desc, gputypes.DefaultLimits(), 0)
			var rpve *RenderPassValidationError
			switch {
			case tc.valid && err != nil:
				t.Errorf("valid pass: %v", err)
			case !tc.valid && !errors.As(err, &rpve):
				t.Errorf("invalid pass: %v, want a RenderPassValidationError", err)
			}
		})
	}
}
