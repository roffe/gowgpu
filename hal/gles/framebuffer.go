// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// maxColorAttachments is the most colour attachments a render pass has.
const maxColorAttachments = 8

// fbAttachment is a mip level and layer of a texture as a framebuffer
// attachment; tex is nil for none.
type fbAttachment struct {
	tex   *Texture
	mip   uint32
	layer uint32
}

// viewAttachment is the attachment a view renders to: its first mip level
// and layer.
func viewAttachment(v *TextureView) fbAttachment {
	return fbAttachment{tex: v.texture, mip: v.baseMip, layer: v.baseLayer}
}

// framebuffers are the framebuffer objects render passes and copies go
// through, as wgpu-hal's draw and copy framebuffers: each pass and copy
// attaches its own mip levels and layers. They live on the queue's context,
// which they are made on, framebuffers not being shared between contexts.
type framebuffers struct {
	draw, read, resolve uint32

	// drawn is what the draw framebuffer has attached: its colour
	// attachments, then its depth/stencil one, and readColor and readDepth
	// what the read framebuffer has. Only what differs is attached again.
	drawn                [maxColorAttachments + 1]fbAttachment
	readColor, readDepth fbAttachment
}

// newFramebuffers makes the framebuffers on the current context.
func newFramebuffers(ctx *gl.Context) *framebuffers {
	return &framebuffers{draw: ctx.GenFramebuffers(1), read: ctx.GenFramebuffers(1), resolve: ctx.GenFramebuffers(1)}
}

// destroy deletes the framebuffers; the context must be current.
func (f *framebuffers) destroy(ctx *gl.Context) {
	ctx.DeleteFramebuffers(f.draw, f.read, f.resolve)
	*f = framebuffers{}
}

// attach attaches a to the framebuffer bound to target at attachment, a
// texture level, an array or 3D texture's layer, or a cube map's face; a
// without a texture detaches what is there.
func attach(ctx *gl.Context, target, attachment uint32, a fbAttachment) {
	switch {
	case a.tex == nil:
		ctx.FramebufferTexture2D(target, attachment, gl.TEXTURE_2D, 0, 0)
	case a.tex.layered():
		ctx.FramebufferTextureLayer(target, attachment, a.tex.id, int32(a.mip), int32(a.layer))
	case a.tex.target == gl.TEXTURE_CUBE_MAP:
		ctx.FramebufferTexture2D(target, attachment, gl.TEXTURE_CUBE_MAP_POSITIVE_X+a.layer, a.tex.id, int32(a.mip))
	default:
		ctx.FramebufferTexture2D(target, attachment, a.tex.target, a.tex.id, int32(a.mip))
	}
}

// attachmentPoint is where a texture of format attaches: a colour
// attachment, or the depth, stencil or combined point.
func attachmentPoint(format gputypes.TextureFormat, color uint32) uint32 {
	if format.IsDepthStencil() {
		return depthStencilAttachmentPoint(format)
	}
	return gl.COLOR_ATTACHMENT0 + color
}

// BindRenderTargetsCommand binds the draw framebuffer with a render pass's
// attachments.
type BindRenderTargetsCommand struct {
	fbs    *framebuffers
	colors [maxColorAttachments]fbAttachment
	depth  fbAttachment
}

func (c *BindRenderTargetsCommand) Execute(ctx *gl.Context) {
	f := c.fbs
	ctx.BindFramebuffer(gl.FRAMEBUFFER, f.draw)
	for i, a := range c.colors {
		if f.drawn[i] != a {
			attach(ctx, gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0+uint32(i), a)
			f.drawn[i] = a
		}
	}
	if d := &f.drawn[maxColorAttachments]; *d != c.depth {
		if d.tex != nil {
			// Detaching the combined point detaches depth and stencil both.
			ctx.FramebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.TEXTURE_2D, 0, 0)
		}
		if c.depth.tex != nil {
			attach(ctx, gl.FRAMEBUFFER, depthStencilAttachmentPoint(c.depth.tex.format), c.depth)
		}
		*d = c.depth
	}
}

// bindRead binds the read framebuffer with a attached alone, at colour
// attachment 0 or at its depth/stencil point, for a copy to read from.
func (f *framebuffers) bindRead(ctx *gl.Context, a fbAttachment) {
	ctx.BindFramebuffer(gl.READ_FRAMEBUFFER, f.read)
	color, depth := a, fbAttachment{}
	if a.tex.format.IsDepthStencil() {
		color, depth = fbAttachment{}, a
	}
	if f.readColor != color {
		attach(ctx, gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, color)
		f.readColor = color
	}
	if f.readDepth != depth {
		if f.readDepth.tex != nil {
			ctx.FramebufferTexture2D(gl.READ_FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, gl.TEXTURE_2D, 0, 0)
		}
		if depth.tex != nil {
			attach(ctx, gl.READ_FRAMEBUFFER, depthStencilAttachmentPoint(depth.tex.format), depth)
		}
		f.readDepth = depth
	}
}
