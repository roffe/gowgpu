//go:build !(js && wasm)

package core

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
)

// validateRenderPassAttachments checks each attachment of a render pass the
// way WebGPU does: a renderable view (a RenderAttachment texture, one 2D mip
// level and layer, all aspects), a color-renderable format for a color
// attachment and a depth/stencil one for the depth/stencil attachment, the
// load and store ops each channel needs and no others, a depth clear value
// in [0, 1], a resolve target that fits its multisampled attachment, and at
// least one attachment.
//
// Reference: Dawn CommandEncoder.cpp ValidateRenderPassColorAttachment,
// ValidateRenderPassDepthStencilAttachment and ValidateResolveTarget;
// wgpu-core command/render.rs (PassChannel::resolve, resolve target checks)
func validateRenderPassAttachments(desc *RenderPassDescriptor, features gputypes.Features) error {
	attachments := 0
	for i := range desc.ColorAttachments {
		ca := &desc.ColorAttachments[i]
		if ca.View == nil {
			if ca.ResolveTarget != nil {
				return invalidAttachment(desc, i, "a resolve target without a view")
			}
			continue
		}
		attachments++
		if ca.View.Parent == nil {
			continue
		}
		if reason := renderableView(ca.View); reason != "" {
			return invalidAttachment(desc, i, reason)
		}
		format := ca.View.ViewFormat()
		ff := GuaranteedFormatFeatures(format, features)
		if format.IsDepthStencil() || ff.Usages&gputypes.TextureUsageRenderAttachment == 0 {
			return invalidAttachment(desc, i, fmt.Sprintf("format %v is not color-renderable", format))
		}
		if ca.LoadOp == gputypes.LoadOpUndefined || ca.StoreOp == gputypes.StoreOpUndefined {
			return invalidAttachment(desc, i, "the load and store ops are required")
		}
		if ca.ResolveTarget != nil {
			if reason := resolveTarget(ca.View, ca.ResolveTarget, ff); reason != "" {
				return invalidAttachment(desc, i, reason)
			}
		}
	}

	if ds := desc.DepthStencilAttachment; ds != nil && ds.View != nil {
		attachments++
		if ds.View.Parent != nil {
			if reason := depthStencilAttachment(ds); reason != "" {
				return invalidAttachment(desc, -1, reason)
			}
		}
	}
	if attachments == 0 {
		return &RenderPassValidationError{Kind: RenderPassErrorNoAttachments, Label: desc.Label}
	}
	return nil
}

func invalidAttachment(desc *RenderPassDescriptor, index int, reason string) error {
	return &RenderPassValidationError{
		Kind:            RenderPassErrorInvalidAttachment,
		Label:           desc.Label,
		AttachmentIndex: index,
		Reason:          reason,
	}
}

// renderableView returns why v can't be a render pass attachment, or "".
//
// Reference: WebGPU "renderable texture view"
func renderableView(v *TextureView) string {
	switch {
	case v.Parent.Usage()&gputypes.TextureUsageRenderAttachment == 0:
		return "the texture was not made with RenderAttachment"
	case v.Dimension != gputypes.TextureViewDimensionUndefined && v.Dimension != gputypes.TextureViewDimension2D &&
		v.Dimension != gputypes.TextureViewDimension2DArray:
		return fmt.Sprintf("a %v view is not renderable", v.Dimension)
	case v.MipLevelCount > 1:
		return fmt.Sprintf("the view has %d mip levels, an attachment one", v.MipLevelCount)
	case v.ArrayLayerCount > 1:
		return fmt.Sprintf("the view has %d array layers, an attachment one", v.ArrayLayerCount)
	case v.Aspect != gputypes.TextureAspectUndefined && v.Aspect != gputypes.TextureAspectAll:
		return "the view is of one aspect, an attachment of all"
	}
	return ""
}

// resolveTarget returns why target can't take the resolve of view, a color
// attachment of a format with features ff, or "".
func resolveTarget(view, target *TextureView, ff FormatFeatures) string {
	if target.Parent == nil {
		return ""
	}
	if reason := renderableView(target); reason != "" {
		return "resolve target: " + reason
	}
	switch {
	case view.Parent.SampleCount() <= 1:
		return "a resolve target for an attachment that is not multisampled"
	case target.Parent.SampleCount() > 1:
		return "the resolve target is multisampled"
	case target.ViewFormat() != view.ViewFormat():
		return fmt.Sprintf("resolve target format %v differs from the attachment's %v", target.ViewFormat(), view.ViewFormat())
	case !ff.Resolve:
		return fmt.Sprintf("format %v can't be resolved", view.ViewFormat())
	}
	if tw, th := renderExtent(target); tw != max(view.Parent.Size().Width>>view.BaseMipLevel, 1) ||
		th != max(view.Parent.Size().Height>>view.BaseMipLevel, 1) {
		return "the resolve target's size differs from the attachment's"
	}
	return ""
}

// depthStencilAttachment returns why ds can't be the depth/stencil
// attachment of a render pass, or "".
func depthStencilAttachment(ds *RenderPassDepthStencilAttachment) string {
	if reason := renderableView(ds.View); reason != "" {
		return reason
	}
	format := ds.View.ViewFormat()
	if !format.IsDepthStencil() {
		return fmt.Sprintf("format %v is not a depth/stencil format", format)
	}
	if reason := attachmentChannel("depth", format.HasDepth(), ds.DepthReadOnly, ds.DepthLoadOp, ds.DepthStoreOp); reason != "" {
		return reason
	}
	if reason := attachmentChannel("stencil", format.HasStencil(), ds.StencilReadOnly, ds.StencilLoadOp, ds.StencilStoreOp); reason != "" {
		return reason
	}
	if format.HasDepth() && ds.DepthLoadOp == gputypes.LoadOpClear && !(ds.DepthClearValue >= 0 && ds.DepthClearValue <= 1) {
		return fmt.Sprintf("depth clear value %v is outside [0, 1]", ds.DepthClearValue)
	}
	return ""
}

// attachmentChannel returns why the ops of the depth or the stencil channel
// of a depth/stencil attachment are wrong, or "": a channel the format has
// takes both ops unless it is read-only, and one it lacks or reads only
// takes none.
func attachmentChannel(name string, has, readOnly bool, load gputypes.LoadOp, store gputypes.StoreOp) string {
	given := load != gputypes.LoadOpUndefined || store != gputypes.StoreOpUndefined
	switch {
	case !has && given:
		return fmt.Sprintf("%s ops for a format without %s", name, name)
	case has && readOnly && given:
		return fmt.Sprintf("%s ops for read-only %s", name, name)
	case has && !readOnly && (load == gputypes.LoadOpUndefined || store == gputypes.StoreOpUndefined):
		return fmt.Sprintf("the %s load and store ops are required", name)
	}
	return ""
}

// passExtent is the render extent of a pass's attachments, which
// validateRenderPassDescriptor has checked agree.
func passExtent(desc *RenderPassDescriptor) (width, height uint32) {
	for i := range desc.ColorAttachments {
		if v := desc.ColorAttachments[i].View; v != nil && v.Parent != nil {
			return renderExtent(v)
		}
	}
	if ds := desc.DepthStencilAttachment; ds != nil && ds.View != nil && ds.View.Parent != nil {
		return renderExtent(ds.View)
	}
	return 0, 0
}

// checkViewport checks a viewport the way wgpu-core does: a size from 0 to
// maxDimension, a position within twice that, which every Vulkan device's
// viewport bounds range covers, and depths in [0, 1] with the minimum
// first. A NaN fails every comparison and so each check.
//
// Reference: wgpu-core command/render.rs set_viewport
func checkViewport(vp gputypes.Viewport, maxDimension uint32) error {
	size := float32(maxDimension)
	bound := 2 * size
	switch {
	case !(vp.Width >= 0 && vp.Width <= size && vp.Height >= 0 && vp.Height <= size):
		return fmt.Errorf("viewport size %vx%v is not within 0 and %d", vp.Width, vp.Height, maxDimension)
	case !(vp.X >= -bound && vp.Y >= -bound && vp.X+vp.Width <= bound-1 && vp.Y+vp.Height <= bound-1):
		return fmt.Errorf("viewport at %v,%v sized %vx%v reaches past %v", vp.X, vp.Y, vp.Width, vp.Height, bound-1)
	case !(vp.MinDepth >= 0 && vp.MinDepth <= vp.MaxDepth && vp.MaxDepth <= 1):
		return fmt.Errorf("viewport depths %v to %v are not an ascending range in [0, 1]", vp.MinDepth, vp.MaxDepth)
	}
	return nil
}

// checkScissor checks that a scissor rect lies inside the render extent.
//
// Reference: wgpu-core command/render.rs set_scissor
func checkScissor(r gputypes.ScissorRect, width, height uint32) error {
	if uint64(r.X)+uint64(r.Width) > uint64(width) || uint64(r.Y)+uint64(r.Height) > uint64(height) {
		return fmt.Errorf("scissor rect at %d,%d sized %dx%d is not inside the %dx%d render extent",
			r.X, r.Y, r.Width, r.Height, width, height)
	}
	return nil
}
