//go:build !(js && wasm)

package core

import (
	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// passZero is a texture of one subresource whose contents a pass of the
// encoder discarded, zeroed ahead of a later pass or dispatch that reads it
// through the view of it: a discarded texture reads as zeros.
type passZero struct {
	index  track.TrackerIndex
	format gputypes.TextureFormat
	raw    hal.Texture
	view   hal.TextureView
}

// ZeroBeforePass zeroes texture, of one subresource and with contents a
// pass of the encoder discarded, through view ahead of the render pass
// being recorded, which reads it: in the command buffer that goes in ahead
// of the pass with the barriers into it. raw is the handle the barriers are
// recorded against.
//
// Reference: wgpu-core command/memory_init.rs fixup_discarded_surfaces
func (e *CoreCommandEncoder) ZeroBeforePass(texture *Texture, raw hal.Texture, view hal.TextureView) {
	z, ok := newPassZero(texture, raw, view)
	if !ok || e.mutable == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.mutable.activePass != nil {
		e.mutable.pass.zeros = append(e.mutable.pass.zeros, z)
	}
}

// ZeroTexture zeroes texture, of one subresource and with contents a pass
// of the encoder discarded, through view on the command buffer being
// recorded, outside any pass: a compute pass that reads it ends its HAL pass
// for this and begins another after.
//
// Reference: wgpu-core command/memory_init.rs fixup_discarded_surfaces
func (e *CoreCommandEncoder) ZeroTexture(texture *Texture, raw hal.Texture, view hal.TextureView) {
	z, ok := newPassZero(texture, raw, view)
	if !ok || e.mutable == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	guard := e.device.snatchLock.Read()
	defer guard.Release()
	halEncoder := e.raw.Get(guard)
	if halEncoder == nil {
		return
	}
	scope, zeros := &e.mutable.pass, [1]passZero{z}
	scope.zeroTransitionsOf(e.mutable.textureTracker, zeros[:])
	scope.recordZeros(*halEncoder, zeros[:])
	scope.clearZeroScratch()
}

func newPassZero(texture *Texture, raw hal.Texture, view hal.TextureView) (passZero, bool) {
	if texture == nil || raw == nil || view == nil {
		return passZero{}, false
	}
	td := texture.TrackingData()
	if td == nil || !td.Index().IsValid() {
		return passZero{}, false
	}
	return passZero{index: td.Index(), format: texture.Format(), raw: raw, view: view}, true
}

// zeroTransitionsOf moves the textures of zeros to the state the passes
// that clear them leave them in, in tracker, and keeps the barriers that
// take them there.
func (p *passScope) zeroTransitionsOf(tracker *track.EncoderTextureTracker, zeros []passZero) {
	for _, z := range zeros {
		uses := track.TextureUsesColorTarget
		if z.format.IsDepthStencil() {
			uses = track.TextureUsesDepthStencilWrite
		}
		n := len(p.zeroTransitions)
		p.zeroTransitions = tracker.SetSubresourceUsage(z.index, track.TextureSelector{}, track.TextureSelector{}, uses, p.zeroTransitions)
		for _, t := range p.zeroTransitions[n:] {
			p.zeroBarriers = append(p.zeroBarriers, t.IntoHAL(z.raw))
		}
	}
}

// recordZeros records on halEncoder the barriers zeroTransitionsOf kept and
// a pass per texture of zeros that clears it to zero and stores.
func (p *passScope) recordZeros(halEncoder hal.CommandEncoder, zeros []passZero) {
	if len(p.zeroBarriers) > 0 {
		halEncoder.TransitionTextures(p.zeroBarriers)
	}
	for _, z := range zeros {
		desc := &hal.RenderPassDescriptor{Label: "(wgpu internal) zero discarded texture"}
		if z.format.IsDepthStencil() {
			ds := &hal.RenderPassDepthStencilAttachment{View: z.view}
			if z.format.HasDepth() {
				ds.DepthLoadOp, ds.DepthStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
			}
			if z.format.HasStencil() {
				ds.StencilLoadOp, ds.StencilStoreOp = gputypes.LoadOpClear, gputypes.StoreOpStore
			}
			desc.DepthStencilAttachment = ds
		} else {
			desc.ColorAttachments = []hal.RenderPassColorAttachment{{View: z.view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}}
		}
		if pass := halEncoder.BeginRenderPass(desc); pass != nil {
			pass.End()
		}
	}
}

// clearZeroScratch empties the transition and barrier lists of the zeroing,
// so they pin no texture.
func (p *passScope) clearZeroScratch() {
	clear(p.zeroBarriers)
	p.zeroTransitions, p.zeroBarriers = p.zeroTransitions[:0], p.zeroBarriers[:0]
}
