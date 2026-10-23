//go:build !(js && wasm)

package wgpu

import (
	"math"
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// SetScissorRect refuses a rect outside the pass's attachments, which Metal
// requires and Vulkan reads as negative past 2^31, and SetViewport a
// negative or oversized viewport, one too far out, a NaN and a depth range
// outside [0, 1] or upside down.
func TestViewportAndScissorAreChecked(t *testing.T) {
	device, h := newValidatedDevice(t)

	color, colorView := pixTexture(t, device, "viewport-scissor", TextureUsageRenderAttachment)
	defer releaseTextureView(color, colorView)
	maxDim := float32(device.Limits().MaxTextureDimension2D)
	nan := float32(math.NaN())

	for _, tc := range []struct {
		name    string
		record  func(p *RenderPassEncoder)
		wantErr string
	}{
		{"scissor over the whole pass", func(p *RenderPassEncoder) {
			p.SetScissorRect(gputypes.ScissorRect{Width: pixSide, Height: pixSide})
		}, ""},
		{"scissor in a corner", func(p *RenderPassEncoder) {
			p.SetScissorRect(gputypes.ScissorRect{X: pixSide - 1, Y: pixSide - 1, Width: 1, Height: 1})
		}, ""},
		{"scissor past the right edge", func(p *RenderPassEncoder) {
			p.SetScissorRect(gputypes.ScissorRect{X: 1, Width: pixSide, Height: pixSide})
		}, "scissor"},
		{"scissor past the bottom edge", func(p *RenderPassEncoder) {
			p.SetScissorRect(gputypes.ScissorRect{Width: pixSide, Height: pixSide + 1})
		}, "scissor"},
		{"scissor at x 2^31", func(p *RenderPassEncoder) {
			p.SetScissorRect(gputypes.ScissorRect{X: 1 << 31, Width: 1, Height: 1})
		}, "scissor"},
		{"viewport over the whole pass", func(p *RenderPassEncoder) {
			p.SetViewport(gputypes.Viewport{Width: pixSide, Height: pixSide, MaxDepth: 1})
		}, ""},
		{"viewport partly outside the pass", func(p *RenderPassEncoder) {
			p.SetViewport(gputypes.Viewport{X: -2, Y: 2, Width: pixSide, Height: pixSide, MinDepth: 0.25, MaxDepth: 0.75})
		}, ""},
		{"negative viewport width", func(p *RenderPassEncoder) {
			p.SetViewport(gputypes.Viewport{Width: -1, Height: pixSide, MaxDepth: 1})
		}, "viewport size"},
		{"viewport larger than a texture can be", func(p *RenderPassEncoder) {
			p.SetViewport(gputypes.Viewport{Width: maxDim + 1, Height: pixSide, MaxDepth: 1})
		}, "viewport size"},
		{"viewport too far out", func(p *RenderPassEncoder) {
			p.SetViewport(gputypes.Viewport{X: 2 * maxDim, Width: pixSide, Height: pixSide, MaxDepth: 1})
		}, "reaches past"},
		{"NaN viewport", func(p *RenderPassEncoder) {
			p.SetViewport(gputypes.Viewport{X: nan, Width: pixSide, Height: pixSide, MaxDepth: 1})
		}, "reaches past"},
		{"viewport depth above 1", func(p *RenderPassEncoder) {
			p.SetViewport(gputypes.Viewport{Width: pixSide, Height: pixSide, MaxDepth: 2})
		}, "depths"},
		{"upside down viewport depths", func(p *RenderPassEncoder) {
			p.SetViewport(gputypes.Viewport{Width: pixSide, Height: pixSide, MinDepth: 0.75, MaxDepth: 0.25})
		}, "depths"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
				ColorAttachments: []RenderPassColorAttachment{{View: colorView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}},
			})
			if err != nil {
				t.Fatalf("BeginRenderPass: %v", err)
			}
			tc.record(pass)
			_ = pass.End()
			cb, err := enc.Finish()
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("Finish: %v", err)
			case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
				t.Errorf("Finish: %v, want an error about %q", err, tc.wantErr)
			}
			if err == nil {
				if _, err := device.Queue().Submit(cb); err != nil {
					t.Fatalf("Submit: %v", err)
				}
			}
		})
	}
	assertNoValidation(t, h)
}
