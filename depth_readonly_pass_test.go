//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestDepthWriteThenReadOnlyPass: a depth texture written by one pass and
// attached read-only (DepthReadOnly && StencilReadOnly, LoadOp Load) by the
// next. The tracker transitions DepthStencilWrite -> DepthStencilRead between
// them, and the barrier has to name the attachment layout: Vulkan's render
// pass keeps a read-only depth attachment in DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
// and a barrier into SHADER_READ_ONLY_OPTIMAL failed the draw under the
// validation layer. In one encoder the second pass is split off with its
// transit CB; across two submits the barrier is the submit's.
func TestDepthWriteThenReadOnlyPass(t *testing.T) {
	for _, tc := range []struct {
		name        string
		sameEncoder bool
	}{
		{"same_encoder", true},
		{"two_submits", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			device, h := newValidatedDevice(t)

			tex, err := device.CreateTexture(&TextureDescriptor{
				Label:         "depth",
				Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
				MipLevelCount: 1,
				SampleCount:   1,
				Dimension:     TextureDimension2D,
				Format:        TextureFormatDepth32Float,
				Usage:         TextureUsageRenderAttachment | TextureUsageTextureBinding,
			})
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			defer tex.Release()
			view, err := device.CreateTextureView(tex, nil)
			if err != nil {
				t.Fatalf("CreateTextureView: %v", err)
			}
			defer view.Release()

			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
				DepthStencilAttachment: &RenderPassDepthStencilAttachment{
					View: view, DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 1,
				},
			})
			if err != nil {
				t.Fatalf("BeginRenderPass(write): %v", err)
			}
			if err := pass.End(); err != nil {
				t.Fatalf("End(write): %v", err)
			}
			if !tc.sameEncoder {
				finishAndSubmit(t, device, enc)
				enc, err = device.CreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("CreateCommandEncoder: %v", err)
				}
			}
			pass, err = enc.BeginRenderPass(&RenderPassDescriptor{
				DepthStencilAttachment: &RenderPassDepthStencilAttachment{
					View: view, DepthLoadOp: gputypes.LoadOpLoad, DepthStoreOp: gputypes.StoreOpStore,
					DepthReadOnly: true, StencilReadOnly: true,
				},
			})
			if err != nil {
				t.Fatalf("BeginRenderPass(read-only): %v", err)
			}
			if err := pass.End(); err != nil {
				t.Fatalf("End(read-only): %v", err)
			}
			finishAndSubmit(t, device, enc)
			if err := device.WaitIdle(); err != nil {
				t.Fatalf("WaitIdle: %v", err)
			}
			assertNoValidation(t, h)
		})
	}
}
