//go:build windows && !(js && wasm)

package dx12

import (
	"encoding/binary"
	"math"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestClearDepthByRenderPass clears a depth texture through the render
// pass call windows/arm64 clears with and reads the depth back.
func TestClearDepthByRenderPass(t *testing.T) {
	dev, queue := openTestDevice(t)
	if underWine() {
		t.Skip("vkd3d's BeginRenderPass and EndRenderPass are stubs")
	}
	const side, pitch = 4, 256
	tex := newTestTexture(t, dev, hal.TextureDescriptor{
		Size:      hal.Extent3D{Width: side, Height: side},
		Dimension: gputypes.TextureDimension2D,
		Format:    gputypes.TextureFormatDepth32Float,
		Usage:     gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopySrc,
	})
	raw, err := dev.CreateTextureView(tex, &hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2D, MipLevelCount: 1, ArrayLayerCount: 1})
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	t.Cleanup(func() { dev.DestroyTextureView(raw) })
	view := raw.(*TextureView)
	out := newTestBuffer(t, dev, pitch*side, gputypes.BufferUsageCopyDst|gputypes.BufferUsageMapRead)
	record(t, dev, queue, func(enc *CommandEncoder) {
		if before, barrier := enc.stateTracker.transitionTexture(tex, 0, d3d12.D3D12_RESOURCE_STATE_DEPTH_WRITE); barrier {
			enc.emitStateBarrierPlans([]stateBarrierPlan{{resource: tex, subresource: 0, before: before, after: d3d12.D3D12_RESOURCE_STATE_DEPTH_WRITE}})
		}
		if err := enc.clearDepthStencilByRenderPass(view.dsvHandle, d3d12.DXGI_FORMAT_D32_FLOAT, d3d12.D3D12_CLEAR_FLAG_DEPTH, 0.25, 0); err != nil {
			t.Fatalf("clearDepthStencilByRenderPass: %v", err)
		}
		enc.CopyTextureToBuffer(tex, out, []hal.BufferTextureCopy{{
			BufferLayout: hal.ImageDataLayout{BytesPerRow: pitch, RowsPerImage: side},
			TextureBase:  hal.ImageCopyTexture{Texture: tex},
			Size:         hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
		}})
	})
	got := readBuffer(t, dev, out, pitch*side)
	for y := 0; y < side; y++ {
		for x := 0; x < side; x++ {
			if d := math.Float32frombits(binary.LittleEndian.Uint32(got[y*pitch+x*4:])); d != 0.25 {
				t.Fatalf("depth at (%d, %d) = %v, want 0.25", x, y, d)
			}
		}
	}
}
