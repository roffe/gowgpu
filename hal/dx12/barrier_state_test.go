// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestTextureBarrierStateDepth: a barrier into RenderAttachment names
// DEPTH_WRITE for a depth texture, which has no RENDER_TARGET state, or
// DEPTH_READ when the fine-grained usage is a read-only attachment; colour
// textures and the other usages are as before.
func TestTextureBarrierStateDepth(t *testing.T) {
	depth := &Texture{format: gputypes.TextureFormatDepth32Float}
	color := &Texture{format: gputypes.TextureFormatRGBA8Unorm}
	srv := d3d12.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | d3d12.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
	tests := []struct {
		name  string
		tex   *Texture
		usage gputypes.TextureUsage
		uses  hal.TextureUses
		want  d3d12.D3D12_RESOURCE_STATES
	}{
		{"depth attachment", depth, gputypes.TextureUsageRenderAttachment, 0, d3d12.D3D12_RESOURCE_STATE_DEPTH_WRITE},
		{"depth attachment written", depth, gputypes.TextureUsageRenderAttachment, hal.TextureUsesDepthStencilWrite, d3d12.D3D12_RESOURCE_STATE_DEPTH_WRITE},
		{"depth attachment read-only", depth, gputypes.TextureUsageRenderAttachment, hal.TextureUsesDepthStencilRead, d3d12.D3D12_RESOURCE_STATE_DEPTH_READ},
		{"depth attachment read-only and sampled", depth, gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding, hal.TextureUsesDepthStencilRead | hal.TextureUsesResource, d3d12.D3D12_RESOURCE_STATE_DEPTH_READ | srv},
		{"depth sampled", depth, gputypes.TextureUsageTextureBinding, hal.TextureUsesResource, srv},
		{"color attachment", color, gputypes.TextureUsageRenderAttachment, hal.TextureUsesColorTarget, d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET},
		{"color copy dst", color, gputypes.TextureUsageCopyDst, hal.TextureUsesCopyDst, d3d12.D3D12_RESOURCE_STATE_COPY_DEST},
		{"nil texture", nil, gputypes.TextureUsageRenderAttachment, 0, d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := textureBarrierState(tt.tex, tt.usage, tt.uses); got != tt.want {
				t.Errorf("textureBarrierState = %#x, want %#x", got, tt.want)
			}
		})
	}
}
