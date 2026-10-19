// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestTextureBarrierStateDepth: a barrier into RenderAttachment names
// DEPTH_WRITE for a depth texture, which has no RENDER_TARGET state; colour
// textures and the other usages are as before.
func TestTextureBarrierStateDepth(t *testing.T) {
	depth := &Texture{format: gputypes.TextureFormatDepth32Float}
	color := &Texture{format: gputypes.TextureFormatRGBA8Unorm}
	tests := []struct {
		name  string
		tex   *Texture
		usage gputypes.TextureUsage
		want  d3d12.D3D12_RESOURCE_STATES
	}{
		{"depth attachment", depth, gputypes.TextureUsageRenderAttachment, d3d12.D3D12_RESOURCE_STATE_DEPTH_WRITE},
		{"depth sampled", depth, gputypes.TextureUsageTextureBinding, d3d12.D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE | d3d12.D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE},
		{"color attachment", color, gputypes.TextureUsageRenderAttachment, d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET},
		{"color copy dst", color, gputypes.TextureUsageCopyDst, d3d12.D3D12_RESOURCE_STATE_COPY_DEST},
		{"nil texture", nil, gputypes.TextureUsageRenderAttachment, d3d12.D3D12_RESOURCE_STATE_RENDER_TARGET},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := textureBarrierState(tt.tex, tt.usage); got != tt.want {
				t.Errorf("textureBarrierState = %#x, want %#x", got, tt.want)
			}
		})
	}
}
