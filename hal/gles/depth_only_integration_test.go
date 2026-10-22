// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build integration && linux && !(js && wasm)

package gles

import (
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// A pass with a depth attachment and no colour attachment draws into a
// framebuffer of its own: the depth texture gets one, the viewport is its
// size, and the clear lands in it.
func TestDepthOnlyPassTargetsItsTexture(t *testing.T) {
	adapterCtx := newTestAdapterContext(t)
	t.Cleanup(adapterCtx.Destroy)
	dev := &Device{ctx: adapterCtx, maxTextureUnits: 16}
	const side = 8
	tex, err := dev.CreateTexture(&hal.TextureDescriptor{
		Size:          hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     gputypes.TextureDimension2D,
		Format:        gputypes.TextureFormatDepth32Float,
		Usage:         gputypes.TextureUsageRenderAttachment,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	t.Cleanup(func() { dev.DestroyTexture(tex) })
	view, err := dev.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	glCtx := adapterCtx.Lock()
	glCtx.Viewport(0, 0, 1, 1)
	adapterCtx.Unlock()

	_ = enc.BeginEncoding("depth-only")
	pass := enc.BeginRenderPass(&hal.RenderPassDescriptor{
		DepthStencilAttachment: &hal.RenderPassDepthStencilAttachment{
			View: view, DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 0.25,
		},
	})
	pass.End()
	cb, err := enc.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding: %v", err)
	}
	if _, err := (&Queue{ctx: adapterCtx}).Submit([]hal.CommandBuffer{cb}); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	glCtx = adapterCtx.Lock()
	defer adapterCtx.Unlock()
	texture := tex.(*Texture)
	if texture.fbo == 0 {
		t.Fatal("the pass made no framebuffer for the depth texture")
	}
	var vp [4]int32
	glCtx.GetIntegerv(gl.VIEWPORT, &vp[0])
	if vp != [4]int32{0, 0, side, side} {
		t.Errorf("viewport after the pass is %v, want the texture's %dx%d", vp, side, side)
	}
	glCtx.BindFramebuffer(gl.READ_FRAMEBUFFER, texture.fbo)
	var depth [side * side]float32
	glCtx.ReadPixels(0, 0, side, side, gl.DEPTH_COMPONENT, gl.FLOAT, unsafe.Pointer(&depth[0]))
	glCtx.BindFramebuffer(gl.READ_FRAMEBUFFER, 0)
	if glErr := glCtx.GetError(); glErr != 0 {
		t.Fatalf("reading the depth back: GL error %#x", glErr)
	}
	for i, d := range depth {
		if d != 0.25 {
			t.Fatalf("depth texel %d is %v, want the cleared 0.25", i, d)
		}
	}
}
