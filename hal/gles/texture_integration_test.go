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

// openTestDevice opens a device on ctx the way an adapter on it would, with
// the capabilities probed from it.
func openTestDevice(t *testing.T, ctx *AdapterContext) *Device {
	t.Helper()
	glCtx := ctx.Lock()
	caps := queryAdapterCapabilities(glCtx)
	ctx.Unlock()
	open, err := (&Adapter{ctx: ctx, caps: caps}).Open(0, gputypes.Limits{})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	return open.Device.(*Device)
}

// Every depth and stencil format makes a texture, on desktop GL and on ES:
// Depth32Float was the unorm GL_DEPTH_COMPONENT32, not an ES format, and
// Depth32FloatStencil8 took a transfer type its internal format doesn't.
func TestDepthStencilTexturesCreate(t *testing.T) {
	for _, tc := range []struct {
		name string
		ctx  func(*testing.T) *AdapterContext
	}{{"desktop", newTestAdapterContext}, {"es", newTestESAdapterContext}} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := tc.ctx(t)
			t.Cleanup(ctx.Destroy)
			dev := openTestDevice(t, ctx)
			for _, format := range []gputypes.TextureFormat{
				gputypes.TextureFormatStencil8, gputypes.TextureFormatDepth16Unorm, gputypes.TextureFormatDepth24Plus,
				gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureFormatDepth32Float, gputypes.TextureFormatDepth32FloatStencil8,
			} {
				tex, err := dev.CreateTexture(&hal.TextureDescriptor{
					Size: hal.Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
					Dimension: gputypes.TextureDimension2D, Format: format,
					Usage: gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding,
				})
				if err != nil {
					t.Errorf("%s: %v", format, err)
					continue
				}
				dev.DestroyTexture(tex)
			}
		})
	}
}

// An ETC2 texture, the compressed format every ES context offers, is made
// with its own internal format and takes its blocks: it was RGBA8, and an
// upload read eight times the blocks' size.
func TestETC2TextureOnES(t *testing.T) {
	ctx := newTestESAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	dev := openTestDevice(t, ctx)
	if !dev.formats.texStorage {
		t.Skip("no glTexStorage2D")
	}
	tex, err := dev.CreateTexture(&hal.TextureDescriptor{
		Size: hal.Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatETC2RGB8Unorm,
		Usage: gputypes.TextureUsageTextureBinding | gputypes.TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	t.Cleanup(func() { dev.DestroyTexture(tex) })
	q := &Queue{ctx: ctx}
	blocks := make([]byte, 4*8)
	if err := q.WriteTexture(&hal.ImageCopyTexture{Texture: tex}, blocks, &hal.ImageDataLayout{BytesPerRow: 16},
		&hal.Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	glCtx := ctx.Lock()
	defer ctx.Unlock()
	if glErr := glCtx.GetError(); glErr != 0 {
		t.Fatalf("GL error %#x after the upload", glErr)
	}
	if got := tex.(*Texture).glf.internal; got != gl.COMPRESSED_RGB8_ETC2 {
		t.Errorf("internal format %#x, want COMPRESSED_RGB8_ETC2", got)
	}
}

// An adapter made without a surface has the capabilities of its context:
// it had none, and its device wrote GLSL 3.30 core for every context, ES
// included, which no ES driver compiles.
func TestHeadlessAdapterHasCapabilities(t *testing.T) {
	for _, tc := range []struct {
		name string
		ctx  func(*testing.T) *AdapterContext
		es   bool
	}{{"desktop", newTestAdapterContext, false}, {"es", newTestESAdapterContext, true}} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := tc.ctx(t)
			t.Cleanup(ctx.Destroy)
			exposed := makeAdapterFromContext(ctx, 0, 0)
			if exposed.Adapter == nil {
				t.Fatal("no adapter")
			}
			if exposed.Features == 0 || exposed.Capabilities.DownlevelCapabilities.Flags == 0 {
				t.Errorf("features %#x, downlevel %#x: the context's capabilities were not probed", uint64(exposed.Features), uint64(exposed.Capabilities.DownlevelCapabilities.Flags))
			}
			open, err := exposed.Adapter.Open(0, gputypes.Limits{})
			if err != nil {
				t.Fatalf("Open: %v", err)
			}
			defer open.Device.Destroy()
			if v := open.Device.(*Device).glslVersion; v.ES != tc.es {
				t.Errorf("the device writes GLSL %s", v)
			}
		})
	}
}

// A 4-sample attachment renders and resolves on ES, which has no
// glTexImage2DMultisample: the texture failed to make on ES, or had no
// storage where the call was a no-op, and the pass drew nowhere.
func TestMultisampledAttachmentOnES(t *testing.T) {
	ctx := newTestESAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	dev := openTestDevice(t, ctx)
	t.Cleanup(dev.Destroy)
	newTex := func(samples uint32, usage gputypes.TextureUsage) hal.Texture {
		tex, err := dev.CreateTexture(&hal.TextureDescriptor{
			Size: hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: samples,
			Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatRGBA8Unorm, Usage: usage,
		})
		if err != nil {
			t.Fatalf("CreateTexture(%d samples): %v", samples, err)
		}
		t.Cleanup(func() { dev.DestroyTexture(tex) })
		return tex
	}
	msaa := newTex(4, gputypes.TextureUsageRenderAttachment)
	resolve := newTex(1, gputypes.TextureUsageRenderAttachment|gputypes.TextureUsageCopySrc)
	if dev.formats.texStorageMS {
		newTex(4, gputypes.TextureUsageRenderAttachment|gputypes.TextureUsageTextureBinding)
	}
	msaaView, _ := dev.CreateTextureView(msaa, nil)
	resolveView, _ := dev.CreateTextureView(resolve, nil)

	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	_ = enc.BeginEncoding("msaa")
	pass := enc.BeginRenderPass(&hal.RenderPassDescriptor{ColorAttachments: []hal.RenderPassColorAttachment{{
		View: msaaView, ResolveTarget: resolveView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
		ClearValue: gputypes.Color{G: 1, A: 1},
	}}})
	pass.End()
	cb, err := enc.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding: %v", err)
	}
	if _, err := (&Queue{ctx: ctx}).Submit([]hal.CommandBuffer{cb}); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	glCtx := ctx.Lock()
	defer ctx.Unlock()
	glCtx.BindFramebuffer(gl.READ_FRAMEBUFFER, dev.fbs.resolve)
	var px [4 * 4 * 4]byte
	glCtx.ReadPixels(0, 0, 4, 4, gl.RGBA, gl.UNSIGNED_BYTE, unsafe.Pointer(&px[0]))
	glCtx.BindFramebuffer(gl.READ_FRAMEBUFFER, 0)
	if glErr := glCtx.GetError(); glErr != 0 {
		t.Fatalf("GL error %#x", glErr)
	}
	if [4]byte(px[:4]) != [4]byte{0, 255, 0, 255} {
		t.Errorf("resolved pixel %v, want green", px[:4])
	}
}
