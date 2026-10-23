// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build integration && linux && !(js && wasm)

package gles

import (
	"testing"

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
