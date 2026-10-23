// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build integration && linux && !(js && wasm)

package gles

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/egl"
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

// The present blit of an sRGB swapchain to an sRGB EGL surface keeps its
// bytes on ES, where the blit decodes the sRGB swapchain and only an sRGB
// destination encodes it again: to a linear window surface a stored 188
// arrived as 128. A pbuffer stands in for the window.
func TestSRGBPresentBlitOnES(t *testing.T) {
	ctx := newTestESAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	eglCtx := ctx.EGL()
	if !eglCtx.SupportsGLColorspace() {
		t.Skip("no EGL_GL_COLORSPACE on this display")
	}
	attribs := []egl.EGLInt{egl.Width, 4, egl.Height, 4, egl.GLColorspaceKHR, egl.GLColorspaceSRGBKHR, egl.None}
	pb := egl.CreatePbufferSurface(eglCtx.Display(), eglCtx.Config(), &attribs[0])
	if pb == egl.NoSurface {
		t.Skipf("no sRGB pbuffer: EGL error %#x", egl.GetError())
	}
	t.Cleanup(func() { egl.DestroySurface(eglCtx.Display(), pb) })

	glCtx := ctx.LockForSurface(pb)
	defer ctx.Unlock()
	s := &Surface{}
	if err := s.reconfigureSwapchainFBOWith(glCtx, gputypes.TextureFormatRGBA8UnormSrgb, 4, 4); err != nil {
		t.Fatalf("swapchain: %v", err)
	}
	defer destroySwapchainFBO(glCtx, s.swapchainFBO, s.colorRenderbuffer)
	glCtx.BindFramebuffer(gl.FRAMEBUFFER, s.swapchainFBO)
	grey := [4]float32{0.5, 0.5, 0.5, 1}
	glCtx.ClearBufferfv(gl.COLOR, 0, &grey)
	var stored, presented [4]byte
	glCtx.ReadPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, unsafe.Pointer(&stored[0]))
	glCtx.BindFramebuffer(gl.FRAMEBUFFER, 0)

	s.blitSwapchainToDefaultWith(glCtx)
	glCtx.BindFramebuffer(gl.READ_FRAMEBUFFER, 0)
	glCtx.ReadPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, unsafe.Pointer(&presented[0]))
	if glErr := glCtx.GetError(); glErr != 0 {
		t.Fatalf("GL error %#x", glErr)
	}
	if d := int(presented[0]) - int(stored[0]); d < -1 || d > 1 {
		t.Errorf("the swapchain held %v and the surface got %v", stored, presented)
	}
}

// Textures are read back on ES in pairs every ES driver reads and repacked
// into their own texels: R8, RG8, BGRA8 and the half float formats were
// read in their upload pairs, which ES lets a driver refuse.
func TestTextureReadbackOnES(t *testing.T) {
	ctx := newTestESAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	dev := openTestDevice(t, ctx)
	t.Cleanup(dev.Destroy)
	q := &Queue{ctx: ctx}
	glCtx := ctx.Lock()
	floats := queryAdapterCapabilities(glCtx).Extensions["GL_EXT_color_buffer_float"]
	ctx.Unlock()
	half := func(hs ...uint16) []byte {
		b := make([]byte, 2*len(hs))
		for i, h := range hs {
			b[2*i], b[2*i+1] = byte(h), byte(h>>8)
		}
		return b
	}
	for _, tc := range []struct {
		format gputypes.TextureFormat
		texels []byte // 2x2
		float  bool
	}{
		{gputypes.TextureFormatRGBA8Unorm, seq(16), false},
		{gputypes.TextureFormatR8Unorm, seq(4), false},
		{gputypes.TextureFormatRG8Unorm, seq(8), false},
		{gputypes.TextureFormatBGRA8Unorm, seq(16), false},
		{gputypes.TextureFormatRGB10A2Unorm, seq(16), false},
		{gputypes.TextureFormatR16Unorm, seq(8), false},
		{gputypes.TextureFormatR8Uint, seq(4), false},
		{gputypes.TextureFormatRG16Sint, append(seq(14), 0xff, 0xff), false},
		{gputypes.TextureFormatRGBA32Uint, seq(64), false},
		{gputypes.TextureFormatRGB10A2Uint, seq(16), false},
		{gputypes.TextureFormatR16Float, half(0x3c00, 0xc000, 0x0001, 0x7bff), true},
		{gputypes.TextureFormatRGBA16Float, half(0x3c00, 0x3555, 0x8000, 0xfbff, 0x0400, 0x03ff, 0x4248, 0x0000,
			0x7bff, 0xbc00, 0x2e66, 0x5640, 0x0001, 0x8001, 0x3800, 0xc500), true},
		{gputypes.TextureFormatRG32Float, half(0, 0x3f80, 0, 0xc000, 0x1234, 0x3eaa, 0, 0x4780, 0, 0x7f7f, 0, 0x8000, 0, 0x3f00, 0, 0x4f00), true},
	} {
		if _, ok := dev.formats.glFormat(tc.format); !ok || tc.float && !floats {
			t.Logf("%s: not renderable here", tc.format)
			continue
		}
		tex, err := dev.CreateTexture(&hal.TextureDescriptor{
			Size: hal.Extent3D{Width: 2, Height: 2, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
			Dimension: gputypes.TextureDimension2D, Format: tc.format,
			Usage: gputypes.TextureUsageCopySrc | gputypes.TextureUsageCopyDst | gputypes.TextureUsageRenderAttachment,
		})
		if err != nil {
			t.Errorf("%s: %v", tc.format, err)
			continue
		}
		row := len(tc.texels) / 2
		size := hal.Extent3D{Width: 2, Height: 2, DepthOrArrayLayers: 1}
		if err := q.WriteTexture(&hal.ImageCopyTexture{Texture: tex}, tc.texels, &hal.ImageDataLayout{BytesPerRow: uint32(row)}, &size); err != nil {
			t.Fatalf("%s: WriteTexture: %v", tc.format, err)
		}
		buf, err := dev.CreateBuffer(&hal.BufferDescriptor{Size: 512, Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		enc, err := dev.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		_ = enc.BeginEncoding("readback")
		enc.CopyTextureToBuffer(tex, buf, []hal.BufferTextureCopy{{
			BufferLayout: hal.ImageDataLayout{BytesPerRow: 256, RowsPerImage: 2}, TextureBase: hal.ImageCopyTexture{Texture: tex}, Size: size,
		}})
		cb, err := enc.EndEncoding()
		if err != nil {
			t.Fatalf("EndEncoding: %v", err)
		}
		if _, err := q.Submit([]hal.CommandBuffer{cb}); err != nil {
			t.Fatalf("Submit: %v", err)
		}

		glCtx := ctx.Lock()
		if glErr := glCtx.GetError(); glErr != 0 {
			t.Errorf("%s: GL error %#x", tc.format, glErr)
		}
		glCtx.BindBuffer(gl.COPY_READ_BUFFER, buf.(*Buffer).id)
		if p := glCtx.MapBufferRange(gl.COPY_READ_BUFFER, 0, 512, gl.MAP_READ_BIT); p != 0 {
			got := unsafe.Slice(*(**byte)(unsafe.Pointer(&p)), 512)
			for y := range 2 {
				if g, w := got[y*256:y*256+row], tc.texels[y*row:(y+1)*row]; !bytes.Equal(g, w) {
					t.Errorf("%s: row %d read % x, want % x", tc.format, y, g, w)
				}
			}
			glCtx.UnmapBuffer(gl.COPY_READ_BUFFER)
		} else {
			t.Errorf("%s: glMapBufferRange failed", tc.format)
		}
		glCtx.BindBuffer(gl.COPY_READ_BUFFER, 0)
		ctx.Unlock()
		dev.DestroyBuffer(buf)
		dev.DestroyTexture(tex)
	}
}

// seq is the bytes 1, 2, ... n.
func seq(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i + 1)
	}
	return b
}

// An error an earlier call left doesn't fail the next texture: its checks
// read glGetError, which returns the oldest error, so one stray error made
// CreateTexture fail with "TexImage2D failed".
func TestCreateTextureIgnoresEarlierErrors(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	dev := openTestDevice(t, ctx)
	t.Cleanup(dev.Destroy)
	glCtx := ctx.Lock()
	glCtx.Enable(0xffff) // GL_INVALID_ENUM, left behind
	ctx.Unlock()
	tex, err := dev.CreateTexture(&hal.TextureDescriptor{
		Size: hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatRGBA8Unorm,
		Usage: gputypes.TextureUsageTextureBinding | gputypes.TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	dev.DestroyTexture(tex)
}
