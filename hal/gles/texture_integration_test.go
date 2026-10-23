// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build integration && linux && !(js && wasm)

package gles

import (
	"bytes"
	"errors"
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

// An adapter made without a surface has the capabilities of its context,
// and its device writes the GLSL of that context, GLSL ES on an ES one.
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

// A pipeline without a fragment stage, as a depth-only pass has, links on
// ES, which links no program without a fragment shader.
func TestVertexOnlyPipelineOnES(t *testing.T) {
	ctx := newTestESAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	dev := openTestDevice(t, ctx)
	t.Cleanup(dev.Destroy)
	module, err := dev.CreateShaderModule(&hal.ShaderModuleDescriptor{Source: hal.ShaderSource{WGSL: `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(vi), 0.0, 0.0, 1.0);
}`}})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer dev.DestroyShaderModule(module)
	layout, err := dev.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer dev.DestroyPipelineLayout(layout)
	p, err := dev.CreateRenderPipeline(&hal.RenderPipelineDescriptor{
		Layout:    layout,
		Vertex:    hal.VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		DepthStencil: &hal.DepthStencilState{Format: gputypes.TextureFormatDepth32Float, DepthWriteEnabled: true,
			DepthCompare: gputypes.CompareFunctionLess},
		Multisample: gputypes.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	dev.DestroyRenderPipeline(p)
}

// The limits are in WebGPU's units: inter-stage variables of four
// components, where GL counts components, and the workgroup sizes the
// context has, where ES 3.1 guarantees fewer than desktop GL.
func TestLimitsInWebGPUUnits(t *testing.T) {
	for _, tc := range []struct {
		name string
		ctx  func(*testing.T) *AdapterContext
	}{{"desktop", newTestAdapterContext}, {"es", newTestESAdapterContext}} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := tc.ctx(t)
			t.Cleanup(ctx.Destroy)
			glCtx := ctx.Lock()
			defer ctx.Unlock()
			caps := queryAdapterCapabilities(glCtx)
			var components int32
			glCtx.GetIntegerv(gl.MAX_VARYING_COMPONENTS, &components)
			if got := caps.Limits.MaxInterStageShaderVariables; components > 0 && got != uint32(components/4) {
				t.Errorf("MaxInterStageShaderVariables %d for %d components", got, components)
			}
			if !caps.compute {
				return
			}
			for i, got := range []uint32{caps.Limits.MaxComputeWorkgroupSizeX, caps.Limits.MaxComputeWorkgroupSizeY, caps.Limits.MaxComputeWorkgroupSizeZ} {
				if want := glCtx.GetIntegeri(gl.MAX_COMPUTE_WORK_GROUP_SIZE, uint32(i)); got != uint32(want) {
					t.Errorf("workgroup size %d is %d, the context has %d", i, got, want)
				}
			}
		})
	}
}

// A WriteTexture given fewer bytes than its rows take is refused: GL read
// on past the data into whatever memory followed it.
func TestWriteTextureRefusesShortData(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	dev := openTestDevice(t, ctx)
	t.Cleanup(dev.Destroy)
	tex, err := dev.CreateTexture(&hal.TextureDescriptor{
		Size: hal.Extent3D{Width: 64, Height: 64, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatRGBA8Unorm, Usage: gputypes.TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer dev.DestroyTexture(tex)
	q := &Queue{ctx: ctx}
	size := hal.Extent3D{Width: 64, Height: 64, DepthOrArrayLayers: 1}
	if err := q.WriteTexture(&hal.ImageCopyTexture{Texture: tex}, make([]byte, 4096), &hal.ImageDataLayout{BytesPerRow: 256}, &size); err == nil {
		t.Error("a write of 16384 bytes from 4096 went through")
	}
	if err := q.WriteTexture(&hal.ImageCopyTexture{Texture: tex}, make([]byte, 64), &hal.ImageDataLayout{Offset: 100, BytesPerRow: 256}, &size); err == nil {
		t.Error("a write from past the end of its data went through")
	}
}

// A write into the stencil of a combined depth/stencil texture keeps its
// depth: GL writes that stencil only together with the depth, which is read
// and written back. A Stencil8 texture is one where the context has no
// stencil-only textures. ES, which reads neither back, refuses the write.
func TestStencilWriteKeepsDepth(t *testing.T) {
	for _, tc := range []struct {
		format     gputypes.TextureFormat
		aspect     gputypes.TextureAspect
		noStencil8 bool
	}{
		{gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureAspectStencilOnly, false},
		{gputypes.TextureFormatDepth32FloatStencil8, gputypes.TextureAspectStencilOnly, false},
		{gputypes.TextureFormatStencil8, gputypes.TextureAspectAll, true},
	} {
		t.Run(tc.format.String(), func(t *testing.T) {
			ctx := newTestAdapterContext(t)
			t.Cleanup(ctx.Destroy)
			dev := openTestDevice(t, ctx)
			if tc.noStencil8 {
				dev.formats.stencil8 = false
			}
			const side = 8
			tex, err := dev.CreateTexture(&hal.TextureDescriptor{
				Size: hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
				Dimension: gputypes.TextureDimension2D, Format: tc.format,
				Usage: gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopyDst | gputypes.TextureUsageCopySrc,
			})
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			t.Cleanup(func() { dev.DestroyTexture(tex) })
			glTex := tex.(*Texture)
			texel := int(pixelSize(glTex.glf.format, glTex.glf.typ))
			stencilAt := 0
			if texel == 8 {
				stencilAt = 4
			}
			read := func() []byte {
				glCtx := ctx.Lock()
				defer ctx.Unlock()
				pixels := make([]byte, side*side*texel)
				withFramebuffer(glCtx, gl.READ_FRAMEBUFFER, func() {
					attach(glCtx, gl.READ_FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, fbAttachment{tex: glTex})
					glCtx.ReadPixels(0, 0, side, side, glTex.glf.format, glTex.glf.typ, unsafe.Pointer(&pixels[0]))
				})
				return pixels
			}
			glCtx := ctx.Lock()
			withFramebuffer(glCtx, gl.FRAMEBUFFER, func() {
				attach(glCtx, gl.FRAMEBUFFER, gl.DEPTH_STENCIL_ATTACHMENT, fbAttachment{tex: glTex})
				glCtx.Disable(gl.SCISSOR_TEST)
				glCtx.DepthMask(true)
				glCtx.ClearDepth(0.25)
				glCtx.Clear(gl.DEPTH_BUFFER_BIT)
			})
			ctx.Unlock()
			before := read()

			stencil := make([]byte, side*side)
			for i := range stencil {
				stencil[i] = byte(3*i + 1)
			}
			q := &Queue{ctx: ctx}
			if err := q.WriteTexture(&hal.ImageCopyTexture{Texture: tex, Aspect: tc.aspect}, stencil,
				&hal.ImageDataLayout{BytesPerRow: side}, &hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}); err != nil {
				t.Fatalf("WriteTexture: %v", err)
			}
			after := read()
			for i := range side * side {
				for b := range texel {
					want := before[i*texel+b]
					if b == stencilAt {
						want = stencil[i]
					}
					if got := after[i*texel+b]; got != want && (texel == 4 || b < 5) {
						t.Fatalf("texel %d byte %d = %#x, want %#x (before % x, after % x)", i, b, got, want, before[i*texel:(i+1)*texel], after[i*texel:(i+1)*texel])
					}
				}
			}
		})
	}
	t.Run("es", func(t *testing.T) {
		ctx := newTestESAdapterContext(t)
		t.Cleanup(ctx.Destroy)
		dev := openTestDevice(t, ctx)
		tex, err := dev.CreateTexture(&hal.TextureDescriptor{
			Size: hal.Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
			Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatDepth24PlusStencil8,
			Usage: gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopyDst,
		})
		if err != nil {
			t.Fatalf("CreateTexture: %v", err)
		}
		t.Cleanup(func() { dev.DestroyTexture(tex) })
		err = (&Queue{ctx: ctx}).WriteTexture(&hal.ImageCopyTexture{Texture: tex, Aspect: gputypes.TextureAspectStencilOnly}, make([]byte, 64),
			&hal.ImageDataLayout{BytesPerRow: 8}, &hal.Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1})
		if !errors.Is(err, errStencilWriteES) {
			t.Errorf("WriteTexture on ES: %v, want errStencilWriteES", err)
		}
	})
}

// sampleLevel draws mip level mip of tex, a 2D texture, into an RGBA8 target
// of its size with a texelFetch per texel and returns the pixels.
func sampleLevel(t *testing.T, glCtx *gl.Context, tex *Texture, mip uint32) []byte {
	t.Helper()
	program := glCtx.CreateProgram()
	defer glCtx.DeleteProgram(program)
	for stage, source := range map[uint32]string{
		gl.VERTEX_SHADER: "#version 330 core\nvoid main() {\n" +
			"gl_Position = vec4(float((gl_VertexID & 1) * 4 - 1), float((gl_VertexID >> 1) * 4 - 1), 0.0, 1.0);\n}\n",
		gl.FRAGMENT_SHADER: "#version 330 core\nuniform sampler2D tex;\nuniform int lod;\nout vec4 color;\n" +
			"void main() {\ncolor = texelFetch(tex, ivec2(gl_FragCoord.xy), lod);\n}\n",
	} {
		shader, err := compileGLSL(glCtx, stage, "test", source)
		if err != nil {
			t.Fatal(err)
		}
		glCtx.AttachShader(program, shader)
		defer glCtx.DeleteShader(shader)
	}
	glCtx.LinkProgram(program)
	vao := glCtx.GenVertexArrays(1)
	defer glCtx.DeleteVertexArrays(vao)
	width, height := tex.mipSize(mip)
	target := glCtx.GenTextures(1)
	defer glCtx.DeleteTextures(target)
	glCtx.BindTexture(gl.TEXTURE_2D, target)
	glCtx.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, int32(width), int32(height), 0, gl.RGBA, gl.UNSIGNED_BYTE, 0)
	pixels := make([]byte, width*height*4)
	withFramebuffer(glCtx, gl.FRAMEBUFFER, func() {
		glCtx.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0)
		glCtx.Viewport(0, 0, int32(width), int32(height))
		glCtx.Disable(gl.SCISSOR_TEST)
		glCtx.BindVertexArray(vao)
		glCtx.UseProgram(program)
		glCtx.ActiveTexture(gl.TEXTURE0)
		glCtx.BindTexture(gl.TEXTURE_2D, tex.id)
		glCtx.BindSampler(0, 0)
		glCtx.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_NEAREST)
		glCtx.Uniform1i(glCtx.GetUniformLocation(program, "tex"), 0)
		glCtx.Uniform1i(glCtx.GetUniformLocation(program, "lod"), int32(mip))
		glCtx.DrawArrays(gl.TRIANGLES, 0, 3)
		glCtx.ReadPixels(0, 0, int32(width), int32(height), gl.RGBA, gl.UNSIGNED_BYTE, unsafe.Pointer(&pixels[0]))
	})
	glCtx.BindTexture(gl.TEXTURE_2D, 0)
	glCtx.BindVertexArray(0)
	glCtx.UseProgram(0)
	return pixels
}

// zero sets every texel of a texture the queue doesn't zero to zero,
// whatever its memory held: each texture here is filled with something
// else first, through a framebuffer or an upload, then zeroed again.
func TestTextureZero(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	dev := openTestDevice(t, ctx)
	create := func(desc *hal.TextureDescriptor) *Texture {
		t.Helper()
		desc.Dimension, desc.SampleCount = gputypes.TextureDimension2D, max(desc.SampleCount, 1)
		tex, err := dev.CreateTexture(desc)
		if err != nil {
			t.Fatalf("CreateTexture(%s): %v", desc.Format, err)
		}
		t.Cleanup(func() { dev.DestroyTexture(tex) })
		return tex.(*Texture)
	}
	zeroAgain := func(tex *Texture) {
		glCtx := ctx.Lock()
		defer ctx.Unlock()
		if tex.target != gl.RENDERBUFFER {
			glCtx.BindTexture(tex.target, tex.id)
			defer glCtx.BindTexture(tex.target, 0)
		}
		tex.zero(glCtx)
		if glErr := glCtx.GetError(); glErr != 0 {
			t.Fatalf("GL error %#x zeroing %s", glErr, tex.format)
		}
	}

	t.Run("depth", func(t *testing.T) {
		tex := create(&hal.TextureDescriptor{
			Size: hal.Extent3D{Width: 16, Height: 16, DepthOrArrayLayers: 2}, MipLevelCount: 2,
			Format: gputypes.TextureFormatDepth32Float, Usage: gputypes.TextureUsageRenderAttachment,
		})
		glCtx := ctx.Lock()
		withFramebuffer(glCtx, gl.FRAMEBUFFER, func() {
			glCtx.DepthMask(true)
			glCtx.ClearDepth(0.75)
			for mip := range uint32(2) {
				for layer := range uint32(2) {
					attach(glCtx, gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, fbAttachment{tex: tex, mip: mip, layer: layer})
					glCtx.Clear(gl.DEPTH_BUFFER_BIT)
				}
			}
		})
		ctx.Unlock()
		zeroAgain(tex)
		glCtx = ctx.Lock()
		defer ctx.Unlock()
		for mip := range uint32(2) {
			for layer := range uint32(2) {
				side := int32(16 >> mip)
				depth := make([]float32, side*side)
				withFramebuffer(glCtx, gl.READ_FRAMEBUFFER, func() {
					attach(glCtx, gl.READ_FRAMEBUFFER, gl.DEPTH_ATTACHMENT, fbAttachment{tex: tex, mip: mip, layer: layer})
					glCtx.ReadPixels(0, 0, side, side, gl.DEPTH_COMPONENT, gl.FLOAT, unsafe.Pointer(&depth[0]))
				})
				for i, d := range depth {
					if d != 0 {
						t.Fatalf("mip %d layer %d: depth %d is %v, want 0", mip, layer, i, d)
					}
				}
			}
		}
	})

	t.Run("multisampled", func(t *testing.T) {
		tex := create(&hal.TextureDescriptor{
			Size: hal.Extent3D{Width: 16, Height: 16, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 4,
			Format: gputypes.TextureFormatRGBA8Uint, Usage: gputypes.TextureUsageRenderAttachment,
		})
		resolved := create(&hal.TextureDescriptor{
			Size: hal.Extent3D{Width: 16, Height: 16, DepthOrArrayLayers: 1}, MipLevelCount: 1,
			Format: gputypes.TextureFormatRGBA8Uint, Usage: gputypes.TextureUsageRenderAttachment,
		})
		glCtx := ctx.Lock()
		withFramebuffer(glCtx, gl.FRAMEBUFFER, func() {
			attach(glCtx, gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, fbAttachment{tex: tex})
			ab := [4]uint32{0xab, 0xab, 0xab, 0xab}
			glCtx.ClearBufferuiv(gl.COLOR, 0, &ab)
		})
		ctx.Unlock()
		zeroAgain(tex)
		glCtx = ctx.Lock()
		defer ctx.Unlock()
		pixels := make([]byte, 16*16*4)
		withFramebuffer(glCtx, gl.READ_FRAMEBUFFER, func() {
			attach(glCtx, gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, fbAttachment{tex: tex})
			withFramebuffer(glCtx, gl.DRAW_FRAMEBUFFER, func() {
				attach(glCtx, gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, fbAttachment{tex: resolved})
				glCtx.BlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, gl.COLOR_BUFFER_BIT, gl.NEAREST)
			})
			attach(glCtx, gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, fbAttachment{tex: resolved})
			glCtx.ReadPixels(0, 0, 16, 16, gl.RGBA_INTEGER, gl.UNSIGNED_BYTE, unsafe.Pointer(&pixels[0]))
		})
		if !bytes.Equal(pixels, make([]byte, len(pixels))) {
			t.Errorf("resolved pixels % x..., want zeros", pixels[:8])
		}
	})

	t.Run("BC1", func(t *testing.T) {
		glCtx := ctx.Lock()
		s3tc := queryAdapterCapabilities(glCtx).Extensions["GL_EXT_texture_compression_s3tc"]
		ctx.Unlock()
		if !s3tc || !dev.formats.texStorage {
			t.Skip("no S3TC textures")
		}
		const side, mips = 64, 7
		tex := create(&hal.TextureDescriptor{
			Size: hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}, MipLevelCount: mips,
			Format: gputypes.TextureFormatBC1RGBAUnorm, Usage: gputypes.TextureUsageTextureBinding | gputypes.TextureUsageCopyDst,
		})
		q := &Queue{ctx: ctx}
		for mip := range uint32(5) {
			w := uint32(side) >> mip
			blocks := bytes.Repeat([]byte{0xab}, int(w/4*8*(w/4)))
			if err := q.WriteTexture(&hal.ImageCopyTexture{Texture: tex, MipLevel: mip}, blocks,
				&hal.ImageDataLayout{BytesPerRow: w / 4 * 8}, &hal.Extent3D{Width: w, Height: w, DepthOrArrayLayers: 1}); err != nil {
				t.Fatalf("WriteTexture: %v", err)
			}
		}
		zeroAgain(tex)
		glCtx = ctx.Lock()
		defer ctx.Unlock()
		for mip := range uint32(mips) {
			pixels := sampleLevel(t, glCtx, tex, mip)
			for i := 0; i < len(pixels); i += 4 {
				if string(pixels[i:i+4]) != "\x00\x00\x00\xff" {
					t.Fatalf("mip %d texel %d samples % x, want opaque black", mip, i/4, pixels[i:i+4])
				}
			}
		}
	})
}

// A view of one layer, or of another dimension, of a texture the shaders
// bind is sampled through a texture view of its own, where the context has
// them: a 2D view of layer 1 of an array texture took the array texture,
// which a sampler2D reads nothing from. Without texture views a bind group
// refuses such a view to sample, and takes one layer of it to store to.
func TestLayerViewsSample(t *testing.T) {
	for _, tc := range []struct {
		name  string
		ctx   func(*testing.T) *AdapterContext
		views bool
	}{{"views", newTestAdapterContext, true}, {"no views", newTestAdapterContext, false}, {"es views", newTestESAdapterContext, true}} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := tc.ctx(t)
			t.Cleanup(ctx.Destroy)
			dev := openTestDevice(t, ctx)
			if tc.views && !dev.formats.textureView {
				t.Skip("no texture views on this context")
			}
			dev.formats.textureView = tc.views
			tex, err := dev.CreateTexture(&hal.TextureDescriptor{
				Size: hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 2}, MipLevelCount: 1, SampleCount: 1,
				Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatRGBA8Unorm,
				Usage: gputypes.TextureUsageTextureBinding | gputypes.TextureUsageStorageBinding,
			})
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			t.Cleanup(func() { dev.DestroyTexture(tex) })
			whole, err := dev.CreateTextureView(tex, &hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2DArray})
			if err != nil {
				t.Fatalf("CreateTextureView(array): %v", err)
			}
			t.Cleanup(func() { dev.DestroyTextureView(whole) })
			layer, err := dev.CreateTextureView(tex, &hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2D, BaseArrayLayer: 1, ArrayLayerCount: 1})
			if err != nil {
				t.Fatalf("CreateTextureView(layer 1): %v", err)
			}
			if whole.(*TextureView).id != 0 {
				t.Error("the view of the whole array has a texture view of its own")
			}
			lv := layer.(*TextureView)
			if got := lv.id != 0; got != tc.views {
				t.Errorf("the view of layer 1 has a texture view: %v, want %v", got, tc.views)
			}
			if tc.views && lv.target != gl.TEXTURE_2D {
				t.Errorf("its target is %#x, want TEXTURE_2D", lv.target)
			}

			group := func(entry gputypes.BindGroupLayoutEntry) error {
				t.Helper()
				entry.Binding, entry.Visibility = 0, gputypes.ShaderStageFragment
				bgl, err := dev.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{entry}})
				if err != nil {
					t.Fatalf("CreateBindGroupLayout: %v", err)
				}
				_, err = dev.CreateBindGroup(&hal.BindGroupDescriptor{Layout: bgl, Entries: []gputypes.BindGroupEntry{
					{Binding: 0, Resource: gputypes.TextureViewBinding{TextureView: layer.NativeHandle()}},
				}})
				return err
			}
			sampled := group(gputypes.BindGroupLayoutEntry{Texture: &gputypes.TextureBindingLayout{
				SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}})
			if (sampled == nil) != tc.views {
				t.Errorf("a bind group sampling layer 1: %v", sampled)
			}
			if err := group(gputypes.BindGroupLayoutEntry{StorageTexture: &gputypes.StorageTextureBindingLayout{
				Access: gputypes.StorageTextureAccessWriteOnly, Format: gputypes.TextureFormatRGBA8Unorm, ViewDimension: gputypes.TextureViewDimension2D}}); err != nil {
				t.Errorf("a bind group storing to layer 1: %v", err)
			}

			dev.DestroyTextureView(layer)
			if lv.id != 0 {
				t.Error("the texture view outlived DestroyTextureView")
			}
		})
	}
}

// mipStoreShader stores the texel of a sampled view under each texel of a
// storage view of another size, its channels turned one place (g, b, r, a).
const mipStoreShader = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
	let at = id.xy * textureDimensions(src) / textureDimensions(dst);
	textureStore(dst, vec2<i32>(id.xy), textureLoad(src, vec2<i32>(at), 0).gbra);
}
`

// A dispatch stores into the mip level of its storage view while it
// samples a view of another level of the same texture: into level 2 from
// level 0, then into level 1 from level 2, below the sampled level. A view
// of one level sampled through its texture set the texture's base and max
// level to it, and an image unit storing to a level outside them stores
// nothing; such a view is a texture view of its own now.
func TestStoreIntoAMipBesideASampledMip(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	dev := openTestDevice(t, ctx)
	if !dev.formats.textureView {
		t.Skip("no texture views on this context")
	}
	const side = 8
	raw, err := dev.CreateTexture(&hal.TextureDescriptor{
		Size: hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}, MipLevelCount: 3, SampleCount: 1,
		Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatRGBA8Unorm,
		Usage: gputypes.TextureUsageTextureBinding | gputypes.TextureUsageStorageBinding | gputypes.TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	t.Cleanup(func() { dev.DestroyTexture(raw) })
	tex := raw.(*Texture)
	q := &Queue{ctx: ctx}
	green := bytes.Repeat([]byte{0, 255, 0, 255}, side*side)
	if err := q.WriteTexture(&hal.ImageCopyTexture{Texture: tex}, green, &hal.ImageDataLayout{BytesPerRow: side * 4},
		&hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	level := func(mip uint32) hal.TextureView {
		t.Helper()
		v, err := dev.CreateTextureView(tex, &hal.TextureViewDescriptor{Dimension: gputypes.TextureViewDimension2D, BaseMipLevel: mip, MipLevelCount: 1})
		if err != nil {
			t.Fatalf("CreateTextureView(level %d): %v", mip, err)
		}
		t.Cleanup(func() { dev.DestroyTextureView(v) })
		return v
	}
	bgl, err := dev.CreateBindGroupLayout(&hal.BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: gputypes.ShaderStageCompute, Texture: &gputypes.TextureBindingLayout{
			SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 1, Visibility: gputypes.ShaderStageCompute, StorageTexture: &gputypes.StorageTextureBindingLayout{
			Access: gputypes.StorageTextureAccessWriteOnly, Format: gputypes.TextureFormatRGBA8Unorm, ViewDimension: gputypes.TextureViewDimension2D}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	pl, err := dev.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{BindGroupLayouts: []hal.BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	module, err := dev.CreateShaderModule(&hal.ShaderModuleDescriptor{Source: hal.ShaderSource{WGSL: mipStoreShader}})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(func() { dev.DestroyShaderModule(module) })
	pipe, err := dev.CreateComputePipeline(&hal.ComputePipelineDescriptor{Layout: pl, Compute: hal.ComputeState{Module: module, EntryPoint: "cs_main"}})
	if err != nil {
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	t.Cleanup(func() { dev.DestroyComputePipeline(pipe) })
	enc, err := dev.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	_ = enc.BeginEncoding("store into a mip level")
	for _, step := range []struct{ src, dst uint32 }{{0, 2}, {2, 1}} {
		group, err := dev.CreateBindGroup(&hal.BindGroupDescriptor{Layout: bgl, Entries: []gputypes.BindGroupEntry{
			{Binding: 0, Resource: gputypes.TextureViewBinding{TextureView: level(step.src).NativeHandle()}},
			{Binding: 1, Resource: gputypes.TextureViewBinding{TextureView: level(step.dst).NativeHandle()}},
		}})
		if err != nil {
			t.Fatalf("CreateBindGroup: %v", err)
		}
		pass := enc.BeginComputePass(&hal.ComputePassDescriptor{})
		pass.SetPipeline(pipe)
		pass.SetBindGroup(0, group, nil)
		pass.Dispatch(side>>step.dst, side>>step.dst, 1)
		pass.End()
		enc.TransitionTextures([]hal.TextureBarrier{{Texture: tex, Usage: hal.TextureUsageTransition{
			OldUsage: gputypes.TextureUsageStorageBinding, NewUsage: gputypes.TextureUsageTextureBinding}}})
	}
	cb, err := enc.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding: %v", err)
	}
	if _, err := q.Submit([]hal.CommandBuffer{cb}); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	glCtx := ctx.Lock()
	defer ctx.Unlock()
	for mip, want := range map[uint32][4]byte{1: {0, 0, 255, 255}, 2: {255, 0, 0, 255}} {
		n := side >> mip
		pixels := make([]byte, n*n*4)
		withFramebuffer(glCtx, gl.READ_FRAMEBUFFER, func() {
			attach(glCtx, gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, fbAttachment{tex: tex, mip: mip})
			glCtx.ReadPixels(0, 0, int32(n), int32(n), gl.RGBA, gl.UNSIGNED_BYTE, unsafe.Pointer(&pixels[0]))
		})
		for i := 0; i < len(pixels); i += 4 {
			if [4]byte(pixels[i:i+4]) != want {
				t.Fatalf("mip level %d texel %d = % x, want % x", mip, i/4, pixels[i:i+4], want)
			}
		}
	}
}
