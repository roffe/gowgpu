//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/noop"
)

type copyCaptureEncoder struct {
	*noop.CommandEncoder
	bases []hal.ImageCopyTexture
}

func (e *copyCaptureEncoder) CopyTextureToBuffer(_ hal.Texture, _ hal.Buffer, regions []hal.BufferTextureCopy) {
	for _, r := range regions {
		e.bases = append(e.bases, r.TextureBase)
	}
}

func (e *copyCaptureEncoder) CopyBufferToTexture(_ hal.Buffer, _ hal.Texture, regions []hal.BufferTextureCopy) {
	for _, r := range regions {
		e.bases = append(e.bases, r.TextureBase)
	}
}

func (e *copyCaptureEncoder) CopyTextureToTexture(_, _ hal.Texture, regions []hal.TextureCopy) {
	for _, r := range regions {
		e.bases = append(e.bases, r.SrcBase, r.DstBase)
	}
}

// A copy region is on the textures the copy is given, whatever its Texture
// holds: one without a Texture neither panics nor loses its mip level,
// origin and aspect, and a copy from a buffer keeps the aspect too.
func TestCopyRegionsAreOnTheCopysTextures(t *testing.T) {
	enc := &copyCaptureEncoder{CommandEncoder: &noop.CommandEncoder{}}
	device := newCaptureDevice(&captureDevice{Device: &noop.Device{}, commandEncoder: enc})
	texture := func() *Texture {
		tex, err := device.CreateTexture(&TextureDescriptor{
			Size: Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 1}, MipLevelCount: 2, SampleCount: 1,
			Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageCopySrc | TextureUsageCopyDst,
		})
		if err != nil {
			t.Fatalf("CreateTexture: %v", err)
		}
		return tex
	}
	src, dst := texture(), texture()
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: 1024, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	base := ImageCopyTexture{MipLevel: 1, Origin: Origin3D{X: 1, Y: 2}, Aspect: gputypes.TextureAspectAll}
	size := Extent3D{Width: 2, Height: 2, DepthOrArrayLayers: 1}
	layout := ImageDataLayout{BytesPerRow: 256}

	encoder, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	encoder.CopyTextureToBuffer(src, buf, []BufferTextureCopy{{BufferLayout: layout, TextureBase: base, Size: size}})
	encoder.CopyBufferToTexture(buf, dst, []BufferTextureCopy{{BufferLayout: layout, TextureBase: base, Size: size}})
	encoder.CopyTextureToTexture(src, dst, []TextureCopy{{Source: base, Destination: base, Size: size}})
	if _, err := encoder.Finish(); err != nil {
		t.Fatalf("Finish: %v", err)
	}

	on := func(tex *Texture) hal.ImageCopyTexture {
		return hal.ImageCopyTexture{Texture: tex.resolveHAL(), MipLevel: 1, Origin: hal.Origin3D{X: 1, Y: 2}, Aspect: gputypes.TextureAspectAll}
	}
	want := []hal.ImageCopyTexture{on(src), on(dst), on(src), on(dst)}
	if len(enc.bases) != len(want) {
		t.Fatalf("HAL copies saw %d texture bases, want %d", len(enc.bases), len(want))
	}
	for i := range want {
		if enc.bases[i] != want[i] {
			t.Errorf("texture base %d = %+v, want %+v", i, enc.bases[i], want[i])
		}
	}
}

// A copy of all of a depth-only or stencil-only texture reaches the backend
// as a copy of its one aspect, which Vulkan otherwise took for color.
func TestCopiesOfOneAspectFormatsNameTheAspect(t *testing.T) {
	enc := &copyCaptureEncoder{CommandEncoder: &noop.CommandEncoder{}}
	device := newCaptureDevice(&captureDevice{Device: &noop.Device{}, commandEncoder: enc})
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: 1024, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	encoder, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	formats := []gputypes.TextureFormat{gputypes.TextureFormatDepth32Float, gputypes.TextureFormatStencil8}
	for _, f := range formats {
		tex, err := device.CreateTexture(&TextureDescriptor{
			Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
			Dimension: TextureDimension2D, Format: f, Usage: TextureUsageCopySrc,
		})
		if err != nil {
			t.Fatalf("CreateTexture(%v): %v", f, err)
		}
		encoder.CopyTextureToBuffer(tex, buf, []BufferTextureCopy{{
			BufferLayout: ImageDataLayout{BytesPerRow: 256},
			TextureBase:  ImageCopyTexture{Aspect: gputypes.TextureAspectAll},
			Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		}})
	}
	if _, err := encoder.Finish(); err != nil {
		t.Fatalf("Finish: %v", err)
	}
	want := []gputypes.TextureAspect{gputypes.TextureAspectDepthOnly, gputypes.TextureAspectStencilOnly}
	if len(enc.bases) != len(want) {
		t.Fatalf("HAL copies saw %d texture bases, want %d", len(enc.bases), len(want))
	}
	for i, w := range want {
		if got := enc.bases[i].Aspect; got != w {
			t.Errorf("%v copy aspect = %v, want %v", formats[i], got, w)
		}
	}
}
