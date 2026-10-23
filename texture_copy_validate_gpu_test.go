//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// The texture side of a copy is checked before the backend sees it: the
// region inside the mip level, a mip level the texture has, the buffer's
// usage and offset, copy-compatible formats, one aspect of a depth/stencil
// format that buffer copies take and the bytes a compressed copy reads.
func TestTextureCopiesAreChecked(t *testing.T) {
	device, h := newValidatedDevice(t)

	texture := func(format gputypes.TextureFormat, side, mips uint32) *Texture {
		tex, err := device.CreateTexture(&TextureDescriptor{
			Size: Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}, MipLevelCount: mips, SampleCount: 1,
			Dimension: TextureDimension2D, Format: format, Usage: TextureUsageCopySrc | TextureUsageCopyDst | TextureUsageRenderAttachment,
		})
		if err != nil {
			t.Fatalf("CreateTexture(%v): %v", format, err)
		}
		t.Cleanup(tex.Release)
		return tex
	}
	buffer := func(size uint64, usage gputypes.BufferUsage) *Buffer {
		b, err := device.CreateBuffer(&BufferDescriptor{Size: size, Usage: usage})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		t.Cleanup(b.Release)
		return b
	}
	rgba, other := texture(TextureFormatRGBA8Unorm, 4, 1), texture(TextureFormatRGBA8Unorm, 4, 1)
	half := texture(gputypes.TextureFormatRGBA16Float, 4, 1)
	depthStencil := texture(gputypes.TextureFormatDepth24PlusStencil8, 4, 1)
	depth24 := texture(gputypes.TextureFormatDepth24Plus, 4, 1)
	depth32 := texture(gputypes.TextureFormatDepth32Float, 4, 1)
	buf := buffer(1024, BufferUsageCopySrc|BufferUsageCopyDst)
	storage := buffer(1024, BufferUsageStorage|BufferUsageCopySrc)
	uniform := buffer(1024, BufferUsageUniform|BufferUsageCopyDst)

	region := func(x, y, mip uint32, aspect gputypes.TextureAspect, offset uint64, w, hgt uint32) []BufferTextureCopy {
		return []BufferTextureCopy{{
			BufferLayout: ImageDataLayout{Offset: offset, BytesPerRow: 256, RowsPerImage: hgt},
			TextureBase:  ImageCopyTexture{MipLevel: mip, Origin: Origin3D{X: x, Y: y}, Aspect: aspect},
			Size:         Extent3D{Width: w, Height: hgt, DepthOrArrayLayers: 1},
		}}
	}
	toTexture := func(x, y uint32) []TextureCopy {
		return []TextureCopy{{Destination: ImageCopyTexture{Origin: Origin3D{X: x, Y: y}}, Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}}}
	}
	all, depthOnly, stencilOnly := gputypes.TextureAspectAll, gputypes.TextureAspectDepthOnly, gputypes.TextureAspectStencilOnly
	for _, tc := range []struct {
		name    string
		record  func(e *CommandEncoder)
		wantErr string
	}{
		{"texture to buffer", func(e *CommandEncoder) { e.CopyTextureToBuffer(rgba, buf, region(2, 2, 0, all, 0, 2, 2)) }, ""},
		{"buffer to texture", func(e *CommandEncoder) { e.CopyBufferToTexture(buf, rgba, region(0, 0, 0, all, 4, 4, 4)) }, ""},
		{"texture to texture", func(e *CommandEncoder) { e.CopyTextureToTexture(rgba, other, toTexture(0, 0)) }, ""},
		{"stencil of a depth/stencil texture", func(e *CommandEncoder) {
			e.CopyTextureToBuffer(depthStencil, buf, region(0, 0, 0, stencilOnly, 0, 4, 4))
		}, ""},
		{"depth32float to a buffer", func(e *CommandEncoder) { e.CopyTextureToBuffer(depth32, buf, region(0, 0, 0, all, 0, 4, 4)) }, ""},
		{"past the right edge", func(e *CommandEncoder) { e.CopyTextureToBuffer(rgba, buf, region(3, 0, 0, all, 0, 4, 1)) }, "reaches past"},
		{"past the bottom edge", func(e *CommandEncoder) { e.CopyBufferToTexture(buf, rgba, region(0, 2, 0, all, 0, 4, 4)) }, "reaches past"},
		{"texture to texture past the edge", func(e *CommandEncoder) { e.CopyTextureToTexture(rgba, other, toTexture(2, 2)) }, "reaches past"},
		{"mip level the texture lacks", func(e *CommandEncoder) { e.CopyTextureToBuffer(rgba, buf, region(0, 0, 2, all, 0, 1, 1)) }, "mip level"},
		{"destination buffer without CopyDst", func(e *CommandEncoder) { e.CopyTextureToBuffer(rgba, storage, region(0, 0, 0, all, 0, 4, 4)) }, "CopyDst"},
		{"source buffer without CopySrc", func(e *CommandEncoder) { e.CopyBufferToTexture(uniform, rgba, region(0, 0, 0, all, 0, 4, 4)) }, "CopySrc"},
		{"buffer offset inside a texel", func(e *CommandEncoder) { e.CopyBufferToTexture(buf, rgba, region(0, 0, 0, all, 2, 4, 4)) }, "offset"},
		{"formats of different texels", func(e *CommandEncoder) { e.CopyTextureToTexture(rgba, half, toTexture(0, 0)) }, "formats"},
		{"both aspects to a buffer", func(e *CommandEncoder) { e.CopyTextureToBuffer(depthStencil, buf, region(0, 0, 0, all, 0, 4, 4)) }, "both aspects"},
		{"depth24plus to a buffer", func(e *CommandEncoder) { e.CopyTextureToBuffer(depth24, buf, region(0, 0, 0, depthOnly, 0, 4, 4)) }, "depth"},
		{"buffer to depth32float", func(e *CommandEncoder) { e.CopyBufferToTexture(buf, depth32, region(0, 0, 0, all, 0, 4, 4)) }, "only copies out"},
		{"part of a depth/stencil texture", func(e *CommandEncoder) {
			e.CopyTextureToBuffer(depthStencil, buf, region(0, 0, 0, stencilOnly, 0, 2, 2))
		}, "whole mip levels"},
		{"rows past the buffer", func(e *CommandEncoder) { e.CopyTextureToBuffer(rgba, buf, region(0, 0, 0, all, 512, 4, 4)) }, "past the buffer"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			tc.record(enc)
			cb, err := enc.Finish()
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("Finish: %v", err)
			case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
				t.Errorf("Finish: %v, want an error about %q", err, tc.wantErr)
			}
			if err == nil {
				if _, err := device.Queue().Submit(cb); err != nil {
					t.Fatalf("Submit: %v", err)
				}
			}
		})
	}

	assertNoValidation(t, h)
}

// A copy of a compressed texture is in whole blocks, and its rows are rows of
// blocks: 64x64 texels of BC1 need 32 rows of 128 bytes, not 256 bytes.
func TestCompressedTextureCopiesAreChecked(t *testing.T) {
	device, h := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureTextureCompressionBC))
	bc, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: 64, Height: 64, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: gputypes.TextureFormatBC1RGBAUnorm, Usage: TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer bc.Release()
	short, err := device.CreateBuffer(&BufferDescriptor{Size: 256, Usage: BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer short.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToTexture(short, bc, []BufferTextureCopy{{
		BufferLayout: ImageDataLayout{BytesPerRow: 256, RowsPerImage: 16},
		Size:         Extent3D{Width: 64, Height: 64, DepthOrArrayLayers: 1},
	}})
	if _, err := enc.Finish(); err == nil || !strings.Contains(err.Error(), "past the buffer") {
		t.Errorf("BC1 64x64 from 256 bytes: Finish = %v, want an error about the buffer", err)
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: bc}, make([]byte, 8),
		&ImageDataLayout{BytesPerRow: 8}, &Extent3D{Width: 2, Height: 2, DepthOrArrayLayers: 1}); err == nil || !strings.Contains(err.Error(), "blocks") {
		t.Errorf("WriteTexture of 2x2 texels of BC1 = %v, want an error about whole blocks", err)
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: bc}, make([]byte, 8),
		&ImageDataLayout{BytesPerRow: 8}, &Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}); err != nil {
		t.Errorf("WriteTexture of one BC1 block: %v", err)
	}
	assertNoValidation(t, h)
}

// A copy or texture write with nothing in it is valid and does nothing: it
// used to reach the backends, which Vulkan's rules forbid, DX12 failed the
// command buffer over and GL panicked on for empty data.
func TestEmptyTextureCopiesDoNothing(t *testing.T) {
	device, h := newValidatedDevice(t)
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageCopySrc | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	other, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageCopySrc | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer other.Release()
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: 1024, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()

	region := func(size Extent3D) []BufferTextureCopy {
		return []BufferTextureCopy{{BufferLayout: ImageDataLayout{BytesPerRow: 256}, Size: size}}
	}
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToTexture(buf, tex, region(Extent3D{Height: 4, DepthOrArrayLayers: 1}))
	enc.CopyTextureToBuffer(tex, buf, region(Extent3D{Width: 4, DepthOrArrayLayers: 1}))
	enc.CopyTextureToTexture(tex, other, []TextureCopy{{Size: Extent3D{Width: 4, Height: 4}}})
	finishAndSubmit(t, device, enc)

	q := device.Queue()
	if err := q.WriteTexture(&ImageCopyTexture{Texture: tex}, nil, &ImageDataLayout{}, &Extent3D{}); err != nil {
		t.Errorf("WriteTexture of nothing: %v", err)
	}
	if err := q.WriteTexture(&ImageCopyTexture{Texture: tex}, make([]byte, 64), &ImageDataLayout{BytesPerRow: 16},
		&Extent3D{Height: 4, DepthOrArrayLayers: 1}); err != nil {
		t.Errorf("WriteTexture of no columns: %v", err)
	}
	enc, err = device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	finishAndSubmit(t, device, enc)
	assertNoValidation(t, h)
}
