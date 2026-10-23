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

func testTexture(format gputypes.TextureFormat, dimension gputypes.TextureDimension, width, height, depth, mips uint32) *Texture {
	return &Texture{format: format, dimension: dimension, size: hal.Extent3D{Width: width, Height: height, DepthOrArrayLayers: depth}, mipLevels: mips}
}

func TestPlanBufferTextureCopiesPlacementAndArraySlices(t *testing.T) {
	texture := testTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 128, 16, 3, 2)
	region := hal.ImageCopyTexture{MipLevel: 1, Origin: hal.Origin3D{X: 4, Y: 2, Z: 1}, Aspect: gputypes.TextureAspectAll}
	size := hal.Extent3D{Width: 8, Height: 2, DepthOrArrayLayers: 2}
	plans := planBufferTextureCopies(texture, region, hal.ImageDataLayout{Offset: 0, BytesPerRow: 256, RowsPerImage: 4}, size)
	if len(plans) != 2 {
		t.Fatalf("got %d plans, want 2", len(plans))
	}
	for i, plan := range plans {
		if plan.bufferOffset%512 != 0 {
			t.Errorf("plan %d offset %d is not 512-aligned", i, plan.bufferOffset)
		}
		if plan.subresource != texture.subresourceIndex(1, 1+uint32(i)) {
			t.Errorf("plan %d subresource %d", i, plan.subresource)
		}
		if plan.textureOriginX != region.Origin.X || plan.textureOriginY != region.Origin.Y || plan.textureOriginZ != 0 {
			t.Errorf("plan %d texture origin = (%d,%d,%d)", i, plan.textureOriginX, plan.textureOriginY, plan.textureOriginZ)
		}
	}
}

func TestPlanBufferTextureCopiesReconstructsPlacedDelta(t *testing.T) {
	texture := testTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 128, 64, 1, 1)
	region := hal.ImageCopyTexture{Origin: hal.Origin3D{}, Aspect: gputypes.TextureAspectAll}
	for _, tc := range []struct {
		name   string
		offset uint64
		pitch  uint32
	}{
		{name: "zero", offset: 0, pitch: 256},
		{name: "one-block", offset: 4, pitch: 256},
		{name: "256", offset: 256, pitch: 768},
		{name: "512", offset: 512, pitch: 256},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plans := planBufferTextureCopies(texture, region, hal.ImageDataLayout{Offset: tc.offset, BytesPerRow: tc.pitch, RowsPerImage: 2}, hal.Extent3D{Width: 8, Height: 2, DepthOrArrayLayers: 1})
			if len(plans) != 1 {
				t.Fatalf("got %d plans", len(plans))
			}
			plan := plans[0]
			logical := plan.bufferOffset + uint64(plan.bufferOriginY)*uint64(plan.rowPitch)/uint64(1) + uint64(plan.bufferOriginX)*4
			if logical != tc.offset {
				t.Fatalf("reconstructed offset %d, want %d", logical, tc.offset)
			}
		})
	}
}

func TestPlanBufferTextureCopiesBCAndRejectsUnrepresentableDelta(t *testing.T) {
	texture := testTexture(gputypes.TextureFormatBC1RGBAUnorm, gputypes.TextureDimension2D, 64, 64, 1, 1)
	region := hal.ImageCopyTexture{Aspect: gputypes.TextureAspectAll}
	plans := planBufferTextureCopies(texture, region, hal.ImageDataLayout{Offset: 256, BytesPerRow: 512, RowsPerImage: 4}, hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1})
	if len(plans) != 1 {
		t.Fatalf("got %d BC plans", len(plans))
	}
	if plans[0].bufferOriginX != 128 || plans[0].footprintWidth < 132 {
		t.Fatalf("BC plan origin/footprint = (%d,%d)", plans[0].bufferOriginX, plans[0].footprintWidth)
	}
	plan := plans[0]
	reconstructed := plan.bufferOffset + uint64(plan.bufferOriginY/4)*uint64(plan.rowPitch) + uint64(plan.bufferOriginX/4)*8
	if reconstructed != 256 {
		t.Fatalf("BC reconstructed offset %d, want 256", reconstructed)
	}
	if got := planBufferTextureCopies(texture, region, hal.ImageDataLayout{Offset: 4, BytesPerRow: 256, RowsPerImage: 4}, hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}); got != nil {
		t.Fatal("expected byte delta not divisible by BC block size to be rejected")
	}
}

func TestPlanBufferTextureCopiesZeroPitchUsesNativeAlignment(t *testing.T) {
	rgba := testTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 8, 8, 1, 1)
	rgbaPlans := planBufferTextureCopies(rgba, hal.ImageCopyTexture{Aspect: gputypes.TextureAspectAll}, hal.ImageDataLayout{}, hal.Extent3D{Width: 8, Height: 1, DepthOrArrayLayers: 1})
	if len(rgbaPlans) != 1 || rgbaPlans[0].rowPitch != 256 {
		t.Fatalf("RGBA zero-pitch plan = %#v, want one 256-byte row", rgbaPlans)
	}
	bc := testTexture(gputypes.TextureFormatBC1RGBAUnorm, gputypes.TextureDimension2D, 8, 8, 1, 1)
	bcPlans := planBufferTextureCopies(bc, hal.ImageCopyTexture{Aspect: gputypes.TextureAspectAll}, hal.ImageDataLayout{}, hal.Extent3D{Width: 8, Height: 4, DepthOrArrayLayers: 1})
	if len(bcPlans) != 1 || bcPlans[0].rowPitch != 256 {
		t.Fatalf("BC zero-pitch plan = %#v, want one 256-byte block row", bcPlans)
	}
	if planBufferTextureCopies(rgba, hal.ImageCopyTexture{Aspect: gputypes.TextureAspectAll}, hal.ImageDataLayout{}, hal.Extent3D{Width: 8, Height: 2, DepthOrArrayLayers: 1}) != nil {
		t.Fatal("zero BytesPerRow must remain invalid for multi-row RGBA copies")
	}
}

func TestTextureWriteSourceOffsetUsesBlockRows(t *testing.T) {
	// BC1's four texel rows occupy one source block row. A two-layer upload
	// with RowsPerImage=4 must therefore advance by one row pitch per layer.
	if got := textureWriteSourceOffset(16, 256, 1, 1, 0); got != 272 {
		t.Fatalf("BC layer offset = %d, want 272", got)
	}
	if got := textureWriteSourceOffset(16, 256, 2, 1, 0); got != 528 {
		t.Fatalf("two block-row layer offset = %d, want 528", got)
	}
}

func TestWriteTextureNativeLayoutRepacksUnalignedSourcePitch(t *testing.T) {
	texture := testTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 64, 8, 1, 1)
	_, native, sourceBPR, _, ok := writeTextureNativeLayout(texture, gputypes.TextureAspectAll, hal.ImageDataLayout{BytesPerRow: 300, RowsPerImage: 2}, hal.Extent3D{Width: 64, Height: 2, DepthOrArrayLayers: 1})
	if !ok || native.BytesPerRow != 256 || sourceBPR != 300 {
		t.Fatalf("native layout = %#v, source BPR = %d, ok=%v", native, sourceBPR, ok)
	}
}

func TestWriteTextureNativeLayoutKeepsSourcePaddingOutOfStaging(t *testing.T) {
	texture := testTexture(gputypes.TextureFormatBC1RGBAUnorm, gputypes.TextureDimension2D, 8, 8, 2, 1)
	_, native, _, sourceBlockRows, ok := writeTextureNativeLayout(texture, gputypes.TextureAspectAll, hal.ImageDataLayout{BytesPerRow: 300, RowsPerImage: 8}, hal.Extent3D{Width: 8, Height: 4, DepthOrArrayLayers: 2})
	if !ok || sourceBlockRows != 2 || native.RowsPerImage != 4 {
		t.Fatalf("native layout = %#v, source block rows = %d, ok=%v", native, sourceBlockRows, ok)
	}
}

func TestPlanBufferTextureCopiesPadded3DUsesAbsoluteDepth(t *testing.T) {
	texture := testTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension3D, 32, 32, 8, 1)
	region := hal.ImageCopyTexture{Origin: hal.Origin3D{X: 2, Y: 3, Z: 2}, Aspect: gputypes.TextureAspectAll}
	plans := planBufferTextureCopies(texture, region, hal.ImageDataLayout{BytesPerRow: 256, RowsPerImage: 8}, hal.Extent3D{Width: 4, Height: 2, DepthOrArrayLayers: 3})
	if len(plans) != 3 {
		t.Fatalf("got %d plans", len(plans))
	}
	for i, plan := range plans {
		if plan.subresource != 0 || plan.textureOriginZ != 2+uint32(i) {
			t.Errorf("plan %d subresource/depth = %d/%d", i, plan.subresource, plan.textureOriginZ)
		}
	}
}

func TestPlanTextureTextureCopiesArraysAnd3D(t *testing.T) {
	array := testTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 32, 32, 4, 2)
	arrayCopy := hal.TextureCopy{
		SrcBase: hal.ImageCopyTexture{MipLevel: 1, Origin: hal.Origin3D{Z: 1}, Aspect: gputypes.TextureAspectAll},
		DstBase: hal.ImageCopyTexture{MipLevel: 0, Origin: hal.Origin3D{Z: 2}, Aspect: gputypes.TextureAspectAll},
		Size:    hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 2},
	}
	if plans := planTextureTextureCopies(array, array, arrayCopy); len(plans) != 2 || plans[0].srcFront != 0 || plans[0].dstZ != 0 {
		t.Fatalf("array plans = %#v", plans)
	}
	volume := testTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension3D, 32, 32, 8, 1)
	volumeCopy := hal.TextureCopy{SrcBase: hal.ImageCopyTexture{Origin: hal.Origin3D{Z: 2}, Aspect: gputypes.TextureAspectAll}, DstBase: hal.ImageCopyTexture{Origin: hal.Origin3D{Z: 3}, Aspect: gputypes.TextureAspectAll}, Size: hal.Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 3}}
	plans := planTextureTextureCopies(volume, volume, volumeCopy)
	if len(plans) != 1 || plans[0].srcFront != 2 || plans[0].srcBack != 5 || plans[0].dstZ != 3 {
		t.Fatalf("3D plans = %#v", plans)
	}
	volumeCopy.Size.DepthOrArrayLayers = 6
	if planTextureTextureCopies(volume, volume, volumeCopy) != nil {
		t.Fatal("expected absolute 3D bounds to reject an oversized copy")
	}
}

func TestTextureViewMipExtentIsAtLeastOneTexel(t *testing.T) {
	view := &TextureView{texture: testTexture(gputypes.TextureFormatRGBA8Unorm, gputypes.TextureDimension2D, 8, 2, 1, 4), baseMip: 3}
	if w, h := view.mipExtent(); w != 1 || h != 1 {
		t.Fatalf("mip 3 of 8x2 = %dx%d, want 1x1", w, h)
	}
	view.baseMip = 1
	if w, h := view.mipExtent(); w != 4 || h != 1 {
		t.Fatalf("mip 1 of 8x2 = %dx%d, want 4x1", w, h)
	}
}

func TestTextureCopyFormatIsThePlanesOwn(t *testing.T) {
	tests := []struct {
		name   string
		format gputypes.TextureFormat
		plane  uint32
		expect d3d12.DXGI_FORMAT
		bytes  uint32
	}{
		{"colour", gputypes.TextureFormatRGBA8Unorm, 0, d3d12.DXGI_FORMAT_R8G8B8A8_UNORM, 4},
		{"depth16", gputypes.TextureFormatDepth16Unorm, 0, d3d12.DXGI_FORMAT_R16_UNORM, 2},
		{"depth32", gputypes.TextureFormatDepth32Float, 0, d3d12.DXGI_FORMAT_R32_FLOAT, 4},
		{"depth32 of depth32 stencil8", gputypes.TextureFormatDepth32FloatStencil8, 0, d3d12.DXGI_FORMAT_R32_FLOAT, 4},
		{"stencil of depth32 stencil8", gputypes.TextureFormatDepth32FloatStencil8, 1, d3d12.DXGI_FORMAT_R8_UINT, 1},
		{"stencil of depth24 stencil8", gputypes.TextureFormatDepth24PlusStencil8, 1, d3d12.DXGI_FORMAT_R8_UINT, 1},
		{"stencil8", gputypes.TextureFormatStencil8, 1, d3d12.DXGI_FORMAT_R8_UINT, 1},
		{"depth of depth24 stencil8", gputypes.TextureFormatDepth24PlusStencil8, 0, d3d12.DXGI_FORMAT_UNKNOWN, 0},
		{"depth24", gputypes.TextureFormatDepth24Plus, 0, d3d12.DXGI_FORMAT_UNKNOWN, 0},
		{"stencil of depth32", gputypes.TextureFormatDepth32Float, 1, d3d12.DXGI_FORMAT_UNKNOWN, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			format, bytes := textureCopyFormat(tt.format, tt.plane)
			if format != tt.expect || bytes != tt.bytes {
				t.Fatalf("textureCopyFormat = (%d, %d), want (%d, %d)", format, bytes, tt.expect, tt.bytes)
			}
		})
	}
}

func TestPlanBufferTextureCopiesDepthStencilPlanes(t *testing.T) {
	texture := testTexture(gputypes.TextureFormatDepth32FloatStencil8, gputypes.TextureDimension2D, 64, 4, 1, 1)
	size := hal.Extent3D{Width: 64, Height: 4, DepthOrArrayLayers: 1}

	depth := planBufferTextureCopies(texture, hal.ImageCopyTexture{Aspect: gputypes.TextureAspectDepthOnly}, hal.ImageDataLayout{BytesPerRow: 256}, size)
	if len(depth) != 1 || depth[0].plane != 0 || depth[0].subresource != 0 || depth[0].rowPitch != 256 {
		t.Fatalf("depth plans = %#v, want one plane-0 plan of 256-byte rows", depth)
	}
	stencil := planBufferTextureCopies(texture, hal.ImageCopyTexture{Aspect: gputypes.TextureAspectStencilOnly}, hal.ImageDataLayout{BytesPerRow: 256}, size)
	if len(stencil) != 1 || stencil[0].plane != 1 || stencil[0].subresource != 1 {
		t.Fatalf("stencil plans = %#v, want one plane-1 plan", stencil)
	}
	if planBufferTextureCopies(texture, hal.ImageCopyTexture{Aspect: gputypes.TextureAspectAll}, hal.ImageDataLayout{BytesPerRow: 256}, size) != nil {
		t.Fatal("both planes of a depth/stencil texture cannot be one copy")
	}
	if planBufferTextureCopies(texture, hal.ImageCopyTexture{Aspect: gputypes.TextureAspectDepthOnly}, hal.ImageDataLayout{BytesPerRow: 128}, size) != nil {
		t.Fatal("a 128-byte row cannot hold 64 depth values")
	}

	stencil8 := testTexture(gputypes.TextureFormatStencil8, gputypes.TextureDimension2D, 64, 4, 1, 1)
	if plans := planBufferTextureCopies(stencil8, hal.ImageCopyTexture{Aspect: gputypes.TextureAspectAll}, hal.ImageDataLayout{BytesPerRow: 256}, size); len(plans) != 1 || plans[0].plane != 1 {
		t.Fatalf("stencil8 plans = %#v, want one plane-1 plan", plans)
	}
	depth24 := testTexture(gputypes.TextureFormatDepth24Plus, gputypes.TextureDimension2D, 64, 4, 1, 1)
	if planBufferTextureCopies(depth24, hal.ImageCopyTexture{Aspect: gputypes.TextureAspectDepthOnly}, hal.ImageDataLayout{BytesPerRow: 256}, size) != nil {
		t.Fatal("24-bit depth has no buffer copy")
	}
}

// TestPlanBufferTextureCopiesSplitsRowsPastThePitch checks, for every
// block-aligned offset into a 512-byte window, that the plans of a copy
// place each texel at the byte the WebGPU layout puts it at, cover each
// texel once and keep their footprints at 512-aligned offsets inside the
// row pitch. Rows that run past the pitch from the aligned offset are split
// in two copies.
func TestPlanBufferTextureCopiesSplitsRowsPastThePitch(t *testing.T) {
	for _, tc := range []struct {
		format  gputypes.TextureFormat
		pitch   uint32
		w, h    uint32
		originX uint32
	}{
		{gputypes.TextureFormatRGBA8Unorm, 512, 128, 16, 0},
		{gputypes.TextureFormatRGBA8Unorm, 256, 64, 4, 0},
		{gputypes.TextureFormatRGBA8Unorm, 768, 192, 3, 0},
		{gputypes.TextureFormatRGBA8Unorm, 768, 150, 3, 7},
		{gputypes.TextureFormatR8Unorm, 256, 256, 5, 0},
		{gputypes.TextureFormatRGBA32Float, 1280, 80, 2, 3},
		{gputypes.TextureFormatBC1RGBAUnorm, 256, 128, 8, 4},
	} {
		texture := testTexture(tc.format, gputypes.TextureDimension2D, 512, 64, 1, 1)
		info, _ := textureBlockInfoForPlane(tc.format, 0)
		for base := uint64(3072); base < 3072+512; base += uint64(info.bytes) {
			region := hal.ImageCopyTexture{Origin: hal.Origin3D{X: tc.originX}, Aspect: gputypes.TextureAspectAll}
			layout := hal.ImageDataLayout{Offset: base, BytesPerRow: tc.pitch, RowsPerImage: tc.h}
			plans := planBufferTextureCopies(texture, region, layout, hal.Extent3D{Width: tc.w, Height: tc.h, DepthOrArrayLayers: 1})
			if plans == nil {
				t.Fatalf("%v pitch %d offset %d: no plan", tc.format, tc.pitch, base)
			}
			if len(plans) > 2 {
				t.Fatalf("%v pitch %d offset %d: %d plans", tc.format, tc.pitch, base, len(plans))
			}
			covered := make(map[[2]uint32]bool)
			for _, p := range plans {
				if p.bufferOffset%512 != 0 || uint64(p.footprintWidth/info.width)*uint64(info.bytes) > uint64(tc.pitch) {
					t.Fatalf("%v pitch %d offset %d: footprint at %d is %d texels wide", tc.format, tc.pitch, base, p.bufferOffset, p.footprintWidth)
				}
				for y := uint32(0); y < p.copyHeight; y += info.height {
					for x := uint32(0); x < p.copyWidth; x += info.width {
						texel := [2]uint32{p.textureOriginX + x - tc.originX, y}
						if covered[texel] {
							t.Fatalf("%v pitch %d offset %d: texel %v copied twice", tc.format, tc.pitch, base, texel)
						}
						covered[texel] = true
						placed := p.bufferOffset + uint64((p.bufferOriginY+y)/info.height)*uint64(tc.pitch) + uint64((p.bufferOriginX+x)/info.width)*uint64(info.bytes)
						logical := base + uint64(y/info.height)*uint64(tc.pitch) + uint64(texel[0]/info.width)*uint64(info.bytes)
						if placed != logical {
							t.Fatalf("%v pitch %d offset %d: texel %v read from byte %d, want %d", tc.format, tc.pitch, base, texel, placed, logical)
						}
					}
				}
			}
			if want := int(tc.w/info.width) * int(tc.h/info.height); len(covered) != want {
				t.Fatalf("%v pitch %d offset %d: %d blocks copied, want %d", tc.format, tc.pitch, base, len(covered), want)
			}
		}
	}
}
