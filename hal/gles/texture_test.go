// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func TestNewTextureViewResolvesItsRange(t *testing.T) {
	tex := &Texture{format: gputypes.TextureFormatRGBA8Unorm, dimension: gputypes.TextureDimension2D,
		size: hal.Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 6}, mipLevels: 4}
	v, err := newTextureView(tex, &hal.TextureViewDescriptor{BaseMipLevel: 1, BaseArrayLayer: 2})
	if err != nil {
		t.Fatal(err)
	}
	defer v.Destroy()
	if v.mipCount != 3 || v.layerCount != 4 {
		t.Errorf("a view from mip 1 and layer 2 covers %d mips and %d layers, want 3 and 4", v.mipCount, v.layerCount)
	}
	if viewByHandle(v.NativeHandle()) != v {
		t.Error("the view's handle does not find it")
	}
}

// GL has no views that change the format, so one is refused rather than
// sampling and rendering as the texture's own format.
func TestNewTextureViewRefusesAnotherFormat(t *testing.T) {
	tex := &Texture{format: gputypes.TextureFormatRGBA8Unorm, mipLevels: 1, size: hal.Extent3D{Width: 1, Height: 1, DepthOrArrayLayers: 1}}
	if _, err := newTextureView(tex, &hal.TextureViewDescriptor{Format: gputypes.TextureFormatRGBA8UnormSrgb}); err == nil {
		t.Error("an sRGB view of an RGBA8Unorm texture was made")
	}
	ds := &Texture{format: gputypes.TextureFormatDepth24PlusStencil8, mipLevels: 1, size: hal.Extent3D{Width: 1, Height: 1, DepthOrArrayLayers: 1}}
	v, err := newTextureView(ds, &hal.TextureViewDescriptor{Format: gputypes.TextureFormatStencil8, Aspect: gputypes.TextureAspectStencilOnly})
	if err != nil {
		t.Fatalf("the stencil view of a depth/stencil texture: %v", err)
	}
	v.Destroy()
}
