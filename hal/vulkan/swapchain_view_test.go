//go:build !(js && wasm)

package vulkan

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// A surface whose textures are viewed in another format needs a mutable
// format swapchain, which a device without VK_KHR_swapchain_mutable_format
// cannot make: the configuration is refused before anything is made.
func TestSurfaceViewFormatsNeedMutableFormatSwapchains(t *testing.T) {
	inst := &Instance{}
	surface := &Surface{handle: 1, instance: inst}
	config := validSwapchainConfig()
	config.Width, config.Height = 4, 4
	config.ViewFormats = []gputypes.TextureFormat{gputypes.TextureFormatRGBA8Unorm, gputypes.TextureFormatRGBA8UnormSrgb}
	err := surface.createSwapchain(&Device{instance: inst}, config)
	if err == nil || !strings.Contains(err.Error(), "VK_KHR_swapchain_mutable_format") {
		t.Fatalf("createSwapchain with an sRGB view format and no mutable format swapchains: %v, want the missing extension", err)
	}
	if surface.swapchain != nil {
		t.Fatal("the refused configuration made a swapchain")
	}
}

// A 2D view of a swapchain image in the surface's format is the swapchain's
// own view. A view in a format that is not one of the surface's view
// formats, of another dimension than 2D or a 2D array, or of another mip
// level, layer or aspect than the image's one is refused.
func TestSwapchainViewTakesTheDescriptor(t *testing.T) {
	sc := &Swapchain{viewFormats: []gputypes.TextureFormat{gputypes.TextureFormatBGRA8UnormSrgb}}
	texture := &SwapchainTexture{handle: 1, view: 2, swapchain: sc, format: gputypes.TextureFormatBGRA8Unorm}
	d := &Device{}

	for _, desc := range []*hal.TextureViewDescriptor{
		nil,
		{},
		{Format: gputypes.TextureFormatBGRA8Unorm, Dimension: gputypes.TextureViewDimension2D, Aspect: gputypes.TextureAspectAll, MipLevelCount: 1, ArrayLayerCount: 1},
	} {
		view, err := d.swapchainView(texture, desc)
		if err != nil {
			t.Fatalf("swapchainView(%+v): %v", desc, err)
		}
		if view.handle != texture.view || view != texture.textureView || !view.isSwapchain || view.swapchain != sc {
			t.Fatalf("swapchainView(%+v) = %+v, want the swapchain's own view", desc, view)
		}
	}

	for _, desc := range []hal.TextureViewDescriptor{
		{Format: gputypes.TextureFormatRGBA8UnormSrgb},
		{Format: gputypes.TextureFormatBGRA8UnormSrgb, Dimension: gputypes.TextureViewDimension3D},
		{Dimension: gputypes.TextureViewDimensionCube},
		{BaseMipLevel: 1},
		{MipLevelCount: 2},
		{BaseArrayLayer: 1},
		{ArrayLayerCount: 2},
		{Aspect: gputypes.TextureAspectDepthOnly},
	} {
		if view, err := d.swapchainView(texture, &desc); err == nil {
			t.Errorf("swapchainView(%+v) = %+v, want it refused", desc, view)
		}
	}
	if len(texture.views) != 0 {
		t.Fatalf("refused views were kept: %+v", texture.views)
	}
}
