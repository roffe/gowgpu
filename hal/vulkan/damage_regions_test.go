//go:build !(js && wasm)

package vulkan

import (
	"image"
	"slices"
	"testing"

	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

func TestDamageRegionsClipToExtent(t *testing.T) {
	got := damageRegions(nil, []image.Rectangle{
		image.Rect(10, 10, 20, 20),
		image.Rect(90, 40, 120, 60),                    // past the right edge
		{Min: image.Pt(30, 30), Max: image.Pt(20, 40)}, // inverted
		image.Rect(200, 0, 300, 10),                    // outside
		image.Rect(-5, -5, 5, 5),                       // over the top-left corner
	}, vk.Extent2D{Width: 100, Height: 50})
	want := []vk.RectLayerKHR{
		{Offset: vk.Offset2D{X: 10, Y: 10}, Extent: vk.Extent2D{Width: 10, Height: 10}},
		{Offset: vk.Offset2D{X: 90, Y: 40}, Extent: vk.Extent2D{Width: 10, Height: 10}},
		{Offset: vk.Offset2D{X: 0, Y: 0}, Extent: vk.Extent2D{Width: 5, Height: 5}},
	}
	if !slices.Equal(got, want) {
		t.Errorf("damageRegions = %+v, want %+v", got, want)
	}
}
