//go:build !(js && wasm)

package vulkan

import (
	"testing"

	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// TestDiscardEncodingForgetsTheSwapchainImage: a recording that used a
// swapchain image, a pass on it or a barrier of it, the present barrier
// included, is discarded, so the layout it would have left the image in is
// not the image's: that goes back to UNDEFINED, for a pass that loads the
// image to clear it and the present to transition it from there. Another
// image keeps its layout.
func TestDiscardEncodingForgetsTheSwapchainImage(t *testing.T) {
	sc := &Swapchain{
		imageLayouts: []vk.ImageLayout{vk.ImageLayoutPresentSrcKhr, vk.ImageLayoutPresentSrcKhr},
		currentImage: 1,
		presentReady: true,
	}
	e := &CommandEncoder{poolManaged: true, swapchain: sc}

	e.DiscardEncoding()

	if got := sc.imageLayouts[1]; got != vk.ImageLayoutUndefined {
		t.Errorf("discarded image layout = %v, want UNDEFINED", got)
	}
	if sc.presentReady {
		t.Error("present still takes the discarded image as presentable")
	}
	if got := sc.imageLayouts[0]; got != vk.ImageLayoutPresentSrcKhr {
		t.Errorf("other image layout = %v, want PRESENT_SRC_KHR untouched", got)
	}
	if e.swapchain != nil {
		t.Error("swapchain still set after discard")
	}
}

// TestResetAllForgetsTheSwapchain: once the command buffers are spent, a
// discard of the next recording has nothing of theirs to take back.
func TestResetAllForgetsTheSwapchain(t *testing.T) {
	e := &CommandEncoder{poolManaged: true, swapchain: &Swapchain{}}
	e.ResetAll(nil)
	if e.swapchain != nil {
		t.Error("swapchain still set after ResetAll")
	}
}
