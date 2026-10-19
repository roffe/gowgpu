//go:build !(js && wasm)

package vulkan

import (
	"testing"

	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// TestDiscardEncodingResetsPresentedLayout: a swapchain pass split off into a
// command buffer of its own is closed with the inline present barrier, which
// marks the image PRESENT_SRC_KHR. Discarding the encoder after that drops
// that command buffer, so the mark has to go: present would otherwise skip
// its barrier on an image that never got there.
func TestDiscardEncodingResetsPresentedLayout(t *testing.T) {
	sc := &Swapchain{imageLayouts: []vk.ImageLayout{vk.ImageLayoutPresentSrcKhr, vk.ImageLayoutPresentSrcKhr}}
	e := &CommandEncoder{poolManaged: true, presented: sc, presentedImage: 1}

	e.DiscardEncoding()

	if got := sc.imageLayouts[1]; got != vk.ImageLayoutUndefined {
		t.Errorf("discarded image layout = %v, want UNDEFINED", got)
	}
	if got := sc.imageLayouts[0]; got != vk.ImageLayoutPresentSrcKhr {
		t.Errorf("other image layout = %v, want PRESENT_SRC_KHR untouched", got)
	}
	if e.presented != nil {
		t.Error("presented swapchain still set after discard")
	}
}

// TestResetAllForgetsPresented: once the command buffers are spent the
// present barrier they carried is history.
func TestResetAllForgetsPresented(t *testing.T) {
	e := &CommandEncoder{poolManaged: true, presented: &Swapchain{}}
	e.ResetAll(nil)
	if e.presented != nil {
		t.Error("presented swapchain still set after ResetAll")
	}
}
