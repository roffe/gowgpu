//go:build !(js && wasm)

package vulkan

import (
	"slices"
	"testing"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// fakeSwapchainQueue is a queue on a device without a driver, with a
// swapchain per acquire semaphore given, each with its image acquired and
// one present semaphore, acquire+10. Its vkQueueSubmit fails, after Submit
// has put together what to wait for and signal.
func fakeSwapchainQueue(acquireSems ...vk.Semaphore) (*Queue, []*Swapchain) {
	device := &Device{cmds: &vk.Commands{}, timelineFence: &deviceFence{isTimeline: true, timelineSemaphore: 99}}
	q := &Queue{device: device}
	device.queue = q
	var swapchains []*Swapchain
	for _, sem := range acquireSems {
		sc := &Swapchain{
			device:             device,
			imageAcquired:      true,
			acquireSemaphores:  []vk.Semaphore{sem},
			acquireFenceValues: []uint64{0},
			currentAcquireSem:  sem,
			presentPools:       []presentSemaphorePool{{semaphores: []vk.Semaphore{sem + 10}}},
		}
		q.hold(sc)
		swapchains = append(swapchains, sc)
	}
	return q, swapchains
}

// Two windows acquired together: a submit waits for both acquire semaphores
// and signals a present semaphore of each. The queue had one active
// swapchain, the last acquired, and the other window's present failed.
func TestSubmitSynchronizesEveryAcquiredSwapchain(t *testing.T) {
	q, sc := fakeSwapchainQueue(1, 2)

	_, err := q.Submit([]hal.CommandBuffer{&CommandBuffer{handle: 5}})
	if err == nil {
		t.Fatal("Submit without a driver succeeded")
	}
	if got := q.submitScratch.waits; !slices.Equal(got, []vk.Semaphore{1, 2}) {
		t.Errorf("submit waits for %v, want both acquire semaphores", got)
	}
	if got := q.submitScratch.signals; len(got) < 2 || !slices.Equal(got[:2], []vk.Semaphore{11, 12}) {
		t.Errorf("submit signals %v, want a present semaphore of each first", got)
	}
	// A present semaphore of each was taken for a submission that never
	// happened, so neither image may be presented.
	if !sc[0].broken || !sc[1].broken {
		t.Errorf("failed submit left the swapchains usable: %v, %v", sc[0].broken, sc[1].broken)
	}
}

// Discarding one window's image gives it back and leaves the other
// window's acquire semaphore to the next submit.
func TestDiscardOneOfTwoAcquiredSwapchains(t *testing.T) {
	q, sc := fakeSwapchainQueue(1, 2)
	surface := &Surface{swapchain: sc[0], device: q.device}

	surface.DiscardTexture(nil)

	if q.holds(sc[0]) || !q.holds(sc[1]) {
		t.Fatalf("after discarding the first: held %v and %v, want only the second", q.holds(sc[0]), q.holds(sc[1]))
	}
	_, _ = q.Submit([]hal.CommandBuffer{&CommandBuffer{handle: 5}})
	if got := q.submitScratch.waits; !slices.Equal(got, []vk.Semaphore{2}) {
		t.Errorf("submit after the discard waits for %v, want the second's acquire semaphore", got)
	}
}

// A discarded image stays acquired from Vulkan, so the next acquire hands
// it out again, its acquire semaphore no submit has to wait for: a few
// discarded images kept acquired stopped vkAcquireNextImageKHR returning
// any, every acquire timing out after a second.
func TestDiscardedImageIsAcquiredAgain(t *testing.T) {
	q, sc := fakeSwapchainQueue(1)
	swapchain := sc[0]
	swapchain.handle = 7
	swapchain.images = []vk.Image{30, 31}
	swapchain.surfaceTextures = []*SwapchainTexture{{index: 0}, {index: 1}}
	swapchain.presentPools = append(swapchain.presentPools, presentSemaphorePool{})
	swapchain.imageLayouts = []vk.ImageLayout{vk.ImageLayoutUndefined, vk.ImageLayoutPresentSrcKhr}
	swapchain.currentImage = 1
	surface := &Surface{swapchain: swapchain, device: q.device}

	surface.DiscardTexture(nil)
	got, err := surface.AcquireTexture(nil)
	if err != nil {
		t.Fatalf("AcquireTexture after a discard: %v", err)
	}
	if got.Texture != swapchain.surfaceTextures[1] {
		t.Errorf("AcquireTexture after a discard returned %v, want the discarded image", got.Texture)
	}
	if !swapchain.imageAcquired || !swapchain.acquireUsed || !q.holds(swapchain) {
		t.Errorf("image acquired %v, acquire semaphore waited %v, held %v; want all", swapchain.imageAcquired, swapchain.acquireUsed, q.holds(swapchain))
	}
}
