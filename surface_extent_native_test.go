//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/noop"
)

// extentSurface is a noop surface whose swapchain has the size extent says,
// whatever it was configured with, like a Vulkan surface with a defined
// current extent.
type extentSurface struct {
	noop.Surface
	width, height uint32
}

func (s *extentSurface) ActualExtent() (uint32, uint32) { return s.width, s.height }

// The surface texture has the size of the swapchain's images, which can
// differ from the configured size and change without a Configure: render
// passes check the other attachments against it.
func TestSurfaceTextureHasTheSwapchainExtent(t *testing.T) {
	rawDevice := &noop.Device{}
	coreDevice := core.NewDevice(rawDevice, nil, 0, gputypes.DefaultLimits(), "surface-extent-test")
	queue := &Queue{hal: &noop.Queue{}, halDevice: rawDevice}
	device := &Device{core: coreDevice, queue: queue}
	queue.device = device
	defer device.Release()

	raw := &extentSurface{width: 640, height: 480}
	surface := &Surface{
		core:           core.NewSurface(raw, "surface-extent-test"),
		surfaceCreated: true,
		currentBackend: gputypes.BackendEmpty,
	}
	defer surface.Release()
	if err := surface.Configure(device, &SurfaceConfiguration{
		Width: 320, Height: 240,
		Format:      gputypes.TextureFormatRGBA8Unorm,
		Usage:       gputypes.TextureUsageRenderAttachment,
		PresentMode: gputypes.PresentModeFifo,
		AlphaMode:   gputypes.CompositeAlphaModeAuto,
	}); err != nil {
		t.Fatalf("Configure: %v", err)
	}

	for _, want := range []gputypes.Extent3D{
		{Width: 640, Height: 480, DepthOrArrayLayers: 1},
		{Width: 800, Height: 600, DepthOrArrayLayers: 1},
	} {
		raw.width, raw.height = want.Width, want.Height
		texture, _, err := surface.GetCurrentTexture()
		if err != nil {
			t.Fatalf("GetCurrentTexture: %v", err)
		}
		if got := texture.coreTexture.Size(); got != want {
			t.Errorf("surface texture size = %+v, want the swapchain's %+v", got, want)
		}
		surface.DiscardTexture()
	}
}
