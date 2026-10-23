//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/noop"
)

// pickySurface is a noop HAL surface that, like Vulkan's, refuses to replace
// a configuration with one of another device until it is unconfigured, and
// refuses every configuration on refuse.
type pickySurface struct {
	noop.Surface
	device hal.Device
	refuse hal.Device
}

func (s *pickySurface) Configure(device hal.Device, config *hal.SurfaceConfiguration) error {
	if device == s.refuse || s.device != nil && s.device != device {
		return errors.New("configure refused")
	}
	s.device = device
	return s.Surface.Configure(device, config)
}

func (s *pickySurface) Unconfigure(device hal.Device) {
	s.device = nil
	s.Surface.Unconfigure(device)
}

// A surface configured on one device and then on another is the second
// device's, and one whose configuration on another device fails is no
// device's, rather than the new device's with the old one's swapchain.
func TestSurfaceConfigureOntoAnotherDevice(t *testing.T) {
	newDevice := func(label string) *Device {
		device, err := NewDeviceFromHAL(&releaseCountingDevice{}, &noop.Queue{}, 0, gputypes.DefaultLimits(), label)
		if err != nil {
			t.Fatalf("NewDeviceFromHAL: %v", err)
		}
		t.Cleanup(device.Release)
		return device
	}
	first, second, third := newDevice("first"), newDevice("second"), newDevice("third")
	raw := &pickySurface{refuse: third.halDevice()}
	surface := &Surface{core: core.NewSurface(raw, "switch"), surfaceCreated: true, currentBackend: gputypes.BackendEmpty}
	defer surface.Release()
	config := &SurfaceConfiguration{
		Width: 4, Height: 4,
		Format:      gputypes.TextureFormatRGBA8Unorm,
		Usage:       gputypes.TextureUsageRenderAttachment,
		PresentMode: gputypes.PresentModeFifo,
		AlphaMode:   gputypes.CompositeAlphaModeAuto,
	}

	if err := surface.Configure(first, config); err != nil {
		t.Fatalf("Configure(first): %v", err)
	}
	if err := surface.Configure(second, config); err != nil {
		t.Fatalf("Configure(second): %v", err)
	}
	st, _, err := surface.GetCurrentTexture()
	if err != nil {
		t.Fatalf("GetCurrentTexture: %v", err)
	}
	if st.device != second || raw.device != second.halDevice() {
		t.Error("the surface texture is not the second device's")
	}
	surface.DiscardTexture()

	if err := surface.Configure(third, config); err == nil {
		t.Fatal("Configure(third) succeeded on a backend that refuses it")
	}
	if _, _, err := surface.GetCurrentTexture(); err == nil {
		t.Error("GetCurrentTexture after a failed Configure onto another device succeeded")
	}
}
