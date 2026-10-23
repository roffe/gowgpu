//go:build !(js && wasm)

package wgpu

import (
	"slices"
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/noop"
)

// viewFormatDevice records the format of every view the HAL makes.
type viewFormatDevice struct {
	noop.Device
	viewFormats []gputypes.TextureFormat
}

func (d *viewFormatDevice) CreateTextureView(texture hal.Texture, desc *hal.TextureViewDescriptor) (hal.TextureView, error) {
	d.viewFormats = append(d.viewFormats, desc.Format)
	return d.Device.CreateTextureView(texture, desc)
}

// newViewFormatSurface returns a surface and a device whose adapter has the
// downlevel flags flags.
func newViewFormatSurface(t *testing.T, flags gputypes.DownlevelFlags) (*Surface, *Device, *viewFormatDevice) {
	t.Helper()
	raw := &viewFormatDevice{}
	adapter := &core.Adapter{DownlevelCapabilities: gputypes.DownlevelCapabilities{Flags: flags}}
	queue := &Queue{hal: &noop.Queue{}, halDevice: raw}
	device := &Device{core: core.NewDevice(raw, adapter, 0, gputypes.DefaultLimits(), "view-format-test"), queue: queue}
	queue.device = device
	surface := &Surface{
		core:           core.NewSurface(&noop.Surface{}, "view-format-test"),
		surfaceCreated: true,
		currentBackend: gputypes.BackendEmpty,
	}
	t.Cleanup(func() {
		surface.Release()
		device.Release()
	})
	return surface, device, raw
}

func viewFormatConfig(viewFormats ...gputypes.TextureFormat) *SurfaceConfiguration {
	return &SurfaceConfiguration{
		Width:       4,
		Height:      4,
		Format:      gputypes.TextureFormatBGRA8Unorm,
		Usage:       gputypes.TextureUsageRenderAttachment,
		PresentMode: gputypes.PresentModeFifo,
		ViewFormats: viewFormats,
	}
}

// A surface takes view formats that differ from its format only in being
// sRGB, as WebGPU's canvas does, and only on an adapter with
// DownlevelFlagsSurfaceViewFormats; its own format it takes anywhere.
func TestSurfaceConfigurationViewFormats(t *testing.T) {
	for _, tc := range []struct {
		name        string
		flags       gputypes.DownlevelFlags
		viewFormats []gputypes.TextureFormat
		ok          bool
	}{
		{"own format without the flag", 0, []gputypes.TextureFormat{gputypes.TextureFormatBGRA8Unorm}, true},
		{"sRGB twin without the flag", 0, []gputypes.TextureFormat{gputypes.TextureFormatBGRA8UnormSrgb}, false},
		{"sRGB twin", gputypes.DownlevelFlagsSurfaceViewFormats, []gputypes.TextureFormat{gputypes.TextureFormatBGRA8UnormSrgb}, true},
		{"another format", gputypes.DownlevelFlagsSurfaceViewFormats, []gputypes.TextureFormat{gputypes.TextureFormatRGBA8UnormSrgb}, false},
		{"texture view formats only", gputypes.DownlevelFlagsViewFormats, []gputypes.TextureFormat{gputypes.TextureFormatBGRA8UnormSrgb}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			surface, device, _ := newViewFormatSurface(t, tc.flags)
			err := surface.Configure(device, viewFormatConfig(tc.viewFormats...))
			if tc.ok && err != nil {
				t.Fatalf("Configure with view formats %v: %v", tc.viewFormats, err)
			}
			if !tc.ok && err == nil {
				t.Fatalf("Configure with view formats %v succeeded, want it refused", tc.viewFormats)
			}
		})
	}
}

// A view of a surface texture is in the surface's format or one of its view
// formats, and reaches the HAL in it; a view in another format is refused,
// as is any but its own format after a reconfigure without view formats.
func TestSurfaceTextureViewTakesTheViewFormats(t *testing.T) {
	surface, device, raw := newViewFormatSurface(t, gputypes.DownlevelFlagsSurfaceViewFormats)
	config := viewFormatConfig(gputypes.TextureFormatBGRA8UnormSrgb)
	if err := surface.Configure(device, config); err != nil {
		t.Fatalf("Configure: %v", err)
	}
	config.ViewFormats[0] = gputypes.TextureFormatRGBA8Unorm // the surface keeps its own copy

	texture, _, err := surface.GetCurrentTexture()
	if err != nil {
		t.Fatalf("GetCurrentTexture: %v", err)
	}
	for _, format := range []gputypes.TextureFormat{gputypes.TextureFormatUndefined, gputypes.TextureFormatBGRA8Unorm, gputypes.TextureFormatBGRA8UnormSrgb} {
		view, err := texture.CreateView(&TextureViewDescriptor{Format: format})
		if err != nil {
			t.Fatalf("CreateView(%v): %v", format, err)
		}
		view.Release()
	}
	view, err := device.CreateTextureView(texture.AsTexture(), &TextureViewDescriptor{Format: gputypes.TextureFormatBGRA8UnormSrgb})
	if err != nil {
		t.Fatalf("Device.CreateTextureView(AsTexture, BGRA8UnormSrgb): %v", err)
	}
	view.Release()
	want := []gputypes.TextureFormat{gputypes.TextureFormatBGRA8Unorm, gputypes.TextureFormatBGRA8Unorm, gputypes.TextureFormatBGRA8UnormSrgb, gputypes.TextureFormatBGRA8UnormSrgb}
	if !slices.Equal(raw.viewFormats, want) {
		t.Fatalf("the HAL made views in %v, want %v", raw.viewFormats, want)
	}
	for _, desc := range []TextureViewDescriptor{
		{Format: gputypes.TextureFormatRGBA8UnormSrgb},
		{Format: gputypes.TextureFormatRGBA8Unorm},
		{Dimension: gputypes.TextureViewDimension3D},
	} {
		if view, err := texture.CreateView(&desc); err == nil {
			view.Release()
			t.Errorf("CreateView(%+v) succeeded, want it refused", desc)
		}
	}
	if err := surface.Present(texture); err != nil {
		t.Fatalf("Present: %v", err)
	}

	if err := surface.Configure(device, viewFormatConfig()); err != nil {
		t.Fatalf("Configure without view formats: %v", err)
	}
	texture, _, err = surface.GetCurrentTexture()
	if err != nil {
		t.Fatalf("GetCurrentTexture: %v", err)
	}
	if view, err := texture.CreateView(&TextureViewDescriptor{Format: gputypes.TextureFormatBGRA8UnormSrgb}); err == nil {
		view.Release()
		t.Error("CreateView in the old view format succeeded after a reconfigure without it")
	}
	surface.DiscardTexture()
}
