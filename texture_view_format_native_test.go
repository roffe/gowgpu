//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/noop"
)

// A texture takes view formats besides its own format only on an adapter
// with DownlevelFlagsViewFormats, as in wgpu-core, and its own format on
// any.
func TestTextureViewFormatsNeedTheDownlevelFlag(t *testing.T) {
	for _, tc := range []struct {
		name       string
		flags      gputypes.DownlevelFlags
		viewFormat gputypes.TextureFormat
		ok         bool
	}{
		{"own format without the flag", 0, gputypes.TextureFormatRGBA8Unorm, true},
		{"sRGB twin without the flag", 0, gputypes.TextureFormatRGBA8UnormSrgb, false},
		{"sRGB twin", gputypes.DownlevelFlagsViewFormats, gputypes.TextureFormatRGBA8UnormSrgb, true},
		{"surface view formats only", gputypes.DownlevelFlagsSurfaceViewFormats, gputypes.TextureFormatRGBA8UnormSrgb, false},
		{"another format", gputypes.DownlevelFlagsViewFormats, gputypes.TextureFormatBGRA8UnormSrgb, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			adapter := &core.Adapter{DownlevelCapabilities: gputypes.DownlevelCapabilities{Flags: tc.flags}}
			device := &Device{core: core.NewDevice(&noop.Device{}, adapter, 0, DefaultLimits(), "texture-view-format-test")}
			tex, err := device.CreateTexture(&TextureDescriptor{
				Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
				Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageTextureBinding,
				ViewFormats: []gputypes.TextureFormat{tc.viewFormat},
			})
			if tc.ok != (err == nil) {
				t.Fatalf("CreateTexture with view format %v: %v", tc.viewFormat, err)
			}
			if tex != nil {
				tex.Release()
			}
		})
	}
}
