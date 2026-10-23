//go:build !(js && wasm)

package core

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

func TestValidateSurfaceConfiguration(t *testing.T) {
	caps := &hal.SurfaceCapabilities{
		Formats:      []gputypes.TextureFormat{gputypes.TextureFormatBGRA8Unorm, gputypes.TextureFormatRGBA16Float},
		PresentModes: []gputypes.PresentMode{gputypes.PresentModeFifo},
		AlphaModes:   []gputypes.CompositeAlphaMode{gputypes.CompositeAlphaModeOpaque},
		Usage:        gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopyDst,
	}
	limits := gputypes.DefaultLimits()
	valid := hal.SurfaceConfiguration{Width: 640, Height: 480, Format: gputypes.TextureFormatBGRA8Unorm,
		Usage:       gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopyDst,
		PresentMode: gputypes.PresentModeFifo, AlphaMode: gputypes.CompositeAlphaModeOpaque}
	if err := ValidateSurfaceConfiguration(&valid, caps, limits); err != nil {
		t.Errorf("supported configuration: %v", err)
	}
	auto := valid
	auto.PresentMode, auto.AlphaMode = gputypes.PresentModeUndefined, gputypes.CompositeAlphaModeAuto
	if err := ValidateSurfaceConfiguration(&auto, caps, limits); err != nil {
		t.Errorf("modes left to the backend: %v", err)
	}
	for name, change := range map[string]func(*hal.SurfaceConfiguration){
		"format":       func(c *hal.SurfaceConfiguration) { c.Format = gputypes.TextureFormatRGBA8Unorm },
		"present mode": func(c *hal.SurfaceConfiguration) { c.PresentMode = gputypes.PresentModeMailbox },
		"alpha mode":   func(c *hal.SurfaceConfiguration) { c.AlphaMode = gputypes.CompositeAlphaModePremultiplied },
		"usage":        func(c *hal.SurfaceConfiguration) { c.Usage |= gputypes.TextureUsageCopySrc },
		"zero width":   func(c *hal.SurfaceConfiguration) { c.Width = 0 },
		"too high":     func(c *hal.SurfaceConfiguration) { c.Height = limits.MaxTextureDimension2D + 1 },
	} {
		config := valid
		change(&config)
		if err := ValidateSurfaceConfiguration(&config, caps, limits); !errors.Is(err, ErrSurfaceConfiguration) {
			t.Errorf("unsupported %s: %v, want ErrSurfaceConfiguration", name, err)
		}
	}
}
