//go:build js && wasm

package wgpu

import (
	"errors"
	"slices"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/internal/browser"
)

func TestBrowserConfigureReturnsTheBrowsersException(t *testing.T) {
	canvas := jsEval(`{
		width: 300,
		height: 150,
		getContext() {
			return {
				configure(c) { if (c.format === "rgba16float") throw new TypeError("rgba16float is not yet supported") },
				getCurrentTexture() { throw new DOMException("context lost", "InvalidStateError") },
			}
		},
	}`)
	bs, err := browser.NewSurface(jsEval("undefined"), canvas)
	if err != nil {
		t.Fatalf("NewSurface: %v", err)
	}
	s := &Surface{browser: bs}
	d, _ := newFakeDevice(jsEval("new Promise(() => {})"))

	config := &SurfaceConfiguration{Width: 64, Height: 32, Format: gputypes.TextureFormatRGBA16Float, Usage: gputypes.TextureUsageRenderAttachment}
	if err := s.Configure(d, config); err == nil {
		t.Fatal("Configure(rgba16float): nil error where the browser throws")
	}
	if _, _, err := s.GetCurrentTexture(); err == nil {
		t.Fatal("GetCurrentTexture after a failed Configure: nil error, want not configured")
	}

	config.Format = gputypes.TextureFormatBGRA8Unorm
	if err := s.Configure(d, config); err != nil {
		t.Fatalf("Configure(bgra8unorm): %v", err)
	}
	if _, _, err := s.GetCurrentTexture(); !errors.Is(err, ErrSurfaceLost) {
		t.Fatalf("GetCurrentTexture = %v, want ErrSurfaceLost for a throwing getCurrentTexture", err)
	}
}

func TestBrowserSurfaceCapabilitiesOfferPremultipliedAlpha(t *testing.T) {
	caps := (&Adapter{}).GetSurfaceCapabilities(nil)
	want := []gputypes.CompositeAlphaMode{gputypes.CompositeAlphaModeOpaque, gputypes.CompositeAlphaModePremultiplied}
	if !slices.Equal(caps.AlphaModes, want) {
		t.Fatalf("AlphaModes = %v, want %v", caps.AlphaModes, want)
	}
}
