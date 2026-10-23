//go:build js && wasm

package wgpu

import (
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/internal/browser"
)

// DeviceDescriptor configures device creation.
type DeviceDescriptor struct {
	Label            string
	RequiredFeatures gputypes.Features
	RequiredLimits   gputypes.Limits
}

// Adapter represents a physical GPU.
// On browser, this wraps a GPUAdapter via internal/browser.Adapter.
type Adapter struct {
	browser  *browser.Adapter
	info     gputypes.AdapterInfo
	features gputypes.Features
	limits   gputypes.Limits
	released bool
}

// Info returns adapter metadata.
func (a *Adapter) Info() gputypes.AdapterInfo { return a.info }

// Features returns supported features.
func (a *Adapter) Features() gputypes.Features { return a.features }

// Limits returns the adapter's resource limits.
func (a *Adapter) Limits() gputypes.Limits { return a.limits }

// DownlevelCapabilities returns backend capability flags for downlevel adapters.
// Browsers are fully WebGPU compliant, so this returns the default (full compliance).
func (a *Adapter) DownlevelCapabilities() gputypes.DownlevelCapabilities {
	return gputypes.DefaultDownlevelCapabilities()
}

// RequestDevice creates a logical device from this adapter.
// If desc is nil, no features are enabled. Without RequiredLimits, a nil desc
// included, the device gets the adapter's limits, as natively.
func (a *Adapter) RequestDevice(desc *DeviceDescriptor) (*Device, error) {
	if a.released {
		return nil, ErrReleased
	}

	var label string
	var features gputypes.Features
	var limits gputypes.Limits
	if desc != nil {
		label, features, limits = desc.Label, desc.RequiredFeatures, desc.RequiredLimits
	}
	if limits == (gputypes.Limits{}) {
		limits = a.limits
	}
	jsDesc, err := browser.BuildDeviceDescriptor(label, features, limits)
	if err != nil {
		return nil, fmt.Errorf("wgpu: RequestDevice: %w", err)
	}

	bd, err := a.browser.RequestDevice(jsDesc)
	if err != nil {
		return nil, err
	}

	// Extract the device's features and limits.
	deviceFeatures := browser.ExtractFeatures(bd.Features())
	deviceLimits := browser.ExtractLimits(bd.Limits())

	queue := &Queue{
		browser: bd.Queue(),
	}

	device := &Device{
		browser:  bd,
		queue:    queue,
		features: deviceFeatures,
		limits:   deviceLimits,
	}
	queue.device = device
	if err := device.lostErr(); err != nil {
		return nil, fmt.Errorf("wgpu: RequestDevice: %w", err)
	}
	return device, nil
}

// SurfaceCapabilities describes what a surface supports on this adapter.
type SurfaceCapabilities struct {
	Formats      []gputypes.TextureFormat
	PresentModes []gputypes.PresentMode
	AlphaModes   []gputypes.CompositeAlphaMode
}

// GetSurfaceCapabilities returns the capabilities of a surface for this adapter.
//
// On browser, capabilities are statically defined per the WebGPU spec:
//   - Formats: rgba8unorm, bgra8unorm, rgba16float (preferred format first),
//     rgba16float only where a device has shown the browser takes it
//   - PresentModes: Fifo only (browser controls VSync)
//   - AlphaModes: Opaque and Premultiplied
//
// The preferred format is obtained from navigator.gpu.getPreferredCanvasFormat()
// and placed first in the formats list.
//
// Matches Rust wgpu SurfaceInterface::get_capabilities for WebSurface.
func (a *Adapter) GetSurfaceCapabilities(surface *Surface) *SurfaceCapabilities {
	// Browser WebGPU supports these three formats per spec:
	// https://gpuweb.github.io/gpuweb/#supported-context-formats
	formats := []gputypes.TextureFormat{
		TextureFormatRGBA8Unorm,
		TextureFormatBGRA8Unorm,
	}
	if browser.RGBA16FloatCanvasSupported() {
		formats = append(formats, gputypes.TextureFormatRGBA16Float)
	}

	// Put the preferred format first (Rust wgpu does the same swap).
	if surface != nil && surface.browser != nil {
		preferredStr := surface.browser.GetPreferredCanvasFormat()
		preferredFmt := browser.TextureFormatFromJS(preferredStr)
		for i, f := range formats {
			if f == preferredFmt {
				formats[0], formats[i] = formats[i], formats[0]
				break
			}
		}
	}

	return &SurfaceCapabilities{
		Formats:      formats,
		PresentModes: []gputypes.PresentMode{gputypes.PresentModeFifo},
		AlphaModes:   []gputypes.CompositeAlphaMode{gputypes.CompositeAlphaModeOpaque, gputypes.CompositeAlphaModePremultiplied},
	}
}

// Release releases the adapter.
func (a *Adapter) Release() {
	if a.released {
		return
	}
	a.released = true
}
