//go:build !(js && wasm)

package wgpu

import (
	"sync"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// Texture represents a GPU texture.
type Texture struct {
	hal          hal.Texture
	device       *Device
	format       gputypes.TextureFormat
	released     bool
	surface      *core.Surface
	surfaceLease uint64

	// coreTexture holds the core.Texture for dense TrackerIndex allocation.
	// This enables barrier injection: the TrackerIndex is what a render
	// pass's usage scope (via core.TextureView.Parent) and the command
	// buffer's tracker record the texture's usages under. At submit time the
	// tracker is merged into the device-level DeviceTracker which produces
	// the barrier list.
	//
	// nil for textures created without HAL (e.g., test stubs or browser).
	// Swapchain textures also get a coreTexture so barriers work for
	// surface textures used as render targets.
	//
	// Reference: Rust wgpu-core resource.rs Texture (holds TrackingData)
	coreTexture *core.Texture

	// mark is the last command encoder that tracked this texture.
	mark encoderMark

	// defaultView is the HAL view CreateTextureView hands out for a nil
	// descriptor: one per texture, shared by every TextureView made that way
	// and destroyed with the texture. A view a frame of the same texture is
	// then no new HAL view, and on Vulkan no new framebuffer.
	viewMu      sync.Mutex
	defaultView hal.TextureView
	viewDestroy viewDestroy
}

// sharedDefaultView returns the texture's default HAL view, making it the
// first time. nil means it could not be made, and the caller makes its own.
func (t *Texture) sharedDefaultView(dev hal.Device, raw hal.Texture) hal.TextureView {
	t.viewMu.Lock()
	defer t.viewMu.Unlock()
	if t.defaultView == nil && !t.released {
		v, err := dev.CreateTextureView(raw, &hal.TextureViewDescriptor{})
		if err != nil {
			return nil
		}
		t.defaultView = v
	}
	return t.defaultView
}

// resolveHAL is the single boundary from a public texture wrapper to HAL.
// Surface textures are borrowed and are usable only for their acquisition.
// Encoder validation and HAL conversion may call this more than once during
// one operation; acquisition and presentation are serialized by the render
// loop, so the lease remains stable across those calls.
func (t *Texture) resolveHAL() hal.Texture {
	if t == nil || t.released || t.hal == nil || t.device == nil || t.device.released.Load() {
		return nil
	}
	if t.surface != nil && !t.surface.AcquisitionValid(t.surfaceLease) {
		return nil
	}
	return t.hal
}

// Format returns the texture format.
func (t *Texture) Format() gputypes.TextureFormat { return t.format }

// Release destroys the texture. The underlying HAL texture is not freed
// immediately — destruction is deferred until the GPU completes any submission
// that may reference it. This prevents use-after-free on DX12/Vulkan.
func (t *Texture) Release() {
	if t.released {
		return
	}
	// Surface textures are borrowed swapchain images. The wrapper never owns
	// their HAL lifetime, even while the acquisition is still active.
	if t.surface != nil {
		t.released = true
		return
	}
	t.released = true
	if q := t.device.queue; q != nil && q.pending != nil {
		q.pending.forgetZero(t)
	}

	// ADR-060: Release TrackerIndex to prevent monotonic growth (#307).
	// Surface textures handle this in Surface.destroySwapchainTexture.
	releaseTextureTracking(t.coreTexture)

	halDevice := t.device.halDevice()
	if halDevice == nil {
		return
	}

	t.viewMu.Lock()
	defaultView := t.defaultView
	t.defaultView = nil
	t.viewMu.Unlock()

	halTex := t.hal
	now := func() {
		if defaultView != nil {
			halDevice.DestroyTextureView(defaultView)
		}
		halDevice.DestroyTexture(halTex)
	}
	dq := t.device.destroyQueue()
	if dq == nil {
		now()
		return
	}
	// The shared default view goes first, at the same submission.
	later := func() {
		subIdx := t.device.lastSubmissionIndex()
		if defaultView != nil {
			t.viewDestroy = viewDestroy{dev: halDevice, view: defaultView}
			dq.DeferRelease(subIdx, "TextureView", &t.viewDestroy)
		}
		dq.Defer(subIdx, "Texture", func() {
			halDevice.DestroyTexture(halTex)
		})
	}
	// A texture the queued writes copy into stays until they have run.
	if q := t.device.queue; q != nil && q.pending != nil && q.pending.holdDestroy(t, now, later) {
		return
	}
	later()
}

// releaseTextureTracking gives a texture's tracker index back, after taking
// its state out of the device tracker: the next texture to get the index
// starts untracked instead of inheriting this one's last usage, which would
// make its first submit transition it from a layout it was never in.
func releaseTextureTracking(ct *core.Texture) {
	if ct == nil {
		return
	}
	td := ct.TrackingData()
	if td == nil || !td.Index().IsValid() || td.IsReleased() {
		return
	}
	if dev := ct.Device(); dev != nil && dev.Tracker() != nil {
		dev.Tracker().RemoveTexture(td.Index())
	}
	td.Release()
}

// TextureView represents a view into a texture.
type TextureView struct {
	hal          hal.TextureView
	device       *Device
	texture      *Texture
	baseMipLevel uint32
	released     bool
	surface      *core.Surface
	surfaceLease uint64
	destroy      viewDestroy // filled by Release
	// shared is set on a view of the texture's default HAL view, which the
	// texture owns: Release leaves it alone.
	shared bool
}

// resolveHAL is the single boundary from a public texture-view wrapper to HAL.
// Like Texture.resolveHAL, callers may resolve repeatedly within one serialized
// encoder operation without extending the surface acquisition lifetime.
func (v *TextureView) resolveHAL() hal.TextureView {
	if v == nil || v.released || v.hal == nil || v.device == nil || v.device.released.Load() {
		return nil
	}
	if v.surface != nil && !v.surface.AcquisitionValid(v.surfaceLease) {
		return nil
	}
	if v.surface != nil && v.texture != nil && v.texture.resolveHAL() == nil {
		return nil
	}
	return v.hal
}

// Texture returns the parent Texture that this view was created from.
// Returns nil if the view has been released.
func (v *TextureView) Texture() *Texture {
	if v == nil || v.released {
		return nil
	}
	return v.texture
}

// Release marks the texture view for destruction. The underlying HAL TextureView
// (and its descriptor heap slots) is not freed immediately — it is deferred via
// DestroyQueue until the GPU completes any submission that may reference it.
// This prevents descriptor use-after-free on DX12 with maxFramesInFlight=2
// (BUG-DX12-007).
func (v *TextureView) Release() {
	if v.released {
		return
	}
	v.released = true

	if v.device == nil || v.shared {
		return
	}

	halDevice := v.device.halDevice()
	if halDevice == nil {
		return
	}

	dq := v.device.destroyQueue()
	if dq == nil {
		halDevice.DestroyTextureView(v.hal)
		return
	}

	v.destroy = viewDestroy{dev: halDevice, view: v.hal}
	dq.DeferRelease(v.device.lastSubmissionIndex(), "TextureView", &v.destroy)
}

// viewDestroy destroys a released view's HAL view once the GPU is done with
// it. It lives in the TextureView, so the destroy queue holds it without a
// closure.
type viewDestroy struct {
	dev  hal.Device
	view hal.TextureView
}

// ReleaseDeferred implements core.Releaser.
func (d *viewDestroy) ReleaseDeferred() {
	d.dev.DestroyTextureView(d.view)
	*d = viewDestroy{}
}
