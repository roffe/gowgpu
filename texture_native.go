//go:build !(js && wasm)

package wgpu

import (
	"sync"
	"sync/atomic"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// Texture represents a GPU texture.
type Texture struct {
	hal          hal.Texture
	device       *Device
	format       gputypes.TextureFormat
	viewFormats  []gputypes.TextureFormat
	released     atomic.Bool
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

	// zeroable is set when the queue can zero the texture, as it zeroes new
	// ones, which a texture whose contents a pass discarded needs again.
	zeroable bool

	// defaultView is the HAL view CreateTextureView hands out for a nil
	// descriptor: one per texture, shared by every TextureView made that way
	// and destroyed with the texture. A view a frame of the same texture is
	// then no new HAL view, and on Vulkan no new framebuffer.
	viewMu          sync.Mutex
	defaultView     hal.TextureView
	defaultViewDesc hal.TextureViewDescriptor
	viewDestroy     viewDestroy
}

// sharedDefaultView returns the texture's default HAL view, making it the
// first time. nil means it could not be made, and the caller makes its own.
func (t *Texture) sharedDefaultView(dev hal.Device, raw hal.Texture) (hal.TextureView, hal.TextureViewDescriptor) {
	t.viewMu.Lock()
	defer t.viewMu.Unlock()
	if t.defaultView == nil && !t.released.Load() {
		var desc hal.TextureViewDescriptor
		if t.coreTexture != nil {
			resolved, err := core.ResolveTextureViewDescriptor(t.coreTexture, t.viewFormats, nil)
			if err != nil {
				return nil, desc
			}
			desc = resolved
		}
		v, err := dev.CreateTextureView(raw, &desc)
		if err != nil {
			return nil, desc
		}
		t.defaultView, t.defaultViewDesc = v, desc
	}
	return t.defaultView, t.defaultViewDesc
}

// resolveHAL is the single boundary from a public texture wrapper to HAL.
// Surface textures are borrowed and are usable only for their acquisition.
// Encoder validation and HAL conversion may call this more than once during
// one operation; acquisition and presentation are serialized by the render
// loop, so the lease remains stable across those calls.
func (t *Texture) resolveHAL() hal.Texture {
	if t == nil || t.released.Load() || t.hal == nil || t.device == nil || t.device.released.Load() {
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
// that may reference it. This prevents use-after-free on DX12/Vulkan. A Release
// while another goroutine submits waits for that Submit.
func (t *Texture) Release() {
	// A surface texture's Texture is the acquisition's, which ends at the
	// present or discard.
	if t.surface != nil {
		return
	}
	// A Submit holds the queue lock from checking its textures until its
	// submission is the last one, which the destroy below waits for.
	if q := t.device.queue; q != nil {
		q.mu.Lock()
		defer q.mu.Unlock()
	}
	if t.released.Swap(true) {
		return
	}
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

// dropDefaultView destroys the texture's default view, once the GPU is done
// with the submissions made so far.
func (t *Texture) dropDefaultView() {
	t.viewMu.Lock()
	view := t.defaultView
	t.defaultView = nil
	t.viewMu.Unlock()
	halDevice := t.device.halDevice()
	if view == nil || halDevice == nil {
		return
	}
	dq := t.device.destroyQueue()
	if dq == nil {
		halDevice.DestroyTextureView(view)
		return
	}
	t.viewDestroy = viewDestroy{dev: halDevice, view: view}
	dq.DeferRelease(t.device.lastSubmissionIndex(), "TextureView", &t.viewDestroy)
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
	hal     hal.TextureView
	device  *Device
	texture *Texture
	// desc is the view's descriptor with WebGPU's defaults resolved, as the
	// HAL got it; zero for a view of a texture core does not know.
	desc         hal.TextureViewDescriptor
	released     bool
	surface      *core.Surface
	surfaceLease uint64
	destroy      viewDestroy // filled when the last reference is dropped
	// ref counts the application's reference and those of the bind groups
	// and command buffers that use the view; the HAL view is destroyed once
	// all are dropped. nil on a view of the texture's default HAL view,
	// which the texture owns.
	ref *core.ResourceRef
	// mark is the last command encoder that took a reference.
	mark encoderMark
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

// subresources selects the subresources of its texture the view takes in,
// every one of a texture core does not know.
func (v *TextureView) subresources() track.TextureSelector {
	if v.texture == nil || v.texture.coreTexture == nil {
		return track.TextureSelector{}
	}
	cv := core.TextureView{
		Parent:          v.texture.coreTexture,
		BaseMipLevel:    v.desc.BaseMipLevel,
		MipLevelCount:   v.desc.MipLevelCount,
		BaseArrayLayer:  v.desc.BaseArrayLayer,
		ArrayLayerCount: v.desc.ArrayLayerCount,
		Aspect:          v.desc.Aspect,
	}
	return cv.Subresources()
}

// copySubresources selects the subresources of t a copy of size at base
// reads or writes: one mip level, and its layers, but for a 3D texture,
// whose depth slices are not subresources.
func (t *Texture) copySubresources(base *hal.ImageCopyTexture, size *hal.Extent3D) track.TextureSelector {
	if t == nil || t.coreTexture == nil {
		return track.TextureSelector{}
	}
	sel := track.TextureSelector{
		Mips:    track.Range{Start: base.MipLevel, End: base.MipLevel + 1},
		Aspects: track.AspectsOf(base.Aspect),
	}
	if t.coreTexture.Dimension() == gputypes.TextureDimension3D {
		sel.Layers = t.coreTexture.Subresources().Layers
	} else {
		sel.Layers = track.Range{Start: base.Origin.Z, End: base.Origin.Z + max(size.DepthOrArrayLayers, 1)}
	}
	return sel
}

// boundViewSubresources is the subresources the texture views of entries
// take in, in the order collectBindGroupResources lists their textures.
func boundViewSubresources(entries []BindGroupEntry) []track.TextureSelector {
	var sels []track.TextureSelector
	for i := range entries {
		if v := entries[i].TextureView; v != nil && v.texture != nil {
			sels = append(sels, v.subresources())
		}
	}
	return sels
}

// Texture returns the parent Texture that this view was created from.
// Returns nil if the view has been released.
func (v *TextureView) Texture() *Texture {
	if v == nil || v.released {
		return nil
	}
	return v.texture
}

// Release drops the application's reference to the texture view. The HAL
// view stays while a bind group or an unsubmitted command buffer uses it, and
// after the last of them it is destroyed once the GPU completes the
// submissions that may reference it.
func (v *TextureView) Release() {
	if v.released {
		return
	}
	v.released = true
	if v.ref != nil {
		v.ref.Drop()
	}
}

// withRef gives a view that owns its HAL view the reference count that
// destroys it.
func (v *TextureView) withRef(label string) *TextureView {
	v.ref = core.NewResourceRef("TextureView:"+label, v.destroyHAL)
	return v
}

// destroyHAL destroys the HAL view after the GPU completes the last
// submission.
func (v *TextureView) destroyHAL() {
	if v.device == nil {
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
