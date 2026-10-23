//go:build !(js && wasm)

package core

import (
	"errors"
	"image"
	"slices"
	"unsafe"

	"github.com/roffe/gowgpu/hal"
)

// Surface lifecycle errors.
var (
	// ErrSurfaceNotConfigured is returned when attempting to acquire or present
	// on a surface that has not been configured.
	ErrSurfaceNotConfigured = errors.New("core: surface is not configured")

	// ErrSurfaceAlreadyAcquired is returned when attempting to acquire a texture
	// while one is already acquired.
	ErrSurfaceAlreadyAcquired = errors.New("core: surface texture already acquired")

	// ErrSurfaceNoTextureAcquired is returned when attempting to present or discard
	// without an acquired texture.
	ErrSurfaceNoTextureAcquired = errors.New("core: no surface texture acquired")

	// ErrSurfaceConfigureWhileAcquired is returned when attempting to configure
	// a surface while a texture is still acquired.
	ErrSurfaceConfigureWhileAcquired = errors.New("core: cannot configure surface while texture is acquired")

	// ErrSurfaceNilDevice is returned when a nil device is passed to Configure.
	ErrSurfaceNilDevice = errors.New("core: device must not be nil")

	// ErrSurfaceNilConfig is returned when a nil config is passed to Configure.
	ErrSurfaceNilConfig = errors.New("core: surface configuration must not be nil")
)

// SetPrepareFrame registers a platform hook that is called before acquiring a texture.
//
// The hook returns the current surface dimensions and whether they changed.
// If changed is true, the surface is automatically reconfigured before acquiring.
//
// Pass nil to remove the hook.
func (s *Surface) SetPrepareFrame(fn PrepareFrameFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prepareFrame = fn
}

// Configure configures the surface with the given device and settings.
//
// The surface must not have an acquired texture. If the surface is already
// configured, it will be reconfigured with the new settings. One configured
// with another device is unconfigured on that device first, and stays
// unconfigured if the new configuration fails.
//
// After Configure, the surface enters the Configured state and is ready
// to acquire textures. The configuration is copied; the surface does not
// retain the caller's pointer.
func (s *Surface) Configure(device *Device, config *hal.SurfaceConfiguration) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.waitAcquire()

	if device == nil {
		return ErrSurfaceNilDevice
	}
	if config == nil {
		return ErrSurfaceNilConfig
	}
	if s.state == SurfaceStateAcquired {
		return ErrSurfaceConfigureWhileAcquired
	}
	if s.raw == nil {
		return ErrSurfaceNotConfigured
	}

	halDevice := s.getHALDevice(device)
	if halDevice == nil {
		return ErrDeviceDestroyed
	}

	// A surface configured on another device lets that device's swapchain
	// go first, as wgpu-core unconfigures before it configures: a backend
	// cannot replace a swapchain with one of another device.
	if s.device != nil && s.device != device {
		if old := s.getHALDevice(s.device); old != nil {
			s.raw.Unconfigure(old)
		}
		s.invalidateAcquisitionLocked()
		s.device, s.config = nil, nil
		s.state = SurfaceStateUnconfigured
	}

	// Keep the caller's configuration private to the surface. Backends may
	// inspect the configuration after Configure returns, so pass the copy to
	// the HAL as well as storing it only after a successful configure.
	configCopy := *config
	configCopy.ViewFormats = slices.Clone(config.ViewFormats)
	if err := s.raw.Configure(halDevice, &configCopy); err != nil {
		return err
	}

	s.invalidateAcquisitionLocked()
	s.device = device
	s.config = &configCopy
	s.state = SurfaceStateConfigured
	return nil
}

// Unconfigure removes the surface configuration and returns to the Unconfigured state.
//
// If a texture is currently acquired, it is discarded first.
// If the surface is already unconfigured, this is a no-op.
func (s *Surface) Unconfigure() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.waitAcquire()

	if s.state == SurfaceStateUnconfigured {
		return
	}

	// Discard acquired texture if any
	if s.state == SurfaceStateAcquired && s.acquiredTex != nil {
		s.raw.DiscardTexture(s.acquiredTex)
		s.acquiredTex = nil
	}
	s.invalidateAcquisitionLocked()

	halDevice := s.getHALDevice(s.device)
	if halDevice != nil {
		s.raw.Unconfigure(halDevice)
	}

	s.device = nil
	s.config = nil
	s.state = SurfaceStateUnconfigured
}

// AcquireTexture acquires the next surface texture for rendering.
//
// The surface must be in the Configured state. If a PrepareFrame hook is
// registered and reports that dimensions changed, the surface is automatically
// reconfigured before acquiring.
//
// After a successful acquire, the surface enters the Acquired state.
// The caller must either Present or DiscardTexture before acquiring again.
func (s *Surface) AcquireTexture(fence hal.Fence) (*hal.AcquiredSurfaceTexture, error) {
	result, _, err := s.AcquireTextureWithLease(fence)
	return result, err
}

// AcquireTextureWithLease acquires a texture and returns its opaque lifetime
// lease. Call AcquisitionValid before converting a retained public wrapper to
// HAL; the lease expires on present, discard, unconfigure, or destruction.
//
// The PrepareFrame hook and the backend's acquire, which can wait for the
// next image, run without the surface's lock, so the hook may call back into
// the surface and other goroutines can read it meanwhile. A Configure,
// Unconfigure or Destroy waits for the acquire.
func (s *Surface) AcquireTextureWithLease(fence hal.Fence) (*hal.AcquiredSurfaceTexture, uint64, error) {
	s.mu.Lock()
	if err := s.acquireStateErr(); err != nil {
		s.mu.Unlock()
		return nil, 0, err
	}
	hook := s.prepareFrame
	s.mu.Unlock()

	var width, height uint32
	var changed bool
	if hook != nil {
		width, height, changed = hook()
	}

	s.mu.Lock()
	s.waitAcquire()
	if err := s.acquireStateErr(); err != nil {
		s.mu.Unlock()
		return nil, 0, err
	}
	if changed {
		if err := s.resizeLocked(width, height); err != nil {
			s.mu.Unlock()
			return nil, 0, err
		}
	}
	raw := s.raw
	s.acquiring = true
	s.mu.Unlock()

	result, err := raw.AcquireTexture(fence)

	s.mu.Lock()
	defer s.mu.Unlock()
	s.acquiring = false
	s.acquired.Broadcast()
	if err != nil {
		return nil, 0, err
	}

	s.acquiredTex = result.Texture
	s.nextAcquisition++
	if s.nextAcquisition == 0 {
		s.nextAcquisition++
	}
	s.acquisition = s.nextAcquisition
	s.state = SurfaceStateAcquired
	return result, s.acquisition, nil
}

// AcquisitionValid reports whether lease still identifies this surface's
// current acquired texture.
func (s *Surface) AcquisitionValid(lease uint64) bool {
	if s == nil || lease == 0 {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state == SurfaceStateAcquired && s.acquisition == lease
}

// Present presents the acquired surface texture to the screen.
//
// The surface must be in the Acquired state. After presenting, the surface
// returns to the Configured state and is ready to acquire again.
func (s *Surface) Present(queue hal.Queue) error {
	return s.PresentWithDamage(queue, nil)
}

// PresentWithDamage presents the acquired surface texture to the screen,
// passing optional damage rectangles to the compositor.
//
// damageRects specifies which regions of the surface changed this frame
// (physical pixels, top-left origin). When nil or empty, the entire surface
// is presented — identical code path to Present(). Backends that support
// damage rects use them as compositor hints; others accept and ignore them.
//
// The surface must be in the Acquired state. After presenting, the surface
// returns to the Configured state and is ready to acquire again.
func (s *Surface) PresentWithDamage(queue hal.Queue, damageRects []image.Rectangle) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state != SurfaceStateAcquired {
		return ErrSurfaceNoTextureAcquired
	}

	err := queue.Present(s.raw, s.acquiredTex, damageRects)
	s.acquiredTex = nil
	s.invalidateAcquisitionLocked()
	s.state = SurfaceStateConfigured
	return err
}

// DiscardTexture discards the acquired surface texture without presenting it.
//
// Use this if rendering failed or was canceled. If no texture is acquired,
// this is a no-op.
func (s *Surface) DiscardTexture() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.state != SurfaceStateAcquired {
		return
	}

	if s.acquiredTex != nil {
		s.raw.DiscardTexture(s.acquiredTex)
	}

	s.acquiredTex = nil
	s.invalidateAcquisitionLocked()
	s.state = SurfaceStateConfigured
}

func (s *Surface) invalidateAcquisitionLocked() {
	s.acquisition = 0
}

// RetireDevice clears the logical configuration after its HAL device has been
// destroyed. The public API calls this after backend device teardown so a
// retained surface can be released or configured with another device without
// retaining a valid acquisition from the old device.
func (s *Surface) RetireDevice(device *Device) {
	if s == nil || device == nil {
		return
	}
	s.mu.Lock()
	s.waitAcquire()
	if s.device != device {
		s.mu.Unlock()
		return
	}
	s.acquiredTex = nil
	s.invalidateAcquisitionLocked()
	s.device = nil
	s.config = nil
	s.state = SurfaceStateUnconfigured
	s.mu.Unlock()
}

// Destroy invalidates the active acquisition and releases the owned HAL
// surface. It deliberately does not call DiscardTexture: device-first instance
// teardown may already have retired the backend swapchain.
func (s *Surface) Destroy() {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.waitAcquire()
	raw := s.raw
	if raw == nil {
		s.mu.Unlock()
		return
	}
	acquiredTex := s.acquiredTex
	halDevice := s.getHALDevice(s.device)
	s.raw = nil
	s.acquiredTex = nil
	s.invalidateAcquisitionLocked()
	s.device = nil
	s.config = nil
	s.state = SurfaceStateUnconfigured
	s.mu.Unlock()

	// When the configured device is still alive, retire the current acquisition
	// and configuration before destroying the platform surface. Device-first
	// teardown leaves no HAL device here, so the backend has already retired (or
	// abandoned) its device-owned children and only the platform handle remains.
	if acquiredTex != nil {
		raw.DiscardTexture(acquiredTex)
	}
	if halDevice != nil {
		raw.Unconfigure(halDevice)
	}
	raw.Destroy()
	untrackResource(uintptr(unsafe.Pointer(s))) //nolint:gosec // debug tracking uses pointer as unique ID
}

// State returns the current lifecycle state of the surface.
func (s *Surface) State() SurfaceState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.state
}

// Config returns a copy of the current surface configuration.
// Returns nil if the surface is unconfigured. Mutating the returned value does
// not change the surface or reconfigure its HAL surface.
func (s *Surface) Config() *hal.SurfaceConfiguration {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.config == nil {
		return nil
	}
	configCopy := *s.config
	configCopy.ViewFormats = slices.Clone(s.config.ViewFormats)
	return &configCopy
}

// acquireStateErr is the error of an acquire in the surface's state, nil
// when it can acquire. s.mu is held.
func (s *Surface) acquireStateErr() error {
	switch {
	case s.state == SurfaceStateAcquired:
		return ErrSurfaceAlreadyAcquired
	case s.state != SurfaceStateConfigured:
		return ErrSurfaceNotConfigured
	}
	return nil
}

// waitAcquire waits until no acquire is waiting in the backend. s.mu is
// held.
func (s *Surface) waitAcquire() {
	for s.acquiring {
		if s.acquired.L == nil {
			s.acquired.L = &s.mu
		}
		s.acquired.Wait()
	}
}

// resizeLocked reconfigures the surface to the size the PrepareFrame hook
// reported. s.mu is held.
func (s *Surface) resizeLocked(width, height uint32) error {
	if s.config == nil {
		return nil
	}
	if s.raw == nil {
		return ErrSurfaceNotConfigured
	}

	newConfig := *s.config
	newConfig.Width = width
	newConfig.Height = height

	halDevice := s.getHALDevice(s.device)
	if halDevice == nil {
		return ErrDeviceDestroyed
	}

	if err := s.raw.Configure(halDevice, &newConfig); err != nil {
		return err
	}
	s.config = &newConfig
	return nil
}

// getHALDevice extracts the hal.Device from a core.Device using the snatch lock.
// Returns nil if the device has been destroyed or has no HAL integration.
// Must NOT be called with s.mu held if the device's snatch lock could deadlock;
// in practice the snatch lock is independent so this is safe.
func (s *Surface) getHALDevice(device *Device) hal.Device {
	if device == nil || device.SnatchLock() == nil {
		return nil
	}
	guard := device.SnatchLock().Read()
	defer guard.Release()
	return device.Raw(guard)
}
