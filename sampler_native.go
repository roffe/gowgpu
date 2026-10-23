//go:build !(js && wasm)

package wgpu

import (
	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// Sampler represents a texture sampler.
type Sampler struct {
	hal      hal.Sampler
	device   *Device
	released bool
	// ref counts the application's reference and those of the bind groups
	// that bind the sampler; the HAL sampler is destroyed once all are
	// dropped.
	ref *core.ResourceRef
	// Filter modes and comparison retained for bind-time validation.
	magFilter    gputypes.FilterMode
	minFilter    gputypes.FilterMode
	mipmapFilter gputypes.FilterMode
	compare      gputypes.CompareFunction
}

// Release drops the application's reference to the sampler. The HAL sampler
// stays while a bind group binds it, and after the last of them it is
// destroyed once the GPU completes the submissions that may reference it.
func (s *Sampler) Release() {
	if s.released {
		return
	}
	s.released = true
	if s.ref != nil {
		s.ref.Drop()
	}
}

// withRef gives the sampler the reference count that destroys it.
func (s *Sampler) withRef(label string) *Sampler {
	s.ref = core.NewResourceRef("Sampler:"+label, s.destroyHAL)
	return s
}

// destroyHAL destroys the HAL sampler after the GPU completes the last
// submission.
func (s *Sampler) destroyHAL() {
	if s.device == nil {
		return
	}
	halDevice := s.device.halDevice()
	if halDevice == nil {
		return
	}
	dq := s.device.destroyQueue()
	if dq == nil {
		halDevice.DestroySampler(s.hal)
		return
	}
	halSampler := s.hal
	dq.Defer(s.device.lastSubmissionIndex(), "Sampler", func() {
		halDevice.DestroySampler(halSampler)
	})
}
