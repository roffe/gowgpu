//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package hal

import "github.com/roffe/gowgpu/gputypes"

// Backend notes
//
// The backends are hal/vulkan, hal/metal, hal/dx12 and hal/gles, plus hal/noop
// for tests. Each registers itself from init() with RegisterBackend, and
// hal/allbackends imports the ones a platform has.
//
// Native calls go through cgo trampolines (Vulkan, Metal, EGL and GL) or
// syscall (DX12, WGL), with the libraries opened at run time. See
// docs/ARCHITECTURE.md.
//
// A backend implements every hal.Backend method, reports what it can do
// through its adapters' features, limits and downlevel flags, and refuses a
// surface target kind it can't use before it touches a handle.

// BackendInfo provides metadata about a backend implementation.
type BackendInfo struct {
	// Variant identifies the backend type.
	Variant gputypes.Backend

	// Name is a human-readable backend name.
	Name string

	// Version of the backend implementation.
	Version string

	// Features supported by this backend.
	Features BackendFeatures

	// Limitations of this backend.
	Limitations BackendLimitations
}

// BackendFeatures describes capabilities of a backend.
type BackendFeatures struct {
	// SupportsCompute indicates compute shader support.
	SupportsCompute bool

	// SupportsMultiQueue indicates multiple queue support.
	SupportsMultiQueue bool

	// SupportsRayTracing indicates ray tracing support.
	SupportsRayTracing bool

	// MaxTextureSize is the maximum texture dimension.
	MaxTextureSize uint32

	// MaxBufferSize is the maximum buffer size.
	MaxBufferSize uint64
}

// BackendLimitations describes known limitations.
type BackendLimitations struct {
	// NoAsyncCompute means compute must run on graphics queue.
	NoAsyncCompute bool

	// LimitedFormats means some texture formats unavailable.
	LimitedFormats bool

	// NoBindlessResources means bindless not supported.
	NoBindlessResources bool
}

// BackendFactory creates backend instances.
// This allows lazy initialization of backends.
type BackendFactory func() (Backend, error)

// registeredFactories holds lazy backend factories.
var registeredFactories = make(map[gputypes.Backend]BackendFactory)

// RegisterBackendFactory registers a factory for lazy backend creation.
// This is preferred over RegisterBackend for backends that may fail
// initialization (e.g., missing GPU drivers).
func RegisterBackendFactory(variant gputypes.Backend, factory BackendFactory) {
	backendsMu.Lock()
	defer backendsMu.Unlock()
	registeredFactories[variant] = factory
}

// CreateBackend creates a backend instance using registered factory.
// Returns error if no factory is registered for the variant.
func CreateBackend(variant gputypes.Backend) (Backend, error) {
	backendsMu.RLock()
	factory, ok := registeredFactories[variant]
	backendsMu.RUnlock()

	if !ok {
		return nil, ErrBackendNotFound
	}
	return factory()
}

// ProbeBackend tests if a backend is available without fully initializing it.
// Returns BackendInfo if available, error otherwise.
func ProbeBackend(variant gputypes.Backend) (*BackendInfo, error) {
	// First check if already registered
	_, ok := GetBackend(variant)
	if ok {
		return &BackendInfo{
			Variant: variant,
			Name:    variant.String(),
			Version: "1.0.0",
		}, nil
	}

	// Try factory
	backendsMu.RLock()
	factory, hasFactory := registeredFactories[variant]
	backendsMu.RUnlock()

	if !hasFactory {
		return nil, ErrBackendNotFound
	}

	// Create and immediately check
	b, err := factory()
	if err != nil {
		return nil, err
	}

	// Register for future use
	RegisterBackend(b)

	return &BackendInfo{
		Variant: b.Variant(),
		Name:    b.Variant().String(),
		Version: "1.0.0",
	}, nil
}

// SelectBestBackend chooses the most capable available backend.
// Priority: Vulkan > Metal > DX12 > OpenGL > Noop
func SelectBestBackend() (Backend, error) {
	priority := []gputypes.Backend{
		gputypes.BackendVulkan,
		gputypes.BackendMetal,
		gputypes.BackendDX12,
		gputypes.BackendGL,
		gputypes.BackendEmpty, // noop
	}

	for _, variant := range priority {
		backend, ok := GetBackend(variant)
		if ok {
			return backend, nil
		}

		// Try factory
		backendsMu.RLock()
		factory, hasFactory := registeredFactories[variant]
		backendsMu.RUnlock()

		if hasFactory {
			if b, err := factory(); err == nil {
				RegisterBackend(b)
				return b, nil
			}
		}
	}

	return nil, ErrBackendNotFound
}
