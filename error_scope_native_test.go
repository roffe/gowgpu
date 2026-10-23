//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/hal/noop"
)

// Errors that calls on a device return go to the innermost error scope that
// captures their kind too, as WebGPU's error scopes have them: a texture
// WebGPU refuses, and an encoder's error at Finish.
func TestErrorScopesCaptureTheErrorsCallsReturn(t *testing.T) {
	device := newCaptureDevice(&captureDevice{Device: &noop.Device{}, commandEncoder: &noop.CommandEncoder{}})

	device.PushErrorScope(ErrorFilterValidation)
	device.PushErrorScope(ErrorFilterOutOfMemory)
	if _, err := device.CreateTexture(&TextureDescriptor{Size: Extent3D{Width: 0, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1,
		SampleCount: 1, Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageCopyDst}); err == nil {
		t.Fatal("CreateTexture of width 0 succeeded")
	}
	if got := device.PopErrorScope(); got != nil {
		t.Errorf("out-of-memory scope captured %v", got)
	}
	got := device.PopErrorScope()
	if got == nil || got.Type != ErrorFilterValidation {
		t.Fatalf("validation scope captured %v, want the texture's validation error", got)
	}

	device.PushErrorScope(ErrorFilterValidation)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToBuffer(nil, 0, nil, 0, 4)
	if _, err := enc.Finish(); err == nil {
		t.Fatal("Finish succeeded after a copy of nil buffers")
	}
	if got := device.PopErrorScope(); got == nil {
		t.Error("validation scope captured nothing from the encoder's error")
	}

	if _, err := device.CreateTexture(nil); err == nil {
		t.Fatal("CreateTexture(nil) succeeded")
	}
}
