//go:build !(js && wasm)

package wgpu

import (
	"log/slog"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/roffe/gowgpu/hal"
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

// An error no scope captures goes to the newest handler set, one a scope
// captures goes to no handler, and after SetUncapturedErrorHandler(nil) the
// uncaptured ones are logged again. Setting handlers while other goroutines
// make errors is safe (go test -race).
func TestUncapturedErrorHandler(t *testing.T) {
	device := newCaptureDevice(&captureDevice{Device: &noop.Device{}})
	badTexture := func() error {
		_, err := device.CreateTexture(&TextureDescriptor{Size: Extent3D{Width: 0, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1,
			SampleCount: 1, Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageCopyDst})
		return err
	}

	var got []*GPUError
	device.SetUncapturedErrorHandler(func(*GPUError) { t.Error("the replaced handler got an error") })
	device.SetUncapturedErrorHandler(func(e *GPUError) { got = append(got, e) })
	err := badTexture()
	if err == nil {
		t.Fatal("CreateTexture of width 0 succeeded")
	}
	if len(got) != 1 || got[0].Type != ErrorFilterValidation || got[0].Message != err.Error() {
		t.Fatalf("handler got %v, want the texture's validation error %q", got, err)
	}

	device.PushErrorScope(ErrorFilterValidation)
	_ = badTexture()
	if device.PopErrorScope() == nil {
		t.Error("the scope captured nothing")
	}
	if len(got) != 1 {
		t.Errorf("the handler got the error the scope captured: %v", got[1:])
	}

	logs := &captureHandler{}
	prev := hal.Logger()
	hal.SetLogger(slog.New(logs))
	t.Cleanup(func() { hal.SetLogger(prev) })
	device.SetUncapturedErrorHandler(nil)
	_ = badTexture()
	if len(got) != 1 {
		t.Errorf("the handler got an error after SetUncapturedErrorHandler(nil): %v", got[1:])
	}
	logs.mu.Lock()
	logged := slices.ContainsFunc(logs.msgs, func(m string) bool { return strings.HasPrefix(m, "wgpu: uncaptured error") })
	logs.mu.Unlock()
	if !logged {
		t.Error("no uncaptured error logged without a handler")
	}

	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for range 50 {
				_ = badTexture()
			}
		})
	}
	for range 50 {
		device.SetUncapturedErrorHandler(func(*GPUError) {})
		device.SetUncapturedErrorHandler(nil)
	}
	wg.Wait()
}
