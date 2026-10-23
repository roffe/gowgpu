//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestClearIntegerTargets clears an R32Uint target to 4000 and an RGBA8Sint
// one to -5, 7, -128 and 127 and reads them back. Vulkan passed every clear
// value as float32 bits, so the R32Uint target read 0x457a0000.
func TestClearIntegerTargets(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL {
		t.Skip("GL clears every colour attachment with glClearBufferfv")
	}
	for _, tc := range []struct {
		format gputypes.TextureFormat
		clear  gputypes.Color
		want   []byte
	}{
		{gputypes.TextureFormatR32Uint, gputypes.Color{R: 4000}, binary.LittleEndian.AppendUint32(nil, 4000)},
		{gputypes.TextureFormatRGBA8Sint, gputypes.Color{R: -5, G: 7, B: -128, A: 127}, []byte{0xfb, 7, 0x80, 127}},
	} {
		tex, err := device.CreateTexture(&TextureDescriptor{
			Label:         tc.format.String(),
			Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
			MipLevelCount: 1,
			SampleCount:   1,
			Dimension:     TextureDimension2D,
			Format:        tc.format,
			Usage:         TextureUsageRenderAttachment | TextureUsageCopySrc,
		})
		if err != nil {
			t.Fatalf("CreateTexture(%s): %v", tc.format, err)
		}
		view, err := device.CreateTextureView(tex, nil)
		if err != nil {
			t.Fatalf("CreateTextureView: %v", err)
		}
		defer releaseTextureView(tex, view)
		st := readbackBuffer(t, device, "integer-clear-st")
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		clearPass(t, enc, view, gputypes.LoadOpClear, tc.clear, nil)
		copyOut(enc, tex, st)
		finishAndSubmit(t, device, enc)
		if got := readback(t, st)[:4]; string(got) != string(tc.want) {
			t.Errorf("%s cleared to %v reads % x, want % x", tc.format, tc.clear, got, tc.want)
		}
	}
	assertNoValidation(t, h)
}
