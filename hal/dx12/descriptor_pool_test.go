//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// TestManyViewsAndSamplers keeps more render target and depth views alive
// than one CPU descriptor heap holds, and more samplers than the
// shader-visible sampler heap holds, of a few distinct descriptions.
func TestManyViewsAndSamplers(t *testing.T) {
	dev, _ := openTestDevice(t)
	var views []hal.TextureView
	t.Cleanup(func() {
		for _, v := range views {
			dev.DestroyTextureView(v)
		}
	})
	for i := 0; i < 512; i++ {
		format := gputypes.TextureFormatRGBA8Unorm
		if i%2 == 1 {
			format = gputypes.TextureFormatDepth24Plus
		}
		tex := newTestTexture(t, dev, hal.TextureDescriptor{
			Size:      hal.Extent3D{Width: 4, Height: 4},
			Dimension: gputypes.TextureDimension2D,
			Format:    format,
			Usage:     gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageTextureBinding,
		})
		view, err := dev.CreateTextureView(tex, &hal.TextureViewDescriptor{})
		if err != nil {
			t.Fatalf("CreateTextureView %d (%v): %v", i, format, err)
		}
		views = append(views, view)
	}

	var samplers []hal.Sampler
	t.Cleanup(func() {
		for _, s := range samplers {
			dev.DestroySampler(s)
		}
	})
	for i := 0; i < 3000; i++ {
		filter := gputypes.FilterModeNearest
		if i%2 == 1 {
			filter = gputypes.FilterModeLinear
		}
		s, err := dev.CreateSampler(&hal.SamplerDescriptor{MinFilter: filter, MagFilter: filter, LodMaxClamp: 32})
		if err != nil {
			t.Fatalf("CreateSampler %d: %v", i, err)
		}
		samplers = append(samplers, s)
	}
	if a, b := samplers[0].(*Sampler), samplers[2].(*Sampler); a.samplerPoolSlot != b.samplerPoolSlot {
		t.Errorf("alike samplers take slots %d and %d, want one", a.samplerPoolSlot, b.samplerPoolSlot)
	}
	if a, b := samplers[0].(*Sampler), samplers[1].(*Sampler); a.samplerPoolSlot == b.samplerPoolSlot {
		t.Errorf("different samplers share slot %d", a.samplerPoolSlot)
	}
}
