//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestUnsetMipmapFilterIsNearest samples a 16x16 texture with a mip level
// of its own colour each, minified 4 times into the 4x4 target, with a
// sampler that leaves MipmapFilter unset. WebGPU's default is nearest, so
// level 2 is picked; Metal took the unset filter as not mipmapped and
// sampled level 0.
func TestUnsetMipmapFilterIsNearest(t *testing.T) {
	device, h := newValidatedDevice(t)
	r := layoutRig{t, device}

	const side, levels = 16, 3
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size:          Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
		MipLevelCount: levels,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageTextureBinding | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	colors := [levels][4]byte{{255, 0, 0, 255}, {0, 255, 0, 255}, {0, 0, 255, 255}}
	for level, c := range colors {
		w := uint32(side >> level)
		data := make([]byte, w*w*4)
		for i := 0; i < len(data); i += 4 {
			copy(data[i:], c[:])
		}
		if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex, MipLevel: uint32(level)}, data,
			&ImageDataLayout{BytesPerRow: w * 4, RowsPerImage: w}, &Extent3D{Width: w, Height: w, DepthOrArrayLayers: 1}); err != nil {
			t.Fatalf("WriteTexture level %d: %v", level, err)
		}
	}
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer view.Release()
	sampler, err := device.CreateSampler(&SamplerDescriptor{
		MinFilter:   gputypes.FilterModeNearest,
		MagFilter:   gputypes.FilterModeNearest,
		LodMaxClamp: 32,
	})
	if err != nil {
		t.Fatalf("CreateSampler: %v", err)
	}
	defer sampler.Release()

	bgl := r.bgl(
		gputypes.BindGroupLayoutEntry{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		gputypes.BindGroupLayoutEntry{Binding: 1, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
	)
	pipe := r.renderPipeline(r.layout(bgl), `
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	return textureSample(tex, samp, pos.xy / 4.0);
}
`)
	group := r.group(bgl, BindGroupEntry{Binding: 0, TextureView: view}, BindGroupEntry{Binding: 1, Sampler: sampler})
	pix := r.draw(func(p *RenderPassEncoder) {
		p.SetPipeline(pipe)
		p.SetBindGroup(0, group, nil)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	expectPixels(t, "mip level 2", pix, 0, 0, 255, 255)
	assertNoValidation(t, h)
}
