//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// copyTexelsShader copies a sampled texture into a storage texture, the
// first texture and the first image binding of its layout.
const copyTexelsShader = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
	textureStore(dst, vec2<i32>(id.xy), textureLoad(src, vec2<i32>(id.xy), 0));
}
`

// TestComputeCopiesSampledIntoStorageTexture runs a compute shader that
// loads a sampled texture and stores into a storage texture. GL bound the
// storage texture to the texture unit numbered like its image unit, over the
// sampled texture, and never to an image unit, so nothing was stored.
func TestComputeCopiesSampledIntoStorageTexture(t *testing.T) {
	device, h := newValidatedDevice(t)
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 1, Visibility: ShaderStageCompute, StorageTexture: &gputypes.StorageTextureBindingLayout{
			Access: gputypes.StorageTextureAccessWriteOnly, Format: gputypes.TextureFormatRGBA8Unorm, ViewDimension: gputypes.TextureViewDimension2D}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer layout.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{layout}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "copy-texels", WGSL: copyTexelsShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "copy-texels", Layout: pl, Module: module, EntryPoint: "cs_main"})
	if err != nil {
		if strings.Contains(err.Error(), "ComputeShaders") {
			t.Skipf("no compute shaders on this adapter: %v", err)
		}
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipe.Release()

	src, srcView := pixTexture(t, device, "src", TextureUsageRenderAttachment|TextureUsageTextureBinding)
	defer releaseTextureView(src, srcView)
	dst, dstView := pixTexture(t, device, "dst", TextureUsageStorageBinding|TextureUsageCopySrc)
	defer releaseTextureView(dst, dstView)
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{
		{Binding: 0, TextureView: srcView}, {Binding: 1, TextureView: dstView},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()
	st := readbackBuffer(t, device, "dst-staging")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, srcView, gputypes.LoadOpClear, green, nil)
	cpass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cpass.SetPipeline(pipe)
	cpass.SetBindGroup(0, group, nil)
	cpass.Dispatch(pixSide, pixSide, 1)
	if err := cpass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	copyOut(enc, dst, st)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "the stored copy", readback(t, st), 0, 255, 0, 255)
	assertNoValidation(t, h)
}
