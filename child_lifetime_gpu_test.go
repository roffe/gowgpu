//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/noop"
)

// sampleAtShader fills the target with the texel a sampler picks at u=0.375
// of a 2x1 texture: texel 0 with nearest filtering, a 3:1 blend of texels 0
// and 1 with linear.
const sampleAtShader = `
@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[vid], 0.0, 1.0);
}

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return textureSampleLevel(t, s, vec2<f32>(0.375, 0.5), 0.0);
}
`

// A bind group keeps the view and the sampler it binds, and a pipeline its
// layout, after the application releases them: the group still samples the
// red texel with its nearest sampler, after a linear sampler made since
// could have taken the released one's slot.
func TestReleasedViewSamplerAndLayoutStayWithTheirUsers(t *testing.T) {
	device, h := newValidatedDevice(t)
	queue := device.Queue()

	src, err := device.CreateTexture(&TextureDescriptor{
		Label:         "lifetime-src",
		Size:          Extent3D{Width: 2, Height: 1, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageTextureBinding | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer src.Release()
	if err := queue.WriteTexture(&ImageCopyTexture{Texture: src}, []byte{255, 0, 0, 255, 0, 255, 0, 255},
		&ImageDataLayout{BytesPerRow: 8, RowsPerImage: 1}, &Extent3D{Width: 2, Height: 1, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	view, err := device.CreateTextureView(src, &TextureViewDescriptor{
		Format: TextureFormatRGBA8Unorm, Dimension: gputypes.TextureViewDimension2D,
		MipLevelCount: 1, ArrayLayerCount: 1,
	})
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	nearest, err := device.CreateSampler(&SamplerDescriptor{
		Label:        "nearest",
		AddressModeU: gputypes.AddressModeClampToEdge, AddressModeV: gputypes.AddressModeClampToEdge,
		AddressModeW: gputypes.AddressModeClampToEdge,
		MagFilter:    gputypes.FilterModeNearest, MinFilter: gputypes.FilterModeNearest,
		MipmapFilter: gputypes.FilterModeNearest, LodMaxClamp: 32,
	})
	if err != nil {
		t.Fatalf("CreateSampler: %v", err)
	}
	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "sample-at", WGSL: sampleAtShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: gputypes.ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{
			SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D,
		}},
		{Binding: 1, Visibility: gputypes.ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	layout, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{
		{Binding: 0, TextureView: view},
		{Binding: 1, Sampler: nearest},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	pipeline, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Layout: layout,
		Vertex: VertexState{Module: shader, EntryPoint: "vs_main"},
		Fragment: &FragmentState{Module: shader, EntryPoint: "fs_main", Targets: []gputypes.ColorTargetState{
			{Format: TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll},
		}},
		Primitive:   gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Multisample: gputypes.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}

	view.Release()
	nearest.Release()
	layout.Release()
	for name, n := range map[string]int32{
		"view": view.ref.RefCount(), "sampler": nearest.ref.RefCount(), "layout": layout.ref.RefCount(),
	} {
		if n != 1 {
			t.Errorf("released %s: %d references, want the 1 its user holds", name, n)
		}
	}
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	linear, err := device.CreateSampler(&SamplerDescriptor{
		Label:        "linear",
		AddressModeU: gputypes.AddressModeClampToEdge, AddressModeV: gputypes.AddressModeClampToEdge,
		AddressModeW: gputypes.AddressModeClampToEdge,
		MagFilter:    gputypes.FilterModeLinear, MinFilter: gputypes.FilterModeLinear,
		MipmapFilter: gputypes.FilterModeLinear, LodMaxClamp: 32,
	})
	if err != nil {
		t.Fatalf("CreateSampler(linear): %v", err)
	}
	defer linear.Release()

	target, targetView := pixTexture(t, device, "lifetime-target", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(target, targetView)
	staging := readbackBuffer(t, device, "lifetime-staging")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, targetView, gputypes.LoadOpClear, black, func(p *RenderPassEncoder) {
		p.SetPipeline(pipeline)
		p.SetBindGroup(0, group, nil)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, target, staging)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "sampled through the released view and sampler", readback(t, staging), 255, 0, 0, 255)

	group.Release()
	pipeline.Release()
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	for name, n := range map[string]int32{
		"view": view.ref.RefCount(), "sampler": nearest.ref.RefCount(), "layout": layout.ref.RefCount(),
	} {
		if n != 0 {
			t.Errorf("%s: %d references after its users were released, want 0", name, n)
		}
	}
	assertNoValidation(t, h)
}

// A command buffer keeps the views of its render pass attachments until the
// GPU is done with it, however early the application releases them.
func TestReleasedAttachmentViewStaysForItsCommandBuffer(t *testing.T) {
	device, h := newValidatedDevice(t)

	target, _ := pixTexture(t, device, "attachment-target", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer target.Release()
	view, err := device.CreateTextureView(target, &TextureViewDescriptor{
		Format: TextureFormatRGBA8Unorm, Dimension: gputypes.TextureViewDimension2D,
		MipLevelCount: 1, ArrayLayerCount: 1,
	})
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	staging := readbackBuffer(t, device, "attachment-staging")
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, green, nil)
	copyOut(enc, target, staging)
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}

	view.Release()
	if n := view.ref.RefCount(); n != 1 {
		t.Errorf("released attachment view: %d references, want the 1 its command buffer holds", n)
	}
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	expectPixels(t, "cleared through the released view", readback(t, staging), 0, 255, 0, 255)
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	if n := view.ref.RefCount(); n != 0 {
		t.Errorf("attachment view: %d references after its command buffer ran, want 0", n)
	}
	assertNoValidation(t, h)
}

// Submit refuses a command buffer whose render pass writes timestamps into
// a query set released since.
func TestSubmitRefusesReleasedTimestampQuerySet(t *testing.T) {
	h := &captureDevice{Device: &noop.Device{}}
	device := newCaptureDevice(h)
	h.commandEncoder = &captureCommandEncoder{
		CommandEncoder: &noop.CommandEncoder{},
		renderPass:     &captureRenderPass{RenderPassEncoder: &noop.RenderPassEncoder{}},
	}
	qs := newTestQuerySet(&testHALResource{}, device)
	begin, end := uint32(0), uint32(1)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		TimestampWrites: &RenderPassTimestampWrites{QuerySet: qs, BeginningOfPassWriteIndex: &begin, EndOfPassWriteIndex: &end},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if err := validateCommandBufferForSubmit(cb, 0); err != nil {
		t.Fatalf("submit validation before the release: %v", err)
	}
	qs.Release()
	if err := validateCommandBufferForSubmit(cb, 0); !errors.Is(err, ErrReleased) {
		t.Errorf("submit validation with a released timestamp query set: %v, want ErrReleased", err)
	}
}
