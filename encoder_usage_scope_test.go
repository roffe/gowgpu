//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
)

// These tests pin the WebGPU usage-scope rule: a usage conflict is only
// possible inside ONE usage scope (one render pass, one compute dispatch, one
// copy command). Different commands of the same encoder may use a resource
// with different usages; the encoder transitions between them.
//
// Reference: W3C WebGPU spec "usage scope"; wgpu-core track/mod.rs
// (UsageScope = merge with conflict check, command buffer Tracker =
// barrier + update, no conflict).

func newUsageScopeTexture(t *testing.T, device *Device, label string, usage gputypes.TextureUsage) (*Texture, *TextureView) {
	t.Helper()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         label,
		Size:          Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         usage,
	})
	if err != nil {
		t.Fatalf("CreateTexture(%s): %v", label, err)
	}
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		tex.Release()
		t.Fatalf("CreateView(%s): %v", label, err)
	}
	t.Cleanup(func() {
		view.Release()
		tex.Release()
	})
	return tex, view
}

func newUsageScopeBuffer(t *testing.T, device *Device, label string, usage gputypes.BufferUsage) *Buffer {
	t.Helper()
	// Big enough for the copies these tests record: a 4x4 RGBA8 texture at
	// 256 bytes a row is 784 bytes.
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: label, Size: 1024, Usage: usage})
	if err != nil {
		t.Fatalf("CreateBuffer(%s): %v", label, err)
	}
	t.Cleanup(buf.Release)
	return buf
}

func runColorPass(t *testing.T, enc *CommandEncoder, view *TextureView, load gputypes.LoadOp, record func(*RenderPassEncoder)) {
	t.Helper()
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{
			View:    view,
			LoadOp:  load,
			StoreOp: gputypes.StoreOpStore,
		}},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	if record != nil {
		record(pass)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("RenderPass.End: %v", err)
	}
}

// mustFinish finishes enc and releases the command buffer when the test
// ends. Only for one the test doesn't submit: a submitted command buffer
// lives in a pooled encoder object, which a parallel test may have got
// back by then, and the Release would land on its command buffer.
func mustFinish(t *testing.T, enc *CommandEncoder) *CommandBuffer {
	t.Helper()
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	t.Cleanup(cb.Release)
	return cb
}

// TestRenderPassThenCopyTextureToBufferSameEncoder is the triangle-headless
// readback pattern: render into T, end the pass, copy T to a staging buffer,
// all in one encoder. ColorTarget and CopySrc belong to different usage
// scopes, so this is valid WebGPU.
func TestRenderPassThenCopyTextureToBufferSameEncoder(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	tex, view := newUsageScopeTexture(t, device, "readback-target",
		TextureUsageRenderAttachment|TextureUsageCopySrc)
	staging := newUsageScopeBuffer(t, device, "readback-staging", BufferUsageCopyDst|BufferUsageMapRead)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	runColorPass(t, enc, view, gputypes.LoadOpClear, nil)
	enc.CopyTextureToBuffer(tex, staging, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{BytesPerRow: 256, RowsPerImage: 4},
		Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	cb := mustFinish(t, enc)

	idx := tex.coreTexture.TrackingData().Index()
	tracker := cb.core.TextureScope()
	if got := tracker.StartUsage(idx); got != track.TextureUsesColorTarget {
		t.Errorf("first usage in command buffer = %v, want ColorTarget", got)
	}
	if got := tracker.EndUsage(idx); got != track.TextureUsesCopySrc {
		t.Errorf("last usage in command buffer = %v, want CopySrc", got)
	}
}

// TestCopyBufferToTextureThenRenderAttachmentSameEncoder uploads into T and
// then renders on top of it (LoadOp Load) in the same encoder.
func TestCopyBufferToTextureThenRenderAttachmentSameEncoder(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	tex, view := newUsageScopeTexture(t, device, "upload-then-draw",
		TextureUsageRenderAttachment|TextureUsageCopyDst)
	upload := newUsageScopeBuffer(t, device, "upload-src", BufferUsageCopySrc)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToTexture(upload, tex, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{BytesPerRow: 256, RowsPerImage: 4},
		Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	runColorPass(t, enc, view, gputypes.LoadOpLoad, nil)
	cb := mustFinish(t, enc)

	idx := tex.coreTexture.TrackingData().Index()
	tracker := cb.core.TextureScope()
	if got := tracker.StartUsage(idx); got != track.TextureUsesCopyDst {
		t.Errorf("first usage in command buffer = %v, want CopyDst", got)
	}
	if got := tracker.EndUsage(idx); got != track.TextureUsesColorTarget {
		t.Errorf("last usage in command buffer = %v, want ColorTarget", got)
	}
}

// TestCopyBufferToTextureThenSampledInRenderPassSameEncoder uploads into T and
// samples T (TextureBinding) from a later render pass of the same encoder.
func TestCopyBufferToTextureThenSampledInRenderPassSameEncoder(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	sampled, sampledView := newUsageScopeTexture(t, device, "upload-then-sample",
		TextureUsageTextureBinding|TextureUsageCopyDst)
	_, targetView := newUsageScopeTexture(t, device, "sample-target", TextureUsageRenderAttachment)
	upload := newUsageScopeBuffer(t, device, "upload-src", BufferUsageCopySrc)

	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{
		Entries: []gputypes.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: ShaderStageFragment,
			Texture: &gputypes.TextureBindingLayout{
				SampleType:    gputypes.TextureSampleTypeFloat,
				ViewDimension: gputypes.TextureViewDimension2D,
			},
		}},
	})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer layout.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{
		Layout:  layout,
		Entries: []BindGroupEntry{{Binding: 0, TextureView: sampledView}},
	})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToTexture(upload, sampled, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: sampled},
		BufferLayout: ImageDataLayout{BytesPerRow: 256, RowsPerImage: 4},
		Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	runColorPass(t, enc, targetView, gputypes.LoadOpClear, func(pass *RenderPassEncoder) {
		pass.SetBindGroup(0, group, nil)
	})
	mustFinish(t, enc)
}

// TestCopyBufferToBufferThenVertexBufferSameEncoder writes vertex data with a
// copy and then binds that buffer as a vertex buffer in a later render pass.
func TestCopyBufferToBufferThenVertexBufferSameEncoder(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	staging := newUsageScopeBuffer(t, device, "vertex-staging", BufferUsageCopySrc)
	vertices := newUsageScopeBuffer(t, device, "vertices", BufferUsageVertex|BufferUsageCopyDst)
	_, targetView := newUsageScopeTexture(t, device, "vertex-target", TextureUsageRenderAttachment)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToBuffer(staging, 0, vertices, 0, 64)
	runColorPass(t, enc, targetView, gputypes.LoadOpClear, func(pass *RenderPassEncoder) {
		pass.SetVertexBuffer(0, vertices, 0)
	})
	cb := mustFinish(t, enc)

	idx := vertices.core.TrackingData().Index()
	tracker := cb.core.BufferScope()
	if got := tracker.StartUsage(idx); got != track.BufferUsesCopyDst {
		t.Errorf("first usage in command buffer = %v, want CopyDst", got)
	}
	if got := tracker.EndUsage(idx); got != track.BufferUsesVertex {
		t.Errorf("last usage in command buffer = %v, want Vertex", got)
	}
}

// TestRenderCopyRenderSameTextureSameEncoder renders into T, copies T to U and
// renders into T again — three usage scopes, each valid on its own.
func TestRenderCopyRenderSameTextureSameEncoder(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	tex, view := newUsageScopeTexture(t, device, "render-copy-render",
		TextureUsageRenderAttachment|TextureUsageCopySrc)
	dst, _ := newUsageScopeTexture(t, device, "render-copy-dst", TextureUsageCopyDst)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	runColorPass(t, enc, view, gputypes.LoadOpClear, nil)
	enc.CopyTextureToTexture(tex, dst, []TextureCopy{{
		Source:      ImageCopyTexture{Texture: tex},
		Destination: ImageCopyTexture{Texture: dst},
		Size:        Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	runColorPass(t, enc, view, gputypes.LoadOpLoad, nil)
	cb := mustFinish(t, enc)

	idx := tex.coreTexture.TrackingData().Index()
	tracker := cb.core.TextureScope()
	if got := tracker.StartUsage(idx); got != track.TextureUsesColorTarget {
		t.Errorf("first usage = %v, want ColorTarget", got)
	}
	if got := tracker.EndUsage(idx); got != track.TextureUsesColorTarget {
		t.Errorf("last usage = %v, want ColorTarget", got)
	}
}

// newSampledGroup makes a bind group that samples view at binding 0.
func newSampledGroup(t *testing.T, device *Device, view *TextureView) *BindGroup {
	t.Helper()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{
		Entries: []gputypes.BindGroupLayoutEntry{{
			Binding:    0,
			Visibility: ShaderStageFragment,
			Texture: &gputypes.TextureBindingLayout{
				SampleType:    gputypes.TextureSampleTypeFloat,
				ViewDimension: gputypes.TextureViewDimension2D,
			},
		}},
	})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	t.Cleanup(layout.Release)
	group, err := device.CreateBindGroup(&BindGroupDescriptor{
		Layout:  layout,
		Entries: []BindGroupEntry{{Binding: 0, TextureView: view}},
	})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(group.Release)
	return group
}

// TestRenderThenSampleSameEncoderSplitsThePass renders into T and samples T
// from a later pass of the same encoder. The second pass's bind groups are
// only known once it ends, so it is recorded in a HAL command buffer of its
// own and the ColorTarget -> Resource barrier goes in one inserted ahead of
// it: the finished command buffer is [before, transit, pass, after], and the
// first pass, which nothing preceded, stayed in the first one.
func TestRenderThenSampleSameEncoderSplitsThePass(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	tex, view := newUsageScopeTexture(t, device, "render-then-sample",
		TextureUsageRenderAttachment|TextureUsageTextureBinding)
	_, targetView := newUsageScopeTexture(t, device, "sample-target", TextureUsageRenderAttachment)
	group := newSampledGroup(t, device, view)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	runColorPass(t, enc, view, gputypes.LoadOpClear, nil)
	runColorPass(t, enc, targetView, gputypes.LoadOpClear, func(pass *RenderPassEncoder) {
		pass.SetBindGroup(0, group, nil)
	})
	cb, err := enc.Finish() // submitted below, see mustFinish
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}

	// Nothing is reopened after the split pass: an encoder that ends with
	// it submits no empty command buffer.
	if got := len(cb.core.HalBufferList()); got != 3 {
		t.Errorf("HAL command buffers = %d, want 3: before, transit, pass", got)
	}
	idx := tex.coreTexture.TrackingData().Index()
	tracker := cb.core.TextureScope()
	if got := tracker.StartUsage(idx); got != track.TextureUsesColorTarget {
		t.Errorf("first usage = %v, want ColorTarget", got)
	}
	if got := tracker.EndUsage(idx); got != track.TextureUsesResource {
		t.Errorf("last usage = %v, want Resource", got)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
}

// TestBindGroupTexturesRecordedInEveryPass binds the same group in two passes
// with an upload into its texture between them. The group's textures go in
// each pass's usage scope, so the second pass gets the CopyDst -> Resource
// barrier ahead of it, and the copy got Resource -> CopyDst ahead of itself.
func TestBindGroupTexturesRecordedInEveryPass(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	sampled, sampledView := newUsageScopeTexture(t, device, "sampled-twice",
		TextureUsageTextureBinding|TextureUsageCopyDst)
	_, targetView := newUsageScopeTexture(t, device, "target", TextureUsageRenderAttachment)
	upload := newUsageScopeBuffer(t, device, "upload-src", BufferUsageCopySrc)
	group := newSampledGroup(t, device, sampledView)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	bind := func(pass *RenderPassEncoder) { pass.SetBindGroup(0, group, nil) }
	runColorPass(t, enc, targetView, gputypes.LoadOpClear, bind)
	enc.CopyBufferToTexture(upload, sampled, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: sampled},
		BufferLayout: ImageDataLayout{BytesPerRow: 256, RowsPerImage: 4},
		Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	runColorPass(t, enc, targetView, gputypes.LoadOpLoad, bind)
	cb, err := enc.Finish() // submitted below, see mustFinish
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}

	if got := len(cb.core.HalBufferList()); got != 3 {
		t.Errorf("HAL command buffers = %d, want 3: before (first pass and copy), transit, pass", got)
	}
	idx := sampled.coreTexture.TrackingData().Index()
	tracker := cb.core.TextureScope()
	if got := tracker.StartUsage(idx); got != track.TextureUsesResource {
		t.Errorf("first usage = %v, want Resource", got)
	}
	if got := tracker.EndUsage(idx); got != track.TextureUsesResource {
		t.Errorf("last usage = %v, want Resource (sampled again after the upload)", got)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
}

// TestCopyPastBufferEndIsAnError: a copy between a buffer and a texture
// whose buffer side runs past the end of the buffer is a validation error
// at Finish, instead of a copy the GPU runs past the buffer.
func TestCopyPastBufferEndIsAnError(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	tex, _ := newUsageScopeTexture(t, device, "short-target", TextureUsageCopySrc|TextureUsageCopyDst|TextureUsageRenderAttachment)
	short, err := device.CreateBuffer(&BufferDescriptor{Label: "short", Size: 512, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(short.Release)
	// 4 rows at 256 bytes a row: 784 bytes, the buffer has 512.
	region := []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{BytesPerRow: 256, RowsPerImage: 4},
		Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}}

	for _, tc := range []struct {
		name   string
		record func(enc *CommandEncoder)
	}{
		{"CopyTextureToBuffer", func(enc *CommandEncoder) { enc.CopyTextureToBuffer(tex, short, region) }},
		{"CopyBufferToTexture", func(enc *CommandEncoder) { enc.CopyBufferToTexture(short, tex, region) }},
	} {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		tc.record(enc)
		if _, err := enc.Finish(); err == nil || !strings.Contains(err.Error(), "past the buffer") {
			t.Errorf("%s past the buffer end: Finish = %v, want an error naming the overrun", tc.name, err)
		}
	}

	// One full row at 256 bytes and a 4 byte last row from offset 252 end
	// at 512 exactly: that fits.
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyTextureToBuffer(tex, short, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{Offset: 252, BytesPerRow: 256},
		Size:         Extent3D{Width: 1, Height: 2, DepthOrArrayLayers: 1},
	}})
	if _, err := enc.Finish(); err != nil {
		t.Errorf("copy that fits: Finish = %v, want nil", err)
	}
}

// TestCopyBufferToBufferValidation: CopyBufferToBuffer refuses what WebGPU's
// copyBufferToBuffer refuses, at Finish, and a copy of 0 bytes is no copy.
func TestCopyBufferToBufferValidation(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	newBuffer := func(usage gputypes.BufferUsage) *Buffer {
		t.Helper()
		buf, err := device.CreateBuffer(&BufferDescriptor{Size: 256, Usage: usage})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		t.Cleanup(buf.Release)
		return buf
	}
	src := newBuffer(BufferUsageCopySrc)
	dst := newBuffer(BufferUsageCopyDst)

	for _, tc := range []struct {
		name                string
		src, dst            *Buffer
		srcOff, dstOff, len uint64
		valid               bool
	}{
		{"valid", src, dst, 4, 8, 128, true},
		{"empty", src, dst, 0, 0, 0, true},
		{"source without CopySrc", dst, dst, 0, 0, 4, false},
		{"destination without CopyDst", src, src, 0, 0, 4, false},
		{"unaligned source offset", src, dst, 2, 0, 4, false},
		{"unaligned destination offset", src, dst, 0, 2, 4, false},
		{"unaligned size", src, dst, 0, 0, 6, false},
		{"past the source end", src, dst, 128, 0, 132, false},
		{"past the destination end", src, dst, 0, 192, 128, false},
		{"source offset past the end", src, dst, 260, 0, 0, false},
	} {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		enc.CopyBufferToBuffer(tc.src, tc.srcOff, tc.dst, tc.dstOff, tc.len)
		cb, err := enc.Finish()
		if tc.valid != (err == nil) {
			t.Errorf("%s: Finish = %v, want valid %v", tc.name, err, tc.valid)
		}
		if cb != nil {
			cb.Release()
		}
	}
}

// TestSecondPassWithoutBarriersIsTwoCommandBuffers: a second render pass of
// an encoder is recorded in a command buffer of its own, since the barriers
// it may need ahead of it are only known once it ends. When it turns out to
// need none there is no transit command buffer, and nothing is reopened
// after it: two command buffers in all, the one the encoder started with
// and the pass's.
func TestSecondPassWithoutBarriersIsTwoCommandBuffers(t *testing.T) {
	t.Parallel()
	_, _, device := newTestDeviceWithTracker(t)
	defer device.Release()

	first, firstView := newUsageScopeTexture(t, device, "first-target", TextureUsageRenderAttachment)
	_ = first
	second, secondView := newUsageScopeTexture(t, device, "second-target", TextureUsageRenderAttachment)
	_ = second

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	runColorPass(t, enc, firstView, gputypes.LoadOpClear, nil)
	runColorPass(t, enc, secondView, gputypes.LoadOpClear, nil)
	if enc.core.IsCBOpen() {
		t.Error("a command buffer is open after the second pass, want none until the next command")
	}
	cb := mustFinish(t, enc)
	if got := len(cb.core.HalBufferList()); got != 2 {
		t.Errorf("HAL command buffers = %d, want 2: the first pass's and the second's, no transit, nothing after", got)
	}
}
