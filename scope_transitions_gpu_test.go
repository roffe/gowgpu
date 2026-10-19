//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
	"encoding/binary"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// GPU tests of the usage scopes and the barriers between them: they record
// the mid-encoder and submit-time barrier paths on the real GPU and read the
// pixels back, and when the Khronos validation layer is on the instance
// (VK_LAYER_PATH + VK_INSTANCE_LAYERS) they fail on any validation message,
// since the debug messenger logs through hal.Logger. GOWGPU_TEST_BACKEND
// (vulkan, gl, dx12) picks the backend; the default is the first adapter.
//
// They share the process-wide logger, so none of them is parallel.

// captureHandler collects slog records so a test can assert on validation
// messages, and remembers the backend the device it came with runs on.
type captureHandler struct {
	mu      sync.Mutex
	msgs    []string
	backend gputypes.Backend
}

func (h *captureHandler) Enabled(context.Context, slog.Level) bool { return true }
func (h *captureHandler) Handle(_ context.Context, r slog.Record) error {
	var sb strings.Builder
	sb.WriteString(r.Message)
	r.Attrs(func(a slog.Attr) bool {
		sb.WriteString(" ")
		sb.WriteString(a.String())
		return true
	})
	h.mu.Lock()
	h.msgs = append(h.msgs, sb.String())
	h.mu.Unlock()
	return nil
}
func (h *captureHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *captureHandler) WithGroup(string) slog.Handler      { return h }

// validation returns the validation-layer messages captured so far. The
// Vulkan messenger drains every 50ms, so it waits for the drain first.
func (h *captureHandler) validation() []string {
	time.Sleep(250 * time.Millisecond)
	h.mu.Lock()
	defer h.mu.Unlock()
	var out []string
	for _, m := range h.msgs {
		if strings.Contains(m, "Validation") || strings.Contains(m, "VUID") {
			out = append(out, m)
		}
	}
	return out
}

// testBackends is the backend GOWGPU_TEST_BACKEND (gl, dx12, vulkan) names,
// and whether it names one; all of them when it doesn't.
func testBackends() (gputypes.Backends, bool) {
	switch os.Getenv("GOWGPU_TEST_BACKEND") {
	case "gl":
		return gputypes.BackendsGL, true
	case "dx12":
		return gputypes.BackendsDX12, true
	case "vulkan":
		return gputypes.BackendsVulkan, true
	}
	return gputypes.BackendsAll, false
}

// newValidatedDevice makes a device on a debug instance, with the logger
// captured. Without the validation layer installed the capture stays empty.
func newValidatedDevice(t *testing.T) (*Device, *captureHandler) {
	t.Helper()
	return newValidatedDeviceWith(t, 0)
}

// newValidatedDeviceWith is newValidatedDevice with features turned on. It
// skips when the adapter lacks them.
func newValidatedDeviceWith(t *testing.T, features gputypes.Features) (*Device, *captureHandler) {
	t.Helper()
	h := &captureHandler{}
	prev := hal.Logger()
	hal.SetLogger(slog.New(h))
	t.Cleanup(func() { hal.SetLogger(prev) })

	backends, _ := testBackends()
	inst, err := CreateInstance(&InstanceDescriptor{Backends: backends, Flags: gputypes.InstanceFlagsDebug})
	if err != nil {
		t.Fatalf("CreateInstance: %v", err)
	}
	adapter, err := inst.RequestAdapter(nil)
	if err != nil {
		if strings.Contains(err.Error(), "no adapters available") {
			t.Skipf("no GPU adapter on this machine: %v", err)
		}
		t.Fatalf("RequestAdapter: %v", err)
	}
	if !adapter.Features().ContainsAll(features) {
		t.Skipf("%s on %s lacks features %#x", adapter.Info().Backend, adapter.Info().Name, uint64(features&^adapter.Features()))
	}
	device, err := adapter.RequestDevice(&DeviceDescriptor{RequiredFeatures: features})
	if err != nil {
		t.Fatalf("RequestDevice: %v", err)
	}
	if device.core == nil || device.core.Tracker() == nil {
		t.Skip("device has no tracker")
	}
	t.Logf("adapter: %s on %s", adapter.Info().Backend, adapter.Info().Name)
	h.backend = adapter.Info().Backend
	t.Cleanup(device.Release)
	return device, h
}

func assertNoValidation(t *testing.T, h *captureHandler) {
	t.Helper()
	for _, m := range h.validation() {
		t.Errorf("validation: %s", m)
	}
}

const pixSide = 4
const pixRow = 256 // bytes per row, 256-aligned for CopyTextureToBuffer
const pixStaging = pixRow * pixSide

// pixTexture makes a pixSide square RGBA8 texture, with a view when the
// usage calls for one.
func pixTexture(t *testing.T, device *Device, label string, usage gputypes.TextureUsage) (*Texture, *TextureView) {
	t.Helper()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         label,
		Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         usage,
	})
	if err != nil {
		t.Fatalf("CreateTexture(%s): %v", label, err)
	}
	if usage&(TextureUsageRenderAttachment|TextureUsageTextureBinding|TextureUsageStorageBinding) == 0 {
		return tex, nil
	}
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView(%s): %v", label, err)
	}
	return tex, view
}

// releaseTextureView releases a view (which may be nil) and its texture.
func releaseTextureView(tex *Texture, view *TextureView) {
	if view != nil {
		view.Release()
	}
	tex.Release()
}

// readbackBuffer makes a staging buffer for one pixTexture.
func readbackBuffer(t *testing.T, device *Device, label string) *Buffer {
	t.Helper()
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: label, Size: pixStaging, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer(%s): %v", label, err)
	}
	t.Cleanup(buf.Release)
	return buf
}

// clearPass records a render pass on view with the load op and clear
// colour, and record's draws if any.
func clearPass(t *testing.T, enc *CommandEncoder, view *TextureView, load gputypes.LoadOp, c gputypes.Color, record func(*RenderPassEncoder)) {
	t.Helper()
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments: []RenderPassColorAttachment{{View: view, LoadOp: load, StoreOp: gputypes.StoreOpStore, ClearValue: c}},
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

// copyOut copies a pixTexture into a readbackBuffer.
func copyOut(enc *CommandEncoder, tex *Texture, dst *Buffer) {
	enc.CopyTextureToBuffer(tex, dst, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
		Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
	}})
}

// finishAndSubmit finishes enc and submits it.
func finishAndSubmit(t *testing.T, device *Device, enc *CommandEncoder) {
	t.Helper()
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
}

// readback maps the staging buffer and returns the pixels, rows packed.
func readback(t *testing.T, buf *Buffer) []byte {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, MapModeRead, 0, pixStaging); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, pixStaging)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	data := rng.Bytes()
	out := make([]byte, 0, pixSide*pixSide*4)
	for y := 0; y < pixSide; y++ {
		out = append(out, data[y*pixRow:y*pixRow+pixSide*4]...)
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

// expectPixels fails the test on the first pixel that is not (r, g, b, a).
func expectPixels(t *testing.T, what string, pix []byte, r, g, b, a byte) {
	t.Helper()
	for i := 0; i+3 < len(pix); i += 4 {
		if pix[i] != r || pix[i+1] != g || pix[i+2] != b || pix[i+3] != a {
			t.Errorf("%s: pixel %d = %v, want [%d %d %d %d]", what, i/4, pix[i:i+4], r, g, b, a)
			return
		}
	}
}

var (
	red   = gputypes.Color{R: 1, G: 0, B: 0, A: 1}
	green = gputypes.Color{R: 0, G: 1, B: 0, A: 1}
	black = gputypes.Color{R: 0, G: 0, B: 0, A: 1}
)

// TestRenderCopyRenderPixels: pass (clear T red), copy T to U, pass (clear
// T green, split off with its transit CB), copy T and U out. The pixels
// prove the HAL command buffers execute in [before, transit, pass, after]
// order with the barriers in the right places.
//
// "unsampled" gives U no TextureBinding: the zeroing in the same submit
// leaves it a copy destination, and the copy into it waits for the zeroing's.
// "sampled" is zeroed in a submit of its own first, which leaves it sampled.
func TestRenderCopyRenderPixels(t *testing.T) {
	for _, tc := range []struct {
		name     string
		midUsage gputypes.TextureUsage
	}{
		{"unsampled", TextureUsageCopyDst | TextureUsageCopySrc},
		{"sampled", TextureUsageCopyDst | TextureUsageCopySrc | TextureUsageTextureBinding},
	} {
		t.Run(tc.name, func(t *testing.T) {
			device, h := newValidatedDevice(t)

			tex, view := pixTexture(t, device, "rcr-target", TextureUsageRenderAttachment|TextureUsageCopySrc)
			defer releaseTextureView(tex, view)
			mid, midView := pixTexture(t, device, "rcr-mid", tc.midUsage)
			defer releaseTextureView(mid, midView)
			stT := readbackBuffer(t, device, "rcr-stT")
			stU := readbackBuffer(t, device, "rcr-stU")
			if tc.name == "sampled" {
				// Flush the zeroing in a submit of its own, so the barriers
				// below are all tracker barriers.
				device.Queue().Submit()
			}

			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			clearPass(t, enc, view, gputypes.LoadOpClear, red, nil)
			enc.CopyTextureToTexture(tex, mid, []TextureCopy{{
				Source:      ImageCopyTexture{Texture: tex},
				Destination: ImageCopyTexture{Texture: mid},
				Size:        Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
			}})
			clearPass(t, enc, view, gputypes.LoadOpClear, green, nil)
			copyOut(enc, tex, stT)
			copyOut(enc, mid, stU)
			cb, err := enc.Finish()
			if err != nil {
				t.Fatalf("Finish: %v", err)
			}
			if got := len(cb.core.HalBufferList()); got != 4 {
				t.Errorf("HAL command buffers = %d, want 4 (before, transit, pass, after)", got)
			}
			if _, err := device.Queue().Submit(cb); err != nil {
				t.Fatalf("Submit: %v", err)
			}

			expectPixels(t, "T after render-copy-render", readback(t, stT), 0, 255, 0, 255)
			expectPixels(t, "U copied between the passes", readback(t, stU), 255, 0, 0, 255)
			assertNoValidation(t, h)
		})
	}
}

// TestUploadThenLoadPassPreservesPixels: CopyBufferToTexture into T, then a
// LoadOp Load pass on T (split, CopyDst -> ColorTarget ahead of it), then
// copy out. The pass must see the upload. "unsampled" (no TextureBinding)
// is zeroed in the same submit, which leaves it a copy destination, so its
// submit barrier is between that copy and the upload; "sampled" starts from
// the sampled state the zeroing in a submit of its own leaves, and its Load
// pass starts in GENERAL, the layout a sampled color target ends in.
func TestUploadThenLoadPassPreservesPixels(t *testing.T) {
	for _, tc := range []struct {
		name  string
		usage gputypes.TextureUsage
	}{
		{"unsampled", TextureUsageRenderAttachment | TextureUsageCopyDst | TextureUsageCopySrc},
		{"sampled", TextureUsageRenderAttachment | TextureUsageCopyDst | TextureUsageCopySrc | TextureUsageTextureBinding},
	} {
		t.Run(tc.name, func(t *testing.T) {
			device, h := newValidatedDevice(t)
			if h.backend == gputypes.BackendGL {
				t.Skip("CopyBufferToTexture ignores the row pitch on GL (README known issues)")
			}

			tex, view := pixTexture(t, device, "upload-load", tc.usage)
			defer releaseTextureView(tex, view)
			upload, err := device.CreateBuffer(&BufferDescriptor{Label: "upload", Size: pixStaging, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
			if err != nil {
				t.Fatalf("CreateBuffer: %v", err)
			}
			defer upload.Release()
			src := make([]byte, pixStaging)
			for y := 0; y < pixSide; y++ {
				for x := 0; x < pixSide; x++ {
					p := src[y*pixRow+x*4:]
					p[0], p[1], p[2], p[3] = 0, 0, 255, 255
				}
			}
			if err := device.Queue().WriteBuffer(upload, 0, src); err != nil {
				t.Fatalf("WriteBuffer: %v", err)
			}
			st := readbackBuffer(t, device, "upload-load-st")
			if tc.name == "sampled" {
				device.Queue().Submit()
			}

			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			enc.CopyBufferToTexture(upload, tex, []BufferTextureCopy{{
				TextureBase:  ImageCopyTexture{Texture: tex},
				BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
				Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
			}})
			clearPass(t, enc, view, gputypes.LoadOpLoad, black, nil)
			copyOut(enc, tex, st)
			finishAndSubmit(t, device, enc)

			expectPixels(t, "T after upload then Load pass", readback(t, st), 0, 0, 255, 255)
			assertNoValidation(t, h)
		})
	}
}

const blitShader = `
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {
	var o: VOut;
	let x = f32(i32(vi & 1u) * 4 - 1);
	let y = f32(i32(vi >> 1u) * 4 - 1);
	o.pos = vec4<f32>(x, y, 0.0, 1.0);
	o.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
	return o;
}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> { return textureSample(t, s, in.uv); }
`

// blitRig is a pipeline that draws a sampled texture over the whole target.
type blitRig struct {
	layout   *BindGroupLayout
	pipeline *RenderPipeline
	sampler  *Sampler
}

func newBlitRig(t *testing.T, device *Device) *blitRig {
	t.Helper()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		{Binding: 1, Visibility: ShaderStageFragment, Sampler: &gputypes.SamplerBindingLayout{Type: gputypes.SamplerBindingTypeFiltering}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	t.Cleanup(layout.Release)
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{layout}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	t.Cleanup(pl.Release)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "scope-blit", WGSL: blitShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(module.Release)
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "scope-blit",
		Layout:    pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	t.Cleanup(pipe.Release)
	sampler, err := device.CreateSampler(&SamplerDescriptor{MagFilter: gputypes.FilterModeNearest, MinFilter: gputypes.FilterModeNearest, LodMaxClamp: 32})
	if err != nil {
		t.Fatalf("CreateSampler: %v", err)
	}
	t.Cleanup(sampler.Release)
	return &blitRig{layout: layout, pipeline: pipe, sampler: sampler}
}

func (b *blitRig) group(t *testing.T, device *Device, view *TextureView) *BindGroup {
	t.Helper()
	g, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: b.layout, Entries: []BindGroupEntry{{Binding: 0, TextureView: view}, {Binding: 1, Sampler: b.sampler}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(g.Release)
	return g
}

// TestRenderThenSampleDrawPixels: pass 1 clears S red; pass 2 draws S over
// U with a sampling pipeline; U is copied out. Pass 2 binds S only after the
// HAL pass began, so its ColorTarget -> Resource barrier can only be in the
// transit CB ahead of it; the pixels prove it got there, and a layout
// mismatch at the draw would show as a validation message.
func TestRenderThenSampleDrawPixels(t *testing.T) {
	device, h := newValidatedDevice(t)
	blit := newBlitRig(t, device)

	src, srcView := pixTexture(t, device, "sampled-src", TextureUsageRenderAttachment|TextureUsageTextureBinding)
	defer releaseTextureView(src, srcView)
	dst, dstView := pixTexture(t, device, "sampled-dst", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(dst, dstView)
	group := blit.group(t, device, srcView)
	st := readbackBuffer(t, device, "sampled-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, srcView, gputypes.LoadOpClear, red, nil)
	clearPass(t, enc, dstView, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) {
		pass.SetPipeline(blit.pipeline)
		pass.SetBindGroup(0, group, nil)
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, dst, st)
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if got := len(cb.core.HalBufferList()); got != 4 {
		t.Errorf("HAL command buffers = %d, want 4 (before, transit, pass, after)", got)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	expectPixels(t, "U sampled from S rendered earlier in the encoder", readback(t, st), 255, 0, 0, 255)
	assertNoValidation(t, h)
}

// TestSampleRenderSampleOneEncoder: S is sampled (pass 1 into U), then
// rendered (pass 2, split), then sampled again (pass 3, split) with the same
// bind group. Pass 3 needs ColorTarget -> Resource ahead of it even though
// the group was already bound in pass 1: the group's textures must be in
// every pass's scope.
func TestSampleRenderSampleOneEncoder(t *testing.T) {
	device, h := newValidatedDevice(t)
	blit := newBlitRig(t, device)

	src, srcView := pixTexture(t, device, "srs-src", TextureUsageRenderAttachment|TextureUsageTextureBinding)
	defer releaseTextureView(src, srcView)
	dst, dstView := pixTexture(t, device, "srs-dst", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(dst, dstView)
	group := blit.group(t, device, srcView)
	st := readbackBuffer(t, device, "srs-st")

	// First submit: S red, so pass 1 below samples something defined.
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, srcView, gputypes.LoadOpClear, red, nil)
	finishAndSubmit(t, device, enc)

	enc, err = device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	draw := func(pass *RenderPassEncoder) {
		pass.SetPipeline(blit.pipeline)
		pass.SetBindGroup(0, group, nil)
		pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	}
	clearPass(t, enc, dstView, gputypes.LoadOpClear, black, draw)
	clearPass(t, enc, srcView, gputypes.LoadOpClear, green, nil)
	clearPass(t, enc, dstView, gputypes.LoadOpClear, black, draw)
	copyOut(enc, dst, st)
	finishAndSubmit(t, device, enc)

	expectPixels(t, "U after sample, render, sample", readback(t, st), 0, 255, 0, 255)
	assertNoValidation(t, h)
}

// TestFreshTextureCopyAfterZeroing: a texture created this frame is zeroed
// by the pending writes at Submit (ending in Resource) and an encoder of the
// same submit copies into it (start CopyDst). The submit barrier Resource ->
// CopyDst has to run after the pending-writes command buffer, whose zeroing
// puts the texture in Resource: ahead of it, the copy ran against a texture
// still in the layout it was made in. Any validation message here is that
// layout mismatch.
func TestFreshTextureCopyAfterZeroing(t *testing.T) {
	device, h := newValidatedDevice(t)

	upload, err := device.CreateBuffer(&BufferDescriptor{Label: "fresh-upload", Size: pixStaging, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer upload.Release()
	src := make([]byte, pixStaging)
	for i := range src {
		src[i] = 0x7f
	}
	if err := device.Queue().WriteBuffer(upload, 0, src); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	// Created now: the zeroing is queued for the next Submit.
	tex, view := pixTexture(t, device, "fresh-this-frame", TextureUsageTextureBinding|TextureUsageCopyDst|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "fresh-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToTexture(upload, tex, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
		Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
	}})
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	expectPixels(t, "fresh texture after upload", readback(t, st), 0x7f, 0x7f, 0x7f, 0x7f)
	assertNoValidation(t, h)
}

// TestQueueWriteBufferIsOrdered: a queue write is a CopyDst use of the buffer
// in the device tracker, so the command buffers around it get their barriers:
// a copy into a buffer the same submit wrote (write after write), a copy out
// of it after a write in a later submit, whose start state the device knew as
// CopySrc (read after write), and that write itself after the earlier submit
// read the buffer (write after read). Under the synchronization validation
// layer any message here is one of those hazards.
func TestQueueWriteBufferIsOrdered(t *testing.T) {
	device, h := newValidatedDevice(t)
	const size = 1024
	q := device.Queue()
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "written", Size: size, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	src, err := device.CreateBuffer(&BufferDescriptor{Label: "src", Size: size, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer src.Release()
	st := make([]*Buffer, 2)
	for i := range st {
		if st[i], err = device.CreateBuffer(&BufferDescriptor{Label: "st", Size: size, Usage: BufferUsageCopyDst | BufferUsageMapRead}); err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		defer st[i].Release()
	}
	fill := func(b *Buffer, v byte) {
		t.Helper()
		if err := q.WriteBuffer(b, 0, bytes.Repeat([]byte{v}, size)); err != nil {
			t.Fatalf("WriteBuffer: %v", err)
		}
	}
	copyBuf := func(from, to *Buffer) {
		t.Helper()
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		enc.CopyBufferToBuffer(from, 0, to, 0, size)
		finishAndSubmit(t, device, enc)
	}

	fill(src, 0x11)
	fill(buf, 0x22)
	copyBuf(src, buf) // writes buf after the queue write of the same submit
	copyBuf(buf, st[0])
	fill(buf, 0x33) // after the read above, and read below as CopySrc again
	copyBuf(buf, st[1])
	device.Poll(PollWait)

	for i, want := range []byte{0x11, 0x33} {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		err := st[i].Map(ctx, MapModeRead, 0, size)
		cancel()
		if err != nil {
			t.Fatalf("Map: %v", err)
		}
		rng, err := st[i].MappedRange(0, size)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		if got := rng.Bytes(); !bytes.Equal(got, bytes.Repeat([]byte{want}, size)) {
			t.Errorf("readback %d starts %#x, want all %#x", i, got[0], want)
		}
		if err := st[i].Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
	}
	assertNoValidation(t, h)
}

// TestQueueWriteTextureIsOrdered: a texture that isn't sampled stays a copy
// destination after a queue write, so the next write into it, in the same
// batch or after a command buffer that copied into it, needs a CopyDst to
// CopyDst barrier, not none.
func TestQueueWriteTextureIsOrdered(t *testing.T) {
	device, h := newValidatedDevice(t)
	tex, view := pixTexture(t, device, "copy-dst-only", TextureUsageCopyDst|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	upload, err := device.CreateBuffer(&BufferDescriptor{Label: "upload", Size: pixStaging, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer upload.Release()
	write := func() {
		t.Helper()
		err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, make([]byte, pixSide*pixSide*4),
			&ImageDataLayout{BytesPerRow: pixSide * 4, RowsPerImage: pixSide},
			&Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1})
		if err != nil {
			t.Fatalf("WriteTexture: %v", err)
		}
	}

	write()
	write()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToTexture(upload, tex, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
		Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)
	write()
	if _, err := device.Queue().Submit(); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	device.Poll(PollWait)
	assertNoValidation(t, h)
}

// bindFirstShader draws a full-screen triangle, or stores from a compute
// shader, in the colour of its uniform.
const bindFirstShader = `
@group(0) @binding(0) var<uniform> color: vec4<f32>;
@group(1) @binding(0) var<storage, read_write> out: vec4<f32>;
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return color; }
@compute @workgroup_size(1) fn cs_main() { out = color; }
`

// TestBindGroupBeforePipeline sets the bind groups before the pipeline, which
// WebGPU allows. DX12 set the group's tables before there was a root
// signature to take them, which WARP crashed on and which the root
// signature dropped anyway, so the draw and the dispatch read nothing. Metal
// bound group 1 at the slots of group 0, which only the layout knows.
func TestBindGroupBeforePipeline(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL {
		t.Skip("GL binds a group against the pipeline set at the time, see the README's known issues")
	}
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "bind-first", WGSL: bindFirstShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment | ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer layout.Release()
	outLayout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer outLayout.Release()
	pl, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{layout, outLayout}})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer pl.Release()
	color, err := device.CreateBuffer(&BufferDescriptor{Label: "color", Size: 16, Usage: BufferUsageUniform | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer color.Release()
	out, err := device.CreateBuffer(&BufferDescriptor{Label: "out", Size: 16, Usage: BufferUsageStorage | BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer out.Release()
	green := []float32{0, 1, 0, 1}
	if err := device.Queue().WriteBuffer(color, 0, unsafe.Slice((*byte)(unsafe.Pointer(&green[0])), 16)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: color}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()
	outGroup, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: outLayout, Entries: []BindGroupEntry{{Binding: 0, Buffer: out}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer outGroup.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "bind-first",
		Layout:    pl,
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()
	cpipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "bind-first", Layout: pl, Module: module, EntryPoint: "cs_main"})
	if err != nil {
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer cpipe.Release()

	tex, view := pixTexture(t, device, "bind-first", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "bind-first-st")
	outSt := readbackBuffer(t, device, "bind-first-out")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{}, func(p *RenderPassEncoder) {
		p.SetBindGroup(0, group, nil)
		p.SetBindGroup(1, outGroup, nil)
		p.SetPipeline(pipe)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
	copyOut(enc, tex, st)
	cp, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cp.SetBindGroup(1, outGroup, nil)
	cp.SetBindGroup(0, group, nil)
	cp.SetPipeline(cpipe)
	cp.Dispatch(1, 1, 1)
	if err := cp.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	enc.CopyBufferToBuffer(out, 0, outSt, 0, 16)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	expectPixels(t, "draw", readback(t, st), 0, 255, 0, 255)
	if got := readback(t, outSt)[:16]; !bytes.Equal(got, unsafe.Slice((*byte)(unsafe.Pointer(&green[0])), 16)) {
		t.Errorf("dispatch stored % x, want the uniform's green", got)
	}
	assertNoValidation(t, h)
}

const countShader = `
@group(0) @binding(0) var<storage, read_write> count: atomic<u32>;
@compute @workgroup_size(1) fn main() { atomicAdd(&count, 1u); }
`

// TestDispatchIndirectValidated records validated indirect dispatches at
// offsets on both sides of the storage alignment, one over the limit, in one
// submit. They used to share one params buffer, so every validation read the
// counts of the last dispatch recorded, and the validation's bind group was
// destroyed before the submit.
func TestDispatchIndirectValidated(t *testing.T) {
	device, h := newValidatedDevice(t)
	if device.core.IndirectValidation() == nil {
		t.Skip("no indirect dispatch validation on this device")
	}
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "count", WGSL: countShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
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
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "count", Layout: pl, Module: module, EntryPoint: "main"})
	if err != nil {
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipe.Release()
	count, err := device.CreateBuffer(&BufferDescriptor{Label: "count", Size: 4, Usage: BufferUsageStorage | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer count.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: count}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	const argsSize = 1024
	args, err := device.CreateBuffer(&BufferDescriptor{Label: "args", Size: argsSize, Usage: BufferUsageIndirect | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer args.Release()
	dispatches := []struct {
		offset uint64
		x      uint32
	}{
		{0, 2},
		{268, 3},
		{520, device.core.Limits.MaxComputeWorkgroupsPerDimension + 1}, // zeroed
		{argsSize - 12, 5},
	}
	data := make([]byte, argsSize)
	for _, d := range dispatches {
		binary.LittleEndian.PutUint32(data[d.offset:], d.x)
		binary.LittleEndian.PutUint32(data[d.offset+4:], 1)
		binary.LittleEndian.PutUint32(data[d.offset+8:], 1)
	}
	if err := device.Queue().WriteBuffer(args, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	if err := device.Queue().WriteBuffer(count, 0, make([]byte, 4)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	st := readbackBuffer(t, device, "count-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cp, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	cp.SetPipeline(pipe)
	cp.SetBindGroup(0, group, nil)
	for _, d := range dispatches {
		cp.DispatchIndirect(args, d.offset)
	}
	if err := cp.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	enc.CopyBufferToBuffer(count, 0, st, 0, 4)
	finishAndSubmit(t, device, enc)
	device.Poll(PollWait)

	if got := binary.LittleEndian.Uint32(readback(t, st)); got != 2+3+5 {
		t.Errorf("%d workgroups ran, want %d", got, 2+3+5)
	}
	assertNoValidation(t, h)
}

// TestTransitionThenCopyStartUsage: an explicit TransitionTextures as the
// first use sets the start state to its OldUsage, so the submit barrier
// brings the texture into the state the explicit barrier expects.
func TestTransitionThenCopyStartUsage(t *testing.T) {
	device, h := newValidatedDevice(t)

	tex, view := pixTexture(t, device, "explicit", TextureUsageRenderAttachment|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	st := readbackBuffer(t, device, "explicit-st")

	// Frame 1 leaves the texture a colour target.
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, red, nil)
	finishAndSubmit(t, device, enc)

	// Frame 2 transitions it explicitly, then copies.
	enc, err = device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.TransitionTextures([]TextureBarrier{{Texture: tex, Usage: TextureUsageTransition{OldUsage: TextureUsageRenderAttachment, NewUsage: TextureUsageCopySrc}}})
	copyOut(enc, tex, st)
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	idx := tex.coreTexture.TrackingData().Index()
	if got := cb.core.TextureScope().StartUsage(idx); got != track.TextureUsesColorTarget {
		t.Errorf("start usage = %v, want ColorTarget (the explicit barrier's OldUsage)", got)
	}
	if got := cb.core.TextureScope().EndUsage(idx); got != track.TextureUsesCopySrc {
		t.Errorf("end usage = %v, want CopySrc", got)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	expectPixels(t, "explicit transition then copy", readback(t, st), 255, 0, 0, 255)
	assertNoValidation(t, h)
}

const storeShader = `
@group(0) @binding(0) var img: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
	textureStore(img, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 1.0, 1.0));
}
`

// TestStorageTextureComputeStore: clear S red, store blue into S through a
// storage binding in a compute pass, copy S out, each in a submit of its
// own. The bind group has to describe S as a storage image (Vulkan: a
// STORAGE_IMAGE descriptor in GENERAL, DX12: a UAV), the tracker takes it
// ColorTarget -> StorageWrite -> CopySrc, and the pixels prove the store
// landed.
func TestStorageTextureComputeStore(t *testing.T) {
	device, h := newValidatedDevice(t)

	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, StorageTexture: &gputypes.StorageTextureBindingLayout{
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
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "scope-store", WGSL: storeShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "scope-store", Layout: pl, Module: module, EntryPoint: "cs_main"})
	if err != nil {
		if strings.Contains(err.Error(), "ComputeShaders") {
			t.Skipf("no compute shaders on this adapter: %v", err)
		}
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipe.Release()

	tex, view := pixTexture(t, device, "storage-target", TextureUsageRenderAttachment|TextureUsageStorageBinding|TextureUsageCopySrc)
	defer releaseTextureView(tex, view)
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, TextureView: view}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()
	st := readbackBuffer(t, device, "storage-st")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, red, nil)
	finishAndSubmit(t, device, enc)

	enc, err = device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
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
	finishAndSubmit(t, device, enc)

	enc, err = device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	copyOut(enc, tex, st)
	finishAndSubmit(t, device, enc)

	expectPixels(t, "S after render, compute store, copy in three submits", readback(t, st), 0, 0, 255, 255)
	assertNoValidation(t, h)
}

// BenchmarkSplitPassFrame is a frame with the shape fyne's blur will have
// once it uses one encoder: pass, copy out of its target, pass that samples
// the copy. Reports allocations per frame on the split-pass path.
func BenchmarkSplitPassFrame(b *testing.B) {
	inst, err := CreateInstance(&InstanceDescriptor{Backends: gputypes.BackendsAll})
	if err != nil {
		b.Fatalf("CreateInstance: %v", err)
	}
	adapter, err := inst.RequestAdapter(nil)
	if err != nil {
		b.Skipf("no adapter: %v", err)
	}
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		b.Fatalf("RequestDevice: %v", err)
	}
	defer device.Release()
	frame, err := device.CreateTexture(&TextureDescriptor{Size: Extent3D{Width: 64, Height: 64, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageRenderAttachment | TextureUsageCopySrc | TextureUsageTextureBinding})
	if err != nil {
		b.Fatal(err)
	}
	defer frame.Release()
	frameView, _ := device.CreateTextureView(frame, nil)
	defer frameView.Release()
	snap, err := device.CreateTexture(&TextureDescriptor{Size: Extent3D{Width: 64, Height: 64, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageCopyDst | TextureUsageTextureBinding})
	if err != nil {
		b.Fatal(err)
	}
	defer snap.Release()
	snapView, _ := device.CreateTextureView(snap, nil)
	defer snapView.Release()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
	}})
	if err != nil {
		b.Fatal(err)
	}
	defer layout.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, TextureView: snapView}}})
	if err != nil {
		b.Fatal(err)
	}
	defer group.Release()
	queue := device.Queue()
	copyRegion := []TextureCopy{{Source: ImageCopyTexture{Texture: frame}, Destination: ImageCopyTexture{Texture: snap}, Size: Extent3D{Width: 64, Height: 64, DepthOrArrayLayers: 1}}}
	desc := &RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{View: frameView, LoadOp: gputypes.LoadOpLoad, StoreOp: gputypes.StoreOpStore}}}

	frameFn := func() {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			b.Fatal(err)
		}
		pass, err := enc.BeginRenderPass(desc)
		if err != nil {
			b.Fatal(err)
		}
		if err := pass.End(); err != nil {
			b.Fatal(err)
		}
		enc.CopyTextureToTexture(frame, snap, copyRegion)
		pass, err = enc.BeginRenderPass(desc)
		if err != nil {
			b.Fatal(err)
		}
		pass.SetBindGroup(0, group, nil)
		if err := pass.End(); err != nil {
			b.Fatal(err)
		}
		cb, err := enc.Finish()
		if err != nil {
			b.Fatal(err)
		}
		if _, err := queue.Submit(cb); err != nil {
			b.Fatal(err)
		}
	}
	for i := 0; i < 20; i++ {
		frameFn()
	}
	device.Poll(PollWait)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		frameFn()
	}
}
