//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// aspectTexture is a pixSide square depth/stencil texture of a format with
// both aspects that can be attached, sampled and copied from, with a view
// of each aspect beside the attachment view.
type aspectTexture struct {
	device                 *Device
	tex                    *Texture
	attach, depth, stencil *TextureView
	format                 gputypes.TextureFormat
	stencilCopies          bool
}

// aspectRig draws a full-screen triangle at z 0.5 that fetches one aspect
// of a depth/stencil texture into a colour target: the stencil in red, or
// the depth in red, green and blue.
type aspectRig struct {
	layout   *BindGroupLayout
	pipeline *RenderPipeline
}

func newAspectRig(t *testing.T, device *Device, stencil bool, ds *DepthStencilState) *aspectRig {
	t.Helper()
	sampleType, shader := gputypes.TextureSampleTypeDepth, depthLoadShader
	if stencil {
		sampleType, shader = gputypes.TextureSampleTypeUint, stencilLoadShader
	}
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{SampleType: sampleType, ViewDimension: gputypes.TextureViewDimension2D}},
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
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "aspect-load", WGSL: shader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	t.Cleanup(module.Release)
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:        "aspect-load",
		Layout:       pl,
		Vertex:       VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive:    gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		DepthStencil: ds,
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	t.Cleanup(pipe.Release)
	return &aspectRig{layout: layout, pipeline: pipe}
}

func (r *aspectRig) draw(t *testing.T, device *Device, pass *RenderPassEncoder, view *TextureView) {
	t.Helper()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: r.layout, Entries: []BindGroupEntry{{Binding: 0, TextureView: view}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	t.Cleanup(group.Release)
	pass.SetPipeline(r.pipeline)
	pass.SetBindGroup(0, group, nil)
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
}

var keepStencil = StencilFaceState{Compare: gputypes.CompareFunctionAlways, FailOp: gputypes.StencilOperationKeep, DepthFailOp: gputypes.StencilOperationKeep, PassOp: gputypes.StencilOperationKeep}

var replaceStencil = StencilFaceState{Compare: gputypes.CompareFunctionAlways, FailOp: gputypes.StencilOperationKeep, DepthFailOp: gputypes.StencilOperationKeep, PassOp: gputypes.StencilOperationReplace}

// rig is an aspectRig that fetches the stencil, or the depth, with the
// texture attached as ds has it, nil for a colour-only pass.
func (a *aspectTexture) rig(t *testing.T, stencil bool, ds *DepthStencilState) *aspectRig {
	t.Helper()
	return newAspectRig(t, a.device, stencil, ds)
}

// depthWrite is the depth/stencil state of a pipeline that writes the
// depth of the texture, test Less, and keeps the stencil.
func (a *aspectTexture) depthWrite() *DepthStencilState {
	return &DepthStencilState{
		Format: a.format, DepthWriteEnabled: true, DepthCompare: gputypes.CompareFunctionLess,
		StencilFront: keepStencil, StencilBack: keepStencil, StencilReadMask: 0xFF,
	}
}

// stencilWrite is the depth/stencil state of a pipeline that replaces the
// stencil of the texture with the reference and keeps the depth.
func (a *aspectTexture) stencilWrite() *DepthStencilState {
	return &DepthStencilState{
		Format: a.format, DepthCompare: gputypes.CompareFunctionAlways,
		StencilFront: replaceStencil, StencilBack: replaceStencil, StencilReadMask: 0xFF, StencilWriteMask: 0xFF,
	}
}

// newAspectTexture makes a texture of format on a device of its own, which
// the test fails on validation messages from.
func newAspectTexture(t *testing.T, format gputypes.TextureFormat) *aspectTexture {
	t.Helper()
	var features gputypes.Features
	if format == gputypes.TextureFormatDepth32FloatStencil8 {
		features = gputypes.Features(gputypes.FeatureDepth32FloatStencil8)
	}
	device, h := newValidatedDeviceWith(t, features)
	t.Cleanup(func() { assertNoValidation(t, h) })
	tex, attach := depthStencilTexture(t, device, format, TextureUsageRenderAttachment|TextureUsageTextureBinding|TextureUsageCopySrc)
	view := func(aspect gputypes.TextureAspect) *TextureView {
		v, err := device.CreateTextureView(tex, &TextureViewDescriptor{Aspect: aspect})
		if err != nil {
			t.Fatalf("CreateTextureView(%v): %v", aspect, err)
		}
		t.Cleanup(v.Release)
		return v
	}
	stencilCopies := device.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsDepthTextureAndBufferCopies) &&
		(h.backend != gputypes.BackendDX12 || !upstreamVkd3d())
	return &aspectTexture{
		device: device, tex: tex, attach: attach, format: format, stencilCopies: stencilCopies,
		depth:   view(gputypes.TextureAspectDepthOnly),
		stencil: view(gputypes.TextureAspectStencilOnly),
	}
}

// clear records a pass that clears the depth and the stencil.
func (a *aspectTexture) clear(t *testing.T, enc *CommandEncoder, depth float32, stencil uint32) {
	t.Helper()
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		DepthStencilAttachment: &RenderPassDepthStencilAttachment{
			View:        a.attach,
			DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: depth,
			StencilLoadOp: gputypes.LoadOpClear, StencilStoreOp: gputypes.StoreOpStore, StencilClearValue: stencil,
		},
	})
	if err != nil {
		t.Fatalf("BeginRenderPass(clear): %v", err)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End(clear): %v", err)
	}
}

// fetch records a colour-only pass into a new target that fetches view
// with rig, and the copy of the target into a staging buffer it returns.
func (a *aspectTexture) fetch(t *testing.T, enc *CommandEncoder, rig *aspectRig, view *TextureView) *Buffer {
	t.Helper()
	dst, dstView := pixTexture(t, a.device, "aspect-fetch", TextureUsageRenderAttachment|TextureUsageCopySrc)
	t.Cleanup(func() { releaseTextureView(dst, dstView) })
	st := readbackBuffer(t, a.device, "aspect-fetch-st")
	clearPass(t, enc, dstView, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) { rig.draw(t, a.device, pass, view) })
	copyOut(enc, dst, st)
	return st
}

// attachedPass records a pass that draws with rig, which fetches view,
// into a new colour target with the texture attached as ds says, and the
// copy of the target into a staging buffer it returns.
func (a *aspectTexture) attachedPass(t *testing.T, enc *CommandEncoder, ds RenderPassDepthStencilAttachment, rig *aspectRig, view *TextureView, stencilRef uint32) *Buffer {
	t.Helper()
	dst, dstView := pixTexture(t, a.device, "aspect-attached", TextureUsageRenderAttachment|TextureUsageCopySrc)
	t.Cleanup(func() { releaseTextureView(dst, dstView) })
	st := readbackBuffer(t, a.device, "aspect-attached-st")
	ds.View = a.attach
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{
		ColorAttachments:       []RenderPassColorAttachment{{View: dstView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: black}},
		DepthStencilAttachment: &ds,
	})
	if err != nil {
		t.Fatalf("BeginRenderPass(attached): %v", err)
	}
	pass.SetStencilReference(stencilRef)
	rig.draw(t, a.device, pass, view)
	if err := pass.End(); err != nil {
		t.Fatalf("End(attached): %v", err)
	}
	copyOut(enc, dst, st)
	return st
}

// newEncoder makes a command encoder.
func (a *aspectTexture) newEncoder(t *testing.T) *CommandEncoder {
	t.Helper()
	enc, err := a.device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	return enc
}

// next submits enc and makes a new encoder when split is set, and returns
// enc otherwise.
func (a *aspectTexture) next(t *testing.T, enc *CommandEncoder, split bool) *CommandEncoder {
	t.Helper()
	if !split {
		return enc
	}
	finishAndSubmit(t, a.device, enc)
	return a.newEncoder(t)
}

var aspectFormats = []gputypes.TextureFormat{gputypes.TextureFormatDepth24PlusStencil8, gputypes.TextureFormatDepth32FloatStencil8}

var aspectSplits = []struct {
	name  string
	split bool
}{
	{"same_encoder", false},
	{"submits_between", true},
}

// TestDepthWrittenStencilSampled draws into a depth/stencil texture with
// the depth written and the stencil read-only, while the fragment shader
// fetches that stencil through a stencil-only view, then fetches the depth
// it wrote in a later pass. The depth and the stencil are subresources of
// their own in WebGPU, so the pass is one the tracker has to take: it
// refused it as a texture written and sampled. On Vulkan the image is in
// DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL for the pass, on DX12 its
// depth plane is DEPTH_WRITE and its stencil plane DEPTH_READ and shader
// resource.
func TestDepthWrittenStencilSampled(t *testing.T) {
	for _, format := range aspectFormats {
		for _, tc := range aspectSplits {
			t.Run(format.String()+"/"+tc.name, func(t *testing.T) {
				a := newAspectTexture(t, format)
				enc := a.newEncoder(t)
				a.clear(t, enc, 1, 7)
				enc = a.next(t, enc, tc.split)
				sampled := a.attachedPass(t, enc, RenderPassDepthStencilAttachment{
					DepthLoadOp: gputypes.LoadOpLoad, DepthStoreOp: gputypes.StoreOpStore, StencilReadOnly: true,
				}, a.rig(t, true, a.depthWrite()), a.stencil, 0)
				enc = a.next(t, enc, tc.split)
				written := a.fetch(t, enc, a.rig(t, false, nil), a.depth)
				finishAndSubmit(t, a.device, enc)

				expectPixels(t, "stencil fetched while the depth is written", readback(t, sampled), 7, 0, 0, 255)
				expectDepthPixels(t, "depth written beside the fetched stencil", readback(t, written), 0.5)
			})
		}
	}
}

// TestStencilWrittenDepthSampled is TestDepthWrittenStencilSampled the
// other way round: the stencil is written by the stencil test's Replace,
// with the depth read-only and fetched through a depth-only view. A copy
// of the stencil it wrote and a pass that fetches the depth again follow,
// each a use of one aspect of the texture the barriers take the other
// along in.
func TestStencilWrittenDepthSampled(t *testing.T) {
	for _, format := range aspectFormats {
		for _, tc := range aspectSplits {
			t.Run(format.String()+"/"+tc.name, func(t *testing.T) {
				a := newAspectTexture(t, format)
				enc := a.newEncoder(t)
				a.clear(t, enc, 0.25, 0)
				enc = a.next(t, enc, tc.split)
				sampled := a.attachedPass(t, enc, RenderPassDepthStencilAttachment{
					DepthReadOnly: true, StencilLoadOp: gputypes.LoadOpLoad, StencilStoreOp: gputypes.StoreOpStore,
				}, a.rig(t, false, a.stencilWrite()), a.depth, 9)
				enc = a.next(t, enc, tc.split)
				var stencil *Buffer
				if a.stencilCopies {
					stencil = readbackBuffer(t, a.device, "aspect-stencil-st")
					enc.CopyTextureToBuffer(a.tex, stencil, []BufferTextureCopy{{
						TextureBase:  ImageCopyTexture{Texture: a.tex, Aspect: gputypes.TextureAspectStencilOnly},
						BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
						Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
					}})
				}
				again := a.fetch(t, enc, a.rig(t, false, nil), a.depth)
				written := a.fetch(t, enc, a.rig(t, true, nil), a.stencil)
				finishAndSubmit(t, a.device, enc)

				expectDepthPixels(t, "depth fetched while the stencil is written", readback(t, sampled), 0.25)
				expectDepthPixels(t, "depth fetched after the stencil copy", readback(t, again), 0.25)
				expectPixels(t, "stencil written beside the fetched depth", readback(t, written), 9, 0, 0, 255)
				if stencil != nil {
					got := readback(t, stencil)
					for y := range pixSide {
						if row := got[y*pixSide*4 : y*pixSide*4+pixSide]; string(row) != "\x09\x09\x09\x09" {
							t.Errorf("copied stencil row %d = % x, want 09s", y, row)
						}
					}
				}
			})
		}
	}
}

// TestAspectWrittenAndSampledConflicts: an aspect a pass writes as its
// attachment and binds to be sampled is still a usage conflict, and so is
// either aspect of an attachment with both written.
func TestAspectWrittenAndSampledConflicts(t *testing.T) {
	a := newAspectTexture(t, gputypes.TextureFormatDepth24PlusStencil8)
	for _, tc := range []struct {
		name    string
		ds      RenderPassDepthStencilAttachment
		rig     *aspectRig
		sampled *TextureView
	}{
		{"depth written and sampled", RenderPassDepthStencilAttachment{
			DepthLoadOp: gputypes.LoadOpLoad, DepthStoreOp: gputypes.StoreOpStore, StencilReadOnly: true,
		}, a.rig(t, false, a.depthWrite()), a.depth},
		{"stencil written and sampled", RenderPassDepthStencilAttachment{
			DepthReadOnly: true, StencilLoadOp: gputypes.LoadOpLoad, StencilStoreOp: gputypes.StoreOpStore,
		}, a.rig(t, true, a.stencilWrite()), a.stencil},
		{"both written, stencil sampled", RenderPassDepthStencilAttachment{
			DepthLoadOp: gputypes.LoadOpLoad, DepthStoreOp: gputypes.StoreOpStore,
			StencilLoadOp: gputypes.LoadOpLoad, StencilStoreOp: gputypes.StoreOpStore,
		}, a.rig(t, true, a.depthWrite()), a.stencil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			enc := a.newEncoder(t)
			ds := tc.ds
			ds.View = a.attach
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{DepthStencilAttachment: &ds})
			if err != nil {
				t.Fatalf("BeginRenderPass: %v", err)
			}
			group, err := a.device.CreateBindGroup(&BindGroupDescriptor{Layout: tc.rig.layout, Entries: []BindGroupEntry{{Binding: 0, TextureView: tc.sampled}}})
			if err != nil {
				t.Fatalf("CreateBindGroup: %v", err)
			}
			defer group.Release()
			pass.SetBindGroup(0, group, nil)
			_ = pass.End()
			cb, err := enc.Finish()
			if err == nil {
				cb.Release()
				t.Fatal("an aspect written as the attachment and sampled in one pass was not refused")
			}
			if !strings.Contains(err.Error(), "usage conflict") {
				t.Errorf("Finish = %v, want a usage conflict", err)
			}
		})
	}
}
