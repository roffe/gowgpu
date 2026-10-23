//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// A texture whose contents a pass discarded reads as zeros, as WebGPU has
// it: a later pass that loads it in the same encoder, a copy in the same
// encoder and a pass that loads it in a later submit all see zeros, not the
// red the discarding pass cleared it to.
func TestDiscardedTextureReadsZeros(t *testing.T) {
	device, h := newValidatedDevice(t)
	red := gputypes.Color{R: 1, A: 1}
	target := func() (*Texture, *TextureView) {
		tex, view := pixTexture(t, device, "discarded", TextureUsageRenderAttachment|TextureUsageCopySrc)
		t.Cleanup(func() { releaseTextureView(tex, view) })
		return tex, view
	}
	pass := func(enc *CommandEncoder, view *TextureView, load gputypes.LoadOp, store gputypes.StoreOp) {
		p, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{
			{View: view, LoadOp: load, StoreOp: store, ClearValue: red},
		}})
		if err != nil {
			t.Fatalf("BeginRenderPass: %v", err)
		}
		if err := p.End(); err != nil {
			t.Fatalf("End: %v", err)
		}
	}
	encoder := func() *CommandEncoder {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		return enc
	}

	tex, view := target()
	out := readbackBuffer(t, device, "loaded")
	enc := encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	pass(enc, view, gputypes.LoadOpLoad, gputypes.StoreOpStore)
	copyOut(enc, tex, out)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "loaded after a discard in the encoder", readback(t, out), 0, 0, 0, 0)

	tex, view = target()
	out = readbackBuffer(t, device, "copied")
	enc = encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	copyOut(enc, tex, out)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "copied after a discard in the encoder", readback(t, out), 0, 0, 0, 0)

	tex, view = target()
	out = readbackBuffer(t, device, "next submit")
	enc = encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	finishAndSubmit(t, device, enc)
	enc = encoder()
	pass(enc, view, gputypes.LoadOpLoad, gputypes.StoreOpStore)
	copyOut(enc, tex, out)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "loaded in the submit after the discard", readback(t, out), 0, 0, 0, 0)

	tex, view = target()
	out = readbackBuffer(t, device, "cleared again")
	enc = encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	finishAndSubmit(t, device, enc)
	enc = encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpStore)
	copyOut(enc, tex, out)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "cleared and stored after the discard", readback(t, out), 255, 0, 0, 255)
	assertNoValidation(t, h)
}

// A texture whose contents a pass discarded reads as zeros through a bind
// group too: sampled by a later pass or dispatch of the same encoder, it
// reads zeros, not the red the discarding pass cleared it to.
func TestDiscardedTextureSampledReadsZeros(t *testing.T) {
	device, h := newValidatedDevice(t)
	r := layoutRig{t, device}
	texEntry := gputypes.BindGroupLayoutEntry{Binding: 0, Visibility: ShaderStageFragment | ShaderStageCompute,
		Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}}
	bgl := r.bgl(texEntry)
	pipe := r.renderPipeline(r.layout(bgl), `
@group(0) @binding(0) var src: texture_2d<f32>;
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	return textureLoad(src, vec2<i32>(pos.xy), 0) + vec4<f32>(0.0, 0.0, 1.0, 1.0);
}
`)
	encoder := func() *CommandEncoder {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		return enc
	}
	discarded := func(enc *CommandEncoder) *BindGroup {
		tex, view := pixTexture(t, device, "discarded", TextureUsageRenderAttachment|TextureUsageTextureBinding)
		t.Cleanup(func() { releaseTextureView(tex, view) })
		p, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{
			{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpDiscard, ClearValue: red},
		}})
		if err != nil {
			t.Fatalf("BeginRenderPass: %v", err)
		}
		if err := p.End(); err != nil {
			t.Fatalf("End: %v", err)
		}
		return r.group(bgl, BindGroupEntry{Binding: 0, TextureView: view})
	}
	sample := func(enc *CommandEncoder, group *BindGroup) *Buffer {
		tex, view := pixTexture(t, device, "sampled into", TextureUsageRenderAttachment|TextureUsageCopySrc)
		t.Cleanup(func() { releaseTextureView(tex, view) })
		clearPass(t, enc, view, gputypes.LoadOpClear, black, func(p *RenderPassEncoder) {
			p.SetPipeline(pipe)
			p.SetBindGroup(0, group, nil)
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		})
		out := readbackBuffer(t, device, "sampled")
		copyOut(enc, tex, out)
		return out
	}

	enc := encoder()
	out := sample(enc, discarded(enc))
	finishAndSubmit(t, device, enc)
	expectPixels(t, "sampled by a pass of the encoder", readback(t, out), 0, 0, 255, 255)

	if h.backend != gputypes.BackendGL || device.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsComputeShaders) {
		sbgl := r.bgl(storageEntry(0))
		cpipe := r.computePipeline(r.layout(bgl, sbgl), `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(0) var<storage, read_write> dst: vec4<f32>;
@compute @workgroup_size(1) fn cs_main() {
	dst = textureLoad(src, vec2<i32>(0, 0), 0) + vec4<f32>(0.0, 0.0, 1.0, 1.0);
}
`)
		buf := r.buffer(gputypes.BufferUsageStorage|gputypes.BufferUsageCopySrc, [4]float32{})
		dst := r.group(sbgl, BindGroupEntry{Binding: 0, Buffer: buf, Size: 16})
		enc = encoder()
		group := discarded(enc)
		pass, err := enc.BeginComputePass(nil)
		if err != nil {
			t.Fatalf("BeginComputePass: %v", err)
		}
		pass.SetPipeline(cpipe)
		pass.SetBindGroup(0, group, nil)
		pass.SetBindGroup(1, dst, nil)
		pass.Dispatch(1, 1, 1)
		if err := pass.End(); err != nil {
			t.Fatalf("ComputePass.End: %v", err)
		}
		st := readbackBuffer(t, device, "dispatched")
		enc.CopyBufferToBuffer(buf, 0, st, 0, 16)
		finishAndSubmit(t, device, enc)
		expectFloats4(t, "sampled by a dispatch of the encoder", readFloats(t, st, 4), [4]float32{0, 0, 1, 1})
	}
	assertNoValidation(t, h)
}

// A texture whose contents a pass of one command buffer discarded reads as
// zeros in a later command buffer of the same submit: sampled, loaded by a
// pass and copied, it reads zeros, and a pass that clears it first sets it.
func TestDiscardedTextureReadByALaterCommandBuffer(t *testing.T) {
	device, h := newValidatedDevice(t)
	r := layoutRig{t, device}
	bgl := r.bgl(gputypes.BindGroupLayoutEntry{Binding: 0, Visibility: ShaderStageFragment,
		Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}})
	pipe := r.renderPipeline(r.layout(bgl), `
@group(0) @binding(0) var src: texture_2d<f32>;
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	return textureLoad(src, vec2<i32>(pos.xy), 0) + vec4<f32>(0.0, 0.0, 1.0, 1.0);
}
`)
	encoder := func() *CommandEncoder {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		return enc
	}
	pass := func(enc *CommandEncoder, view *TextureView, load gputypes.LoadOp, store gputypes.StoreOp) {
		p, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{
			{View: view, LoadOp: load, StoreOp: store, ClearValue: red},
		}})
		if err != nil {
			t.Fatalf("BeginRenderPass: %v", err)
		}
		if err := p.End(); err != nil {
			t.Fatalf("End: %v", err)
		}
	}
	// submit discards a texture in one command buffer and has read record
	// what reads it in the next one, submitted together.
	submit := func(read func(*CommandEncoder, *Texture, *TextureView) *Buffer) []byte {
		tex, view := pixTexture(t, device, "discarded", TextureUsageRenderAttachment|TextureUsageTextureBinding|TextureUsageCopySrc)
		t.Cleanup(func() { releaseTextureView(tex, view) })
		first, second := encoder(), encoder()
		pass(first, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
		out := read(second, tex, view)
		cbs := make([]*CommandBuffer, 2)
		for i, enc := range []*CommandEncoder{first, second} {
			var err error
			if cbs[i], err = enc.Finish(); err != nil {
				t.Fatalf("Finish: %v", err)
			}
		}
		if _, err := device.Queue().Submit(cbs...); err != nil {
			t.Fatalf("Submit: %v", err)
		}
		return readback(t, out)
	}

	expectPixels(t, "sampled", submit(func(enc *CommandEncoder, _ *Texture, view *TextureView) *Buffer {
		group := r.group(bgl, BindGroupEntry{Binding: 0, TextureView: view})
		tex, target := pixTexture(t, device, "sampled into", TextureUsageRenderAttachment|TextureUsageCopySrc)
		t.Cleanup(func() { releaseTextureView(tex, target) })
		clearPass(t, enc, target, gputypes.LoadOpClear, black, func(p *RenderPassEncoder) {
			p.SetPipeline(pipe)
			p.SetBindGroup(0, group, nil)
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		})
		out := readbackBuffer(t, device, "sampled")
		copyOut(enc, tex, out)
		return out
	}), 0, 0, 255, 255)
	expectPixels(t, "loaded", submit(func(enc *CommandEncoder, tex *Texture, view *TextureView) *Buffer {
		pass(enc, view, gputypes.LoadOpLoad, gputypes.StoreOpStore)
		out := readbackBuffer(t, device, "loaded")
		copyOut(enc, tex, out)
		return out
	}), 0, 0, 0, 0)
	expectPixels(t, "copied", submit(func(enc *CommandEncoder, tex *Texture, _ *TextureView) *Buffer {
		out := readbackBuffer(t, device, "copied")
		copyOut(enc, tex, out)
		return out
	}), 0, 0, 0, 0)
	expectPixels(t, "cleared first", submit(func(enc *CommandEncoder, tex *Texture, view *TextureView) *Buffer {
		pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpStore)
		out := readbackBuffer(t, device, "cleared")
		copyOut(enc, tex, out)
		return out
	}), 255, 0, 0, 255)
	assertNoValidation(t, h)
}

// A depth texture whose contents a pass discarded reads as zeros, as a
// color one does: loaded by a later pass of the encoder, copied by the
// encoder, by a later command buffer of the submit and in the next submit.
// The discarding passes clear it to 0.75.
func TestDiscardedDepthTextureReadsZeros(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL && !device.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsDepthTextureAndBufferCopies) {
		t.Skip("GL ES reads no depth back")
	}
	target := func() (*Texture, *TextureView) {
		return depthStencilTexture(t, device, gputypes.TextureFormatDepth16Unorm, TextureUsageRenderAttachment|TextureUsageCopySrc)
	}
	pass := func(enc *CommandEncoder, view *TextureView, load gputypes.LoadOp, store gputypes.StoreOp) {
		passOn(t, enc, &RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{
			View: view, DepthLoadOp: load, DepthStoreOp: store, DepthClearValue: 0.75,
		}})
	}
	copyDepth := func(enc *CommandEncoder, tex *Texture, what string) *Buffer {
		out := readbackBuffer(t, device, what)
		enc.CopyTextureToBuffer(tex, out, []BufferTextureCopy{{
			TextureBase:  ImageCopyTexture{Texture: tex, Aspect: gputypes.TextureAspectDepthOnly},
			BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
			Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
		}})
		return out
	}
	// depthOf is the depth copyDepth copied into out, its rows packed.
	depthOf := func(out *Buffer) []byte {
		pix, depth := readback(t, out), make([]byte, 0, pixSide*pixSide*2)
		for y := range pixSide {
			depth = append(depth, pix[y*pixSide*4:y*pixSide*4+pixSide*2]...)
		}
		return depth
	}
	encoder := func() *CommandEncoder {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		return enc
	}
	discarded := func() (*Texture, *TextureView) {
		tex, view := target()
		enc := encoder()
		pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
		finishAndSubmit(t, device, enc)
		return tex, view
	}

	tex, view := target()
	enc := encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	pass(enc, view, gputypes.LoadOpLoad, gputypes.StoreOpStore)
	out := copyDepth(enc, tex, "loaded")
	finishAndSubmit(t, device, enc)
	expectBytes(t, "loaded after a discard in the encoder", depthOf(out), 0)

	tex, view = target()
	enc = encoder()
	pass(enc, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	out = copyDepth(enc, tex, "copied")
	finishAndSubmit(t, device, enc)
	expectBytes(t, "copied after a discard in the encoder", depthOf(out), 0)

	tex, view = target()
	first, second := encoder(), encoder()
	pass(first, view, gputypes.LoadOpClear, gputypes.StoreOpDiscard)
	out = copyDepth(second, tex, "later command buffer")
	cbs := make([]*CommandBuffer, 2)
	for i, enc := range []*CommandEncoder{first, second} {
		var err error
		if cbs[i], err = enc.Finish(); err != nil {
			t.Fatalf("Finish: %v", err)
		}
	}
	if _, err := device.Queue().Submit(cbs...); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	expectBytes(t, "copied by a later command buffer of the submit", depthOf(out), 0)

	tex, view = discarded()
	enc = encoder()
	pass(enc, view, gputypes.LoadOpLoad, gputypes.StoreOpStore)
	out = copyDepth(enc, tex, "next submit")
	finishAndSubmit(t, device, enc)
	expectBytes(t, "loaded in the submit after the discard", depthOf(out), 0)

	tex, _ = discarded()
	enc = encoder()
	out = copyDepth(enc, tex, "copied next submit")
	finishAndSubmit(t, device, enc)
	expectBytes(t, "copied in the submit after the discard", depthOf(out), 0)

	assertNoValidation(t, h)
}

// A multisampled texture whose contents a pass discarded reads as zeros: a
// pass that loads it resolves zeros, later in the encoder, in a later
// command buffer of the submit and in the next submit. The discarding
// passes clear it to red.
func TestDiscardedMultisampledTextureReadsZeros(t *testing.T) {
	device, h := newValidatedDevice(t)
	target := func() *TextureView {
		tex, err := device.CreateTexture(&TextureDescriptor{
			Label: "discarded multisampled", Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
			MipLevelCount: 1, SampleCount: 4, Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm,
			Usage: TextureUsageRenderAttachment,
		})
		if err != nil {
			t.Fatalf("CreateTexture: %v", err)
		}
		view, err := device.CreateTextureView(tex, nil)
		if err != nil {
			t.Fatalf("CreateTextureView: %v", err)
		}
		t.Cleanup(func() { releaseTextureView(tex, view) })
		return view
	}
	discard := func(enc *CommandEncoder, view *TextureView) {
		passOn(t, enc, &RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpDiscard, ClearValue: red,
		}}})
	}
	// resolve records a pass that loads view and resolves it, and the copy
	// of what it resolved into the buffer it returns.
	resolve := func(enc *CommandEncoder, view *TextureView) *Buffer {
		tex, into := pixTexture(t, device, "resolved", TextureUsageRenderAttachment|TextureUsageCopySrc)
		t.Cleanup(func() { releaseTextureView(tex, into) })
		passOn(t, enc, &RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: view, ResolveTarget: into, LoadOp: gputypes.LoadOpLoad, StoreOp: gputypes.StoreOpStore,
		}}})
		out := readbackBuffer(t, device, "resolved")
		copyOut(enc, tex, out)
		return out
	}
	encoder := func() *CommandEncoder {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		return enc
	}

	view := target()
	enc := encoder()
	discard(enc, view)
	out := resolve(enc, view)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "resolved after a discard in the encoder", readback(t, out), 0, 0, 0, 0)

	view = target()
	first, second := encoder(), encoder()
	discard(first, view)
	out = resolve(second, view)
	cbs := make([]*CommandBuffer, 2)
	for i, enc := range []*CommandEncoder{first, second} {
		var err error
		if cbs[i], err = enc.Finish(); err != nil {
			t.Fatalf("Finish: %v", err)
		}
	}
	if _, err := device.Queue().Submit(cbs...); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	expectPixels(t, "resolved by a later command buffer of the submit", readback(t, out), 0, 0, 0, 0)

	view = target()
	enc = encoder()
	discard(enc, view)
	finishAndSubmit(t, device, enc)
	enc = encoder()
	out = resolve(enc, view)
	finishAndSubmit(t, device, enc)
	expectPixels(t, "resolved in the submit after the discard", readback(t, out), 0, 0, 0, 0)
	assertNoValidation(t, h)
}

// A write into the stencil of a combined depth/stencil texture whose
// contents a pass discarded zeroes the texture first and keeps what it
// wrote, on GL too, where the write goes to the texture straight away and
// the zeroing is a pass.
func TestStencilWriteAfterADiscard(t *testing.T) {
	device, h := newValidatedDevice(t)
	skipStencilWritesOnGLES(t, device, h)
	skipStencilCopiesOnVkd3d(t, h)
	tex, view := depthStencilTexture(t, device, gputypes.TextureFormatDepth24PlusStencil8,
		TextureUsageRenderAttachment|TextureUsageCopySrc|TextureUsageCopyDst)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	passOn(t, enc, &RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{
		View:        view,
		DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpDiscard, DepthClearValue: 0.75,
		StencilLoadOp: gputypes.LoadOpClear, StencilStoreOp: gputypes.StoreOpDiscard, StencilClearValue: 0x55,
	}})
	finishAndSubmit(t, device, enc)
	rows := make([]byte, pixSide*pixSide)
	for i := range rows {
		rows[i] = byte(0x10 + i)
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex, Aspect: gputypes.TextureAspectStencilOnly}, rows,
		&ImageDataLayout{BytesPerRow: pixSide, RowsPerImage: pixSide}, &Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	got := readTextureAspect(t, device, tex, gputypes.TextureAspectStencilOnly)
	for y := range pixSide {
		if row, want := got[y*pixRow:y*pixRow+pixSide], rows[y*pixSide:(y+1)*pixSide]; !bytes.Equal(row, want) {
			t.Errorf("stencil row %d = % x, want % x", y, row, want)
		}
	}
	assertNoValidation(t, h)
}
