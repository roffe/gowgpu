//go:build !(js && wasm)

package wgpu

import (
	"context"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// GPU tests of usage per subresource: WebGPU tracks the usages of a texture
// per mip level and array layer, so one level or layer may be rendered or
// copied into while another is sampled or copied from, in one pass or one
// copy. Run with the validation layer and sync validation on (see
// scope_transitions_gpu_test.go), they also check the barriers each
// subresource gets.

// subresSide is the width and height of mip level 0 of the textures here.
const subresSide = 16

// halveFS draws the texel of the bound view under the pixel's position on
// the level above, at half its color, so every level a chain of passes
// makes from the one above is half as bright as that one.
const halveFS = `
@group(0) @binding(0) var src: texture_2d<f32>;
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	let c = textureLoad(src, vec2<i32>(pos.xy) * 2, 0);
	return vec4<f32>(c.rgb * 0.5, 1.0);
}
`

// sameLayerFS is halveFS reading the bound view at the pixel's position,
// for a view of another layer of the same size.
const sameLayerFS = `
@group(0) @binding(0) var src: texture_2d<f32>;
@fragment fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
	let c = textureLoad(src, vec2<i32>(pos.xy), 0);
	return vec4<f32>(c.rgb * 0.5, 1.0);
}
`

// subresColor is what level 0 holds, and level n each of its channels
// halved n times: 200, 120 and 80 halve without rounding three times.
var subresColor = gputypes.Color{R: 200.0 / 255, G: 120.0 / 255, B: 80.0 / 255, A: 1}

func halved(n uint) [4]byte {
	return [4]byte{byte(200 >> n), byte(120 >> n), byte(80 >> n), 255}
}

// subresTexture makes a subresSide square RGBA8 texture with levels mip
// levels and layers array layers that can be rendered, sampled and copied.
func subresTexture(t *testing.T, device *Device, levels, layers uint32, extra gputypes.TextureUsage) *Texture {
	t.Helper()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         "subresource",
		Size:          Extent3D{Width: subresSide, Height: subresSide, DepthOrArrayLayers: layers},
		MipLevelCount: levels,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageRenderAttachment | TextureUsageTextureBinding | TextureUsageCopySrc | TextureUsageCopyDst | extra,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	t.Cleanup(tex.Release)
	return tex
}

// subresView is the 2D view of one level and layer of tex.
func subresView(t *testing.T, device *Device, tex *Texture, level, layer uint32) *TextureView {
	t.Helper()
	view, err := device.CreateTextureView(tex, &TextureViewDescriptor{
		Dimension:       gputypes.TextureViewDimension2D,
		BaseMipLevel:    level,
		MipLevelCount:   1,
		BaseArrayLayer:  layer,
		ArrayLayerCount: 1,
	})
	if err != nil {
		t.Fatalf("CreateTextureView(level %d, layer %d): %v", level, layer, err)
	}
	t.Cleanup(view.Release)
	return view
}

// subresRig holds the pipeline and layout the passes below draw with.
type subresRig struct {
	layoutRig
	srcLayout *BindGroupLayout
	pipe      *RenderPipeline
}

func newSubresRig(t *testing.T, device *Device, fs string) subresRig {
	r := subresRig{layoutRig: layoutRig{t, device}}
	r.srcLayout = r.bgl(gputypes.BindGroupLayoutEntry{Binding: 0, Visibility: ShaderStageFragment,
		Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}})
	r.pipe = r.renderPipeline(r.layout(r.srcLayout), fs)
	return r
}

// pass records a pass that draws into dst from src, both views of the same
// texture.
func (r subresRig) pass(enc *CommandEncoder, dst, src *TextureView) {
	r.t.Helper()
	group := r.group(r.srcLayout, BindGroupEntry{Binding: 0, TextureView: src})
	clearPass(r.t, enc, dst, gputypes.LoadOpClear, black, func(p *RenderPassEncoder) {
		p.SetPipeline(r.pipe)
		p.SetBindGroup(0, group, nil)
		p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	})
}

// levelBuffer makes a buffer one level of a subresTexture is copied into,
// rows 256 bytes apart.
func levelBuffer(t *testing.T, device *Device) *Buffer {
	t.Helper()
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: pixRow * subresSide, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(buf.Release)
	return buf
}

// copyLevelOut records a copy of level and layer of tex into buf.
func copyLevelOut(enc *CommandEncoder, tex *Texture, level, layer uint32, buf *Buffer) {
	side := uint32(subresSide >> level)
	enc.CopyTextureToBuffer(tex, buf, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex, MipLevel: level, Origin: Origin3D{Z: layer}},
		BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: side},
		Size:         Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
	}})
}

// expectLevel checks that the copy of level in buf holds want everywhere.
func expectLevel(t *testing.T, what string, buf *Buffer, level uint32, want [4]byte) {
	t.Helper()
	side := uint32(subresSide >> level)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, MapModeRead, 0, buf.Size()); err != nil {
		t.Fatalf("%s: Map: %v", what, err)
	}
	rng, err := buf.MappedRange(0, buf.Size())
	if err != nil {
		t.Fatalf("%s: MappedRange: %v", what, err)
	}
	data := rng.Bytes()
	pix := make([]byte, 0, side*side*4)
	for y := range side {
		pix = append(pix, data[y*pixRow:y*pixRow+side*4]...)
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("%s: Unmap: %v", what, err)
	}
	expectPixels(t, what, pix, want[0], want[1], want[2], want[3])
}

// TestMipGenerationInOneEncoder renders each mip level of a texture from
// the one above it in one encoder, a pass per level that draws into the
// level while it samples the level above, the way mip chains are made on
// the GPU, then copies every level out in the same encoder.
func TestMipGenerationInOneEncoder(t *testing.T) {
	device, h := newValidatedDevice(t)
	const levels = 3
	tex := subresTexture(t, device, levels, 1, 0)
	r := newSubresRig(t, device, halveFS)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, subresView(t, device, tex, 0, 0), gputypes.LoadOpClear, subresColor, nil)
	for level := uint32(1); level < levels; level++ {
		r.pass(enc, subresView(t, device, tex, level, 0), subresView(t, device, tex, level-1, 0))
	}
	var bufs [levels]*Buffer
	for level := range uint32(levels) {
		bufs[level] = levelBuffer(t, device)
		copyLevelOut(enc, tex, level, 0, bufs[level])
	}
	finishAndSubmit(t, device, enc)
	for level := range uint32(levels) {
		expectLevel(t, "mip level "+string(rune('0'+level)), bufs[level], level, halved(uint(level)))
	}
	assertNoValidation(t, h)
}

// TestMipGenerationAcrossEncoders is TestMipGenerationInOneEncoder with an
// encoder per level, submitted together and one by one: the submit-time
// barriers take each level from what the encoder before left it in.
func TestMipGenerationAcrossEncoders(t *testing.T) {
	for _, together := range []bool{true, false} {
		name := "one submit"
		if !together {
			name = "a submit each"
		}
		t.Run(name, func(t *testing.T) {
			device, h := newValidatedDevice(t)
			const levels = 3
			tex := subresTexture(t, device, levels, 1, 0)
			r := newSubresRig(t, device, halveFS)

			var cbs []*CommandBuffer
			record := func(f func(*CommandEncoder)) {
				t.Helper()
				enc, err := device.CreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("CreateCommandEncoder: %v", err)
				}
				f(enc)
				cb, err := enc.Finish()
				if err != nil {
					t.Fatalf("Finish: %v", err)
				}
				if together {
					cbs = append(cbs, cb)
					return
				}
				if _, err := device.Queue().Submit(cb); err != nil {
					t.Fatalf("Submit: %v", err)
				}
			}
			record(func(enc *CommandEncoder) {
				clearPass(t, enc, subresView(t, device, tex, 0, 0), gputypes.LoadOpClear, subresColor, nil)
			})
			for level := uint32(1); level < levels; level++ {
				record(func(enc *CommandEncoder) {
					r.pass(enc, subresView(t, device, tex, level, 0), subresView(t, device, tex, level-1, 0))
				})
			}
			var bufs [levels]*Buffer
			for level := range uint32(levels) {
				bufs[level] = levelBuffer(t, device)
				record(func(enc *CommandEncoder) { copyLevelOut(enc, tex, level, 0, bufs[level]) })
			}
			if together {
				if _, err := device.Queue().Submit(cbs...); err != nil {
					t.Fatalf("Submit: %v", err)
				}
			}
			for level := range uint32(levels) {
				expectLevel(t, "mip level "+string(rune('0'+level)), bufs[level], level, halved(uint(level)))
			}
			assertNoValidation(t, h)
		})
	}
}

// TestCopyBetweenMipsOfOneTexture copies level 0 of a texture into level 1
// and level 1 into level 2 of the same texture, each copy one usage scope
// with the texture as source and destination, and refuses a copy within
// one subresource.
func TestCopyBetweenMipsOfOneTexture(t *testing.T) {
	device, h := newValidatedDevice(t)
	tex := subresTexture(t, device, 3, 1, 0)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, subresView(t, device, tex, 0, 0), gputypes.LoadOpClear, subresColor, nil)
	for level := uint32(1); level < 3; level++ {
		side := uint32(subresSide >> level)
		enc.CopyTextureToTexture(tex, tex, []TextureCopy{{
			Source:      ImageCopyTexture{Texture: tex, MipLevel: level - 1},
			Destination: ImageCopyTexture{Texture: tex, MipLevel: level},
			Size:        Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
		}})
	}
	var bufs [3]*Buffer
	for level := range uint32(3) {
		bufs[level] = levelBuffer(t, device)
		copyLevelOut(enc, tex, level, 0, bufs[level])
	}
	finishAndSubmit(t, device, enc)
	for level := range uint32(3) {
		expectLevel(t, "mip level "+string(rune('0'+level)), bufs[level], level, halved(0))
	}
	assertNoValidation(t, h)

	enc, err = device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyTextureToTexture(tex, tex, []TextureCopy{{
		Source:      ImageCopyTexture{Texture: tex, MipLevel: 1},
		Destination: ImageCopyTexture{Texture: tex, MipLevel: 1, Origin: Origin3D{X: 4}},
		Size:        Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	if cb, err := enc.Finish(); err == nil {
		cb.Release()
		t.Error("a copy within one mip level of a texture was not refused")
	}
}

// TestLayersUsedDifferentlyInOnePass draws into layer 1 of an array texture
// while the pass samples layer 0 of it, and copies layer 0 into layer 2 of
// the same texture.
func TestLayersUsedDifferentlyInOnePass(t *testing.T) {
	device, h := newValidatedDevice(t)
	skipWithoutGLTextureViews(t, device, h)
	tex := subresTexture(t, device, 1, 3, 0)
	r := newSubresRig(t, device, sameLayerFS)

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, subresView(t, device, tex, 0, 0), gputypes.LoadOpClear, subresColor, nil)
	r.pass(enc, subresView(t, device, tex, 0, 1), subresView(t, device, tex, 0, 0))
	enc.CopyTextureToTexture(tex, tex, []TextureCopy{{
		Source:      ImageCopyTexture{Texture: tex},
		Destination: ImageCopyTexture{Texture: tex, Origin: Origin3D{Z: 2}},
		Size:        Extent3D{Width: subresSide, Height: subresSide, DepthOrArrayLayers: 1},
	}})
	var bufs [3]*Buffer
	for layer := range uint32(3) {
		bufs[layer] = levelBuffer(t, device)
		copyLevelOut(enc, tex, 0, layer, bufs[layer])
	}
	finishAndSubmit(t, device, enc)
	expectLevel(t, "layer 0", bufs[0], 0, halved(0))
	expectLevel(t, "layer 1, drawn from layer 0", bufs[1], 0, halved(1))
	expectLevel(t, "layer 2, copied from layer 0", bufs[2], 0, halved(0))
	assertNoValidation(t, h)
}

// TestMipGenerationByDispatch makes level 1 of a texture from level 0 in a
// compute dispatch that reads level 0 and writes level 1 as a storage
// texture, then samples level 1 in a pass that draws into level 2.
func TestMipGenerationByDispatch(t *testing.T) {
	device, h := newValidatedDevice(t)
	skipWithoutGLTextureViews(t, device, h)
	tex := subresTexture(t, device, 3, 1, TextureUsageStorageBinding)
	r := newSubresRig(t, device, halveFS)

	cbgl := r.bgl(
		gputypes.BindGroupLayoutEntry{Binding: 0, Visibility: ShaderStageCompute,
			Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D}},
		gputypes.BindGroupLayoutEntry{Binding: 1, Visibility: ShaderStageCompute,
			StorageTexture: &gputypes.StorageTextureBindingLayout{Access: gputypes.StorageTextureAccessWriteOnly, Format: TextureFormatRGBA8Unorm, ViewDimension: gputypes.TextureViewDimension2D}},
	)
	cpipe := r.computePipeline(r.layout(cbgl), `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
	let c = textureLoad(src, vec2<i32>(id.xy) * 2, 0);
	textureStore(dst, vec2<i32>(id.xy), vec4<f32>(c.rgb * 0.5, 1.0));
}
`)
	group := r.group(cbgl,
		BindGroupEntry{Binding: 0, TextureView: subresView(t, device, tex, 0, 0)},
		BindGroupEntry{Binding: 1, TextureView: subresView(t, device, tex, 1, 0)})

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, subresView(t, device, tex, 0, 0), gputypes.LoadOpClear, subresColor, nil)
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	pass.SetPipeline(cpipe)
	pass.SetBindGroup(0, group, nil)
	pass.Dispatch(1, 1, 1)
	if err := pass.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	r.pass(enc, subresView(t, device, tex, 2, 0), subresView(t, device, tex, 1, 0))
	var bufs [3]*Buffer
	for level := range uint32(3) {
		bufs[level] = levelBuffer(t, device)
		copyLevelOut(enc, tex, level, 0, bufs[level])
	}
	finishAndSubmit(t, device, enc)
	for level := range uint32(3) {
		expectLevel(t, "mip level "+string(rune('0'+level)), bufs[level], level, halved(uint(level)))
	}
	assertNoValidation(t, h)
}
