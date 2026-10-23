//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
	"slices"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// TestCompressedCopies uploads both layers of mip 0 of an 8x8 BC1 texture
// in one write, rowsPerImage 2 rows of blocks, and the 2x2 mip 2 as the one
// block a WebGPU copy of it is, and reads them back. Vulkan and GL took
// rowsPerImage as texel rows, so the second layer came from the first's
// bytes, and Vulkan copied the block's 4x4 into the 2x2 mip. GL reads no
// compressed texture back, so there the texels the blocks decode to are
// compared instead (sampleCompressedCopies).
func TestCompressedCopies(t *testing.T) {
	device, h := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureTextureCompressionBC))
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         "bc1",
		Size:          Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 2},
		MipLevelCount: 4,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        gputypes.TextureFormatBC1RGBAUnorm,
		Usage:         TextureUsageCopySrc | TextureUsageCopyDst | TextureUsageTextureBinding,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	const blockBytes = 8
	layers := make([]byte, 2*2*2*blockBytes) // two layers of 2x2 blocks
	for i := range layers {
		layers[i] = byte(i + 1)
	}
	q := device.Queue()
	if err := q.WriteTexture(&ImageCopyTexture{Texture: tex}, layers,
		&ImageDataLayout{BytesPerRow: 2 * blockBytes, RowsPerImage: 2}, &Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 2}); err != nil {
		t.Fatalf("WriteTexture(mip 0): %v", err)
	}
	block := []byte{0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8}
	if err := q.WriteTexture(&ImageCopyTexture{Texture: tex, MipLevel: 2}, block,
		&ImageDataLayout{BytesPerRow: blockBytes}, &Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture(mip 2): %v", err)
	}
	if h.backend == gputypes.BackendGL {
		sampleCompressedCopies(t, device, tex, layers, block)
		assertNoValidation(t, h)
		return
	}

	const pitch = 256
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "bc1-read", Size: 5 * pitch, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyTextureToBuffer(tex, buf, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex},
		BufferLayout: ImageDataLayout{BytesPerRow: pitch, RowsPerImage: 2},
		Size:         Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 2},
	}, {
		TextureBase:  ImageCopyTexture{Texture: tex, MipLevel: 2},
		BufferLayout: ImageDataLayout{Offset: 4 * pitch, BytesPerRow: pitch},
		Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, MapModeRead, 0, 5*pitch); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, 5*pitch)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	got := rng.Bytes()
	for row := range 4 { // two block rows of each layer
		want := layers[row*2*blockBytes : (row+1)*2*blockBytes]
		if have := got[row*pitch : row*pitch+2*blockBytes]; string(have) != string(want) {
			t.Errorf("layer %d block row %d = % x, want % x", row/2, row%2, have, want)
		}
	}
	if have := got[4*pitch : 4*pitch+blockBytes]; string(have) != string(block) {
		t.Errorf("mip 2 = % x, want % x", have, block)
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	assertNoValidation(t, h)
}

// sampleCompressedCopies checks tex, which TestCompressedCopies filled, by
// the texels its blocks decode to: mip 0 has to sample like the same blocks
// written one layer at a time, mip 2 like the top left of its block written
// into a texture of one block. A texture filled from a buffer the same way,
// rowsPerImage 2 rows of blocks and the block copied into mip 2, has to
// sample the same.
func sampleCompressedCopies(t *testing.T, device *Device, tex *Texture, layers, block []byte) {
	t.Helper()
	bc1 := func(label string, side, mips uint32) *Texture {
		t.Helper()
		bt, err := device.CreateTexture(&TextureDescriptor{
			Label: label, Size: Extent3D{Width: side, Height: side, DepthOrArrayLayers: 2}, MipLevelCount: mips, SampleCount: 1,
			Dimension: TextureDimension2D, Format: gputypes.TextureFormatBC1RGBAUnorm, Usage: TextureUsageCopyDst | TextureUsageTextureBinding,
		})
		if err != nil {
			t.Fatalf("CreateTexture(%s): %v", label, err)
		}
		t.Cleanup(bt.Release)
		return bt
	}
	q := device.Queue()
	write := func(dst *Texture, layer uint32, data []byte, side uint32) {
		t.Helper()
		if err := q.WriteTexture(&ImageCopyTexture{Texture: dst, Origin: Origin3D{Z: layer}}, data,
			&ImageDataLayout{BytesPerRow: side / 4 * 8}, &Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}); err != nil {
			t.Fatalf("WriteTexture: %v", err)
		}
	}
	const layerBytes = 2 * 2 * 8
	byLayer := bc1("bc1-by-layer", 8, 1)
	write(byLayer, 0, layers[:layerBytes], 8)
	write(byLayer, 1, layers[layerBytes:], 8)
	oneBlock := bc1("bc1-one-block", 4, 1)
	write(oneBlock, 0, block, 4)

	const pitch = 256
	staged := make([]byte, 5*pitch)
	for row := range 4 { // two block rows of each layer
		copy(staged[row*pitch:], layers[row*16:(row+1)*16])
	}
	copy(staged[4*pitch:], block)
	src, err := device.CreateBuffer(&BufferDescriptor{Label: "bc1-src", Size: uint64(len(staged)), Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer src.Release()
	if err := q.WriteBuffer(src, 0, staged); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	copied := bc1("bc1-copied", 8, 4)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToTexture(src, copied, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: copied},
		BufferLayout: ImageDataLayout{BytesPerRow: pitch, RowsPerImage: 2},
		Size:         Extent3D{Width: 8, Height: 8, DepthOrArrayLayers: 2},
	}, {
		TextureBase:  ImageCopyTexture{Texture: copied, MipLevel: 2},
		BufferLayout: ImageDataLayout{Offset: 4 * pitch, BytesPerRow: pitch},
		Size:         Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)

	wantMip0 := sampleTexels(t, device, byLayer)
	blockTexels := sampleTexels(t, device, oneBlock)
	wantMip2 := slices.Concat(blockTexels[0:8], blockTexels[16:24]) // texels 0, 1, 4 and 5 of the block
	const mip2 = (2*64 + 2*16) * 4                                  // the layers of mip 0 and 1 before it
	for name, got := range map[string][]byte{"written": sampleTexels(t, device, tex), "copied": sampleTexels(t, device, copied)} {
		if !bytes.Equal(got[:len(wantMip0)], wantMip0) {
			t.Errorf("%s: mip 0 samples % x, want % x", name, got[:len(wantMip0)], wantMip0)
		}
		if have := got[mip2 : mip2+len(wantMip2)]; !bytes.Equal(have, wantMip2) {
			t.Errorf("%s: layer 0 of mip 2 samples % x, want % x", name, have, wantMip2)
		}
	}
}

// texelsShader stores every texel of every mip level and layer of t, packed
// to RGBA8, into out: the levels one after the other, each layer by layer.
const texelsShader = `
@group(0) @binding(0) var t: texture_2d_array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
@compute @workgroup_size(1) fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
	let layers = textureNumLayers(t);
	let mip = id.z / layers;
	let layer = id.z % layers;
	let size = textureDimensions(t, mip);
	if (id.x >= size.x || id.y >= size.y) {
		return;
	}
	var at = 0u;
	for (var m = 0u; m < mip; m++) {
		let s = textureDimensions(t, m);
		at += s.x * s.y * layers;
	}
	out[at + (layer * size.y + id.y) * size.x + id.x] = pack4x8unorm(textureLoad(t, vec2<i32>(id.xy), i32(layer), i32(mip)));
}
`

// sampleTexels returns what texelsShader stores for tex, a 2D texture of
// more than one layer.
func sampleTexels(t *testing.T, device *Device, tex *Texture) []byte {
	t.Helper()
	ct := tex.coreTexture
	side, layers, mips := ct.Size().Width, ct.Size().DepthOrArrayLayers, ct.MipLevelCount()
	var texels uint64
	for mip := range mips {
		texels += uint64(max(side>>mip, 1)) * uint64(max(side>>mip, 1)) * uint64(layers)
	}
	size := texels * 4
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer view.Release()
	layout, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Texture: &gputypes.TextureBindingLayout{SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2DArray}},
		{Binding: 1, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
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
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "texels", WGSL: texelsShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Label: "texels", Layout: pl, Module: module, EntryPoint: "cs_main"})
	if err != nil {
		skipWithoutComputeShaders(t, err)
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipe.Release()
	out := newStorageBuffer(t, device, size)
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, TextureView: view}, {Binding: 1, Buffer: out, Size: size}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()
	read, err := device.CreateBuffer(&BufferDescriptor{Label: "texels-read", Size: size, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer read.Release()
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
	cp.Dispatch(side, side, mips*layers)
	if err := cp.End(); err != nil {
		t.Fatalf("ComputePass.End: %v", err)
	}
	enc.CopyBufferToBuffer(out, 0, read, 0, size)
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := read.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := read.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	texelBytes := bytes.Clone(rng.Bytes())
	if err := read.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return texelBytes
}
