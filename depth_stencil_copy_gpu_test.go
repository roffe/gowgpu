//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"math"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// readTextureAspect copies one aspect of a pixSide square texture into a
// buffer, rows pixRow bytes apart, and returns the buffer's bytes.
func readTextureAspect(t *testing.T, device *Device, tex *Texture, aspect gputypes.TextureAspect) []byte {
	t.Helper()
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "aspect-readback", Size: pixStaging, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyTextureToBuffer(tex, buf, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: tex, Aspect: aspect},
		BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
		Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, MapModeRead, 0, pixStaging); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, pixStaging)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	out := append([]byte(nil), rng.Bytes()...)
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

func depthStencilTexture(t *testing.T, device *Device, format gputypes.TextureFormat, usage gputypes.TextureUsage) (*Texture, *TextureView) {
	t.Helper()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Label:         format.String(),
		Size:          Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        format,
		Usage:         usage,
	})
	if err != nil {
		t.Fatalf("CreateTexture(%s): %v", format, err)
	}
	t.Cleanup(tex.Release)
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	t.Cleanup(view.Release)
	return tex, view
}

// TestDepthCopyWithoutAspect clears a Depth32Float texture to 0.5 and reads
// it back with the copy's aspect left unset, which names the only aspect the
// format has. Vulkan copied the colour aspect, and NVIDIA read zeros.
func TestDepthCopyWithoutAspect(t *testing.T) {
	device, h := newValidatedDevice(t)
	if !device.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsDepthTextureAndBufferCopies) {
		t.Skip("no depth texture copies on this adapter")
	}
	tex, view := depthStencilTexture(t, device, gputypes.TextureFormatDepth32Float, TextureUsageRenderAttachment|TextureUsageCopySrc)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearDepth(t, enc, view, 0.5)
	finishAndSubmit(t, device, enc)

	got := readTextureAspect(t, device, tex, gputypes.TextureAspectUndefined)
	for y := range pixSide {
		for x := range pixSide {
			at := y*pixRow + x*4
			if d := math.Float32frombits(binary.LittleEndian.Uint32(got[at:])); d != 0.5 {
				t.Fatalf("depth at %d,%d = %v, want 0.5", x, y, d)
			}
		}
	}
	assertNoValidation(t, h)
}

// TestStencilAspectUploads writes the stencil aspect of a
// Depth24PlusStencil8 texture, a different value in each row, with
// Queue.WriteTexture and with CopyBufferToTexture, and reads it back. A
// stencil texel is one byte: Vulkan took four for the row length, so only
// the first row of an upload arrived, and the readback's rows landed a
// quarter of bytesPerRow apart. CopyBufferToTexture also dropped the aspect.
// GL refused the stencil of a combined format, which it writes only with
// the depth beside it.
func TestStencilAspectUploads(t *testing.T) {
	device, h := newValidatedDevice(t)
	skipStencilWritesOnGLES(t, device, h)
	skipStencilCopiesOnVkd3d(t, h)
	rows := make([]byte, pixSide*pixSide)
	for y := range pixSide {
		for x := range pixSide {
			rows[y*pixSide+x] = byte(0x10*(y+1) + x)
		}
	}
	check := func(what string, tex *Texture) {
		t.Helper()
		got := readTextureAspect(t, device, tex, gputypes.TextureAspectStencilOnly)
		for y := range pixSide {
			if row, want := got[y*pixRow:y*pixRow+pixSide], rows[y*pixSide:(y+1)*pixSide]; string(row) != string(want) {
				t.Errorf("%s: stencil row %d = % x, want % x", what, y, row, want)
			}
		}
	}
	usage := TextureUsageRenderAttachment | TextureUsageCopySrc | TextureUsageCopyDst

	written, _ := depthStencilTexture(t, device, gputypes.TextureFormatDepth24PlusStencil8, usage)
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: written, Aspect: gputypes.TextureAspectStencilOnly}, rows,
		&ImageDataLayout{BytesPerRow: pixSide, RowsPerImage: pixSide}, &Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	check("Queue.WriteTexture", written)

	copied, _ := depthStencilTexture(t, device, gputypes.TextureFormatDepth24PlusStencil8, usage)
	src, err := device.CreateBuffer(&BufferDescriptor{Label: "stencil-src", Size: pixStaging, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer src.Release()
	padded := make([]byte, pixStaging)
	for y := range pixSide {
		copy(padded[y*pixRow:], rows[y*pixSide:(y+1)*pixSide])
	}
	if err := device.Queue().WriteBuffer(src, 0, padded); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyBufferToTexture(src, copied, []BufferTextureCopy{{
		TextureBase:  ImageCopyTexture{Texture: copied, Aspect: gputypes.TextureAspectStencilOnly},
		BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
		Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
	}})
	finishAndSubmit(t, device, enc)
	check("CopyBufferToTexture", copied)
	assertNoValidation(t, h)
}

// skipStencilCopiesOnVkd3d skips a buffer copy of a stencil aspect on DX12
// under Wine's own vkd3d. D3D12 copies the stencil plane as plane 1 through an
// R8_UINT footprint, and vkd3d takes the aspect of a copy from the
// footprint's format and the subresource index as a mip level and an array
// layer: it copies the colour of array layer 1, which the texture doesn't
// have, so nothing reaches the stencil and it reads back as zeros.
// vkd3d-proton copies the plane.
func skipStencilCopiesOnVkd3d(t *testing.T, h *captureHandler) {
	t.Helper()
	if h.backend == gputypes.BackendDX12 && upstreamVkd3d() {
		t.Skip("vkd3d copies plane 1 through an R8_UINT footprint as the colour of array layer 1, not the stencil")
	}
}

// skipStencilWritesOnGLES skips a test that writes the stencil of a
// combined depth/stencil texture on GL without depth and stencil copies:
// ES reads no depth or stencil back, which GL needs to write the stencil.
func skipStencilWritesOnGLES(t *testing.T, device *Device, h *captureHandler) {
	t.Helper()
	if h.backend == gputypes.BackendGL && !device.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsDepthTextureAndBufferCopies) {
		t.Skip("GL ES writes no stencil of a combined depth/stencil texture")
	}
}

// TestDepth16Upload writes a Depth16Unorm texture with Queue.WriteTexture
// and reads it back. Vulkan copied it as the colour aspect.
func TestDepth16Upload(t *testing.T) {
	device, h := newValidatedDevice(t)
	if !device.core.DownlevelCapabilitiesRef().Flags.Contains(gputypes.DownlevelFlagsDepthTextureAndBufferCopies) {
		t.Skip("no depth texture copies on this adapter")
	}
	tex, _ := depthStencilTexture(t, device, gputypes.TextureFormatDepth16Unorm, TextureUsageRenderAttachment|TextureUsageCopySrc|TextureUsageCopyDst)
	data := make([]byte, pixSide*pixSide*2)
	for i := range pixSide * pixSide {
		binary.LittleEndian.PutUint16(data[i*2:], uint16(1000*i+7))
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, data,
		&ImageDataLayout{BytesPerRow: pixSide * 2, RowsPerImage: pixSide}, &Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	got := readTextureAspect(t, device, tex, gputypes.TextureAspectAll)
	for y := range pixSide {
		if row, want := got[y*pixRow:y*pixRow+pixSide*2], data[y*pixSide*2:(y+1)*pixSide*2]; string(row) != string(want) {
			t.Errorf("depth row %d = % x, want % x", y, row, want)
		}
	}
	assertNoValidation(t, h)
}
