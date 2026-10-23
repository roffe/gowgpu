//go:build !(js && wasm)

package wgpu

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// The rest of TestNewTextureIsZeroed: the textures a render pass that clears
// them zeroes (depth, multisampled, and render targets nothing samples) and
// compressed ones, which the copies zero a row of blocks at a time. Each
// test leaves textures holding something else behind first, for the new one
// to land on, and fails on any validation message.

// zeroRegion is the room readSubresources gives each subresource in its
// buffer.
const zeroRegion = 16 << 10

// subresource is a mip level and layer to read back: width and height in
// texels, and rows of rowBytes bytes, rows of blocks for a compressed format.
type subresource struct {
	mip, layer, width, height, rowBytes, rows uint32
}

// readSubresources copies subs of tex out and returns the bytes of each, the
// rows packed.
func readSubresources(t *testing.T, device *Device, tex *Texture, aspect gputypes.TextureAspect, subs []subresource) [][]byte {
	t.Helper()
	size := uint64(len(subs)) * zeroRegion
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "zero-readback", Size: size, Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	for i, s := range subs {
		enc.CopyTextureToBuffer(tex, buf, []BufferTextureCopy{{
			TextureBase:  ImageCopyTexture{Texture: tex, MipLevel: s.mip, Origin: Origin3D{Z: s.layer}, Aspect: aspect},
			BufferLayout: ImageDataLayout{Offset: uint64(i) * zeroRegion, BytesPerRow: alignUp(s.rowBytes, copyBytesPerRowAlignment)},
			Size:         Extent3D{Width: s.width, Height: s.height, DepthOrArrayLayers: 1},
		}})
	}
	finishAndSubmit(t, device, enc)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	data := rng.Bytes()
	out := make([][]byte, len(subs))
	for i, s := range subs {
		pitch := alignUp(s.rowBytes, copyBytesPerRowAlignment)
		for r := range s.rows {
			at := uint32(i)*zeroRegion + r*pitch
			out[i] = append(out[i], data[at:at+s.rowBytes]...)
		}
	}
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

// expectBytes fails on the first byte of got that isn't want.
func expectBytes(t *testing.T, what string, got []byte, want byte) {
	t.Helper()
	for i, b := range got {
		if b != want {
			t.Errorf("%s: byte %d is %#x, want %#x", what, i, b, want)
			return
		}
	}
}

// leaveBehind makes textures from desc, has fill put something in each,
// waits for the GPU and releases them, for the next texture made from desc
// to land on their memory.
func leaveBehind(t *testing.T, device *Device, desc *TextureDescriptor, fill func(old *Texture)) {
	t.Helper()
	for range 4 {
		old, err := device.CreateTexture(desc)
		if err != nil {
			t.Fatalf("CreateTexture: %v", err)
		}
		fill(old)
		if _, err := device.Queue().Submit(); err != nil {
			t.Fatalf("Submit: %v", err)
		}
		if err := device.WaitIdle(); err != nil {
			t.Fatalf("WaitIdle: %v", err)
		}
		old.Release()
	}
}

// passOn records a render pass on enc with the given attachments and nothing
// in it.
func passOn(t *testing.T, enc *CommandEncoder, desc *RenderPassDescriptor) {
	t.Helper()
	pass, err := enc.BeginRenderPass(desc)
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("RenderPass.End: %v", err)
	}
}

// TestNewDepthTextureIsZeroed: a new depth texture reads 0 at every mip
// level and layer, which the zeroing clears with a pass each. The textures
// left behind are cleared to 0.75.
func TestNewDepthTextureIsZeroed(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL {
		t.Skip("GL doesn't zero depth textures (README known issues), and reads no depth back")
	}
	const side, mips, layers = 64, 3, 2
	desc := &TextureDescriptor{
		Size:          Extent3D{Width: side, Height: side, DepthOrArrayLayers: layers},
		MipLevelCount: mips,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatDepth32Float,
		Usage:         TextureUsageRenderAttachment | TextureUsageCopySrc,
	}
	leaveBehind(t, device, desc, func(old *Texture) {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		views := make([]*TextureView, 0, mips*layers)
		for mip := range uint32(mips) {
			for layer := range uint32(layers) {
				view, err := device.CreateTextureView(old, &TextureViewDescriptor{
					Dimension: gputypes.TextureViewDimension2DArray, BaseMipLevel: mip, MipLevelCount: 1, BaseArrayLayer: layer, ArrayLayerCount: 1,
				})
				if err != nil {
					t.Fatalf("CreateTextureView: %v", err)
				}
				views = append(views, view)
				passOn(t, enc, &RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{
					View: view, DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 0.75,
				}})
			}
		}
		finishAndSubmit(t, device, enc)
		for _, v := range views {
			v.Release()
		}
	})

	tex, err := device.CreateTexture(desc)
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	subs := make([]subresource, 0, mips*layers)
	for mip := range uint32(mips) {
		w := uint32(side) >> mip
		for layer := range uint32(layers) {
			subs = append(subs, subresource{mip: mip, layer: layer, width: w, height: w, rowBytes: w * 4, rows: w})
		}
	}
	for i, got := range readSubresources(t, device, tex, gputypes.TextureAspectDepthOnly, subs) {
		expectBytes(t, fmt.Sprintf("mip %d layer %d", subs[i].mip, subs[i].layer), got, 0)
	}
	assertNoValidation(t, h)
}

// TestNewRenderTargetIsZeroed: a render target nothing samples reads back
// as zeros where no pass cleared it, read straight from a color target and
// through a resolve from a multisampled one, which a pass that loads it
// resolves into a texture of its own. The targets left behind are cleared
// to 0xab.
func TestNewRenderTargetIsZeroed(t *testing.T) {
	const side = 64
	ab := gputypes.Color{R: 0xab / 255.0, G: 0xab / 255.0, B: 0xab / 255.0, A: 0xab / 255.0}
	for _, samples := range []uint32{1, 4} {
		t.Run(fmt.Sprintf("samples=%d", samples), func(t *testing.T) {
			device, h := newValidatedDevice(t)
			if samples > 1 && h.backend == gputypes.BackendGL {
				t.Skip("GL doesn't zero multisampled textures (README known issues)")
			}
			desc := &TextureDescriptor{
				Size:          Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
				MipLevelCount: 1,
				SampleCount:   samples,
				Dimension:     TextureDimension2D,
				Format:        TextureFormatRGBA8Unorm,
				Usage:         TextureUsageRenderAttachment | TextureUsageCopySrc,
			}
			if samples > 1 {
				desc.Usage = TextureUsageRenderAttachment
			}
			leaveBehind(t, device, desc, func(old *Texture) {
				view, err := device.CreateTextureView(old, nil)
				if err != nil {
					t.Fatalf("CreateTextureView: %v", err)
				}
				defer view.Release()
				enc, err := device.CreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("CreateCommandEncoder: %v", err)
				}
				clearPass(t, enc, view, gputypes.LoadOpClear, ab, nil)
				finishAndSubmit(t, device, enc)
			})

			tex, err := device.CreateTexture(desc)
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			defer tex.Release()
			read := tex
			if samples > 1 {
				read = resolveLoaded(t, device, tex)
			}
			got := readSubresources(t, device, read, gputypes.TextureAspectAll, []subresource{{width: side, height: side, rowBytes: side * 4, rows: side}})
			expectBytes(t, "pixels", got[0], 0)
			assertNoValidation(t, h)
		})
	}
}

// resolveLoaded resolves multisampled tex, as a pass that loads it leaves it,
// into a new texture it returns, released at the end of the test.
func resolveLoaded(t *testing.T, device *Device, tex *Texture) *Texture {
	t.Helper()
	size := Extent3D{Width: tex.coreTexture.Size().Width, Height: tex.coreTexture.Size().Height, DepthOrArrayLayers: 1}
	resolved, err := device.CreateTexture(&TextureDescriptor{
		Size: size, MipLevelCount: 1, SampleCount: 1, Dimension: TextureDimension2D, Format: tex.Format(),
		Usage: TextureUsageRenderAttachment | TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture(resolve): %v", err)
	}
	t.Cleanup(resolved.Release)
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer view.Release()
	resolve, err := device.CreateTextureView(resolved, nil)
	if err != nil {
		t.Fatalf("CreateTextureView(resolve): %v", err)
	}
	defer resolve.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	passOn(t, enc, &RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
		View: view, ResolveTarget: resolve, LoadOp: gputypes.LoadOpLoad, StoreOp: gputypes.StoreOpStore,
	}}})
	finishAndSubmit(t, device, enc)
	return resolved
}

// TestNewCompressedTextureIsZeroed: a new BC texture reads back as zeros
// where nothing wrote it, at mip levels down to 1x1, which take a block that
// sticks out over the edge. One row of blocks is written before the zeroing
// and has to survive it. The textures left behind have 0xab in every block
// of the levels of a block or more.
func TestNewCompressedTextureIsZeroed(t *testing.T) {
	for _, format := range []gputypes.TextureFormat{gputypes.TextureFormatBC1RGBAUnorm, gputypes.TextureFormatBC7RGBAUnorm} {
		t.Run(format.String(), func(t *testing.T) {
			device, h := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureTextureCompressionBC))
			if h.backend == gputypes.BackendGL {
				t.Skip("GL doesn't zero compressed textures (README known issues)")
			}
			const side, mips = 64, 7 // down to 1x1
			blockSize := format.BlockCopySize()
			desc := &TextureDescriptor{
				Size:          Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
				MipLevelCount: mips,
				SampleCount:   1,
				Dimension:     TextureDimension2D,
				Format:        format,
				Usage:         TextureUsageTextureBinding | TextureUsageCopySrc | TextureUsageCopyDst,
			}
			q := device.Queue()
			// write fills blockRows rows of blocks of a mip level w texels wide
			// from block row y with b.
			write := func(tex *Texture, mip, w, y, blockRows uint32, b byte) {
				t.Helper()
				rowBytes := (w + 3) / 4 * blockSize
				data := make([]byte, rowBytes*blockRows)
				for i := range data {
					data[i] = b
				}
				err := q.WriteTexture(&ImageCopyTexture{Texture: tex, MipLevel: mip, Origin: Origin3D{Y: y * 4}}, data,
					&ImageDataLayout{BytesPerRow: rowBytes},
					&Extent3D{Width: w, Height: blockRows * 4, DepthOrArrayLayers: 1})
				if err != nil {
					t.Fatalf("WriteTexture: %v", err)
				}
			}
			leaveBehind(t, device, desc, func(old *Texture) {
				for mip := range uint32(5) { // the levels of 4x4 texels or more
					w := uint32(side) >> mip
					write(old, mip, w, 0, w/4, 0xab)
				}
			})

			tex, err := device.CreateTexture(desc)
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			defer tex.Release()
			const row = 2 // the row of blocks written
			write(tex, 0, side, row, 1, 0xff)

			subs := make([]subresource, 0, mips)
			for mip := range uint32(mips) {
				w := max(uint32(side)>>mip, 1)
				blocks := (w + 3) / 4 // a copy takes whole blocks, the edge of one included
				subs = append(subs, subresource{mip: mip, width: blocks * 4, height: blocks * 4, rowBytes: blocks * blockSize, rows: blocks})
			}
			for i, got := range readSubresources(t, device, tex, gputypes.TextureAspectAll, subs) {
				s := subs[i]
				if s.mip == 0 {
					at, n := row*s.rowBytes, s.rowBytes
					expectBytes(t, "mip 0 above the written row", got[:at], 0)
					expectBytes(t, "mip 0 written row", got[at:at+n], 0xff)
					expectBytes(t, "mip 0 below the written row", got[at+n:], 0)
					continue
				}
				expectBytes(t, fmt.Sprintf("mip %d", s.mip), got, 0)
			}
			assertNoValidation(t, h)
		})
	}
}
