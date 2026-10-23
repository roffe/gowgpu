//go:build windows && !(js && wasm)

package dx12

import (
	"bytes"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestSurfaceTextureCopiesReachTheBackBuffer copies into and out of a
// surface texture, then has the queue bring its back buffer back to PRESENT
// for the present, and copies again from there.
func TestSurfaceTextureCopiesReachTheBackBuffer(t *testing.T) {
	dev, queue := openTestDevice(t)
	const side, pitch = 4, 256
	tex := newTestTexture(t, dev, hal.TextureDescriptor{
		Size:      hal.Extent3D{Width: side, Height: side},
		Dimension: gputypes.TextureDimension2D,
		Format:    gputypes.TextureFormatRGBA8Unorm,
		Usage:     gputypes.TextureUsageCopyDst | gputypes.TextureUsageCopySrc | gputypes.TextureUsageRenderAttachment,
	})
	backBuffer := &Texture{
		raw:          tex.raw,
		format:       tex.format,
		dimension:    gputypes.TextureDimension2D,
		size:         hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
		mipLevels:    1,
		samples:      1,
		usage:        gputypes.TextureUsageRenderAttachment,
		device:       dev,
		isExternal:   true,
		currentState: d3d12.D3D12_RESOURCE_STATE_PRESENT,
	}
	backBuffer.stateOwner.setTextureStates([]d3d12.D3D12_RESOURCE_STATES{d3d12.D3D12_RESOURCE_STATE_PRESENT})
	st := &SurfaceTexture{resource: tex.raw, stateOwner: backBuffer, format: tex.format, width: side, height: side}

	region := hal.BufferTextureCopy{
		BufferLayout: hal.ImageDataLayout{BytesPerRow: pitch, RowsPerImage: side},
		TextureBase:  hal.ImageCopyTexture{Texture: st},
		Size:         hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
	}
	src := newTestBuffer(t, dev, pitch*side, gputypes.BufferUsageCopyDst|gputypes.BufferUsageCopySrc)
	if err := queue.WriteBuffer(src, 0, bytes.Repeat([]byte{0xAB}, pitch*side)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	dst := newTestBuffer(t, dev, pitch*side, gputypes.BufferUsageCopyDst|gputypes.BufferUsageMapRead)
	record(t, dev, queue, func(enc *CommandEncoder) {
		enc.CopyBufferToTexture(src, st, []hal.BufferTextureCopy{region})
		enc.TransitionTextures([]hal.TextureBarrier{{Texture: st, Usage: hal.TextureUsageTransition{NewUsage: gputypes.TextureUsageCopySrc}}})
		enc.CopyTextureToBuffer(st, dst, []hal.BufferTextureCopy{region})
	})
	if got := readBuffer(t, dev, dst, side*4); !bytes.Equal(got, bytes.Repeat([]byte{0xAB}, side*4)) {
		t.Fatalf("copied out of the surface texture: % x, want ab", got)
	}
	if state := backBuffer.scheduledStateSnapshot(0); state != d3d12.D3D12_RESOURCE_STATE_COPY_SOURCE {
		t.Fatalf("back buffer state after the copies = %#x, want COPY_SOURCE", state)
	}

	queue.state.submitMu.Lock()
	err := queue.ensurePresentState(backBuffer)
	queue.state.submitMu.Unlock()
	if err != nil {
		t.Fatalf("ensurePresentState: %v", err)
	}
	if state := backBuffer.scheduledStateSnapshot(0); state != d3d12.D3D12_RESOURCE_STATE_PRESENT {
		t.Fatalf("back buffer state for the present = %#x, want PRESENT", state)
	}

	data := bytes.Repeat([]byte{0xCD}, side*side*4)
	if err := queue.WriteTexture(&hal.ImageCopyTexture{Texture: st}, data, &hal.ImageDataLayout{BytesPerRow: side * 4, RowsPerImage: side}, &region.Size); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	record(t, dev, queue, func(enc *CommandEncoder) {
		enc.CopyTextureToBuffer(st, dst, []hal.BufferTextureCopy{region})
	})
	if got := readBuffer(t, dev, dst, side*4); !bytes.Equal(got, data[:side*4]) {
		t.Fatalf("written to the surface texture: % x, want cd", got)
	}
}
