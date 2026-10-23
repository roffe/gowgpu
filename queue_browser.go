//go:build js && wasm

package wgpu

import (
	"fmt"
	"syscall/js"

	"github.com/roffe/gowgpu/internal/browser"
)

// Queue handles command submission and data transfers.
// On browser, this wraps a GPUQueue via internal/browser.Queue.
type Queue struct {
	browser  *browser.Queue
	device   *Device
	released bool
}

// Submit submits command buffers for execution.
// Returns 0 for the submission index (browser does not track indices).
// Matches Rust wgpu WebQueue::submit which collects into js_sys::Array.
func (q *Queue) Submit(commandBuffers ...*CommandBuffer) (uint64, error) {
	if q.released {
		return 0, ErrReleased
	}
	if err := q.device.lostErr(); err != nil {
		return 0, err
	}
	jsBuffers := make([]js.Value, 0, len(commandBuffers))
	for _, cb := range commandBuffers {
		if cb != nil && cb.browser != nil && !cb.released {
			jsBuffers = append(jsBuffers, cb.browser.Ref())
			cb.released = true
		}
	}
	q.browser.Submit(jsBuffers)
	return 0, nil
}

// Poll returns the last completed submission index.
// On browser, the GPU is polled automatically. Returns 0.
func (q *Queue) Poll() uint64 {
	return 0
}

// WriteBuffer writes data to a buffer. The buffer must be unmapped and made
// with CopyDst, offset and len(data) must be multiples of 4, and the write
// must stay within the buffer.
func (q *Queue) WriteBuffer(buffer *Buffer, offset uint64, data []byte) error {
	if q.released {
		return ErrReleased
	}
	if err := q.device.lostErr(); err != nil {
		return err
	}
	if buffer == nil || buffer.browser == nil {
		return ErrReleased
	}
	if buffer.MapState() != MapStateUnmapped {
		return fmt.Errorf("wgpu: WriteBuffer: buffer is currently mapped")
	}
	if buffer.Usage()&BufferUsageCopyDst == 0 {
		return fmt.Errorf("wgpu: WriteBuffer: buffer missing CopyDst usage")
	}
	if offset%4 != 0 {
		return fmt.Errorf("wgpu: WriteBuffer: offset %d not 4-byte aligned", offset)
	}
	dataSize := uint64(len(data))
	if dataSize%4 != 0 {
		return fmt.Errorf("wgpu: WriteBuffer: data size %d not 4-byte aligned", dataSize)
	}
	if offset+dataSize > buffer.Size() {
		return fmt.Errorf("wgpu: WriteBuffer: offset %d + size %d exceeds buffer size %d", offset, dataSize, buffer.Size())
	}
	return q.browser.WriteBuffer(buffer.browser.Ref(), offset, data)
}

// WriteTexture writes data to a texture.
// Matches Rust wgpu WebQueue::write_texture layout: offset/bytesPerRow/rowsPerImage
// are set on a GPUTexelCopyBufferLayout JS object.
func (q *Queue) WriteTexture(dst *ImageCopyTexture, data []byte, layout *ImageDataLayout, size *Extent3D) error {
	if q.released {
		return ErrReleased
	}
	if err := q.device.lostErr(); err != nil {
		return err
	}
	if dst == nil || dst.Texture == nil || dst.Texture.browser == nil {
		return ErrReleased
	}
	if layout == nil {
		return fmt.Errorf("wgpu: WriteTexture: layout is nil")
	}
	if size == nil {
		return fmt.Errorf("wgpu: WriteTexture: size is nil")
	}

	jsDst := browser.BuildImageCopyTexture(
		dst.Texture.browser.Ref(),
		dst.MipLevel,
		dst.Origin.X, dst.Origin.Y, dst.Origin.Z,
		dst.Aspect,
	)

	jsLayout := browser.BuildTexelCopyBufferLayout(
		layout.Offset,
		layout.BytesPerRow,
		layout.RowsPerImage,
	)
	jsSize := browser.BuildExtent3D(size.Width, size.Height, size.DepthOrArrayLayers)

	return q.browser.WriteTexture(jsDst, data, jsLayout, jsSize)
}

// SetSwapchainSuppressed is a no-op on the browser backend.
// WebGPU browser API handles swapchain synchronization internally.
func (q *Queue) SetSwapchainSuppressed(_ bool) {}

// LastSubmissionIndex returns the most recent submission index.
// On browser, submission indices are not tracked. Returns 0.
func (q *Queue) LastSubmissionIndex() uint64 {
	return 0
}
