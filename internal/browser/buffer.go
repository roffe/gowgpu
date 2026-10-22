//go:build js && wasm

package browser

import (
	"fmt"
	"syscall/js"
)

// Buffer wraps a browser GPUBuffer with mapping state tracking.
//
// Holds a reference to the JavaScript GPUBuffer object and caches the size
// to avoid repeated JS property lookups. Tracks the mapped ArrayBuffer
// so that multiple Go-side GetMappedRange calls reuse a single JS mapping,
// matching the Rust wgpu WebBuffer/WebBufferMapState pattern -- the WebGPU
// spec forbids calling GPUBuffer.getMappedRange more than once for the same
// mapped region.
type Buffer struct {
	// ref_ is the GPUBuffer JavaScript object.
	ref_ js.Value

	// size is the buffer's byte size, cached at creation time.
	size uint64

	// usage is the buffer's usage flags, cached at creation time.
	usage uint32

	// mappedBuffer is the cached JS ArrayBuffer from getMappedRange.
	// Nil (zero Value) when the buffer is not mapped. The WebGPU spec
	// forbids calling getMappedRange on a GPUBuffer more than once, so
	// we cache the result and create sub-range Uint8Array views into it.
	// Matches Rust wgpu WebBufferMapState.mapped_buffer.
	mappedBuffer js.Value

	// mappedOffset is the start of the overall mapped range.
	mappedOffset uint64

	// mappedSize is the length of the overall mapped range.
	mappedSize uint64
}

// NewBuffer constructs a Buffer from a GPUBuffer js.Value.
func NewBuffer(ref js.Value) *Buffer {
	return &Buffer{
		ref_:  ref,
		size:  uint64(ref.Get("size").Float()), //nolint:gosec // JS API returns safe integers
		usage: uint32(ref.Get("usage").Int()),  //nolint:gosec // JS API returns safe integers
	}
}

// Ref returns the underlying GPUBuffer js.Value.
func (b *Buffer) Ref() js.Value { return b.ref_ }

// Size returns the buffer size in bytes.
func (b *Buffer) Size() uint64 { return b.size }

// Usage returns the buffer usage flags.
func (b *Buffer) Usage() uint32 { return b.usage }

// Destroy calls GPUBuffer.destroy() to release GPU memory.
func (b *Buffer) Destroy() {
	b.mappedBuffer = js.Value{}
	destroy := b.ref_.Get("destroy")
	if !destroy.IsUndefined() && !destroy.IsNull() {
		b.ref_.Call("destroy")
	}
}

// MapAsync maps the buffer for CPU access by calling GPUBuffer.mapAsync(mode, offset, size).
// Blocks the calling goroutine until the JS Promise resolves or rejects.
//
// mode uses WebGPU GPUMapMode flags: 1 = MAP_READ, 2 = MAP_WRITE.
// The caller MUST be on a goroutine (not the main goroutine) or the program
// will deadlock, because AwaitPromise yields via a channel.
//
// On success the mapped range is recorded so that subsequent GetMappedRangeBytes
// calls can cache the JS ArrayBuffer (matching Rust wgpu WebBuffer.set_mapped_range).
func (b *Buffer) MapAsync(mode uint32, offset, size uint64) error {
	promise := b.ref_.Call("mapAsync", mode, float64(offset), float64(size))
	_, err := AwaitPromise(promise)
	if err != nil {
		return fmt.Errorf("GPUBuffer.mapAsync: %w", err)
	}
	// Record the mapped range so GetMappedRangeBytes can lazily fetch
	// the ArrayBuffer on the first call (Rust wgpu set_mapped_range).
	b.mappedOffset = offset
	b.mappedSize = size
	return nil
}

// ensureMappedBuffer calls GPUBuffer.getMappedRange once for the entire
// mapped region and caches the ArrayBuffer, which the spec lets a mapping
// hand out only once. It returns the exception getMappedRange throws, if
// any, as an error.
func (b *Buffer) ensureMappedBuffer() (js.Value, error) {
	if b.mappedBuffer.IsUndefined() || b.mappedBuffer.IsNull() || b.mappedBuffer.Equal(js.Value{}) {
		err := catchJS(func() {
			b.mappedBuffer = b.ref_.Call("getMappedRange", float64(b.mappedOffset), float64(b.mappedSize))
		})
		if err != nil {
			return js.Value{}, fmt.Errorf("GPUBuffer.getMappedRange: %w", err)
		}
	}
	return b.mappedBuffer, nil
}

// mappedView returns a Uint8Array view of size bytes of the mapped region
// from offset on. It returns the exception the browser throws when the
// buffer is not mapped or the view falls outside the mapping, if any, as
// an error.
func (b *Buffer) mappedView(offset uint64, size int) (js.Value, error) {
	ab, err := b.ensureMappedBuffer()
	if err != nil {
		return js.Value{}, err
	}
	var view js.Value
	err = catchJS(func() {
		view = js.Global().Get("Uint8Array").New(ab, int(offset-b.mappedOffset), size) //nolint:gosec // validated by caller
	})
	return view, err
}

// GetMappedRangeBytes copies the mapped region [offset, offset+size) into a
// new Go byte slice. The buffer must be mapped, by MapAsync or at creation.
func (b *Buffer) GetMappedRangeBytes(offset, size uint64) ([]byte, error) {
	view, err := b.mappedView(offset, int(size)) //nolint:gosec // validated by caller
	if err != nil {
		return nil, err
	}
	data := make([]byte, size)
	js.CopyBytesToGo(data, view)
	return data, nil
}

// WriteMappedRange writes data into the mapped region at [offset, offset+len(data)).
// The buffer must be mapped for writing, by MapAsync or at creation; the
// browser flushes the data to the GPU on Unmap.
func (b *Buffer) WriteMappedRange(offset uint64, data []byte) error {
	view, err := b.mappedView(offset, len(data))
	if err != nil {
		return err
	}
	js.CopyBytesToJS(view, data)
	return nil
}

// SetMappedAtCreation records the mapped range for a buffer created with
// mappedAtCreation: true. The entire buffer [0, size) is mapped. This must
// be called immediately after NewBuffer for mappedAtCreation buffers so that
// GetMappedRangeBytes / WriteMappedRange work correctly.
func (b *Buffer) SetMappedAtCreation() {
	b.mappedOffset = 0
	b.mappedSize = b.size
}

// Unmap unmaps the buffer, making its mapped ranges invalid, and clears the
// cached ArrayBuffer. Matches Rust wgpu WebBuffer.unmap which calls
// inner.unmap() and sets mapped_buffer = None.
func (b *Buffer) Unmap() {
	b.ref_.Call("unmap")
	b.mappedBuffer = js.Value{}
	b.mappedOffset = 0
	b.mappedSize = 0
}
