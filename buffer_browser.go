//go:build js && wasm

package wgpu

import (
	"context"
	"fmt"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/internal/browser"
)

// Buffer represents a GPU buffer.
type Buffer struct {
	browser  *browser.Buffer
	device   *Device
	size     uint64
	usage    gputypes.BufferUsage
	released bool

	// mapState tracks the current mapping state. Updated by MapAsync/Unmap.
	mapState MapState

	// mapSerial numbers the MapAsync calls, so that a map settling after
	// Unmap or Release canceled it can tell it is not the current one.
	mapSerial uint64

	// ranges are the MappedRange handles out for the current mapping.
	ranges []*MappedRange
}

// Size returns the buffer size in bytes.
func (b *Buffer) Size() uint64 {
	return b.size
}

// Usage returns the buffer's usage flags.
func (b *Buffer) Usage() gputypes.BufferUsage {
	return b.usage
}

// Label returns the buffer's debug label.
// Browser WebGPU does not expose label on the object; returns empty string.
func (b *Buffer) Label() string {
	return ""
}

// Release destroys the buffer.
func (b *Buffer) Release() {
	if b.released {
		return
	}
	b.released = true
	b.mapState = MapStateDestroyed
	b.detachRanges()
	if b.browser != nil {
		b.browser.Destroy()
	}
}

// MapState returns the current mapping state of the buffer.
func (b *Buffer) MapState() MapState {
	if b == nil || b.released {
		return MapStateUnmapped
	}
	return b.mapState
}

// mapModeToJS converts a MapMode to the WebGPU GPUMapMode flags.
// Our MapMode values already match WebGPU spec: MapModeRead=1, MapModeWrite=2.
// This function validates and converts explicitly for clarity.
func mapModeToJS(mode MapMode) uint32 {
	var jsMode uint32
	if mode&MapModeRead != 0 {
		jsMode |= 1 // GPUMapMode.READ
	}
	if mode&MapModeWrite != 0 {
		jsMode |= 2 // GPUMapMode.WRITE
	}
	return jsMode
}

// Map blocks until a CPU-visible mapping is established for the given
// byte range, or until ctx is canceled.
//
// The buffer must have been created with BufferUsageMapRead or
// BufferUsageMapWrite matching mode. offset must be a multiple of 8 and
// size must be a multiple of 4 (WebGPU MAP_ALIGNMENT).
//
// After Map succeeds, call MappedRange to obtain a byte view and Unmap
// when finished:
//
//	if err := buf.Map(ctx, wgpu.MapModeRead, 0, size); err != nil {
//	    return err
//	}
//	defer buf.Unmap()
//	rng, _ := buf.MappedRange(0, size)
//	data := rng.Bytes()
//
// On the browser, Map waits for the GPUBuffer.mapAsync promise. The ctx
// parameter allows cancellation.
func (b *Buffer) Map(ctx context.Context, mode MapMode, offset, size uint64) error {
	if b == nil || b.browser == nil {
		return ErrReleased
	}
	if b.released {
		return ErrBufferDestroyed
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := validateMapArgs(b, mode, offset, size); err != nil {
		return err
	}

	// MapAsync sets the state to pending, then blocks on the JS Promise.
	// We use a channel + goroutine to support context cancellation.
	pending, err := b.MapAsync(mode, offset, size)
	if err != nil {
		return err
	}
	return pending.Wait(ctx)
}

// MapAsync initiates a buffer map without blocking the caller.
//
// Returns a *MapPending handle that resolves once the browser's
// GPUBuffer.mapAsync Promise settles. The caller can poll with
// Status() or block with Wait(ctx).
//
// Validation errors (alignment, usage mismatch, range overflow,
// already-mapped state) surface synchronously.
func (b *Buffer) MapAsync(mode MapMode, offset, size uint64) (*MapPending, error) {
	if b == nil || b.browser == nil {
		return nil, ErrReleased
	}
	if b.released {
		return nil, ErrBufferDestroyed
	}
	if err := validateMapArgs(b, mode, offset, size); err != nil {
		return nil, err
	}
	if b.device.lostErr() != nil {
		return nil, ErrMapDeviceLost
	}

	b.mapState = MapStatePending
	b.mapSerial++
	serial := b.mapSerial
	done := make(chan error, 1)
	b.browser.MapAsync(mapModeToJS(mode), offset, size, func(err error) {
		done <- b.mapSettled(serial, err)
	})
	return &MapPending{buf: b, done: done}, nil
}

// mapSettled moves the buffer to the state the map numbered serial ended
// in and returns what its waiter gets. A map that Unmap or Release
// canceled, and that a later MapAsync may have replaced, leaves the state
// alone and ends canceled.
func (b *Buffer) mapSettled(serial uint64, err error) error {
	if serial != b.mapSerial || b.mapState != MapStatePending {
		return ErrMapCanceled
	}
	if err == nil {
		b.mapState = MapStateMapped
		return nil
	}
	b.mapState = MapStateUnmapped
	if b.device.lostErr() != nil {
		return ErrMapDeviceLost
	}
	return fmt.Errorf("mapAsync: %w", err)
}

// MappedRange returns a safe view over the mapped region [offset, offset+size).
//
// The buffer must be in the Mapped state (Map or MapAsync resolved, or the
// buffer was created with MappedAtCreation: true). The returned MappedRange
// is invalidated by Unmap.
func (b *Buffer) MappedRange(offset, size uint64) (*MappedRange, error) {
	if b == nil || b.browser == nil {
		return nil, ErrReleased
	}
	if b.released {
		return nil, ErrBufferDestroyed
	}
	if b.mapState != MapStateMapped {
		return nil, ErrMapNotMapped
	}
	if offset%8 != 0 || size%4 != 0 {
		return nil, ErrMapAlignment
	}
	if offset+size > b.size {
		return nil, ErrMapRangeOverflow
	}

	r := &MappedRange{
		buf:    b,
		offset: offset,
		size:   size,
		valid:  true,
	}
	b.ranges = append(b.ranges, r)
	return r, nil
}

// Unmap flushes the ranges written through BytesMut, releases the current
// mapping and invalidates all outstanding MappedRange handles. The buffer
// is unmapped even when a flush fails, and the first such error is
// returned. Safe to call multiple times; a second call returns
// ErrMapNotMapped.
func (b *Buffer) Unmap() error {
	if b == nil || b.browser == nil {
		return ErrReleased
	}
	if b.released {
		return ErrBufferDestroyed
	}
	if b.mapState != MapStateMapped && b.mapState != MapStatePending {
		return ErrMapNotMapped
	}
	var err error
	for _, r := range b.ranges {
		if e := r.Flush(); e != nil && err == nil {
			err = e
		}
	}
	b.browser.Unmap()
	b.mapState = MapStateUnmapped
	b.detachRanges()
	return err
}

func (b *Buffer) detachRanges() {
	for _, r := range b.ranges {
		r.detach()
	}
	b.ranges = nil
}

// validateMapArgs performs synchronous validation of mapping parameters.
func validateMapArgs(b *Buffer, mode MapMode, offset, size uint64) error {
	if b.mapState == MapStatePending {
		return ErrMapAlreadyPending
	}
	if b.mapState == MapStateMapped {
		return ErrMapAlreadyMapped
	}
	if b.mapState == MapStateDestroyed {
		return ErrBufferDestroyed
	}

	// Check usage flags.
	if mode&MapModeRead != 0 && b.usage&BufferUsageMapRead == 0 {
		return ErrMapInvalidMode
	}
	if mode&MapModeWrite != 0 && b.usage&BufferUsageMapWrite == 0 {
		return ErrMapInvalidMode
	}

	// Alignment: offset must be multiple of 8, size must be multiple of 4.
	if offset%8 != 0 || size%4 != 0 {
		return ErrMapAlignment
	}

	// Range overflow.
	if offset+size > b.size {
		return ErrMapRangeOverflow
	}

	return nil
}
