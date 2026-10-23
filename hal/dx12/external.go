//go:build windows && !(js && wasm)

package dx12

import (
	"fmt"
	"unsafe"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// D3D12Device returns the ID3D12Device, for code sharing the device with
// gowgpu: a video decoder, or another API filling resources it reads.
func (d *Device) D3D12Device() uintptr {
	return uintptr(unsafe.Pointer(d.raw))
}

// externalPoint is a value of a fence of code sharing the device.
type externalPoint struct {
	fence *d3d12.ID3D12Fence
	value uint64
}

// ExternalSyncFull reports false: a submit takes any number of points of
// AddExternalSync.
func (q *Queue) ExternalSyncFull() bool {
	return false
}

// AddExternalSync has the next Submit wait for fence, an ID3D12Fence of code
// sharing the device such as a video decoder's, to reach wait before its
// command lists run, and signal signal once they complete.
func (q *Queue) AddExternalSync(fence, wait, signal uint64) error {
	if err := q.lockOpen(); err != nil {
		return err
	}
	defer q.state.submitMu.Unlock()
	f := fromHandle[d3d12.ID3D12Fence](fence)
	q.state.externalWaits = append(q.state.externalWaits, externalPoint{f, wait})
	q.state.externalSignals = append(q.state.externalSignals, externalPoint{f, signal})
	return nil
}

// waitExternal has the queue wait for the fences AddExternalSync was given.
func (q *Queue) waitExternal() error {
	for _, p := range q.state.externalWaits {
		if err := q.raw.Wait(p.fence, p.value); err != nil {
			return fmt.Errorf("dx12: waiting for an external fence: %w", err)
		}
	}
	q.state.externalWaits = q.state.externalWaits[:0]
	return nil
}

// signalExternal has the queue signal the fences AddExternalSync was given.
func (q *Queue) signalExternal() error {
	for _, p := range q.state.externalSignals {
		if err := q.raw.Signal(p.fence, p.value); err != nil {
			return fmt.Errorf("dx12: signalling an external fence: %w", err)
		}
	}
	q.state.externalSignals = q.state.externalSignals[:0]
	return nil
}

// CopyExternalBuffer copies src, an ID3D12Resource buffer made outside
// gowgpu that another API fills, such as OpenGL through a shared handle, into
// dst whole, its rows bytesPerRow apart, a multiple of 256. The buffer is left
// in the COMMON state, which the copy promotes it from and it decays back to.
// A fence the submit waits for orders the copy after the filling.
func (e *CommandEncoder) CopyExternalBuffer(src uint64, bytesPerRow uint32, dst hal.Texture) {
	t, ok := textureOf(dst)
	if !e.isRecording || src == 0 || !ok {
		return
	}
	if before, barrier := e.stateTracker.transitionTexture(t, 0, d3d12.D3D12_RESOURCE_STATE_COPY_DEST); barrier {
		e.emitStateBarrierPlans([]stateBarrierPlan{{resource: t, subresource: 0, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_DEST}})
	}
	srcLoc := d3d12.D3D12_TEXTURE_COPY_LOCATION{Resource: fromHandle[d3d12.ID3D12Resource](src), Type: d3d12.D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}
	srcLoc.SetPlacedFootprint(d3d12.D3D12_PLACED_SUBRESOURCE_FOOTPRINT{
		Footprint: d3d12.D3D12_SUBRESOURCE_FOOTPRINT{
			Format:   textureFormatToD3D12(t.format),
			Width:    t.size.Width,
			Height:   t.size.Height,
			Depth:    1,
			RowPitch: bytesPerRow,
		},
	})
	dstLoc := subresourceLocation(t, 0)
	e.cmdList.CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nil)
}

// CopyExternalPlanes copies the planes of src, a planar ID3D12Resource such
// as a video decoder's NV12 texture, layer slice of it when it is an array,
// into dsts, plane i into dsts[i], as much of it as they hold. The texture is
// left in the state it is in, layout, which is COMMON for the copy to promote
// it from and for it to decay back to. A fence the submit waits for orders
// the copy after the work that made src.
func (e *CommandEncoder) CopyExternalPlanes(src uint64, layout, slice uint32, dsts []hal.Texture) uint32 {
	if !e.isRecording || src == 0 || len(dsts) > 3 {
		return layout
	}
	res := fromHandle[d3d12.ID3D12Resource](src)
	layers := max(uint32(res.GetDesc().DepthOrArraySize), 1)
	for i, dst := range dsts {
		t, ok := textureOf(dst)
		if !ok {
			continue
		}
		if before, barrier := e.stateTracker.transitionTexture(t, 0, d3d12.D3D12_RESOURCE_STATE_COPY_DEST); barrier {
			e.emitStateBarrierPlans([]stateBarrierPlan{{resource: t, subresource: 0, before: before, after: d3d12.D3D12_RESOURCE_STATE_COPY_DEST}})
		}
		srcLoc := d3d12.D3D12_TEXTURE_COPY_LOCATION{Resource: res, Type: d3d12.D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}
		srcLoc.SetSubresourceIndex(slice + uint32(i)*layers) // one mip level
		dstLoc := subresourceLocation(t, 0)
		box := d3d12.D3D12_BOX{Right: t.size.Width, Bottom: t.size.Height, Back: 1}
		e.cmdList.CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, &box)
	}
	return layout
}
