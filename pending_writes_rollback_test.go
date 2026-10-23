//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/noop"
)

// batchingQueue is a noop queue whose writes are recorded into command
// buffers, as the GPU backends' are.
type batchingQueue struct{ noop.Queue }

func (*batchingQueue) SupportsCommandBufferCopies() bool { return true }

// A batch of queue writes that is dropped, because a surface texture it
// writes was discarded before the Submit, puts back what it changed outside
// itself: the texture it zeroed is zeroed by the next batch, and the device
// tracker holds the states from before it.
func TestDroppedWriteBatchRollsBack(t *testing.T) {
	device, err := NewDeviceFromHAL(&noop.Device{}, &batchingQueue{}, 0, gputypes.DefaultLimits(), "dropped-batch")
	if err != nil {
		t.Fatalf("NewDeviceFromHAL: %v", err)
	}
	defer device.Release()
	surface := &Surface{
		core:           core.NewSurface(&noop.Surface{}, "dropped-batch"),
		surfaceCreated: true,
		currentBackend: gputypes.BackendEmpty,
	}
	defer surface.Release()
	if err := surface.Configure(device, &SurfaceConfiguration{
		Width:       4,
		Height:      4,
		Format:      gputypes.TextureFormatRGBA8Unorm,
		Usage:       gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopyDst,
		PresentMode: gputypes.PresentModeFifo,
		AlphaMode:   gputypes.CompositeAlphaModeAuto,
	}); err != nil {
		t.Fatalf("Configure: %v", err)
	}
	st, _, err := surface.GetCurrentTexture()
	if err != nil {
		t.Fatalf("GetCurrentTexture: %v", err)
	}

	tex, err := device.CreateTexture(&TextureDescriptor{
		Size:          Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageTextureBinding | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	tracker, idx := device.core.Tracker(), tex.coreTexture.TrackingData().Index()
	usesBefore, trackedBefore := tracker.TextureState(idx)

	data := make([]byte, 4*4*4)
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: st.AsTexture()}, data,
		&ImageDataLayout{BytesPerRow: 16, RowsPerImage: 4}, &Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	surface.DiscardTexture()
	if _, err := device.Queue().Submit(); !errors.Is(err, ErrReleased) {
		t.Fatalf("Submit after the surface texture retired = %v, want ErrReleased", err)
	}

	if _, queued := device.queue.pending.zero[tex]; !queued {
		t.Error("the texture the dropped batch zeroed is not on the list to zero")
	}
	if uses, tracked := tracker.TextureState(idx); uses != usesBefore || tracked != trackedBefore {
		t.Errorf("tracker state after the dropped batch = %v (tracked %v), want %v (tracked %v)", uses, tracked, usesBefore, trackedBefore)
	}
	if _, err := device.Queue().Submit(); err != nil {
		t.Fatalf("Submit after the dropped batch: %v", err)
	}
	if _, queued := device.queue.pending.zero[tex]; queued {
		t.Error("the next batch did not zero the texture")
	}
}

// failingQueue is a batchingQueue whose backend refuses submits.
type failingQueue struct{ batchingQueue }

func (*failingQueue) Submit([]hal.CommandBuffer) (uint64, error) {
	return 0, errors.New("submit refused")
}

// A batch the backend refuses to submit puts back what it changed outside
// itself, as a dropped one does.
func TestRefusedWriteBatchRollsBack(t *testing.T) {
	device, err := NewDeviceFromHAL(&noop.Device{}, &failingQueue{}, 0, gputypes.DefaultLimits(), "refused-batch")
	if err != nil {
		t.Fatalf("NewDeviceFromHAL: %v", err)
	}
	defer device.Release()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size:          Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
		MipLevelCount: 1,
		SampleCount:   1,
		Dimension:     TextureDimension2D,
		Format:        TextureFormatRGBA8Unorm,
		Usage:         TextureUsageTextureBinding | TextureUsageCopyDst,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	tracker, idx := device.core.Tracker(), tex.coreTexture.TrackingData().Index()
	usesBefore, trackedBefore := tracker.TextureState(idx)

	if _, err := device.Queue().Submit(); err == nil {
		t.Fatal("Submit succeeded on a queue that refuses submits")
	}
	if _, queued := device.queue.pending.zero[tex]; !queued {
		t.Error("the texture the refused batch zeroed is not on the list to zero")
	}
	if uses, tracked := tracker.TextureState(idx); uses != usesBefore || tracked != trackedBefore {
		t.Errorf("tracker state after the refused batch = %v (tracked %v), want %v (tracked %v)", uses, tracked, usesBefore, trackedBefore)
	}
}

// refusingQueue is a batchingQueue whose backend refuses submits while
// refuse is set.
type refusingQueue struct {
	batchingQueue
	refuse bool
}

func (q *refusingQueue) Submit(cbs []hal.CommandBuffer) (uint64, error) {
	if q.refuse {
		return 0, errors.New("submit refused")
	}
	return q.batchingQueue.Submit(cbs)
}

// A Submit the backend refuses leaves the list of textures to zero as it
// was: a texture its command buffer clears first stays on it, and one it
// discards is not put on it.
func TestRefusedSubmitKeepsTheZeroList(t *testing.T) {
	q := &refusingQueue{}
	device, err := NewDeviceFromHAL(&noop.Device{}, q, 0, gputypes.DefaultLimits(), "refused-submit")
	if err != nil {
		t.Fatalf("NewDeviceFromHAL: %v", err)
	}
	defer device.Release()
	target := func() (*Texture, *TextureView) {
		tex, err := device.CreateTexture(&TextureDescriptor{
			Size:          Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1},
			MipLevelCount: 1,
			SampleCount:   1,
			Dimension:     TextureDimension2D,
			Format:        TextureFormatRGBA8Unorm,
			Usage:         TextureUsageRenderAttachment,
		})
		if err != nil {
			t.Fatalf("CreateTexture: %v", err)
		}
		view, err := device.CreateTextureView(tex, nil)
		if err != nil {
			t.Fatalf("CreateTextureView: %v", err)
		}
		t.Cleanup(func() { releaseTextureView(tex, view) })
		return tex, view
	}
	pass := func(view *TextureView, store gputypes.StoreOp) *CommandBuffer {
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		p, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{
			{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: store},
		}})
		if err != nil {
			t.Fatalf("BeginRenderPass: %v", err)
		}
		if err := p.End(); err != nil {
			t.Fatalf("End: %v", err)
		}
		cb, err := enc.Finish()
		if err != nil {
			t.Fatalf("Finish: %v", err)
		}
		return cb
	}
	zeroing := func(tex *Texture) bool {
		_, ok := device.queue.pending.zero[tex]
		return ok
	}

	discarded, discardedView := target()
	if _, err := device.Queue().Submit(); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	cleared, clearedView := target()
	if !zeroing(cleared) || zeroing(discarded) {
		t.Fatal("the textures are not where the test needs them: a new one to zero and a zeroed one")
	}

	q.refuse = true
	if _, err := device.Queue().Submit(pass(clearedView, gputypes.StoreOpStore), pass(discardedView, gputypes.StoreOpDiscard)); err == nil {
		t.Fatal("Submit succeeded on a queue that refuses submits")
	}
	if !zeroing(cleared) {
		t.Error("the texture the refused submit would have cleared first is not on the list to zero")
	}
	if zeroing(discarded) {
		t.Error("the texture the refused submit would have discarded is on the list to zero")
	}
}

// stalledQueue is a refusingQueue whose submissions never complete.
type stalledQueue struct{ refusingQueue }

func (*stalledQueue) PollCompleted() uint64 { return 0 }

// A Submit the backend refuses loses the device, as in wgpu-core: the
// device tracker holds the states the refused command buffers would have
// left, so the queue takes no more work, and a pending map fails.
func TestRefusedSubmitLosesTheDevice(t *testing.T) {
	q := &stalledQueue{}
	device, err := NewDeviceFromHAL(&noop.Device{}, q, 0, gputypes.DefaultLimits(), "refused-submit-lost")
	if err != nil {
		t.Fatalf("NewDeviceFromHAL: %v", err)
	}
	defer device.Release()
	buffer := func() *Buffer {
		buf, err := device.CreateBuffer(&BufferDescriptor{Size: 16, Usage: BufferUsageMapRead | BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		t.Cleanup(buf.Release)
		return buf
	}
	mapped, written := buffer(), buffer()
	if err := device.Queue().WriteBuffer(mapped, 0, make([]byte, 16)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	pending, err := mapped.MapAsync(MapModeRead, 0, 16)
	if err != nil {
		t.Fatalf("MapAsync: %v", err)
	}
	defer pending.Release()
	if ready, _ := pending.Status(); ready {
		t.Fatal("the map resolved before the write it waits for completed")
	}

	q.refuse = true
	if err := device.Queue().WriteBuffer(written, 0, make([]byte, 16)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	if _, err := device.Queue().Submit(); !errors.Is(err, ErrDeviceLost) {
		t.Fatalf("refused Submit = %v, want ErrDeviceLost", err)
	}
	if ready, err := pending.Status(); !ready || !errors.Is(err, ErrMapDeviceLost) {
		t.Errorf("pending map after the device was lost = %v, %v, want ErrMapDeviceLost", ready, err)
	}

	q.refuse = false
	if _, err := device.Queue().Submit(); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("Submit on the lost device = %v, want ErrDeviceLost", err)
	}
	if err := device.Queue().WriteBuffer(written, 0, make([]byte, 16)); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("WriteBuffer on the lost device = %v, want ErrDeviceLost", err)
	}
}
