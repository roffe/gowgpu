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
