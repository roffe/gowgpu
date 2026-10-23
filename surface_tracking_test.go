//go:build !(js && wasm)

package wgpu

import (
	"slices"
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/noop"
)

// orderEncoder keeps the texture barriers of each command buffer it records
// in the command buffer.
type orderEncoder struct {
	noop.CommandEncoder
	barriers []hal.TextureBarrier
}

func (e *orderEncoder) BeginEncoding(string) error {
	e.barriers = nil
	return nil
}

func (e *orderEncoder) TransitionTextures(barriers []hal.TextureBarrier) {
	e.barriers = append(e.barriers, barriers...)
}

func (e *orderEncoder) EndEncoding() (hal.CommandBuffer, error) {
	cb := &orderCommandBuffer{barriers: e.barriers}
	e.barriers = nil
	return cb, nil
}

type orderCommandBuffer struct {
	noop.Resource
	barriers []hal.TextureBarrier
}

type orderDevice struct{ noop.Device }

func (*orderDevice) CreateCommandEncoder(*hal.CommandEncoderDescriptor) (hal.CommandEncoder, error) {
	return &orderEncoder{}, nil
}

// orderQueue is a batchingQueue that keeps the texture barriers of the
// command buffers it submits, in the order they run.
type orderQueue struct {
	batchingQueue
	run []hal.TextureBarrier
}

func (q *orderQueue) Submit(cbs []hal.CommandBuffer) (uint64, error) {
	for _, cb := range cbs {
		if o, ok := cb.(*orderCommandBuffer); ok {
			q.run = append(q.run, o.barriers...)
		}
	}
	return q.batchingQueue.Submit(cbs)
}

// A surface texture is tracked in the order its work runs in, like any
// texture: a queue write into it recorded after a pass on it runs first, so
// the barrier into the pass goes from the write's state, and the submit
// leaves the texture in its present state for the present.
func TestSurfaceTextureTrackedInExecutionOrder(t *testing.T) {
	q := &orderQueue{}
	device, err := NewDeviceFromHAL(&orderDevice{}, q, 0, gputypes.DefaultLimits(), "surface-order")
	if err != nil {
		t.Fatalf("NewDeviceFromHAL: %v", err)
	}
	defer device.Release()
	surface := &Surface{
		core:           core.NewSurface(&noop.Surface{}, "surface-order"),
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
	view, err := st.CreateView(nil)
	if err != nil {
		t.Fatalf("CreateView: %v", err)
	}
	defer view.Release()

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, gputypes.LoadOpClear, green, nil)
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: st.AsTexture()}, make([]byte, 4*4*4),
		&ImageDataLayout{BytesPerRow: 16, RowsPerImage: 4}, &Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}); err != nil {
		t.Fatalf("WriteTexture: %v", err)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	var got []hal.TextureUsageTransition
	for _, b := range q.run {
		if b.Texture == st.hal {
			got = append(got, hal.TextureUsageTransition{Old: b.Usage.Old, New: b.Usage.New})
		}
	}
	want := []hal.TextureUsageTransition{
		{Old: hal.TextureUsesUninitialized, New: hal.TextureUsesCopyDst},
		{Old: hal.TextureUsesCopyDst, New: hal.TextureUsesColorTarget},
		{Old: hal.TextureUsesColorTarget, New: hal.TextureUsesPresent},
	}
	if !slices.Equal(got, want) {
		t.Errorf("surface texture barriers in submit order = %+v, want %+v", got, want)
	}
	if err := surface.Present(st); err != nil {
		t.Fatalf("Present: %v", err)
	}
}
