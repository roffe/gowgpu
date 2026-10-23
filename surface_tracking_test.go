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

// orderEncoder keeps what each command buffer it records does to textures:
// its barriers, and its render passes and copies into them.
type orderEncoder struct {
	noop.CommandEncoder
	barriers []hal.TextureBarrier
	ops      []textureOp
}

// textureOp is a render pass on tex, with the load op of its color
// attachment, or a copy into tex.
type textureOp struct {
	tex  hal.Texture
	pass bool
	load gputypes.LoadOp
}

func (e *orderEncoder) BeginEncoding(string) error {
	e.barriers, e.ops = nil, nil
	return nil
}

func (e *orderEncoder) TransitionTextures(barriers []hal.TextureBarrier) {
	e.barriers = append(e.barriers, barriers...)
}

func (e *orderEncoder) BeginRenderPass(desc *hal.RenderPassDescriptor) hal.RenderPassEncoder {
	for _, ca := range desc.ColorAttachments {
		if v, ok := ca.View.(*orderView); ok {
			e.ops = append(e.ops, textureOp{tex: v.tex, pass: true, load: ca.LoadOp})
		}
	}
	return e.CommandEncoder.BeginRenderPass(desc)
}

func (e *orderEncoder) CopyBufferToTexture(_ hal.Buffer, dst hal.Texture, _ []hal.BufferTextureCopy) {
	e.ops = append(e.ops, textureOp{tex: dst})
}

func (e *orderEncoder) EndEncoding() (hal.CommandBuffer, error) {
	cb := &orderCommandBuffer{barriers: e.barriers, ops: e.ops}
	e.barriers, e.ops = nil, nil
	return cb, nil
}

type orderCommandBuffer struct {
	noop.Resource
	barriers []hal.TextureBarrier
	ops      []textureOp
}

// orderView is a view that knows its texture.
type orderView struct {
	noop.Resource
	tex hal.Texture
}

type orderDevice struct{ noop.Device }

func (*orderDevice) CreateCommandEncoder(*hal.CommandEncoderDescriptor) (hal.CommandEncoder, error) {
	return &orderEncoder{}, nil
}

func (*orderDevice) CreateTextureView(tex hal.Texture, _ *hal.TextureViewDescriptor) (hal.TextureView, error) {
	return &orderView{tex: tex}, nil
}

// orderQueue is a batchingQueue that keeps what the command buffers it
// submits do to textures, in the order they run. A direct one writes
// straight away instead of recording the writes, as GL does.
type orderQueue struct {
	batchingQueue
	direct bool
	run    []hal.TextureBarrier
	ops    []textureOp
}

func (q *orderQueue) SupportsCommandBufferCopies() bool { return !q.direct }

func (q *orderQueue) Submit(cbs []hal.CommandBuffer) (uint64, error) {
	for _, cb := range cbs {
		if o, ok := cb.(*orderCommandBuffer); ok {
			q.run = append(q.run, o.barriers...)
			q.ops = append(q.ops, o.ops...)
		}
	}
	return q.batchingQueue.Submit(cbs)
}

// newOrderSurface is a device on an orderQueue, direct or not, and a
// surface configured on it.
func newOrderSurface(t *testing.T, direct bool) (*Device, *orderQueue, *Surface) {
	t.Helper()
	q := &orderQueue{direct: direct}
	device, err := NewDeviceFromHAL(&orderDevice{}, q, 0, gputypes.DefaultLimits(), "surface-order")
	if err != nil {
		t.Fatalf("NewDeviceFromHAL: %v", err)
	}
	t.Cleanup(device.Release)
	surface := &Surface{
		core:           core.NewSurface(&noop.Surface{}, "surface-order"),
		surfaceCreated: true,
		currentBackend: gputypes.BackendEmpty,
	}
	t.Cleanup(surface.Release)
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
	return device, q, surface
}

// A surface texture is tracked in the order its work runs in, like any
// texture: a queue write into it recorded after a pass on it runs first, so
// the barrier into the pass goes from the write's state, and the submit
// leaves the texture in its present state for the present.
func TestSurfaceTextureTrackedInExecutionOrder(t *testing.T) {
	device, q, surface := newOrderSurface(t, false)
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

// Whether a surface texture is zeroed ahead of a pass that loads it is
// decided at the submit, in the order the work runs in: a queue write
// recorded after the pass runs ahead of it and sets the texture, as does a
// pass that clears it in a command buffer submitted first. A surface
// texture nothing set is zeroed at the present. The same goes for a queue
// that writes straight away, as GL does, but for the queue write, which
// runs there and then.
func TestSurfaceTextureZeroedInExecutionOrder(t *testing.T) {
	load, clr := gputypes.LoadOpLoad, gputypes.LoadOpClear
	tests := []struct {
		name    string
		batched bool // only on a queue that records its writes
		run     func(t *testing.T, device *Device, st *SurfaceTexture, view *TextureView)
		want    func(raw hal.Texture) []textureOp
	}{
		{
			name: "a pass loads it first",
			run: func(t *testing.T, device *Device, _ *SurfaceTexture, view *TextureView) {
				submitPass(t, device, view, load)
			},
			want: func(raw hal.Texture) []textureOp {
				return []textureOp{{tex: raw, pass: true, load: clr}, {tex: raw, pass: true, load: load}}
			},
		},
		{
			name:    "a write recorded after the pass runs first",
			batched: true,
			run: func(t *testing.T, device *Device, st *SurfaceTexture, view *TextureView) {
				cb := recordPass(t, device, view, load)
				if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: st.AsTexture()}, make([]byte, 4*4*4),
					&ImageDataLayout{BytesPerRow: 16, RowsPerImage: 4}, &Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}); err != nil {
					t.Fatalf("WriteTexture: %v", err)
				}
				if _, err := device.Queue().Submit(cb); err != nil {
					t.Fatalf("Submit: %v", err)
				}
			},
			want: func(raw hal.Texture) []textureOp {
				return []textureOp{{tex: raw}, {tex: raw, pass: true, load: load}}
			},
		},
		{
			name: "a pass clears it first",
			run: func(t *testing.T, device *Device, _ *SurfaceTexture, view *TextureView) {
				submitPass(t, device, view, clr)
			},
			want: func(raw hal.Texture) []textureOp {
				return []textureOp{{tex: raw, pass: true, load: clr}}
			},
		},
		{
			name: "the clear is recorded last and submitted first",
			run: func(t *testing.T, device *Device, _ *SurfaceTexture, view *TextureView) {
				loads := recordPass(t, device, view, load)
				clears := recordPass(t, device, view, clr)
				if _, err := device.Queue().Submit(clears, loads); err != nil {
					t.Fatalf("Submit: %v", err)
				}
			},
			want: func(raw hal.Texture) []textureOp {
				return []textureOp{{tex: raw, pass: true, load: clr}, {tex: raw, pass: true, load: load}}
			},
		},
		{
			name: "nothing uses it",
			run:  func(*testing.T, *Device, *SurfaceTexture, *TextureView) {},
			want: func(raw hal.Texture) []textureOp {
				return []textureOp{{tex: raw, pass: true, load: clr}}
			},
		},
	}
	for _, direct := range []bool{false, true} {
		for _, tt := range tests {
			if direct && tt.batched {
				continue
			}
			name := tt.name
			if direct {
				name = "direct/" + name
			}
			t.Run(name, func(t *testing.T) {
				testSurfaceZeroOrder(t, direct, tt.run, tt.want)
			})
		}
	}
}

// testSurfaceZeroOrder runs one case of TestSurfaceTextureZeroedInExecutionOrder.
func testSurfaceZeroOrder(t *testing.T, direct bool, run func(*testing.T, *Device, *SurfaceTexture, *TextureView), expected func(hal.Texture) []textureOp) {
	device, q, surface := newOrderSurface(t, direct)
	st, _, err := surface.GetCurrentTexture()
	if err != nil {
		t.Fatalf("GetCurrentTexture: %v", err)
	}
	view, err := st.CreateView(nil)
	if err != nil {
		t.Fatalf("CreateView: %v", err)
	}
	defer view.Release()
	run(t, device, st, view)
	if err := surface.Present(st); err != nil {
		t.Fatalf("Present: %v", err)
	}
	var got []textureOp
	for _, op := range q.ops {
		if op.tex == st.hal {
			got = append(got, op)
		}
	}
	if want := expected(st.hal); !slices.Equal(got, want) {
		t.Errorf("passes and copies on the surface texture = %+v, want %+v", got, want)
	}
}

// recordPass records a command buffer with a pass on view that loads or
// clears it.
func recordPass(t *testing.T, device *Device, view *TextureView, load gputypes.LoadOp) *CommandBuffer {
	t.Helper()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	clearPass(t, enc, view, load, gputypes.Color{G: 1, A: 1}, nil)
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	return cb
}

// submitPass submits a pass on view that loads or clears it.
func submitPass(t *testing.T, device *Device, view *TextureView, load gputypes.LoadOp) {
	t.Helper()
	if _, err := device.Queue().Submit(recordPass(t, device, view, load)); err != nil {
		t.Fatalf("Submit: %v", err)
	}
}
