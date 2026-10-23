//go:build !(js && wasm)

package wgpu

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/noop"
)

// releaseCountingDevice is a noop device that counts the textures and query
// sets it destroys.
type releaseCountingDevice struct {
	noop.Device
	textures, querySets atomic.Int32
}

func (d *releaseCountingDevice) DestroyTexture(texture hal.Texture) {
	d.textures.Add(1)
	d.Device.DestroyTexture(texture)
}

func (d *releaseCountingDevice) CreateQuerySet(*hal.QuerySetDescriptor) (hal.QuerySet, error) {
	return &noop.Resource{}, nil
}

func (d *releaseCountingDevice) DestroyQuerySet(querySet hal.QuerySet) {
	d.querySets.Add(1)
	d.Device.DestroyQuerySet(querySet)
}

// gatedQueue is a noop queue whose GPU has finished submission completed, and
// whose Submit, once entered is set, says so on it and waits in the backend
// for proceed.
type gatedQueue struct {
	noop.Queue
	entered, proceed chan struct{}
	completed        atomic.Uint64
}

func (q *gatedQueue) Submit(commandBuffers []hal.CommandBuffer) (uint64, error) {
	if q.entered != nil {
		q.entered <- struct{}{}
		<-q.proceed
	}
	return q.Queue.Submit(commandBuffers)
}

func (q *gatedQueue) PollCompleted() uint64 { return q.completed.Load() }

// A texture or query set released on another goroutine while a Submit that
// uses it is in the backend is destroyed once that submission is done, not
// at the triage of the Submit itself, which knew only the earlier ones.
func TestReleaseDuringSubmitWaitsForTheSubmission(t *testing.T) {
	run := func(t *testing.T, record func(*Device, *CommandEncoder) func(), destroyed func(*releaseCountingDevice) int32) {
		raw, queue := &releaseCountingDevice{}, &gatedQueue{}
		device, err := NewDeviceFromHAL(raw, queue, 0, gputypes.DefaultLimits(), "release-during-submit")
		if err != nil {
			t.Fatalf("NewDeviceFromHAL: %v", err)
		}
		defer device.Release()
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		release := record(device, enc)
		cb, err := enc.Finish()
		if err != nil {
			t.Fatalf("Finish: %v", err)
		}

		queue.entered, queue.proceed = make(chan struct{}), make(chan struct{})
		submitted := make(chan error, 1)
		go func() {
			_, err := device.Queue().Submit(cb)
			submitted <- err
		}()
		<-queue.entered
		released := make(chan struct{})
		go func() {
			release()
			close(released)
		}()
		time.Sleep(20 * time.Millisecond)
		close(queue.proceed)
		if err := <-submitted; err != nil {
			t.Fatalf("Submit: %v", err)
		}
		<-released
		if n := destroyed(raw); n != 0 {
			t.Fatalf("destroyed %d before the submission that uses it was done", n)
		}

		queue.entered = nil
		queue.completed.Store(device.Queue().LastSubmissionIndex())
		if _, err := device.Queue().Submit(); err != nil {
			t.Fatalf("Submit: %v", err)
		}
		if n := destroyed(raw); n != 1 {
			t.Fatalf("destroyed %d once the submission was done, want 1", n)
		}
	}

	t.Run("texture", func(t *testing.T) {
		run(t, func(device *Device, enc *CommandEncoder) func() {
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
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
				View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore,
			}}})
			if err != nil {
				t.Fatalf("BeginRenderPass: %v", err)
			}
			if err := pass.End(); err != nil {
				t.Fatalf("End: %v", err)
			}
			view.Release()
			return tex.Release
		}, func(raw *releaseCountingDevice) int32 { return raw.textures.Load() })
	})

	t.Run("query set", func(t *testing.T) {
		run(t, func(device *Device, enc *CommandEncoder) func() {
			qs, err := device.CreateQuerySet(&QuerySetDescriptor{Type: QueryTypeOcclusion, Count: 1})
			if err != nil {
				t.Fatalf("CreateQuerySet: %v", err)
			}
			dst, err := device.CreateBuffer(&BufferDescriptor{Size: 8, Usage: BufferUsageQueryResolve})
			if err != nil {
				t.Fatalf("CreateBuffer: %v", err)
			}
			t.Cleanup(dst.Release)
			enc.ResolveQuerySet(qs, 0, 1, dst, 0)
			return qs.Release
		}, func(raw *releaseCountingDevice) int32 { return raw.querySets.Load() })
	})
}
