//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/noop"
)

type testHALResource struct{}

func (*testHALResource) Destroy() {}

func (*testHALResource) NativeHandle() uint64 { return 0 }

type captureRenderPass struct {
	*noop.RenderPassEncoder
	vertexBuffers []hal.Buffer
}

func (p *captureRenderPass) SetVertexBuffer(_ uint32, buffer hal.Buffer, _ uint64) {
	p.vertexBuffers = append(p.vertexBuffers, buffer)
}

type captureCommandEncoder struct {
	*noop.CommandEncoder
	resolved       int
	querySet       hal.QuerySet
	buffer         hal.Buffer
	renderPass     *captureRenderPass
	renderPassDesc *hal.RenderPassDescriptor
}

func (e *captureCommandEncoder) ResolveQuerySet(querySet hal.QuerySet, _, _ uint32, buffer hal.Buffer, _ uint64) {
	e.resolved++
	e.querySet = querySet
	e.buffer = buffer
}

func (e *captureCommandEncoder) BeginRenderPass(desc *hal.RenderPassDescriptor) hal.RenderPassEncoder {
	e.renderPassDesc = desc
	if e.renderPass == nil {
		return nil
	}
	return e.renderPass
}

type captureDevice struct {
	hal.Device
	commandEncoder hal.CommandEncoder
}

func (d *captureDevice) CreateCommandEncoder(*hal.CommandEncoderDescriptor) (hal.CommandEncoder, error) {
	return d.commandEncoder, nil
}

func newCaptureDevice(h *captureDevice) *Device {
	limits := DefaultLimits()
	return &Device{core: core.NewDevice(h, nil, 0, limits, "query-render-bundle-test")}
}

// captureAttachments is the color attachment of a pass on a capture device,
// a pass needing one.
func captureAttachments(t *testing.T, device *Device) []RenderPassColorAttachment {
	t.Helper()
	tex, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageRenderAttachment,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	view, err := device.CreateTextureView(tex, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	return []RenderPassColorAttachment{{View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore}}
}

func newTestQuerySet(raw hal.QuerySet, device *Device) *QuerySet {
	return &QuerySet{
		core:   core.NewQuerySet(raw, device.core, hal.QueryTypeTimestamp, 4, "test query set"),
		device: device,
	}
}

func TestBeginRenderPassTimestampWrites(t *testing.T) {
	begin, end, past := uint32(2), uint32(3), uint32(4)
	resource := &testHALResource{}
	h := &captureDevice{Device: &noop.Device{}}
	device := newCaptureDevice(h)
	active := newTestQuerySet(resource, device)
	released := newTestQuerySet(resource, device)
	released.released.Store(true)
	occlusion := &QuerySet{core: core.NewQuerySet(resource, device.core, hal.QueryTypeOcclusion, 4, "occlusion"), device: device}

	tests := []struct {
		name    string
		writes  *RenderPassTimestampWrites
		wantNil bool
		wantErr bool
	}{
		{name: "absent", writes: nil, wantNil: true},
		{name: "nil query set", writes: &RenderPassTimestampWrites{BeginningOfPassWriteIndex: &begin}, wantErr: true},
		{name: "released query set", writes: &RenderPassTimestampWrites{QuerySet: released, BeginningOfPassWriteIndex: &begin}, wantErr: true},
		{name: "occlusion query set", writes: &RenderPassTimestampWrites{QuerySet: occlusion, BeginningOfPassWriteIndex: &begin}, wantErr: true},
		{name: "no index", writes: &RenderPassTimestampWrites{QuerySet: active}, wantErr: true},
		{name: "index past the set", writes: &RenderPassTimestampWrites{QuerySet: active, EndOfPassWriteIndex: &past}, wantErr: true},
		{name: "one index twice", writes: &RenderPassTimestampWrites{QuerySet: active, BeginningOfPassWriteIndex: &begin, EndOfPassWriteIndex: &begin}, wantErr: true},
		{
			name: "active query set",
			writes: &RenderPassTimestampWrites{
				QuerySet:                  active,
				BeginningOfPassWriteIndex: &begin,
				EndOfPassWriteIndex:       &end,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			command := &captureCommandEncoder{
				CommandEncoder: &noop.CommandEncoder{},
				renderPass:     &captureRenderPass{RenderPassEncoder: &noop.RenderPassEncoder{}},
			}
			h.commandEncoder = command
			encoder, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatal(err)
			}
			pass, err := encoder.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: captureAttachments(t, device), TimestampWrites: tt.writes})
			if tt.wantErr {
				if err == nil {
					t.Fatal("BeginRenderPass succeeded, want an error")
				}
				if command.renderPassDesc != nil {
					t.Fatal("HAL BeginRenderPass was called")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			defer pass.End()
			if command.renderPassDesc == nil {
				t.Fatal("HAL BeginRenderPass was not called")
			}
			got := command.renderPassDesc.TimestampWrites
			if tt.wantNil {
				if got != nil {
					t.Fatalf("TimestampWrites = %+v, want nil", got)
				}
				return
			}
			if got == nil || got.QuerySet != resource || got.BeginningOfPassWriteIndex != &begin || got.EndOfPassWriteIndex != &end {
				t.Fatalf("TimestampWrites = %+v", got)
			}
		})
	}
}

func TestRenderBundleEncoderCommandsAfterFinish(t *testing.T) {
	tests := []struct {
		name string
		call func(*RenderBundleEncoder)
	}{
		{name: "SetPipeline", call: func(e *RenderBundleEncoder) { e.SetPipeline(&RenderPipeline{}) }},
		{name: "SetBindGroup", call: func(e *RenderBundleEncoder) { e.SetBindGroup(0, &BindGroup{}, nil) }},
		{name: "SetVertexBuffer", call: func(e *RenderBundleEncoder) { e.SetVertexBuffer(0, &Buffer{}, 0) }},
		{name: "SetIndexBuffer", call: func(e *RenderBundleEncoder) { e.SetIndexBuffer(&Buffer{}, gputypes.IndexFormatUint16, 0) }},
		{name: "Draw", call: func(e *RenderBundleEncoder) { e.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1}) }},
		{name: "DrawIndexed", call: func(e *RenderBundleEncoder) { e.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 3, InstanceCount: 1}) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoder := &RenderBundleEncoder{}
			tt.call(encoder)
			bundle, err := encoder.Finish(nil)
			if err != nil {
				t.Fatalf("Finish() error = %v", err)
			}
			tt.call(encoder)
			if len(bundle.commands) != 1 || len(encoder.commands) != 0 {
				t.Fatalf("commands = %d in the bundle, %d left in the encoder, want 1 and 0", len(bundle.commands), len(encoder.commands))
			}
		})
	}
}

func TestRenderBundleEncoderFinish(t *testing.T) {
	tests := []struct {
		name      string
		desc      *RenderBundleDescriptor
		wantLabel string
	}{
		{name: "nil descriptor", desc: nil},
		{name: "label", desc: &RenderBundleDescriptor{Label: "test bundle"}, wantLabel: "test bundle"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoder := &RenderBundleEncoder{}
			encoder.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
			bundle, err := encoder.Finish(tt.desc)
			if err != nil {
				t.Fatalf("Finish() error = %v", err)
			}
			if bundle == nil || len(bundle.commands) != 1 || bundle.label != tt.wantLabel {
				t.Fatalf("Finish() bundle = %+v", bundle)
			}
			if second, secondErr := encoder.Finish(nil); second != nil || !errors.Is(secondErr, ErrReleased) {
				t.Fatalf("second Finish() = (%v, %v), want (nil, ErrReleased)", second, secondErr)
			}
		})
	}
}

func TestDeviceCreateRenderBundleEncoder(t *testing.T) {
	tests := []struct {
		name       string
		desc       *RenderBundleEncoderDescriptor
		released   bool
		wantErr    bool
		wantResult bool
	}{
		{name: "nil descriptor", wantErr: true},
		{name: "released device", desc: &RenderBundleEncoderDescriptor{}, released: true, wantErr: true},
		{name: "success", desc: &RenderBundleEncoderDescriptor{}, wantResult: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			device := newCaptureDevice(&captureDevice{Device: &noop.Device{}})
			if tt.released {
				device.released.Store(true)
			}
			got, err := device.CreateRenderBundleEncoder(tt.desc)
			if (err != nil) != tt.wantErr || (got != nil) != tt.wantResult {
				t.Fatalf("CreateRenderBundleEncoder() = (%v, %v)", got, err)
			}
		})
	}
}

func TestRenderBundleRelease(t *testing.T) {
	bundle := &RenderBundle{}
	bundle.Release()
	bundle.Release()
	if !bundle.released.Load() {
		t.Fatal("bundle not released")
	}
	var nilRenderBundle *RenderBundle
	nilRenderBundle.Release()
}

func TestResolveQuerySetAndExecuteBundles(t *testing.T) {
	resource := &testHALResource{}
	pass := &captureRenderPass{RenderPassEncoder: &noop.RenderPassEncoder{}}
	command := &captureCommandEncoder{CommandEncoder: &noop.CommandEncoder{}, renderPass: pass}
	h := &captureDevice{Device: &noop.Device{}, commandEncoder: command}
	device := newCaptureDevice(h)

	buffer, err := device.CreateBuffer(&BufferDescriptor{Size: 256, Usage: BufferUsageQueryResolve | BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer() error = %v", err)
	}
	encoder, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder() error = %v", err)
	}
	querySet := newTestQuerySet(resource, device)
	encoder.ResolveQuerySet(querySet, 0, 1, buffer, 0)
	if command.resolved != 1 || command.querySet != resource || command.buffer == nil {
		t.Fatalf("ResolveQuerySet delegation = (%d, %v, %v)", command.resolved, command.querySet, command.buffer)
	}

	vertices, err := device.CreateBuffer(&BufferDescriptor{Size: 256, Usage: BufferUsageVertex})
	if err != nil {
		t.Fatalf("CreateBuffer() error = %v", err)
	}
	record := func() *RenderBundle {
		bundleEncoder := &RenderBundleEncoder{}
		bundleEncoder.SetVertexBuffer(0, vertices, 0)
		bundle, err := bundleEncoder.Finish(nil)
		if err != nil {
			t.Fatalf("Finish() error = %v", err)
		}
		return bundle
	}
	released := record()
	released.Release()

	renderPass, err := encoder.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: captureAttachments(t, device)})
	if err != nil {
		t.Fatalf("BeginRenderPass() error = %v", err)
	}
	renderPass.ExecuteBundles(nil, released, &RenderBundle{}, record())
	if len(pass.vertexBuffers) != 1 || pass.vertexBuffers[0] == nil {
		t.Fatalf("vertex buffers set = %v, want the one of the active bundle", pass.vertexBuffers)
	}
}

// ResolveQuerySet writes its destination, so the buffer is listed for
// submit validation, and the query set is too: releasing it before the
// submit fails the submit instead of destroying it under the resolve.
func TestResolveQuerySetTracksItsResources(t *testing.T) {
	resource := &testHALResource{}
	command := &captureCommandEncoder{CommandEncoder: &noop.CommandEncoder{}}
	device := newCaptureDevice(&captureDevice{Device: &noop.Device{}, commandEncoder: command})

	buffer, err := device.CreateBuffer(&BufferDescriptor{Size: 512, Usage: BufferUsageQueryResolve | BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer() error = %v", err)
	}
	encoder, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder() error = %v", err)
	}
	querySet := newTestQuerySet(resource, device)
	encoder.ResolveQuerySet(querySet, 0, 1, buffer, 0)
	encoder.ResolveQuerySet(querySet, 1, 1, buffer, 256)
	cb, err := encoder.Finish()
	if err != nil {
		t.Fatalf("Finish() error = %v", err)
	}
	if len(cb.usedBuffers) != 1 || cb.usedBuffers[0] != buffer {
		t.Fatalf("usedBuffers = %v, want the destination once", cb.usedBuffers)
	}
	if len(cb.usedQuerySets) != 1 || cb.usedQuerySets[0] != querySet {
		t.Fatalf("usedQuerySets = %v, want the query set once", cb.usedQuerySets)
	}
	if err := validateCommandBufferForSubmit(cb, 0); err != nil {
		t.Fatalf("validateCommandBufferForSubmit() error = %v", err)
	}
	querySet.released.Store(true)
	if err := validateCommandBufferForSubmit(cb, 0); !errors.Is(err, ErrReleased) {
		t.Fatalf("validateCommandBufferForSubmit() with a released query set = %v, want ErrReleased", err)
	}
}

func TestResolveQuerySetGuards(t *testing.T) {
	resource := &testHALResource{}
	command := &captureCommandEncoder{CommandEncoder: &noop.CommandEncoder{}}
	h := &captureDevice{Device: &noop.Device{}, commandEncoder: command}
	device := newCaptureDevice(h)
	encoder, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder() error = %v", err)
	}
	querySet := newTestQuerySet(resource, device)
	releasedQuerySet := newTestQuerySet(resource, device)
	releasedQuerySet.released.Store(true)
	activeDestination := &Buffer{released: &atomic.Bool{}}
	releasedDestination := &Buffer{released: &atomic.Bool{}}
	releasedDestination.released.Store(true)

	tests := []struct {
		name        string
		encoder     *CommandEncoder
		querySet    *QuerySet
		destination *Buffer
	}{
		{name: "released encoder", encoder: &CommandEncoder{released: true}, querySet: querySet, destination: activeDestination},
		{name: "nil query set", encoder: encoder, destination: activeDestination},
		{name: "released query set", encoder: encoder, querySet: releasedQuerySet, destination: activeDestination},
		{name: "nil destination", encoder: encoder, querySet: querySet},
		{name: "released destination", encoder: encoder, querySet: querySet, destination: releasedDestination},
		{name: "nil HAL query set", encoder: encoder, querySet: &QuerySet{}, destination: activeDestination},
		{name: "nil HAL destination", encoder: encoder, querySet: querySet, destination: activeDestination},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.encoder.ResolveQuerySet(tt.querySet, 0, 1, tt.destination, 0)
			if command.resolved != 0 {
				t.Fatalf("HAL ResolveQuerySet calls = %d, want 0", command.resolved)
			}
		})
	}

	nilRawCommand := &captureCommandEncoder{CommandEncoder: &noop.CommandEncoder{}}
	nilRawDevice := newCaptureDevice(&captureDevice{Device: &noop.Device{}, commandEncoder: nilRawCommand})
	nilRawEncoder, err := nilRawDevice.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder() error = %v", err)
	}
	if _, err := nilRawEncoder.Finish(); err != nil {
		t.Fatalf("Finish() error = %v", err)
	}
	nilRawEncoder.core.TakeHALEncoder()
	nilRawEncoder.released = false
	nilRawEncoder.ResolveQuerySet(querySet, 0, 1, activeDestination, 0)
}

func TestResolveQuerySetRecordsErrors(t *testing.T) {
	resource := &testHALResource{}
	tests := []struct {
		name        string
		querySet    func(*Device) *QuerySet
		destination *Buffer
	}{
		{name: "nil query set", destination: &Buffer{released: &atomic.Bool{}}},
		{name: "nil destination", querySet: func(d *Device) *QuerySet { return newTestQuerySet(resource, d) }},
		{name: "released query set", querySet: func(d *Device) *QuerySet {
			q := newTestQuerySet(resource, d)
			q.released.Store(true)
			return q
		}, destination: &Buffer{released: &atomic.Bool{}}},
		{name: "released destination", querySet: func(d *Device) *QuerySet { return newTestQuerySet(resource, d) }, destination: func() *Buffer {
			b := &Buffer{released: &atomic.Bool{}}
			b.released.Store(true)
			return b
		}()},
		{name: "nil HAL query set", querySet: func(*Device) *QuerySet { return &QuerySet{} }, destination: &Buffer{released: &atomic.Bool{}}},
		{name: "nil HAL destination", querySet: func(d *Device) *QuerySet { return newTestQuerySet(resource, d) }, destination: &Buffer{released: &atomic.Bool{}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			command := &captureCommandEncoder{CommandEncoder: &noop.CommandEncoder{}}
			device := newCaptureDevice(&captureDevice{Device: &noop.Device{}, commandEncoder: command})
			encoder, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder() error = %v", err)
			}
			var querySet *QuerySet
			if tt.querySet != nil {
				querySet = tt.querySet(device)
			}
			encoder.ResolveQuerySet(querySet, 0, 1, tt.destination, 0)
			if command.resolved != 0 {
				t.Fatalf("HAL ResolveQuerySet calls = %d, want 0", command.resolved)
			}
			if _, err := encoder.Finish(); err == nil {
				t.Fatal("Finish() error = nil, want deferred ResolveQuerySet error")
			}
		})
	}
}

// ResolveQuerySet refuses a range outside the query set, a destination
// without QueryResolve, an offset that is not a multiple of 256 and results
// past the end of the destination, before the backend sees them.
func TestResolveQuerySetChecksItsArguments(t *testing.T) {
	resource := &testHALResource{}
	for _, tt := range []struct {
		name         string
		first, count uint32
		usage        gputypes.BufferUsage
		size, offset uint64
		wantErr      string
	}{
		{name: "whole set", count: 4, usage: BufferUsageQueryResolve, size: 32},
		{name: "last query at an offset", first: 3, count: 1, usage: BufferUsageQueryResolve, size: 264, offset: 256},
		{name: "first query past the set", first: 4, usage: BufferUsageQueryResolve, size: 32, wantErr: "among the 4"},
		{name: "range past the set", first: 2, count: 3, usage: BufferUsageQueryResolve, size: 32, wantErr: "among the 4"},
		{name: "destination without QueryResolve", count: 4, usage: BufferUsageCopyDst, size: 32, wantErr: "QueryResolve"},
		{name: "offset not a multiple of 256", count: 1, usage: BufferUsageQueryResolve, size: 512, offset: 8, wantErr: "multiple of 256"},
		{name: "results past the end", count: 2, usage: BufferUsageQueryResolve, size: 264, offset: 256, wantErr: "exceed"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			command := &captureCommandEncoder{CommandEncoder: &noop.CommandEncoder{}}
			device := newCaptureDevice(&captureDevice{Device: &noop.Device{}, commandEncoder: command})
			dst, err := device.CreateBuffer(&BufferDescriptor{Size: tt.size, Usage: tt.usage})
			if err != nil {
				t.Fatalf("CreateBuffer() error = %v", err)
			}
			defer dst.Release()
			encoder, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder() error = %v", err)
			}
			encoder.ResolveQuerySet(newTestQuerySet(resource, device), tt.first, tt.count, dst, tt.offset)
			_, err = encoder.Finish()
			if tt.wantErr == "" {
				if err != nil || command.resolved != 1 {
					t.Fatalf("Finish() = %v with %d HAL resolves, want no error and 1", err, command.resolved)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) || command.resolved != 0 {
				t.Fatalf("Finish() = %v with %d HAL resolves, want an error about %q and none", err, command.resolved, tt.wantErr)
			}
		})
	}
}

func TestExecuteBundlesWithoutRawPass(t *testing.T) {
	command := &captureCommandEncoder{CommandEncoder: &noop.CommandEncoder{}}
	device := newCaptureDevice(&captureDevice{Device: &noop.Device{}, commandEncoder: command})
	encoder, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder() error = %v", err)
	}
	vertices, err := device.CreateBuffer(&BufferDescriptor{Size: 256, Usage: BufferUsageVertex})
	if err != nil {
		t.Fatalf("CreateBuffer() error = %v", err)
	}
	bundleEncoder := &RenderBundleEncoder{}
	bundleEncoder.SetVertexBuffer(0, vertices, 0)
	bundleEncoder.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	bundle, err := bundleEncoder.Finish(nil)
	if err != nil {
		t.Fatalf("Finish() error = %v", err)
	}
	renderPass, err := encoder.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: captureAttachments(t, device)})
	if err != nil {
		t.Fatalf("BeginRenderPass() error = %v", err)
	}
	renderPass.ExecuteBundles(bundle)
}

func TestCreationWithoutHALDevice(t *testing.T) {
	device := &Device{core: nil}
	tests := []struct {
		name string
		call func() error
	}{
		{name: "render bundle encoder", call: func() error { _, err := device.CreateRenderBundleEncoder(&RenderBundleEncoderDescriptor{}); return err }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.call(); !errors.Is(err, ErrReleased) {
				t.Fatalf("error = %v, want ErrReleased", err)
			}
		})
	}
}

func TestRenderPassDescriptorTimestampWritesToHAL(t *testing.T) {
	device := newCaptureDevice(&captureDevice{Device: &noop.Device{}})
	resource := &testHALResource{}
	active := newTestQuerySet(resource, device)
	defer device.destroyQueue().FlushAll()
	defer active.Release()
	released := newTestQuerySet(resource, device)
	released.Release()
	begin, end := uint32(1), uint32(3)
	tests := []struct {
		name   string
		writes *RenderPassTimestampWrites
		want   bool
	}{
		{name: "absent"},
		{name: "nil query set", writes: &RenderPassTimestampWrites{}},
		{name: "released query set", writes: &RenderPassTimestampWrites{QuerySet: released}},
		{name: "both boundaries", writes: &RenderPassTimestampWrites{QuerySet: active, BeginningOfPassWriteIndex: &begin, EndOfPassWriteIndex: &end}, want: true},
		{name: "begin only", writes: &RenderPassTimestampWrites{QuerySet: active, BeginningOfPassWriteIndex: &begin}, want: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := (&RenderPassDescriptor{Label: "pass", TimestampWrites: tt.writes}).toHAL()
			if got.Label != "pass" {
				t.Fatalf("Label = %q", got.Label)
			}
			if !tt.want {
				if got.TimestampWrites != nil {
					t.Fatalf("TimestampWrites = %+v, want nil", got.TimestampWrites)
				}
				return
			}
			writes := got.TimestampWrites
			if writes == nil || writes.QuerySet != resource || writes.BeginningOfPassWriteIndex != tt.writes.BeginningOfPassWriteIndex || writes.EndOfPassWriteIndex != tt.writes.EndOfPassWriteIndex {
				t.Fatalf("TimestampWrites = %+v, want %+v", writes, tt.writes)
			}
		})
	}
}
