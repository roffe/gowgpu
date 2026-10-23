// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin && !(js && wasm)

package metal

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// newSurfaceTest makes a surface on a CAMetalLayer of its own and opens the
// first adapter, all of it destroyed when the test ends.
func newSurfaceTest(t *testing.T) (*Surface, hal.OpenDevice) {
	t.Helper()
	if err := Init(); err != nil {
		t.Fatalf("Init failed: %v", err)
	}

	pool := NewAutoreleasePool()
	defer pool.Drain()

	layer := MsgSend(ID(GetClass("CAMetalLayer")), Sel("new"))
	if layer == 0 {
		t.Fatal("CAMetalLayer new returned nil")
	}
	t.Cleanup(func() { Release(layer) })

	backend := Backend{}
	inst, err := backend.CreateInstance(&hal.InstanceDescriptor{
		Backends: gputypes.Backends(1 << gputypes.BackendMetal),
	})
	if err != nil {
		t.Fatalf("CreateInstance failed: %v", err)
	}
	instance := inst.(*Instance)

	surface, err := instance.CreateSurface(hal.SurfaceTarget{
		Kind:         hal.SurfaceTargetMetalLayer,
		WindowHandle: uintptr(layer),
	})
	if err != nil {
		t.Fatalf("CreateSurface failed: %v", err)
	}
	t.Cleanup(surface.Destroy)

	adapters := instance.EnumerateAdapters(surface)
	if len(adapters) == 0 {
		t.Skip("no Metal adapters available")
	}
	adapter := adapters[0].Adapter
	t.Cleanup(adapter.Destroy)

	open, err := adapter.Open(gputypes.Features(0), gputypes.DefaultLimits())
	if err != nil {
		t.Fatalf("Adapter.Open failed: %v", err)
	}
	t.Cleanup(open.Device.Destroy)
	return surface.(*Surface), open
}

func TestSurfaceTextureCreateView(t *testing.T) {
	surface, open := newSurfaceTest(t)

	config := &hal.SurfaceConfiguration{
		Format:      gputypes.TextureFormatBGRA8Unorm,
		Width:       64,
		Height:      64,
		PresentMode: hal.PresentModeFifo,
		Usage:       gputypes.TextureUsageRenderAttachment,
		AlphaMode:   hal.CompositeAlphaModeOpaque,
	}
	if err := surface.Configure(open.Device, config); err != nil {
		t.Fatalf("Configure failed: %v", err)
	}

	acquired, err := surface.AcquireTexture(nil)
	if err != nil {
		t.Fatalf("AcquireTexture failed: %v", err)
	}
	if acquired.Texture == nil {
		t.Fatal("AcquireTexture returned nil texture")
	}

	view, err := open.Device.CreateTextureView(acquired.Texture, nil)
	if err != nil {
		t.Fatalf("CreateTextureView failed: %v", err)
	}
	if view == nil {
		t.Fatal("CreateTextureView returned nil")
	}
	tv, ok := view.(*TextureView)
	if !ok || tv.raw == 0 {
		t.Fatal("CreateTextureView returned nil raw texture view")
	}
	view.Destroy()

	surface.DiscardTexture(acquired.Texture)
}

// TestSurfaceTextureCopies copies into a drawable, with a command encoder
// and with Queue.WriteTexture, and back out of it.
func TestSurfaceTextureCopies(t *testing.T) {
	surface, open := newSurfaceTest(t)
	device, queue := open.Device.(*Device), open.Queue.(*Queue)

	const side, pitch = 4, 256
	if err := surface.Configure(device, &hal.SurfaceConfiguration{
		Format:      gputypes.TextureFormatBGRA8Unorm,
		Width:       side,
		Height:      side,
		PresentMode: hal.PresentModeFifo,
		Usage:       gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopySrc | gputypes.TextureUsageCopyDst,
		AlphaMode:   hal.CompositeAlphaModeOpaque,
	}); err != nil {
		t.Fatalf("Configure failed: %v", err)
	}
	acquired, err := surface.AcquireTexture(nil)
	if err != nil {
		t.Fatalf("AcquireTexture failed: %v", err)
	}
	defer surface.DiscardTexture(acquired.Texture)

	want := make([]byte, pitch*side)
	for y := range side {
		for x := range side {
			copy(want[y*pitch+x*4:], []byte{byte(40 * x), byte(40 * y), 200, 255})
		}
	}
	region := hal.BufferTextureCopy{
		BufferLayout: hal.ImageDataLayout{BytesPerRow: pitch, RowsPerImage: side},
		TextureBase:  hal.ImageCopyTexture{Texture: acquired.Texture},
		Size:         hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
	}
	readBack := func(name string) []byte {
		t.Helper()
		readback := createMetalTextureCopyTestReadback(t, device, pitch*side)
		defer readback.Destroy()
		encoder := &CommandEncoder{device: device}
		if err := encoder.BeginEncoding(name); err != nil {
			t.Fatalf("BeginEncoding failed: %v", err)
		}
		encoder.CopyTextureToBuffer(acquired.Texture, readback, []hal.BufferTextureCopy{region})
		submitMetalTextureCopyTestEncoder(t, device, queue, encoder)
		got := make([]byte, pitch*side)
		copy(got, unsafe.Slice((*byte)(readback.Contents()), pitch*side))
		return got
	}
	rows := func(b []byte) [][]byte {
		out := make([][]byte, side)
		for y := range side {
			out[y] = b[y*pitch : y*pitch+side*4]
		}
		return out
	}
	expect := func(what string, got []byte) {
		t.Helper()
		for y, row := range rows(got) {
			if !bytes.Equal(row, rows(want)[y]) {
				t.Fatalf("%s: row %d = %v, want %v", what, y, row, rows(want)[y])
			}
		}
	}

	upload := createMetalTextureCopyTestUpload(t, device, 0, want)
	defer upload.Destroy()
	encoder := &CommandEncoder{device: device}
	if err := encoder.BeginEncoding("surface upload"); err != nil {
		t.Fatalf("BeginEncoding failed: %v", err)
	}
	encoder.CopyBufferToTexture(upload, acquired.Texture, []hal.BufferTextureCopy{region})
	submitMetalTextureCopyTestEncoder(t, device, queue, encoder)
	expect("CopyBufferToTexture", readBack("surface readback"))

	for i := range want {
		want[i] ^= 0x55
	}
	if err := queue.WriteTexture(&region.TextureBase, want, &region.BufferLayout, &region.Size); err != nil {
		t.Fatalf("WriteTexture failed: %v", err)
	}
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle failed: %v", err)
	}
	expect("WriteTexture", readBack("surface readback after WriteTexture"))
}
