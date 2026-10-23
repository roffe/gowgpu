//go:build darwin

package wgpu_test

import (
	"context"
	"testing"
	"time"
	"unsafe"

	"github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/metal"
)

// TestCopyExternalBufferMetal copies a picture from an MTLBuffer made outside
// gowgpu into a texture, after the shared event that came with it, and checks
// the event is signalled once the copy is done.
func TestCopyExternalBufferMetal(t *testing.T) {
	device, mtl := newMetalDevice(t)
	pool := metal.NewAutoreleasePool()
	defer pool.Drain()

	const w, h, pitch = 4, 4, 256
	buf := metal.MsgSend(mtl, metal.Sel("newBufferWithLength:options:"), pitch*h, 0) // shared storage
	if buf == 0 {
		t.Fatal("newBufferWithLength failed")
	}
	defer metal.Release(buf)
	want := make([]byte, pitch*h)
	for y := range h {
		for x := range w {
			copy(want[y*pitch+x*4:], []byte{byte(x * 60), byte(y * 60), 7, 255})
		}
	}
	copy(bytesAt(metal.MsgSend(buf, metal.Sel("contents")), pitch*h), want)
	event := newSharedEvent(t, mtl)

	tex := newTexture(t, device, w, h, gputypes.TextureFormatRGBA8Unorm)
	src := wgpu.ExternalBuffer{Buffer: uint64(buf), BytesPerRow: pitch, Semaphore: uint64(event), Value: 1}
	if err := device.Queue().CopyExternalBuffer(src, tex); err != nil {
		t.Fatalf("CopyExternalBuffer: %v", err)
	}
	got := readTextures(t, device, pitch, []textureRead{{tex, 0, w, h}})
	for y := range h {
		for x := range w {
			if g, e := got[y*pitch+x*4:][:4], want[y*pitch+x*4:][:4]; string(g) != string(e) {
				t.Fatalf("texel %d,%d is %v, want %v", x, y, g, e)
			}
		}
	}
	waitSignalled(t, event, 2)
}

// TestCopyExternalPlanesMetal copies the planes of an NV12 IOSurface, as
// VideoToolbox decodes into, into an R8 and an RG8 texture.
func TestCopyExternalPlanesMetal(t *testing.T) {
	device, mtl := newMetalDevice(t)
	pool := metal.NewAutoreleasePool()
	defer pool.Drain()

	const w, h, pitch, uvOffset = 8, 4, 256, 4096
	surface := newNV12Surface(t, w, h, pitch, uvOffset)
	defer metal.Release(surface)
	if kr := int32(metal.MsgSend(surface, metal.Sel("lockWithOptions:seed:"), 0, 0)); kr != 0 {
		t.Fatalf("IOSurface lock: %#x", kr)
	}
	want := make([]byte, uvOffset+pitch*h/2)
	for y := range h {
		for x := range w {
			want[y*pitch+x] = byte(x*16 + y)
		}
	}
	for y := range h / 2 {
		for x := range w {
			want[uvOffset+y*pitch+x] = byte(200 - x*8 - y)
		}
	}
	for i, plane := range [][]byte{want[:pitch*h], want[uvOffset:]} {
		base := metal.MsgSend(surface, metal.Sel("baseAddressOfPlaneAtIndex:"), uintptr(i))
		copy(bytesAt(base, len(plane)), plane)
	}
	metal.MsgSend(surface, metal.Sel("unlockWithOptions:seed:"), 0, 0)
	event := newSharedEvent(t, mtl)

	luma := newTexture(t, device, w, h, gputypes.TextureFormatR8Unorm)
	chroma := newTexture(t, device, w/2, h/2, gputypes.TextureFormatRG8Unorm)
	src := wgpu.ExternalImage{Image: uint64(surface), Semaphore: uint64(event), Value: 1}
	if _, err := device.Queue().CopyExternalPlanes(src, luma, chroma); err != nil {
		t.Fatalf("CopyExternalPlanes: %v", err)
	}
	got := readTextures(t, device, pitch, []textureRead{{luma, 0, w, h}, {chroma, uvOffset, w / 2, h / 2}})
	for y := range h {
		if g, e := got[y*pitch:][:w], want[y*pitch:][:w]; string(g) != string(e) {
			t.Errorf("luma row %d is %v, want %v", y, g, e)
		}
	}
	for y := range h / 2 {
		if g, e := got[uvOffset+y*pitch:][:w], want[uvOffset+y*pitch:][:w]; string(g) != string(e) {
			t.Errorf("chroma row %d is %v, want %v", y, g, e)
		}
	}
	waitSignalled(t, event, 2)
}

func newMetalDevice(t *testing.T) (*wgpu.Device, metal.ID) {
	t.Helper()
	instance, err := wgpu.CreateInstance(&wgpu.InstanceDescriptor{Backends: wgpu.BackendsMetal})
	if err != nil {
		t.Skipf("CreateInstance: %v", err)
	}
	t.Cleanup(instance.Release)
	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		t.Skipf("RequestAdapter: %v", err)
	}
	t.Cleanup(adapter.Release)
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		t.Skipf("RequestDevice: %v", err)
	}
	t.Cleanup(device.Release)
	hd, ok := device.HalDevice().(*metal.Device)
	if !ok {
		t.Skip("not a Metal device")
	}
	return device, metal.ID(hd.MetalDevice())
}

// newSharedEvent returns an MTLSharedEvent at 1, as a producer leaves it
// once its picture is done.
func newSharedEvent(t *testing.T, mtl metal.ID) metal.ID {
	t.Helper()
	event := metal.MsgSend(mtl, metal.Sel("newSharedEvent"))
	if event == 0 {
		t.Fatal("newSharedEvent failed")
	}
	t.Cleanup(func() { metal.Release(event) })
	metal.MsgSend(event, metal.Sel("setSignaledValue:"), 1)
	return event
}

func waitSignalled(t *testing.T, event metal.ID, want uint) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for metal.MsgSendUint(event, metal.Sel("signaledValue")) < want && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if v := metal.MsgSendUint(event, metal.Sel("signaledValue")); v != want {
		t.Errorf("the event is at %d after the copy, want %d", v, want)
	}
}

// newNV12Surface makes a w by h NV12 IOSurface, chroma uvOffset bytes in, rows
// of both planes pitch bytes apart.
func newNV12Surface(t *testing.T, w, h, pitch, uvOffset uint64) metal.ID {
	t.Helper()
	plane := func(w, h, bpe, offset uint64) metal.ID {
		return nsDict(map[string]metal.ID{
			"IOSurfacePlaneWidth": nsNumber(w), "IOSurfacePlaneHeight": nsNumber(h),
			"IOSurfacePlaneBytesPerRow": nsNumber(pitch), "IOSurfacePlaneBytesPerElement": nsNumber(bpe),
			"IOSurfacePlaneOffset": nsNumber(offset), "IOSurfacePlaneSize": nsNumber(pitch * h),
		})
	}
	planes := metal.MsgSend(metal.ID(metal.GetClass("NSMutableArray")), metal.Sel("array"))
	metal.MsgSend(planes, metal.Sel("addObject:"), uintptr(plane(w, h, 1, 0)))
	metal.MsgSend(planes, metal.Sel("addObject:"), uintptr(plane(w/2, h/2, 2, uvOffset)))
	props := nsDict(map[string]metal.ID{
		"IOSurfaceWidth": nsNumber(w), "IOSurfaceHeight": nsNumber(h),
		"IOSurfacePixelFormat": nsNumber(0x34323066), // '420f'
		"IOSurfaceAllocSize":   nsNumber(uvOffset + pitch*h/2),
		"IOSurfacePlaneInfo":   planes,
	})
	surface := metal.MsgSend(metal.ID(metal.GetClass("IOSurface")), metal.Sel("alloc"))
	surface = metal.MsgSend(surface, metal.Sel("initWithProperties:"), uintptr(props))
	if surface == 0 {
		t.Fatal("IOSurface initWithProperties failed")
	}
	return surface
}

func nsNumber(v uint64) metal.ID {
	return metal.MsgSend(metal.ID(metal.GetClass("NSNumber")), metal.Sel("numberWithUnsignedLongLong:"), uintptr(v))
}

func nsDict(kv map[string]metal.ID) metal.ID {
	d := metal.MsgSend(metal.ID(metal.GetClass("NSMutableDictionary")), metal.Sel("dictionary"))
	for k, v := range kv {
		key := metal.NSString(k)
		metal.MsgSend(d, metal.Sel("setObject:forKey:"), uintptr(v), uintptr(key))
		metal.Release(key)
	}
	return d
}

// bytesAt is the n bytes at p, memory outside Go.
func bytesAt(p metal.ID, n int) []byte {
	return unsafe.Slice(*(**byte)(unsafe.Pointer(&p)), n)
}

func newTexture(t *testing.T, device *wgpu.Device, w, h uint32, format gputypes.TextureFormat) *wgpu.Texture {
	t.Helper()
	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Size: wgpu.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: gputypes.TextureDimension2D, Format: format,
		Usage: gputypes.TextureUsageCopyDst | gputypes.TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	t.Cleanup(tex.Release)
	return tex
}

type textureRead struct {
	tex    *wgpu.Texture
	offset uint64
	w, h   uint32
}

// readTextures copies each texture to its offset in a buffer, rows pitch
// bytes apart, and returns the buffer's bytes.
func readTextures(t *testing.T, device *wgpu.Device, pitch uint32, reads []textureRead) []byte {
	t.Helper()
	var size uint64
	for _, r := range reads {
		size = max(size, r.offset+uint64(pitch*r.h))
	}
	staging, err := device.CreateBuffer(&wgpu.BufferDescriptor{Size: size, Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(staging.Release)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	for _, r := range reads {
		enc.CopyTextureToBuffer(r.tex, staging, []wgpu.BufferTextureCopy{{
			BufferLayout: wgpu.ImageDataLayout{Offset: r.offset, BytesPerRow: pitch, RowsPerImage: r.h},
			TextureBase:  wgpu.ImageCopyTexture{Texture: r.tex},
			Size:         wgpu.Extent3D{Width: r.w, Height: r.h, DepthOrArrayLayers: 1},
		}})
	}
	cmd, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if _, err := device.Queue().Submit(cmd); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := staging.Map(ctx, wgpu.MapModeRead, 0, size); err != nil {
		t.Fatalf("Map staging: %v", err)
	}
	defer func() { _ = staging.Unmap() }()
	rng, err := staging.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	return append([]byte(nil), rng.Bytes()...)
}
