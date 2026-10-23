//go:build windows

package wgpu_test

import (
	"context"
	"testing"
	"time"
	"unsafe"

	"github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/dx12"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

// TestCopyExternalBufferDX12 copies a picture from a buffer made outside
// gowgpu into a texture, after the fence that came with it, and checks the
// fence is signalled once the copy is done, as for OpenGL sharing the device.
func TestCopyExternalBufferDX12(t *testing.T) {
	device := newDX12Device(t)
	raw := rawDevice(t, device)

	const w, h, pitch = 4, 4, 256
	desc := d3d12.D3D12_RESOURCE_DESC{
		Dimension: d3d12.D3D12_RESOURCE_DIMENSION_BUFFER, Width: pitch * h, Height: 1, DepthOrArraySize: 1,
		MipLevels: 1, SampleDesc: d3d12.DXGI_SAMPLE_DESC{Count: 1}, Layout: d3d12.D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
	}
	src, err := raw.CreateCommittedResource(&d3d12.D3D12_HEAP_PROPERTIES{Type: d3d12.D3D12_HEAP_TYPE_UPLOAD},
		d3d12.D3D12_HEAP_FLAG_NONE, &desc, d3d12.D3D12_RESOURCE_STATE_GENERIC_READ, nil)
	if err != nil {
		t.Fatalf("CreateCommittedResource: %v", err)
	}
	defer src.Release()
	p, err := src.Map(0, &d3d12.D3D12_RANGE{})
	if err != nil {
		t.Fatalf("Map: %v", err)
	}
	want := make([]byte, pitch*h)
	for y := range h {
		for x := range w {
			copy(want[y*pitch+x*4:], []byte{byte(x * 60), byte(y * 60), 7, 255})
		}
	}
	copy(unsafe.Slice((*byte)(p), pitch*h), want)
	src.Unmap(0, nil)
	fence, err := raw.CreateFence(1, d3d12.D3D12_FENCE_FLAG_NONE) // the drawing done
	if err != nil {
		t.Fatalf("CreateFence: %v", err)
	}
	defer fence.Release()

	tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
		Size: wgpu.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatRGBA8Unorm,
		Usage: gputypes.TextureUsageCopyDst | gputypes.TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	t.Cleanup(tex.Release)
	staging, err := device.CreateBuffer(&wgpu.BufferDescriptor{Size: pitch * h, Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(staging.Release)

	src1 := wgpu.ExternalBuffer{Buffer: uint64(uintptr(unsafe.Pointer(src))), BytesPerRow: pitch, Semaphore: uint64(uintptr(unsafe.Pointer(fence))), Value: 1}
	if err := device.Queue().CopyExternalBuffer(src1, tex); err != nil {
		t.Fatalf("CopyExternalBuffer: %v", err)
	}
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.CopyTextureToBuffer(tex, staging, []wgpu.BufferTextureCopy{{
		BufferLayout: wgpu.ImageDataLayout{BytesPerRow: pitch, RowsPerImage: h},
		TextureBase:  wgpu.ImageCopyTexture{Texture: tex},
		Size:         wgpu.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1},
	}})
	cmd, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if _, err := device.Queue().Submit(cmd); err != nil {
		t.Fatalf("Submit: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := staging.Map(ctx, wgpu.MapModeRead, 0, pitch*h); err != nil {
		t.Fatalf("Map staging: %v", err)
	}
	rng, err := staging.MappedRange(0, pitch*h)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	got := rng.Bytes()
	for y := range h {
		for x := range w {
			if g, e := got[y*pitch+x*4:][:4], want[y*pitch+x*4:][:4]; string(g) != string(e) {
				t.Fatalf("texel %d,%d is %v, want %v", x, y, g, e)
			}
		}
	}
	_ = staging.Unmap()
	if v := fence.GetCompletedValue(); v != 2 {
		t.Errorf("the fence is at %d after the copy, want 2", v)
	}
}

// TestCopyExternalPlanesDX12 copies the planes of an NV12 texture made
// outside gowgpu, in the COMMON state D3D12 video decoding leaves its frames
// in, into an R8 and an RG8 texture.
func TestCopyExternalPlanesDX12(t *testing.T) {
	device := newDX12Device(t)
	raw := rawDevice(t, device)
	const w, h, pitch, uvOffset = 8, 4, 256, 1024 // planes 512-byte aligned in the upload
	nv12, err := raw.CreateCommittedResource(&d3d12.D3D12_HEAP_PROPERTIES{Type: d3d12.D3D12_HEAP_TYPE_DEFAULT}, d3d12.D3D12_HEAP_FLAG_NONE,
		&d3d12.D3D12_RESOURCE_DESC{Dimension: d3d12.D3D12_RESOURCE_DIMENSION_TEXTURE2D, Width: w, Height: h, DepthOrArraySize: 1,
			MipLevels: 1, Format: d3d12.DXGI_FORMAT(103), SampleDesc: d3d12.DXGI_SAMPLE_DESC{Count: 1}}, // NV12
		d3d12.D3D12_RESOURCE_STATE_COMMON, nil)
	if err != nil {
		t.Skipf("no NV12 textures: %v", err)
	}
	defer nv12.Release()
	upload, err := raw.CreateCommittedResource(&d3d12.D3D12_HEAP_PROPERTIES{Type: d3d12.D3D12_HEAP_TYPE_UPLOAD}, d3d12.D3D12_HEAP_FLAG_NONE,
		&d3d12.D3D12_RESOURCE_DESC{Dimension: d3d12.D3D12_RESOURCE_DIMENSION_BUFFER, Width: uvOffset + pitch*h/2, Height: 1, DepthOrArraySize: 1,
			MipLevels: 1, SampleDesc: d3d12.DXGI_SAMPLE_DESC{Count: 1}, Layout: d3d12.D3D12_TEXTURE_LAYOUT_ROW_MAJOR},
		d3d12.D3D12_RESOURCE_STATE_GENERIC_READ, nil)
	if err != nil {
		t.Fatalf("CreateCommittedResource: %v", err)
	}
	defer upload.Release()
	want := make([]byte, uvOffset+pitch*h/2)
	for y := range h {
		for x := range w {
			want[y*pitch+x] = byte(x*16 + y)
		}
	}
	for y := range h / 2 {
		for x := range w / 2 {
			want[uvOffset+y*pitch+x*2], want[uvOffset+y*pitch+x*2+1] = byte(x*20), byte(y*30+1)
		}
	}
	p, err := upload.Map(0, &d3d12.D3D12_RANGE{})
	if err != nil {
		t.Fatalf("Map: %v", err)
	}
	copy(unsafe.Slice((*byte)(p), len(want)), want)
	upload.Unmap(0, nil)

	// as a decoder would leave it: written on a queue of its own, here a copy
	// queue, after which the texture is back in COMMON, and a fence signalled
	fence, err := raw.CreateFence(0, d3d12.D3D12_FENCE_FLAG_NONE)
	if err != nil {
		t.Fatalf("CreateFence: %v", err)
	}
	defer fence.Release()
	queue, err := raw.CreateCommandQueue(&d3d12.D3D12_COMMAND_QUEUE_DESC{Type: d3d12.D3D12_COMMAND_LIST_TYPE_COPY})
	if err != nil {
		t.Fatalf("CreateCommandQueue: %v", err)
	}
	defer queue.Release()
	alloc, err := raw.CreateCommandAllocator(d3d12.D3D12_COMMAND_LIST_TYPE_COPY)
	if err != nil {
		t.Fatalf("CreateCommandAllocator: %v", err)
	}
	defer alloc.Release()
	list, err := raw.CreateCommandList(0, d3d12.D3D12_COMMAND_LIST_TYPE_COPY, alloc, nil)
	if err != nil {
		t.Fatalf("CreateCommandList: %v", err)
	}
	defer list.Release()
	for plane, f := range []struct {
		offset uint64
		format d3d12.DXGI_FORMAT
		w, h   uint32
	}{{0, d3d12.DXGI_FORMAT_R8_UNORM, w, h}, {uvOffset, d3d12.DXGI_FORMAT_R8G8_UNORM, w / 2, h / 2}} {
		dst := d3d12.D3D12_TEXTURE_COPY_LOCATION{Resource: nv12, Type: d3d12.D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX}
		dst.SetSubresourceIndex(uint32(plane))
		src := d3d12.D3D12_TEXTURE_COPY_LOCATION{Resource: upload, Type: d3d12.D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT}
		src.SetPlacedFootprint(d3d12.D3D12_PLACED_SUBRESOURCE_FOOTPRINT{Offset: f.offset,
			Footprint: d3d12.D3D12_SUBRESOURCE_FOOTPRINT{Format: f.format, Width: f.w, Height: f.h, Depth: 1, RowPitch: pitch}})
		list.CopyTextureRegion(&dst, 0, 0, 0, &src, nil)
	}
	if err := list.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	queue.ExecuteCommandLists(1, &list)
	if err := queue.Signal(fence, 1); err != nil {
		t.Fatalf("Signal: %v", err)
	}

	newTex := func(w, h uint32, format gputypes.TextureFormat) *wgpu.Texture {
		tex, err := device.CreateTexture(&wgpu.TextureDescriptor{
			Size: wgpu.Extent3D{Width: w, Height: h, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
			Dimension: gputypes.TextureDimension2D, Format: format, Usage: gputypes.TextureUsageCopyDst | gputypes.TextureUsageCopySrc,
		})
		if err != nil {
			t.Fatalf("CreateTexture: %v", err)
		}
		t.Cleanup(tex.Release)
		return tex
	}
	luma, chroma := newTex(w, h, gputypes.TextureFormatR8Unorm), newTex(w/2, h/2, gputypes.TextureFormatRG8Unorm)
	src := wgpu.ExternalImage{Image: uint64(uintptr(unsafe.Pointer(nv12))), Semaphore: uint64(uintptr(unsafe.Pointer(fence))), Value: 1}
	if _, err := device.Queue().CopyExternalPlanes(src, luma, chroma); err != nil {
		t.Fatalf("CopyExternalPlanes: %v", err)
	}
	staging, err := device.CreateBuffer(&wgpu.BufferDescriptor{Size: uint64(len(want)), Usage: wgpu.BufferUsageCopyDst | wgpu.BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(staging.Release)
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	for _, c := range []struct {
		tex    *wgpu.Texture
		offset uint64
		w, h   uint32
	}{{luma, 0, w, h}, {chroma, uvOffset, w / 2, h / 2}} {
		enc.CopyTextureToBuffer(c.tex, staging, []wgpu.BufferTextureCopy{{
			BufferLayout: wgpu.ImageDataLayout{Offset: c.offset, BytesPerRow: pitch, RowsPerImage: c.h},
			TextureBase:  wgpu.ImageCopyTexture{Texture: c.tex},
			Size:         wgpu.Extent3D{Width: c.w, Height: c.h, DepthOrArrayLayers: 1},
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
	if err := staging.Map(ctx, wgpu.MapModeRead, 0, uint64(len(want))); err != nil {
		t.Fatalf("Map staging: %v", err)
	}
	rng, err := staging.MappedRange(0, uint64(len(want)))
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	got := rng.Bytes()
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
	_ = staging.Unmap()
	if v := fence.GetCompletedValue(); v != 2 {
		t.Errorf("the fence is at %d after the copy, want 2", v)
	}
}

func rawDevice(t *testing.T, device *wgpu.Device) *d3d12.ID3D12Device {
	t.Helper()
	hd, ok := device.HalDevice().(*dx12.Device)
	if !ok {
		t.Skip("not a Direct3D 12 device")
	}
	handle := hd.D3D12Device()
	return *(**d3d12.ID3D12Device)(unsafe.Pointer(&handle))
}

func newDX12Device(t *testing.T) *wgpu.Device {
	t.Helper()
	instance, err := wgpu.CreateInstance(&wgpu.InstanceDescriptor{Backends: wgpu.BackendsDX12})
	if err != nil {
		t.Skipf("CreateInstance: %v", err)
	}
	t.Cleanup(instance.Release)
	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		t.Skipf("RequestAdapter: %v", err)
	}
	t.Cleanup(adapter.Release)
	if got := adapter.Info().Backend; got != gputypes.BackendDX12 {
		t.Skipf("adapter backend is %v, want DX12", got)
	}
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		t.Skipf("RequestDevice: %v", err)
	}
	t.Cleanup(device.Release)
	return device
}
