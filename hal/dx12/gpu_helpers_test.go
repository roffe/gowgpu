//go:build windows && !(js && wasm)

package dx12

import (
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// underWine reports a run under Wine, whose vkd3d leaves out some of D3D12.
func underWine() bool {
	return windows.NewLazySystemDLL("ntdll.dll").NewProc("wine_get_version").Find() == nil
}

// openTestDevice opens the first DX12 adapter, or skips without one.
func openTestDevice(t *testing.T) (*Device, *Queue) {
	t.Helper()
	if testing.Short() {
		t.Skip("skipping GPU test in short mode")
	}
	raw, err := Backend{}.CreateInstance(&hal.InstanceDescriptor{Backends: gputypes.BackendsDX12})
	if err != nil {
		t.Skipf("DX12 unavailable: %v", err)
	}
	t.Cleanup(raw.Destroy)
	adapters := raw.EnumerateAdapters(nil)
	if len(adapters) == 0 {
		t.Skip("no DX12 adapters")
	}
	open, err := adapters[0].Adapter.Open(0, adapters[0].Capabilities.Limits)
	if err != nil {
		t.Skipf("Open: %v", err)
	}
	t.Cleanup(open.Device.Destroy)
	return open.Device.(*Device), open.Queue.(*Queue)
}

func newTestBuffer(t *testing.T, dev *Device, size uint64, usage gputypes.BufferUsage) *Buffer {
	t.Helper()
	b, err := dev.CreateBuffer(&hal.BufferDescriptor{Label: t.Name(), Size: size, Usage: usage})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	t.Cleanup(func() { dev.DestroyBuffer(b) })
	return b.(*Buffer)
}

func newTestTexture(t *testing.T, dev *Device, desc hal.TextureDescriptor) *Texture {
	t.Helper()
	if desc.MipLevelCount == 0 {
		desc.MipLevelCount = 1
	}
	if desc.SampleCount == 0 {
		desc.SampleCount = 1
	}
	if desc.Size.DepthOrArrayLayers == 0 {
		desc.Size.DepthOrArrayLayers = 1
	}
	tex, err := dev.CreateTexture(&desc)
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	t.Cleanup(func() { dev.DestroyTexture(tex) })
	return tex.(*Texture)
}

// record records a command buffer with fn, submits it and waits for the
// GPU to finish it.
func record(t *testing.T, dev *Device, queue *Queue, fn func(enc *CommandEncoder)) {
	t.Helper()
	raw, err := dev.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: t.Name()})
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc := raw.(*CommandEncoder)
	t.Cleanup(enc.Destroy)
	if err := enc.BeginEncoding(t.Name()); err != nil {
		t.Fatalf("BeginEncoding: %v", err)
	}
	fn(enc)
	cmd, err := enc.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding: %v", err)
	}
	if _, err := queue.Submit([]hal.CommandBuffer{cmd}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := dev.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	enc.ResetAll([]hal.CommandBuffer{cmd})
}

// readBuffer returns size bytes of a MapRead buffer.
func readBuffer(t *testing.T, dev *Device, buf *Buffer, size uint64) []byte {
	t.Helper()
	m, err := dev.MapBuffer(buf, 0, size)
	if err != nil {
		t.Fatalf("MapBuffer: %v", err)
	}
	got := make([]byte, size)
	copy(got, unsafe.Slice((*byte)(m.Ptr), size))
	if err := dev.UnmapBuffer(buf); err != nil {
		t.Fatalf("UnmapBuffer: %v", err)
	}
	return got
}
