//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// newGLDevice makes a device on a GL instance of its own, or skips where GL
// has no device to give without a surface.
func newGLDevice(t *testing.T) (*Instance, *Device) {
	t.Helper()
	inst, err := CreateInstance(&InstanceDescriptor{Backends: gputypes.BackendsGL})
	if err != nil {
		t.Skipf("no GL instance: %v", err)
	}
	t.Cleanup(inst.Release)
	adapter, err := inst.RequestAdapter(nil)
	if err != nil {
		t.Skipf("no GL adapter: %v", err)
	}
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		t.Skipf("no GL device without a surface: %v", err)
	}
	t.Cleanup(device.Release)
	return inst, device
}

// writeAndReadBack writes data to a new buffer and maps it back.
func writeAndReadBack(t *testing.T, device *Device, data []byte) []byte {
	t.Helper()
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: uint64(len(data)), Usage: BufferUsageCopyDst | BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	if err := device.Queue().WriteBuffer(buf, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, MapModeRead, 0, uint64(len(data))); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := buf.MappedRange(0, uint64(len(data)))
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	got := bytes.Clone(rng.Bytes())
	if err := buf.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return got
}

// GL instances get the same EGL display, so releasing one must leave the
// contexts of the others working.
func TestGLInstanceReleaseLeavesOtherInstances(t *testing.T) {
	first, _ := newGLDevice(t)
	_, device := newGLDevice(t)
	first.Release()

	want := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	if got := writeAndReadBack(t, device, want); !bytes.Equal(got, want) {
		t.Fatalf("read back %v after releasing the other instance, want %v", got, want)
	}
}
