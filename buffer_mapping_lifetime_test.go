//go:build !(js && wasm)

package wgpu

import (
	"context"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/roffe/gowgpu/core"
)

// A buffer the GC collects while it is mapped keeps its mapping: a slice of
// it holds no reference to the buffer and may still be read.
func TestBufferCollectedWhileMappedKeepsMapping(t *testing.T) {
	inst, adapter, device := newTestDevice(t)
	defer inst.Release()
	defer adapter.Release()
	defer device.Release()

	const size = 1 << 20
	coreBuf, released, data := func() (*core.Buffer, *atomic.Bool, []byte) {
		buf, err := device.CreateBuffer(&BufferDescriptor{
			Label: "collected-while-mapped",
			Size:  size,
			Usage: BufferUsageMapRead | BufferUsageCopyDst,
		})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := buf.Map(ctx, MapModeRead, 0, size); err != nil {
			t.Fatalf("Map: %v", err)
		}
		rng, err := buf.MappedRange(0, size)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		return buf.core, buf.released, rng.Bytes()
	}()

	for i := 0; !released.Load(); i++ {
		if i == 200 {
			t.Fatal("the GC never collected the buffer")
		}
		runtime.GC()
		time.Sleep(5 * time.Millisecond)
	}
	if coreBuf.IsDestroyed() {
		t.Fatal("the GC destroyed a mapped buffer, whose mapping a slice still points into")
	}
	if got := coreBuf.CurrentMapState(); got != core.BufferMapStateMapped {
		t.Fatalf("map state after the GC = %v, want mapped", got)
	}
	var sum byte
	for _, b := range data {
		sum |= b
	}
	if sum != 0 {
		t.Errorf("the mapping of a new buffer holds %#x, want zeros", sum)
	}
}
