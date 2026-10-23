//go:build !(js && wasm)

package vulkan

import (
	"testing"

	"github.com/roffe/gowgpu/hal/vulkan/memory"
)

// TestPoolGrowsPastBlockCap fills the two 1 MiB blocks a pool keeps with
// 256 KiB allocations, then makes four more, and frees them: the four go
// into a third block rather than an allocation each, and the third block
// is freed once it is empty.
func TestPoolGrowsPastBlockCap(t *testing.T) {
	d := openTestDevice(t, 0)
	a, err := memory.NewGpuAllocator(d.handle, d.cmds, d.memoryProperties(), 0, memory.AllocatorConfig{
		BlockSize: 1 << 20, MinAllocationSize: 256, DedicatedThreshold: 512 << 10, MaxBlocksPerHeap: 2,
	})
	if err != nil {
		t.Fatalf("NewGpuAllocator: %v", err)
	}
	defer a.Destroy()
	var blocks []*memory.MemoryBlock
	for range 12 {
		b, err := a.Alloc(memory.AllocationRequest{Size: 256 << 10, Usage: memory.UsageFastDeviceAccess, MemoryTypeBits: ^uint32(0)})
		if err != nil {
			t.Fatalf("Alloc: %v", err)
		}
		blocks = append(blocks, b)
	}
	if s := a.Stats(); s.DedicatedAllocations != 0 || s.TotalAllocated != 3<<20 {
		t.Fatalf("12 allocations of 256 KiB took %d dedicated allocations and %d bytes, want none and 3 MiB", s.DedicatedAllocations, s.TotalAllocated)
	}
	for _, b := range blocks[8:] {
		if err := a.Free(b); err != nil {
			t.Fatalf("Free: %v", err)
		}
	}
	if s := a.Stats(); s.TotalAllocated != 2<<20 {
		t.Fatalf("with the third block empty the pool holds %d bytes, want 2 MiB", s.TotalAllocated)
	}
	for _, b := range blocks[:8] {
		if err := a.Free(b); err != nil {
			t.Fatalf("Free: %v", err)
		}
	}
	if s := a.Stats(); s.TotalAllocated != 2<<20 {
		t.Fatalf("with all blocks empty the pool holds %d bytes, want the 2 MiB it keeps", s.TotalAllocated)
	}
}
