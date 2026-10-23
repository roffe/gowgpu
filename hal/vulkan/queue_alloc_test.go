//go:build !(js && wasm)

package vulkan

import (
	"testing"

	"github.com/roffe/gowgpu/hal"
)

// TestSubmitAllocatesNothing submits a command buffer at a time, each
// waiting for the one before through the relay semaphores, and requires
// Submit to allocate nothing: the arrays its submit info points at go to
// the stack, where pinning them in a Pinner would allocate past five.
func TestSubmitAllocatesNothing(t *testing.T) {
	d := openTestDevice(t, 0)
	const runs = 20
	cbs := make([][]hal.CommandBuffer, runs+1)
	for i := range cbs {
		enc, err := d.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "submit"})
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		if err := enc.BeginEncoding("submit"); err != nil {
			t.Fatalf("BeginEncoding: %v", err)
		}
		cb, err := enc.EndEncoding()
		if err != nil {
			t.Fatalf("EndEncoding: %v", err)
		}
		cbs[i] = []hal.CommandBuffer{cb}
	}
	next := 0
	allocs := testing.AllocsPerRun(runs, func() {
		if _, err := d.queue.Submit(cbs[next]); err != nil {
			t.Fatalf("Submit: %v", err)
		}
		next++
	})
	if err := d.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	for _, cb := range cbs {
		d.FreeCommandBuffer(cb[0])
	}
	if allocs > 0 {
		t.Errorf("Submit allocates %v times", allocs)
	}
}
