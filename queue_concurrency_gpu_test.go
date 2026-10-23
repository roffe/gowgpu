//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

// TestWaitIdleDuringSubmit runs a WaitIdle on one goroutine while another
// submits: a submission the queue reports complete afterwards has to be
// done on the GPU. Metal's WaitIdle read the submission index after its
// wait, unlocked, and published a submission made meanwhile as complete.
func TestWaitIdleDuringSubmit(t *testing.T) {
	_, _, device := newTestDevice(t)
	defer device.Release()
	q := device.Queue()

	const big = 16 << 20
	mk := func(size uint64, usage gputypes.BufferUsage) *Buffer {
		t.Helper()
		buf, err := device.CreateBuffer(&BufferDescriptor{Size: size, Usage: usage})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		t.Cleanup(buf.Release)
		return buf
	}
	a := mk(big, BufferUsageCopySrc|BufferUsageCopyDst)
	b := mk(big, BufferUsageCopySrc|BufferUsageCopyDst)
	rb := mk(256, BufferUsageCopyDst|BufferUsageMapRead)
	marker := bytes.Repeat([]byte{0xab}, 256)
	for _, buf := range []*Buffer{a, b} {
		if err := q.WriteBuffer(buf, 0, marker); err != nil {
			t.Fatalf("WriteBuffer: %v", err)
		}
	}
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}

	// busy copies back and forth between a and b, which keeps the marker at
	// the start of both, and with mark copies the marker into rb.
	busy := func(mark bool) uint64 {
		t.Helper()
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		for i := range 16 {
			if i%2 == 0 {
				enc.CopyBufferToBuffer(a, 0, b, 0, big)
			} else {
				enc.CopyBufferToBuffer(b, 0, a, 0, big)
			}
		}
		if mark {
			enc.CopyBufferToBuffer(a, 0, rb, 0, 256)
		}
		cb, err := enc.Finish()
		if err != nil {
			t.Fatalf("Finish: %v", err)
		}
		idx, err := q.Submit(cb)
		if err != nil {
			t.Fatalf("Submit: %v", err)
		}
		return idx
	}

	busy(false)
	done := make(chan error, 1)
	go func() { done <- device.WaitIdle() }()
	time.Sleep(2 * time.Millisecond) // the WaitIdle is under way behind the first submit
	idx := busy(true)
	if err := <-done; err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}

	if q.Poll() >= idx {
		pending, err := rb.MapAsync(MapModeRead, 0, 256)
		if err != nil {
			t.Fatalf("MapAsync: %v", err)
		}
		device.Poll(PollPoll)
		ready, err := pending.Status()
		if !ready || err != nil {
			t.Fatalf("submission %d reported complete, but a map waiting for it is not: ready %v, %v", idx, ready, err)
		}
		rng, err := rb.MappedRange(0, 256)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		got := append([]byte(nil), rng.Bytes()...)
		if err := rb.Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
		if !bytes.Equal(got, marker) {
			t.Errorf("submission %d reported complete (Poll %d) but its copy has not landed: %#x...", idx, q.Poll(), got[:8])
		}
	}

	// Submits and waits keep working: the throttle has no tokens missing
	// or to spare.
	for range 4 {
		busy(false)
		if err := device.WaitIdle(); err != nil {
			t.Fatalf("WaitIdle: %v", err)
		}
	}
}
