//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/hal"
)

// executedThenFailedQueue submits for real and then reports an error with
// the submission index, as a HAL does when something after handing the
// work to the GPU fails. Nothing completes until the test says so.
type executedThenFailedQueue struct {
	hal.Queue
	completed uint64
}

func (q *executedThenFailedQueue) Submit(cbs []hal.CommandBuffer) (uint64, error) {
	idx, err := q.Queue.Submit(cbs)
	if err != nil {
		return idx, err
	}
	return idx, errors.New("signalling an external fence failed")
}

func (q *executedThenFailedQueue) PollCompleted() uint64 { return q.completed }

// TestSubmitErrorWithAnIndexKeepsTheSubmission submits queued writes
// through a HAL that fails after executing them, and checks Submit reports
// the error with the index and keeps the writes in flight rather than
// cancelling them under the GPU.
func TestSubmitErrorWithAnIndexKeepsTheSubmission(t *testing.T) {
	device, _ := newValidatedDevice(t)
	queue := device.Queue()
	if queue.pending == nil || !queue.pending.usesBatching {
		t.Skip("the backend writes buffers at once, with no queued writes to submit")
	}
	buf, err := device.CreateBuffer(&BufferDescriptor{Label: "submit-error", Size: 256, Usage: BufferUsageCopyDst | BufferUsageCopySrc})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	if err := queue.WriteBuffer(buf, 0, make([]byte, 256)); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	failing := &executedThenFailedQueue{Queue: queue.hal}
	queue.hal = failing
	defer func() { queue.hal = failing.Queue }()

	idx, err := queue.Submit()
	if err == nil || idx == 0 {
		t.Fatalf("Submit = %d, %v; want the submission index with the error", idx, err)
	}
	queue.pending.mu.Lock()
	inflight := len(queue.pending.inflight)
	queue.pending.mu.Unlock()
	if inflight == 0 {
		t.Fatal("the queued writes were cancelled, though the GPU runs them")
	}
	if err := device.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
}
