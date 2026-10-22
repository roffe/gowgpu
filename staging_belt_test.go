//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/hal/noop"
)

func createTestBelt(t *testing.T, chunkSize uint64) *stagingBelt {
	t.Helper()
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	return newStagingBelt(dev, q, chunkSize, 0, 0)
}

func TestStagingBelt_AllocateSubAllocates(t *testing.T) {
	belt := createTestBelt(t, 1024)
	defer belt.destroy()

	// Two small allocations should fit in the same chunk.
	data1 := make([]byte, 32)
	data2 := make([]byte, 64)

	alloc1, err := belt.allocate(32, data1)
	if err != nil {
		t.Fatalf("allocate 1: %v", err)
	}
	alloc2, err := belt.allocate(64, data2)
	if err != nil {
		t.Fatalf("allocate 2: %v", err)
	}

	// Both allocations should be from the same buffer (same chunk).
	if alloc1.buffer != alloc2.buffer {
		t.Error("expected both allocations from the same chunk buffer")
	}

	// Second allocation offset should be after first (aligned to 16).
	expectedOffset := alignUp64(32, stagingBeltDefaultAlignment)
	if alloc2.offset != expectedOffset {
		t.Errorf("expected alloc2.offset=%d, got %d", expectedOffset, alloc2.offset)
	}

	// Only 1 active chunk.
	s := belt.stats()
	active, free, closed := s.ActiveChunks, s.FreeChunks, s.ClosedSubs
	if active != 1 {
		t.Errorf("expected 1 active chunk, got %d", active)
	}
	if free != 0 {
		t.Errorf("expected 0 free chunks, got %d", free)
	}
	if closed != 0 {
		t.Errorf("expected 0 closed submissions, got %d", closed)
	}
}

func TestStagingBelt_AllocateRecyclesChunk(t *testing.T) {
	belt := createTestBelt(t, 1024)
	defer belt.destroy()

	// Allocate to create a chunk.
	data := make([]byte, 16)
	if _, err := belt.allocate(16, data); err != nil {
		t.Fatalf("allocate: %v", err)
	}

	// Finish moves active chunks to closed.
	belt.finish(1)

	s := belt.stats()
	if s.ActiveChunks != 0 {
		t.Errorf("after finish: expected 0 active, got %d", s.ActiveChunks)
	}
	if s.ClosedSubs != 1 {
		t.Errorf("after finish: expected 1 closed, got %d", s.ClosedSubs)
	}

	// Recall recycles closed chunks to free.
	belt.recall(1)

	s = belt.stats()
	if s.FreeChunks != 1 {
		t.Errorf("after recall: expected 1 free, got %d", s.FreeChunks)
	}
	if s.ClosedSubs != 0 {
		t.Errorf("after recall: expected 0 closed, got %d", s.ClosedSubs)
	}

	// Next allocation should reuse the free chunk (no new chunk created).
	alloc, err := belt.allocate(16, data)
	if err != nil {
		t.Fatalf("allocate after recall: %v", err)
	}
	if alloc.buffer == nil {
		t.Error("expected non-nil buffer from recycled chunk")
	}

	s = belt.stats()
	if s.ActiveChunks != 1 {
		t.Errorf("after reuse: expected 1 active, got %d", s.ActiveChunks)
	}
	if s.FreeChunks != 0 {
		t.Errorf("after reuse: expected 0 free (chunk was recycled), got %d", s.FreeChunks)
	}
}

func TestStagingBelt_AllocateOversized(t *testing.T) {
	chunkSize := uint64(256)
	belt := createTestBelt(t, chunkSize)
	defer belt.destroy()

	// Allocate more than chunkSize — should create a one-off buffer.
	oversizedData := make([]byte, chunkSize+100)
	alloc, err := belt.allocate(chunkSize+100, oversizedData)
	if err != nil {
		t.Fatalf("allocate oversized: %v", err)
	}

	if alloc.buffer == nil {
		t.Error("expected non-nil buffer for oversized allocation")
	}
	if alloc.offset != 0 {
		t.Errorf("expected offset=0 for oversized, got %d", alloc.offset)
	}

	// Oversized goes to belt.oversized, not activeChunks.
	s := belt.stats()
	active := s.ActiveChunks
	if active != 0 {
		t.Errorf("expected 0 active chunks for oversized alloc, got %d", active)
	}
	if len(belt.oversized) != 1 {
		t.Errorf("expected 1 oversized buffer, got %d", len(belt.oversized))
	}

	// finish() moves oversized to closedSubmissions (belt owns them).
	belt.finish(1)
	s = belt.stats()
	if s.ClosedSubs != 1 {
		t.Errorf("expected 1 closed submission after finish, got %d", s.ClosedSubs)
	}
}

func TestStagingBelt_DiscardPathsRetireOnlyUnsubmittedBatches(t *testing.T) {
	t.Run("active allocations", func(t *testing.T) {
		belt := createTestBelt(t, 64)
		defer belt.destroy()

		if _, err := belt.allocate(16, make([]byte, 16)); err != nil {
			t.Fatalf("allocate regular chunk: %v", err)
		}
		if _, err := belt.allocate(96, make([]byte, 96)); err != nil {
			t.Fatalf("allocate oversized buffer: %v", err)
		}
		belt.discardActive()
		stats := belt.stats()
		if stats.ActiveChunks != 0 || stats.FreeChunks != 1 || len(belt.oversized) != 0 || len(belt.chunkedAllocs) != 0 {
			t.Fatalf("discarded active state = %+v oversized=%d chunked=%d", stats, len(belt.oversized), len(belt.chunkedAllocs))
		}
	})

	t.Run("closed submissions", func(t *testing.T) {
		belt := createTestBelt(t, 64)
		defer belt.destroy()
		belt.discardLastClosed()

		if _, err := belt.allocate(16, make([]byte, 16)); err != nil {
			t.Fatalf("allocate placeholder batch: %v", err)
		}
		if _, err := belt.allocate(96, make([]byte, 96)); err != nil {
			t.Fatalf("allocate placeholder oversized buffer: %v", err)
		}
		belt.finish(unsubmittedIndex)
		belt.discardLastClosed()
		stats := belt.stats()
		if stats.ClosedSubs != 0 || stats.FreeChunks != 1 {
			t.Fatalf("rolled-back placeholder state = %+v", stats)
		}

		if _, err := belt.allocate(16, make([]byte, 16)); err != nil {
			t.Fatalf("allocate submitted batch: %v", err)
		}
		belt.finish(7)
		belt.discardLastClosed()
		if stats := belt.stats(); stats.ClosedSubs != 1 {
			t.Fatalf("submitted batch was discarded: %+v", stats)
		}
	})
}

// A batch finished ahead of its submit is not recalled before the submit
// names its index: a Poll from another goroutine could otherwise recycle
// its chunks under the copies.
func TestStagingBelt_RecallSkipsUnsubmittedBatch(t *testing.T) {
	belt := createTestBelt(t, 1024)
	defer belt.destroy()

	if _, err := belt.allocate(32, make([]byte, 32)); err != nil {
		t.Fatalf("allocate: %v", err)
	}
	belt.finish(unsubmittedIndex)
	belt.recall(100)
	if s := belt.stats(); s.ClosedSubs != 1 || s.FreeChunks != 0 {
		t.Fatalf("recalled before the submit: %+v", s)
	}
	belt.setLastSubmissionIndex(5)
	belt.recall(100)
	if s := belt.stats(); s.ClosedSubs != 0 || s.FreeChunks != 1 {
		t.Fatalf("not recalled after the submit: %+v", s)
	}
}

func TestStagingBelt_FinishAndRecall(t *testing.T) {
	belt := createTestBelt(t, 1024)
	defer belt.destroy()

	data := make([]byte, 32)

	// Submission 1: allocate, finish.
	if _, err := belt.allocate(32, data); err != nil {
		t.Fatalf("alloc 1: %v", err)
	}
	belt.finish(unsubmittedIndex)
	belt.setLastSubmissionIndex(10)

	// Submission 2: allocate (new chunk since previous was closed), finish.
	if _, err := belt.allocate(32, data); err != nil {
		t.Fatalf("alloc 2: %v", err)
	}
	belt.finish(unsubmittedIndex)
	belt.setLastSubmissionIndex(20)

	s := belt.stats()
	active, free, closed := s.ActiveChunks, s.FreeChunks, s.ClosedSubs
	if active != 0 {
		t.Errorf("expected 0 active, got %d", active)
	}
	if free != 0 {
		t.Errorf("expected 0 free, got %d", free)
	}
	if closed != 2 {
		t.Errorf("expected 2 closed, got %d", closed)
	}

	// Recall submission 1 only (completedIndex=10).
	belt.recall(10)

	s = belt.stats()
	free, closed = s.FreeChunks, s.ClosedSubs
	if free != 1 {
		t.Errorf("after recall(10): expected 1 free, got %d", free)
	}
	if closed != 1 {
		t.Errorf("after recall(10): expected 1 closed, got %d", closed)
	}

	// Recall submission 2 (completedIndex=20).
	belt.recall(20)

	s = belt.stats()
	free, closed = s.FreeChunks, s.ClosedSubs
	if free != 2 {
		t.Errorf("after recall(20): expected 2 free, got %d", free)
	}
	if closed != 0 {
		t.Errorf("after recall(20): expected 0 closed, got %d", closed)
	}
}

func TestStagingBelt_ZeroAllocSteadyState(t *testing.T) {
	belt := createTestBelt(t, 1024)
	defer belt.destroy()

	data := make([]byte, 64)

	// Warmup: create one chunk.
	if _, err := belt.allocate(64, data); err != nil {
		t.Fatalf("warmup alloc: %v", err)
	}
	belt.finish(0)
	belt.setLastSubmissionIndex(1)
	belt.recall(1)

	// After warmup, we should have 1 free chunk.
	s := belt.stats()
	free := s.FreeChunks
	if free != 1 {
		t.Fatalf("expected 1 free chunk after warmup, got %d", free)
	}

	// Steady state: allocate + finish + recall should reuse the same chunk.
	// No new chunks should be created.
	for i := uint64(0); i < 100; i++ {
		if _, err := belt.allocate(64, data); err != nil {
			t.Fatalf("steady state alloc %d: %v", i, err)
		}

		belt.finish(0)
		belt.setLastSubmissionIndex(100 + i)
		belt.recall(100 + i)
	}

	s = belt.stats()
	active, free, closed, totalAllocated := s.ActiveChunks, s.FreeChunks, s.ClosedSubs, s.TotalAllocated
	if active != 0 {
		t.Errorf("steady state: expected 0 active, got %d", active)
	}
	if free != 1 {
		t.Errorf("steady state: expected 1 free chunk (no new allocs), got %d", free)
	}
	if closed != 0 {
		t.Errorf("steady state: expected 0 closed, got %d", closed)
	}
	// Total allocated should be exactly 1 chunk (1024 bytes).
	if totalAllocated != 1024 {
		t.Errorf("steady state: expected totalAllocated=1024, got %d", totalAllocated)
	}
}

func TestStagingBelt_AlignUp64(t *testing.T) {
	tests := []struct {
		name      string
		n         uint64
		alignment uint64
		want      uint64
	}{
		{"already aligned", 16, 16, 16},
		{"needs alignment", 10, 16, 16},
		{"zero", 0, 16, 0},
		{"alignment 0", 42, 0, 42},
		{"alignment 1", 42, 1, 42},
		{"large value", 1000, 256, 1024},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := alignUp64(tt.n, tt.alignment)
			if got != tt.want {
				t.Errorf("alignUp64(%d, %d) = %d, want %d", tt.n, tt.alignment, got, tt.want)
			}
		})
	}
}

func TestStagingBelt_DefaultChunkSize(t *testing.T) {
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	belt := newStagingBelt(dev, q, 0, 0, 0) // 0 = defaults
	defer belt.destroy()

	if belt.chunkSize != stagingBeltDefaultChunkSize {
		t.Errorf("expected default chunkSize=%d, got %d", stagingBeltDefaultChunkSize, belt.chunkSize)
	}
}

func TestStagingBelt_EmptyAllocate(t *testing.T) {
	belt := createTestBelt(t, 1024)
	defer belt.destroy()

	alloc, err := belt.allocate(0, nil)
	if err != nil {
		t.Fatalf("allocate 0 bytes: %v", err)
	}
	if alloc.buffer != nil {
		t.Error("expected nil buffer for 0-byte allocation")
	}

	s := belt.stats()
	active := s.ActiveChunks
	if active != 0 {
		t.Errorf("expected 0 active chunks after 0-byte alloc, got %d", active)
	}
}

func TestStagingBelt_FinishNoWork(t *testing.T) {
	belt := createTestBelt(t, 1024)
	defer belt.destroy()

	belt.finish(1)
	s := belt.stats()
	if s.ClosedSubs != 0 {
		t.Errorf("expected 0 closed submissions from empty finish, got %d", s.ClosedSubs)
	}
}

func TestStagingBelt_ChunkedAllocation(t *testing.T) {
	// Create a belt with maxStagingBufferSize = 100 bytes.
	// An oversized write of 250 bytes should be split into 3 chunks:
	// 100 + 100 + 50.
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	belt := newStagingBelt(dev, q, 64, 0, 100) // chunkSize=64, maxStaging=100
	defer belt.destroy()

	data := make([]byte, 250)
	for i := range data {
		data[i] = byte(i % 256)
	}

	alloc, err := belt.allocate(250, data)
	if err != nil {
		t.Fatalf("allocate chunked: %v", err)
	}

	// First chunk should be returned as primary allocation.
	if alloc.buffer == nil {
		t.Fatal("expected non-nil buffer from chunked allocation")
	}
	if alloc.offset != 0 {
		t.Errorf("expected offset=0 for chunked allocation, got %d", alloc.offset)
	}

	// Should have 3 chunks in chunkedAllocs.
	if len(belt.chunkedAllocs) != 3 {
		t.Fatalf("expected 3 chunked allocs, got %d", len(belt.chunkedAllocs))
	}

	// Verify chunk sizes.
	expectedSizes := []uint64{100, 100, 50}
	for i, ca := range belt.chunkedAllocs {
		if ca.size != expectedSizes[i] {
			t.Errorf("chunk %d: expected size %d, got %d", i, expectedSizes[i], ca.size)
		}
		if ca.buffer == nil {
			t.Errorf("chunk %d: expected non-nil buffer", i)
		}
	}

	// All chunks should be in oversized (not activeChunks).
	if len(belt.oversized) != 3 {
		t.Errorf("expected 3 oversized buffers, got %d", len(belt.oversized))
	}
	s := belt.stats()
	if s.ActiveChunks != 0 {
		t.Errorf("expected 0 active chunks, got %d", s.ActiveChunks)
	}
}

func TestStagingBelt_ChunkedSingleChunk(t *testing.T) {
	// When oversized write fits in maxStagingBufferSize, should NOT chunk.
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	belt := newStagingBelt(dev, q, 64, 0, 200) // chunkSize=64, maxStaging=200
	defer belt.destroy()

	data := make([]byte, 150)
	alloc, err := belt.allocate(150, data)
	if err != nil {
		t.Fatalf("allocate: %v", err)
	}

	if alloc.buffer == nil {
		t.Fatal("expected non-nil buffer")
	}

	// Should NOT have chunked — single oversized buffer.
	if len(belt.chunkedAllocs) != 0 {
		t.Errorf("expected 0 chunked allocs (not chunked), got %d", len(belt.chunkedAllocs))
	}
	if len(belt.oversized) != 1 {
		t.Errorf("expected 1 oversized buffer, got %d", len(belt.oversized))
	}
}

func TestStagingBelt_ChunkedExactMultiple(t *testing.T) {
	// Write is exact multiple of maxStagingBufferSize.
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	belt := newStagingBelt(dev, q, 64, 0, 100) // chunkSize=64, maxStaging=100
	defer belt.destroy()

	data := make([]byte, 300)
	_, err := belt.allocate(300, data)
	if err != nil {
		t.Fatalf("allocate: %v", err)
	}

	// Should have 3 chunks of 100 each.
	if len(belt.chunkedAllocs) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(belt.chunkedAllocs))
	}
	for i, ca := range belt.chunkedAllocs {
		if ca.size != 100 {
			t.Errorf("chunk %d: expected size 100, got %d", i, ca.size)
		}
	}
}

// The chunks of a chunked allocation are the caller's until the next
// allocate, which starts without them: a write after a chunked one used to
// copy from the stale chunks too.
func TestStagingBelt_AllocateResetsChunkedAllocs(t *testing.T) {
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	belt := newStagingBelt(dev, q, 64, 0, 100)
	defer belt.destroy()

	if _, err := belt.allocate(300, make([]byte, 300)); err != nil {
		t.Fatalf("allocate chunked: %v", err)
	}
	if len(belt.chunkedAllocs) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(belt.chunkedAllocs))
	}
	if _, err := belt.allocate(32, make([]byte, 32)); err != nil {
		t.Fatalf("allocate: %v", err)
	}
	if len(belt.chunkedAllocs) != 0 {
		t.Errorf("expected no chunks left from the earlier allocation, got %d", len(belt.chunkedAllocs))
	}
}

// A chunked allocation whose data stops short of its size, as a texture
// write's can, writes the data there is and no more.
func TestStagingBelt_ChunkedShortData(t *testing.T) {
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	belt := newStagingBelt(dev, q, 64, 0, 100)
	defer belt.destroy()

	if _, err := belt.allocate(300, make([]byte, 150)); err != nil {
		t.Fatalf("allocate chunked: %v", err)
	}
	if len(belt.chunkedAllocs) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(belt.chunkedAllocs))
	}
	if q.writeBufferCalls != 3 || len(q.lastWriteData) != 0 {
		t.Errorf("got %d writes, the last of %d bytes; want 3, the last empty", q.writeBufferCalls, len(q.lastWriteData))
	}
}

func TestStagingBelt_MaxStagingBufferSizeDefault(t *testing.T) {
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	belt := newStagingBelt(dev, q, 0, 0, 0) // all defaults
	defer belt.destroy()

	if belt.maxStagingBufferSize != stagingBeltMaxOversizedSize {
		t.Errorf("expected default maxStagingBufferSize=%d, got %d",
			stagingBeltMaxOversizedSize, belt.maxStagingBufferSize)
	}
}

func TestStagingBelt_ChunkedDataIntegrity(t *testing.T) {
	// Verify that chunked allocation splits data correctly across multiple
	// staging buffers. This is the core of BUG-VK-001 Fix 2: writes larger
	// than maxStagingBufferSize must be split, and each chunk must contain
	// the correct slice of the source data.
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	belt := newStagingBelt(dev, q, 32, 0, 80) // chunkSize=32, maxStaging=80
	defer belt.destroy()

	// Write 200 bytes of recognizable data (should produce 3 chunks: 80+80+40).
	data := make([]byte, 200)
	for i := range data {
		data[i] = byte(i)
	}

	alloc, err := belt.allocate(200, data)
	if err != nil {
		t.Fatalf("allocate: %v", err)
	}
	if alloc.buffer == nil {
		t.Fatal("expected non-nil buffer")
	}

	// Verify chunks.
	if len(belt.chunkedAllocs) != 3 {
		t.Fatalf("expected 3 chunks, got %d", len(belt.chunkedAllocs))
	}

	wantSizes := []uint64{80, 80, 40}
	wantDstOffsets := []uint64{0, 80, 160} // cumulative offsets for CopyBufferToBuffer
	dstOffset := uint64(0)
	for i, ca := range belt.chunkedAllocs {
		if ca.size != wantSizes[i] {
			t.Errorf("chunk %d: size=%d, want %d", i, ca.size, wantSizes[i])
		}
		if dstOffset != wantDstOffsets[i] {
			t.Errorf("chunk %d: dst offset=%d, want %d", i, dstOffset, wantDstOffsets[i])
		}
		dstOffset += ca.size
	}

	// Total data transferred must equal original size.
	if dstOffset != 200 {
		t.Errorf("total chunked size=%d, want 200", dstOffset)
	}
}

func TestStagingBelt_SmallMaxStagingBufferSize(t *testing.T) {
	// Simulate a device with very small maxMemoryAllocationSize (e.g., 16 bytes).
	// This exercises the extreme case where even normal-sized writes must be chunked.
	dev := &noop.Device{}
	q := &mockBatchingQueue{}
	belt := newStagingBelt(dev, q, 8, 0, 16) // chunkSize=8, maxStaging=16
	defer belt.destroy()

	// Write 50 bytes. Should produce ceil(50/16)=4 chunks: 16+16+16+2.
	data := make([]byte, 50)
	_, err := belt.allocate(50, data)
	if err != nil {
		t.Fatalf("allocate: %v", err)
	}

	if len(belt.chunkedAllocs) != 4 {
		t.Fatalf("expected 4 chunks, got %d", len(belt.chunkedAllocs))
	}

	wantSizes := []uint64{16, 16, 16, 2}
	for i, ca := range belt.chunkedAllocs {
		if ca.size != wantSizes[i] {
			t.Errorf("chunk %d: size=%d, want %d", i, ca.size, wantSizes[i])
		}
	}
}

// A write too big for a chunk reuses the one-off buffer of an earlier write
// of about the same size once the GPU is done with it, instead of creating
// and destroying one every frame.
func TestStagingBelt_OversizedBufferReused(t *testing.T) {
	const chunkSize = 256
	belt := createTestBelt(t, chunkSize)
	defer belt.destroy()

	first, err := belt.allocate(1000, make([]byte, 1000))
	if err != nil {
		t.Fatal(err)
	}
	belt.finish(0)
	belt.setLastSubmissionIndex(1)

	// Still in flight: a second write gets a buffer of its own.
	second, err := belt.allocate(1000, make([]byte, 1000))
	if err != nil {
		t.Fatal(err)
	}
	if second.buffer == first.buffer {
		t.Fatal("reused a staging buffer the GPU may still read")
	}
	belt.finish(0)
	belt.setLastSubmissionIndex(2)

	belt.recall(2)
	if got := len(belt.freeOversized); got != 2 {
		t.Fatalf("kept %d one-off buffers after recall, want 2", got)
	}
	third, err := belt.allocate(900, make([]byte, 900))
	if err != nil {
		t.Fatal(err)
	}
	if third.buffer != first.buffer && third.buffer != second.buffer {
		t.Fatal("a 900 byte write did not reuse a kept 1024 byte buffer")
	}
	// Under half the kept buffers' size: a new one, not a 1024 byte one.
	small, err := belt.allocate(300, make([]byte, 300))
	if err != nil {
		t.Fatal(err)
	}
	if small.buffer == first.buffer || small.buffer == second.buffer {
		t.Fatal("a 300 byte write took a 1024 byte buffer")
	}
}

// One-off buffers past maxPooledOversizedSize are destroyed after use, not
// kept.
func TestStagingBelt_HugeOversizedBufferNotKept(t *testing.T) {
	belt := createTestBelt(t, 256)
	defer belt.destroy()
	size := uint64(maxPooledOversizedSize + 1)
	if _, err := belt.allocate(size, make([]byte, size)); err != nil {
		t.Fatal(err)
	}
	belt.finish(0)
	belt.setLastSubmissionIndex(1)
	belt.recall(1)
	if got := len(belt.freeOversized); got != 0 {
		t.Fatalf("kept %d huge one-off buffers, want 0", got)
	}
}

// A frame's oversized write, with finish and recall, allocates nothing once
// the belt has warmed up.
func TestStagingBelt_OversizedSteadyStateAllocations(t *testing.T) {
	// noop's queue, not the recording mock: it writes without allocating.
	belt := newStagingBelt(&noop.Device{}, &noop.Queue{}, 256, 0, 0)
	defer belt.destroy()
	data := make([]byte, 1000)
	small := make([]byte, 64)
	var index uint64
	frame := func() {
		if _, err := belt.allocate(1000, data); err != nil {
			t.Fatal(err)
		}
		if _, err := belt.allocate(64, small); err != nil {
			t.Fatal(err)
		}
		index++
		belt.finish(0)
		belt.setLastSubmissionIndex(index)
		belt.recall(index)
	}
	for range 3 {
		frame()
	}
	if allocs := testing.AllocsPerRun(50, frame); allocs > 0 {
		t.Fatalf("a frame costs %.1f allocations, want 0", allocs)
	}
}
