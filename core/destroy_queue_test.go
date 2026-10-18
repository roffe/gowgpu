//go:build !(js && wasm)

package core

import (
	"sync/atomic"
	"testing"
)

func TestDestroyQueue_TrackAndTriage(t *testing.T) {
	var destroyed atomic.Bool
	q := NewDestroyQueue()

	q.Defer(5, "buffer-A", func() { destroyed.Store(true) })

	// Triage with completedIndex < submissionIndex: nothing destroyed.
	q.Triage(4)
	if destroyed.Load() {
		t.Fatal("resource destroyed before its submission completed")
	}
	if q.Len() != 1 {
		t.Fatalf("want 1 pending, got %d", q.Len())
	}

	// Triage with completedIndex >= submissionIndex: destroyed.
	q.Triage(5)
	if !destroyed.Load() {
		t.Fatal("resource was not destroyed after submission completed")
	}
	if q.Len() != 0 {
		t.Fatalf("want 0 pending, got %d", q.Len())
	}
}

func TestDestroyQueue_TriagePartial(t *testing.T) {
	var destroyedA, destroyedB, destroyedC atomic.Bool
	q := NewDestroyQueue()

	q.Defer(1, "A", func() { destroyedA.Store(true) })
	q.Defer(3, "B", func() { destroyedB.Store(true) })
	q.Defer(5, "C", func() { destroyedC.Store(true) })

	// Only submission 1 and 3 completed.
	q.Triage(3)

	if !destroyedA.Load() {
		t.Error("A (index=1) should be destroyed")
	}
	if !destroyedB.Load() {
		t.Error("B (index=3) should be destroyed")
	}
	if destroyedC.Load() {
		t.Error("C (index=5) should NOT be destroyed yet")
	}
	if q.Len() != 1 {
		t.Fatalf("want 1 pending (C), got %d", q.Len())
	}

	// Now complete submission 5.
	q.Triage(5)
	if !destroyedC.Load() {
		t.Error("C should now be destroyed")
	}
	if q.Len() != 0 {
		t.Fatalf("want 0 pending, got %d", q.Len())
	}
}

func TestDestroyQueue_FlushAll(t *testing.T) {
	var count atomic.Int32
	q := NewDestroyQueue()

	q.Defer(10, "A", func() { count.Add(1) })
	q.Defer(20, "B", func() { count.Add(1) })
	q.Defer(30, "C", func() { count.Add(1) })

	q.FlushAll()

	if got := count.Load(); got != 3 {
		t.Fatalf("FlushAll: want 3 destroyed, got %d", got)
	}
	if q.Len() != 0 {
		t.Fatalf("want 0 pending after FlushAll, got %d", q.Len())
	}
}

func TestDestroyQueue_EmptyTriage(t *testing.T) {
	q := NewDestroyQueue()

	// Should not panic on empty queue.
	q.Triage(100)
	q.FlushAll()

	if q.Len() != 0 {
		t.Fatalf("want 0, got %d", q.Len())
	}
}

func TestDestroyQueue_MultipleSubmissions(t *testing.T) {
	order := make([]string, 0, 5)
	q := NewDestroyQueue()

	q.Defer(1, "first", func() { order = append(order, "first") })
	q.Defer(2, "second", func() { order = append(order, "second") })
	q.Defer(3, "third", func() { order = append(order, "third") })
	q.Defer(4, "fourth", func() { order = append(order, "fourth") })
	q.Defer(5, "fifth", func() { order = append(order, "fifth") })

	// Triage incrementally.
	q.Triage(2) // destroys first, second
	if len(order) != 2 || order[0] != "first" || order[1] != "second" {
		t.Fatalf("after Triage(2): want [first second], got %v", order)
	}

	q.Triage(4) // destroys third, fourth
	if len(order) != 4 || order[2] != "third" || order[3] != "fourth" {
		t.Fatalf("after Triage(4): want [... third fourth], got %v", order)
	}

	q.Triage(5) // destroys fifth
	if len(order) != 5 || order[4] != "fifth" {
		t.Fatalf("after Triage(5): want [... fifth], got %v", order)
	}

	if q.Len() != 0 {
		t.Fatalf("want 0 pending, got %d", q.Len())
	}
}

func TestDestroyQueue_DoubleFlush(t *testing.T) {
	var count atomic.Int32
	q := NewDestroyQueue()

	q.Defer(1, "buf", func() { count.Add(1) })
	q.FlushAll()
	q.FlushAll() // second flush should be no-op

	if got := count.Load(); got != 1 {
		t.Fatalf("want destroy called once, got %d", got)
	}
}

func TestDestroyQueue_DeferAfterTriage(t *testing.T) {
	var destroyedA, destroyedB atomic.Bool
	q := NewDestroyQueue()

	q.Defer(1, "A", func() { destroyedA.Store(true) })
	q.Triage(1) // destroys A

	q.Defer(2, "B", func() { destroyedB.Store(true) })
	q.Triage(2) // destroys B

	if !destroyedA.Load() || !destroyedB.Load() {
		t.Fatal("both resources should be destroyed")
	}
}

// Once Triage has dropped a submission's refs, its emptied list comes back
// from RecycledRefs, cleared, and not before.
func TestDestroyQueue_RecyclesRefLists(t *testing.T) {
	q := NewDestroyQueue()
	ref := NewResourceRef("buffer", nil)
	ref.Clone()
	refs := append(make([]*ResourceRef, 0, 16), ref)

	q.TrackSubmission(1, refs)
	if got := q.RecycledRefs(); got != nil {
		t.Fatalf("list came back before its submission completed: %v", got)
	}
	q.Triage(1)
	if got := ref.RefCount(); got != 1 {
		t.Fatalf("ref count after triage = %d, want 1", got)
	}
	got := q.RecycledRefs()
	if len(got) != 0 || cap(got) != 16 {
		t.Fatalf("recycled list len %d cap %d, want 0 and 16", len(got), cap(got))
	}
	if got[:1][0] != nil {
		t.Fatal("recycled list still holds the ref")
	}
	if again := q.RecycledRefs(); again != nil {
		t.Fatal("the same list came back twice")
	}

	// An empty list has nothing to keep alive and comes straight back.
	q.TrackSubmission(2, make([]*ResourceRef, 0, 8))
	if got := q.RecycledRefs(); cap(got) != 8 {
		t.Fatalf("empty list cap %d, want 8", cap(got))
	}
}

// Tracking a submission's refs and triaging it, the way Queue.Submit does
// every frame, allocates nothing once the lists are recycled.
func TestDestroyQueue_SteadyStateAllocations(t *testing.T) {
	q := NewDestroyQueue()
	ref := NewResourceRef("buffer", nil)
	var index uint64
	frame := func() {
		refs := q.RecycledRefs()
		if refs == nil {
			refs = make([]*ResourceRef, 0, 64)
		}
		for range 50 {
			ref.Clone()
			refs = append(refs, ref)
		}
		index++
		q.TrackSubmission(index, refs)
		q.Defer(index, "encoder", func() {})
		q.Triage(index)
	}
	frame()
	if allocs := testing.AllocsPerRun(100, frame); allocs > 0 {
		t.Fatalf("a tracked submission costs %.1f allocations, want 0", allocs)
	}
	if got := ref.RefCount(); got != 1 {
		t.Fatalf("ref count = %d, want 1", got)
	}
}
