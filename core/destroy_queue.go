//go:build !(js && wasm)

package core

import (
	"sync"
)

// DeferredDestroy represents a resource destruction that is deferred until the
// GPU completes the submission that was active when the resource was released.
type DeferredDestroy struct {
	submissionIndex uint64   // GPU submission that must complete before destroy
	destroyFn       func()   // HAL destroy callback
	releaser        Releaser // or this, see DeferRelease
	label           string   // debug label for logging
}

// Releaser is what DeferRelease releases once a submission completes. A
// pointer to a field of an object that exists anyway goes into it without
// allocating, where Defer's closure is an allocation every time.
type Releaser interface {
	ReleaseDeferred()
}

// run releases what d holds.
func (d *DeferredDestroy) run() {
	if d.releaser != nil {
		d.releaser.ReleaseDeferred()
		return
	}
	d.destroyFn()
}

// TrackedSubmission holds Clone'd ResourceRefs associated with a GPU submission.
// When the GPU completes this submission (index <= completedIndex), all refs
// are Drop()'d, decrementing their reference counts and potentially triggering
// destruction when no other owners remain.
//
// This is the Phase 2 per-command-buffer tracking mechanism: resources are
// Clone'd when used during encoding, transferred here on Submit, and Drop'd
// when the GPU finishes.
type TrackedSubmission struct {
	index uint64
	refs  []*ResourceRef
}

// DestroyQueue tracks in-flight GPU submissions and defers resource destruction
// until GPU completion. This prevents use-after-free when the application
// releases a resource while the GPU is still using it in a submitted command.
//
// Matches Rust wgpu-core's LifetimeTracker pattern where resource Drop only
// fires after triage_submissions confirms the fence has passed.
//
// Usage:
//  1. On resource Release(): call Defer(lastSubmissionIndex, destroyFn)
//  2. On Queue.Submit(): call Triage(PollCompleted()) to destroy completed resources
//  3. On Device.Release(): call FlushAll() to destroy everything
//
// Phase 2 adds TrackSubmission() for per-command-buffer resource tracking:
//
//	On Queue.Submit(): call TrackSubmission(subIdx, refs) with Clone'd refs
//	On Triage(): completed submissions' refs are Drop()'d
//
// Thread-safe for concurrent use.
type DestroyQueue struct {
	mu      sync.Mutex
	pending []DeferredDestroy
	tracked []TrackedSubmission

	// Triage's working lists and the ref lists of triaged submissions, kept
	// so that a steady stream of submissions allocates nothing here. A Triage
	// takes the working lists out under the lock and puts them back after,
	// so concurrent or nested calls just allocate their own.
	completedFns  []DeferredDestroy
	completedSubs []TrackedSubmission
	freeRefs      [][]*ResourceRef
}

// maxFreeRefs caps how many emptied ref lists the queue keeps for reuse.
const maxFreeRefs = 8

// NewDestroyQueue creates a new DestroyQueue.
func NewDestroyQueue() *DestroyQueue {
	return &DestroyQueue{}
}

// Defer schedules a resource for destruction after the GPU completes the
// submission identified by index. The destroyFn callback (typically calling
// halDevice.DestroyBuffer, DestroyTexture, etc.) will be invoked by a future
// Triage() call once completedIndex >= index.
//
// Parameters:
//   - index: the submission index that must complete before destruction
//   - label: debug label for the resource (for logging)
//   - fn: callback that performs the actual HAL destruction
func (q *DestroyQueue) Defer(index uint64, label string, fn func()) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.pending = append(q.pending, DeferredDestroy{
		submissionIndex: index,
		destroyFn:       fn,
		label:           label,
	})
}

// DeferRelease is Defer for r.ReleaseDeferred, without a closure.
func (q *DestroyQueue) DeferRelease(index uint64, label string, r Releaser) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.pending = append(q.pending, DeferredDestroy{
		submissionIndex: index,
		releaser:        r,
		label:           label,
	})
}

// TrackSubmission associates Clone'd ResourceRefs with a GPU submission.
// When Triage() confirms the submission has completed, all refs are Drop()'d.
//
// This is the Phase 2 per-command-buffer tracking entry point. Called by
// Queue.Submit() after HAL submit succeeds, once per command buffer.
//
// The queue takes ownership of refs: once Triage has dropped them, the
// emptied slice may come back from RecycledRefs for a new command encoder.
//
// Parameters:
//   - index: the submission index from HAL Queue.Submit()
//   - refs: Clone'd ResourceRef pointers from command encoding
func (q *DestroyQueue) TrackSubmission(index uint64, refs []*ResourceRef) {
	if cap(refs) == 0 {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	if len(refs) == 0 {
		// Nothing to keep alive: the list can go straight back.
		if len(q.freeRefs) < maxFreeRefs {
			q.freeRefs = append(q.freeRefs, refs)
		}
		return
	}
	q.tracked = append(q.tracked, TrackedSubmission{index: index, refs: refs})
}

// RecycledRefs returns an empty slice with room for refs, reusing the list of
// a submission Triage has dropped, or nil when there is none.
func (q *DestroyQueue) RecycledRefs() []*ResourceRef {
	q.mu.Lock()
	defer q.mu.Unlock()
	n := len(q.freeRefs)
	if n == 0 {
		return nil
	}
	refs := q.freeRefs[n-1]
	q.freeRefs[n-1] = nil
	q.freeRefs = q.freeRefs[:n-1]
	return refs
}

// Triage checks which deferred destructions can now proceed because their
// associated GPU submissions have completed (submissionIndex <= completedIndex).
// Resources whose submissions are still in-flight are retained.
//
// Also Drop()'s ResourceRefs from completed TrackedSubmissions (Phase 2).
//
// Callbacks and ref Drop()'s are executed OUTSIDE the mutex to prevent deadlocks
// when an onZero callback re-enters the DestroyQueue (ADR-056: ref.Drop() ->
// onZero -> dq.Defer() must not deadlock with Triage holding the lock).
//
// This should be called after each Queue.Submit() with the result of
// hal.Queue.PollCompleted().
func (q *DestroyQueue) Triage(completedIndex uint64) {
	q.mu.Lock()

	// Collect completed items under the lock; execute callbacks outside.
	completedDestroys, completedSubs := q.completedFns[:0], q.completedSubs[:0]
	q.completedFns, q.completedSubs = nil, nil

	// Triage deferred destroys (Phase 1).
	n := 0
	for i := range q.pending {
		if q.pending[i].submissionIndex <= completedIndex {
			completedDestroys = append(completedDestroys, q.pending[i])
		} else {
			q.pending[n] = q.pending[i]
			n++
		}
	}
	// Clear references in the tail to avoid retaining destroy closures.
	for i := n; i < len(q.pending); i++ {
		q.pending[i] = DeferredDestroy{}
	}
	q.pending = q.pending[:n]

	// Triage tracked submissions (Phase 2).
	tn := 0
	for i := range q.tracked {
		if q.tracked[i].index <= completedIndex {
			completedSubs = append(completedSubs, q.tracked[i])
		} else {
			q.tracked[tn] = q.tracked[i]
			tn++
		}
	}
	for i := tn; i < len(q.tracked); i++ {
		q.tracked[i] = TrackedSubmission{}
	}
	q.tracked = q.tracked[:tn]

	q.mu.Unlock()

	// Execute Phase 1 destroy callbacks outside the lock.
	for i := range completedDestroys {
		completedDestroys[i].run()
	}

	// Drop Phase 2 refs outside the lock.
	// onZero callbacks may call q.Defer() — safe because the lock is released.
	for _, sub := range completedSubs {
		for _, ref := range sub.refs {
			ref.Drop()
		}
		clear(sub.refs)
	}

	clear(completedDestroys)
	q.mu.Lock()
	for i := range completedSubs {
		if refs := completedSubs[i].refs; cap(refs) > 0 && len(q.freeRefs) < maxFreeRefs {
			q.freeRefs = append(q.freeRefs, refs[:0])
		}
		completedSubs[i] = TrackedSubmission{}
	}
	if q.completedFns == nil {
		q.completedFns = completedDestroys
	}
	if q.completedSubs == nil {
		q.completedSubs = completedSubs
	}
	q.mu.Unlock()
}

// FlushAll destroys all pending resources regardless of GPU completion status.
// Called during device shutdown when all GPU work is (or should be) complete.
// Also Drop()'s all tracked submission refs (Phase 2).
//
// Callbacks and ref Drop()'s are executed outside the mutex to prevent deadlocks
// when an onZero callback re-enters the DestroyQueue (ADR-056).
func (q *DestroyQueue) FlushAll() {
	q.mu.Lock()

	// Take ownership of slices and clear under the lock.
	destroys := q.pending
	tracked := q.tracked
	q.pending = nil
	q.tracked = nil

	q.mu.Unlock()

	// Execute Phase 1 destroy callbacks outside the lock.
	for i := range destroys {
		destroys[i].run()
	}

	// Drop Phase 2 refs outside the lock.
	for i := range tracked {
		for _, ref := range tracked[i].refs {
			ref.Drop()
		}
	}
}

// Len returns the number of pending deferred destructions. For testing only.
func (q *DestroyQueue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.pending)
}

// TrackedLen returns the number of tracked submissions. For testing only.
func (q *DestroyQueue) TrackedLen() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.tracked)
}
