//go:build js && wasm

package wgpu

import (
	"context"
	"errors"
	"runtime"
	"strings"
	"syscall/js"
	"testing"

	"github.com/roffe/gowgpu/internal/browser"
)

// fakeMapBuffer returns a GPUBuffer stand-in that keeps its pending map the
// way the WebGPU spec does: unmap and destroy reject it with an AbortError.
func fakeMapBuffer(t *testing.T) (*Buffer, js.Value) {
	t.Helper()
	ref := jsEval(`{
		size: 16,
		usage: 9,
		calls: [],
		pending: null,
		mapAsync() {
			this.calls.push("mapAsync")
			return new Promise((resolve, reject) => { this.pending = {resolve, reject} })
		},
		unmap() {
			this.calls.push("unmap")
			const p = this.pending
			this.pending = null
			if (p) p.reject(new DOMException("unmapped", "AbortError"))
		},
		destroy() { this.calls.push("destroy"); this.unmap() },
		resolvePending() { const p = this.pending; this.pending = null; p.resolve() },
	}`)
	return &Buffer{browser: browser.NewBuffer(ref), size: 16, usage: BufferUsageMapRead | BufferUsageCopyDst}, ref
}

func calls(ref js.Value) string {
	return js.Global().Get("JSON").Call("stringify", ref.Get("calls")).String()
}

func TestBrowserMapAsyncCallsMapAsyncBeforeUnmap(t *testing.T) {
	b, ref := fakeMapBuffer(t)
	first, err := b.MapAsync(MapModeRead, 0, 16)
	if err != nil {
		t.Fatalf("MapAsync: %v", err)
	}
	if err := b.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	second, err := b.MapAsync(MapModeRead, 0, 16)
	if err != nil {
		t.Fatalf("second MapAsync: %v", err)
	}
	if got := calls(ref); got != `["mapAsync","unmap","mapAsync"]` {
		t.Fatalf("calls %s, want mapAsync, unmap, mapAsync in the order they were made", got)
	}
	ref.Call("resolvePending")

	if err := first.Wait(context.Background()); !errors.Is(err, ErrMapCanceled) {
		t.Errorf("canceled map: %v, want ErrMapCanceled", err)
	}
	if err := second.Wait(context.Background()); err != nil {
		t.Errorf("second map: %v", err)
	}
	if b.MapState() != MapStateMapped {
		t.Errorf("state %v after the second map, want mapped", b.MapState())
	}
}

func TestBrowserReleasedBufferCancelsItsMap(t *testing.T) {
	b, ref := fakeMapBuffer(t)
	p, err := b.MapAsync(MapModeRead, 0, 16)
	if err != nil {
		t.Fatalf("MapAsync: %v", err)
	}
	b.Release()
	if got := calls(ref); got != `["mapAsync","destroy","unmap"]` {
		t.Fatalf("calls %s, want mapAsync before destroy", got)
	}
	if err := p.Wait(context.Background()); !errors.Is(err, ErrMapCanceled) {
		t.Fatalf("map of a released buffer: %v, want ErrMapCanceled", err)
	}
}

func TestBrowserReleasingAPendingMapLeavesNoGoroutine(t *testing.T) {
	b, ref := fakeMapBuffer(t)
	p, err := b.MapAsync(MapModeRead, 0, 16)
	if err != nil {
		t.Fatalf("MapAsync: %v", err)
	}
	p.Release()
	ref.Call("resolvePending")
	yieldToJS(t)

	buf := make([]byte, 1<<16)
	if stacks := string(buf[:runtime.Stack(buf, true)]); strings.Contains(stacks, "MapAsync") {
		t.Fatalf("a goroutine is still in MapAsync after its map settled:\n%s", stacks)
	}
	if b.MapState() != MapStateMapped {
		t.Fatalf("state %v, want mapped", b.MapState())
	}
}
