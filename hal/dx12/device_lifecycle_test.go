// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package dx12

import (
	"sync"
	"testing"
	"time"
)

func TestWaitForGPUAfterDeviceCleanupIsNoop(t *testing.T) {
	if err := (&Device{}).waitForGPU(); err != nil {
		t.Fatalf("waitForGPU on cleaned device: %v", err)
	}
}

// TestFenceWaitAfterATimedOutWait times a wait out, lets the fence reach
// the value it waited for, and checks a wait for the next value doesn't
// return early.
func TestFenceWaitAfterATimedOutWait(t *testing.T) {
	dev, _ := openTestDevice(t)
	raw, err := dev.CreateFence()
	if err != nil {
		t.Fatalf("CreateFence: %v", err)
	}
	defer dev.DestroyFence(raw)
	fence := raw.(*Fence)
	if done, err := dev.Wait(fence, 1, 10*time.Millisecond); err != nil || done {
		t.Fatalf("Wait(1) before the signal = %v, %v; want a timeout", done, err)
	}
	if err := fence.raw.Signal(1); err != nil {
		t.Fatalf("Signal: %v", err)
	}
	if done, err := dev.Wait(fence, 2, 50*time.Millisecond); err != nil || done {
		t.Fatalf("Wait(2) with the fence at 1 = %v, %v; want a timeout", done, err)
	}
}

// TestFrameIndexAcrossGoroutines advances frames on one goroutine while
// another recycles frame slots, as two surfaces of one device presenting
// from two goroutines do. It needs -race to see anything.
func TestFrameIndexAcrossGoroutines(t *testing.T) {
	d := &Device{}
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for range 1000 {
			d.advanceFrame()
		}
	}()
	for range 1000 {
		if err := d.recycleFrameSlot(); err != nil {
			t.Fatalf("recycleFrameSlot: %v", err)
		}
	}
	wg.Wait()
}
