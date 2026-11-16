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

// TestFenceValuesPast32Bits moves a fence past 2³² from the CPU and from the
// queue. On 386 each 64-bit value is passed as two words and read back from
// EDX:EAX, so a wrong half shows up here.
func TestFenceValuesPast32Bits(t *testing.T) {
	dev, queue := openTestDevice(t)
	raw, err := dev.CreateFence()
	if err != nil {
		t.Fatalf("CreateFence: %v", err)
	}
	defer dev.DestroyFence(raw)
	fence := raw.(*Fence)

	const cpu = 0x1_0000_0005
	if err := fence.raw.Signal(cpu); err != nil {
		t.Fatalf("Signal: %v", err)
	}
	if got := fence.raw.GetCompletedValue(); got != cpu {
		t.Fatalf("GetCompletedValue after Signal(%#x) = %#x", uint64(cpu), got)
	}

	const gpu = 0x2_0000_0007
	if err := queue.raw.Signal(fence.raw, gpu); err != nil {
		t.Fatalf("queue Signal: %v", err)
	}
	if done, err := dev.Wait(fence, gpu, 5*time.Second); err != nil || !done {
		t.Fatalf("Wait(%#x) = %v, %v", uint64(gpu), done, err)
	}
	if got := fence.raw.GetCompletedValue(); got != gpu {
		t.Fatalf("GetCompletedValue after queue Signal(%#x) = %#x", uint64(gpu), got)
	}
	if done, err := dev.Wait(fence, gpu+1, 10*time.Millisecond); err != nil || done {
		t.Fatalf("Wait(%#x) with the fence at %#x = %v, %v; want a timeout", uint64(gpu+1), uint64(gpu), done, err)
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
