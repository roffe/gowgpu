//go:build !(js && wasm)

package core

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// =============================================================================
// Multi-CB Encoder Tests
// Tests for Rust wgpu-core InnerCommandEncoder parity (command/mod.rs:530-738).
//
// An encoder starts out recording its first command buffer (the HAL encoder
// is recording when the encoder is made), so OpenPass closes that one first
// and every list below counts it.
// =============================================================================

// newTestEncoder creates a CoreCommandEncoder using the mock HAL device.
// Helper to reduce boilerplate across multi-CB tests.
func newTestEncoder(t *testing.T, label string) (*CoreCommandEncoder, *Device) {
	t.Helper()
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")
	encoder, err := device.CreateCommandEncoder(label)
	if err != nil {
		t.Fatalf("CreateCommandEncoder(%q) failed: %v", label, err)
	}
	return encoder, device
}

func TestMultiCBEncoder_SinglePass(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "single-pass")

	// Open a pass, close it -> cbList has 2 CBs: the recording the encoder
	// started with, closed by OpenPass, and the pass.
	if err := encoder.OpenPass("pass1"); err != nil {
		t.Fatalf("OpenPass failed: %v", err)
	}
	if !encoder.IsCBOpen() {
		t.Error("Expected cbListOpen to be true after OpenPass")
	}

	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB failed: %v", err)
	}
	if encoder.IsCBOpen() {
		t.Error("Expected cbListOpen to be false after CloseCB")
	}
	if encoder.CBListLen() != 2 {
		t.Errorf("Expected cbList length 2, got %d", encoder.CBListLen())
	}

	// Finish returns both CBs.
	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatalf("Finish failed: %v", err)
	}
	bufList := cmdBuf.HalBufferList()
	if len(bufList) != 2 {
		t.Errorf("Expected 2 CBs from Finish, got %d", len(bufList))
	}
}

func TestMultiCBEncoder_TwoPasses(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "two-passes")

	// First pass: open -> close
	if err := encoder.OpenPass("pass1"); err != nil {
		t.Fatalf("OpenPass(pass1) failed: %v", err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB(pass1) failed: %v", err)
	}

	// Second pass: open -> close
	if err := encoder.OpenPass("pass2"); err != nil {
		t.Fatalf("OpenPass(pass2) failed: %v", err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB(pass2) failed: %v", err)
	}

	if encoder.CBListLen() != 3 {
		t.Errorf("Expected cbList length 3, got %d", encoder.CBListLen())
	}

	// Finish returns all three CBs: the initial recording and both passes.
	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatalf("Finish failed: %v", err)
	}
	bufList := cmdBuf.HalBufferList()
	if len(bufList) != 3 {
		t.Errorf("Expected 3 CBs from Finish, got %d", len(bufList))
	}
}

func TestMultiCBEncoder_CloseAndSwap(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "close-and-swap")

	// Record a render pass CB.
	if err := encoder.OpenPass("render"); err != nil {
		t.Fatalf("OpenPass(render) failed: %v", err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB(render) failed: %v", err)
	}
	// cbList: [initial, render]

	// Now record a barrier CB and insert it BEFORE the render CB.
	if err := encoder.OpenPass("barrier"); err != nil {
		t.Fatalf("OpenPass(barrier) failed: %v", err)
	}
	if err := encoder.CloseAndSwap(); err != nil {
		t.Fatalf("CloseAndSwap failed: %v", err)
	}
	// cbList: [initial, barrier, render]

	if encoder.CBListLen() != 3 {
		t.Errorf("Expected cbList length 3, got %d", encoder.CBListLen())
	}

	// Finish returns all CBs in correct order.
	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatalf("Finish failed: %v", err)
	}
	bufList := cmdBuf.HalBufferList()
	if len(bufList) != 3 {
		t.Errorf("Expected 3 CBs from Finish, got %d", len(bufList))
	}
	// Both should be non-nil (mockCommandBuffer instances).
	for i, buf := range bufList {
		if buf == nil {
			t.Errorf("CB at index %d is nil", i)
		}
	}
}

func TestMultiCBEncoder_CloseAndSwap_ThreeCBs(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "swap-three")

	// Record: setup, render, then insert barrier before render.
	if err := encoder.OpenPass("setup"); err != nil {
		t.Fatalf("OpenPass(setup) failed: %v", err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB(setup) failed: %v", err)
	}
	// cbList: [initial, setup]

	if err := encoder.OpenPass("render"); err != nil {
		t.Fatalf("OpenPass(render) failed: %v", err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB(render) failed: %v", err)
	}
	// cbList: [initial, setup, render]

	if err := encoder.OpenPass("barrier"); err != nil {
		t.Fatalf("OpenPass(barrier) failed: %v", err)
	}
	if err := encoder.CloseAndSwap(); err != nil {
		t.Fatalf("CloseAndSwap failed: %v", err)
	}
	// cbList: [initial, setup, barrier, render]

	if encoder.CBListLen() != 4 {
		t.Errorf("Expected cbList length 4, got %d", encoder.CBListLen())
	}

	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatalf("Finish failed: %v", err)
	}
	bufList := cmdBuf.HalBufferList()
	if len(bufList) != 4 {
		t.Errorf("Expected 4 CBs from Finish, got %d", len(bufList))
	}
}

func TestMultiCBEncoder_CloseAndPushFront(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "push-front")

	// Record a render pass CB.
	if err := encoder.OpenPass("render"); err != nil {
		t.Fatalf("OpenPass(render) failed: %v", err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB(render) failed: %v", err)
	}
	// cbList: [initial, render]

	// Now record a transit CB and insert at front.
	if err := encoder.OpenPass("transit"); err != nil {
		t.Fatalf("OpenPass(transit) failed: %v", err)
	}
	if err := encoder.CloseAndPushFront(); err != nil {
		t.Fatalf("CloseAndPushFront failed: %v", err)
	}
	// cbList: [transit, initial, render]

	if encoder.CBListLen() != 3 {
		t.Errorf("Expected cbList length 3, got %d", encoder.CBListLen())
	}

	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatalf("Finish failed: %v", err)
	}
	bufList := cmdBuf.HalBufferList()
	if len(bufList) != 3 {
		t.Errorf("Expected 3 CBs from Finish, got %d", len(bufList))
	}
}

func TestMultiCBEncoder_CloseAndPushFront_MultiCB(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "push-front-multi")

	// Build: [initial, A, B], then push-front [C, initial, A, B]
	if err := encoder.OpenPass("A"); err != nil {
		t.Fatal(err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatal(err)
	}
	if err := encoder.OpenPass("B"); err != nil {
		t.Fatal(err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatal(err)
	}
	// cbList: [initial, A, B]

	if err := encoder.OpenPass("C"); err != nil {
		t.Fatal(err)
	}
	if err := encoder.CloseAndPushFront(); err != nil {
		t.Fatal(err)
	}
	// cbList: [C, initial, A, B]

	if encoder.CBListLen() != 4 {
		t.Errorf("Expected 4, got %d", encoder.CBListLen())
	}

	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if len(cmdBuf.HalBufferList()) != 4 {
		t.Errorf("Expected 4 CBs, got %d", len(cmdBuf.HalBufferList()))
	}
}

func TestMultiCBEncoder_CloseIfOpen_WhenOpen(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "close-if-open")

	if err := encoder.OpenPass("pass1"); err != nil {
		t.Fatalf("OpenPass failed: %v", err)
	}
	if !encoder.IsCBOpen() {
		t.Error("Expected open after OpenPass")
	}

	if err := encoder.CloseIfOpen(); err != nil {
		t.Fatalf("CloseIfOpen failed: %v", err)
	}
	if encoder.IsCBOpen() {
		t.Error("Expected closed after CloseIfOpen")
	}
	if encoder.CBListLen() != 2 {
		t.Errorf("Expected 2 CBs in list, got %d", encoder.CBListLen())
	}
}

func TestMultiCBEncoder_CloseIfOpen_WhenClosed(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "close-if-noop")

	// Close the recording the encoder started with; CloseIfOpen is then a
	// no-op.
	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB: %v", err)
	}
	if err := encoder.CloseIfOpen(); err != nil {
		t.Fatalf("CloseIfOpen should succeed when nothing open: %v", err)
	}
	if encoder.CBListLen() != 1 {
		t.Errorf("Expected 1 CB in list, got %d", encoder.CBListLen())
	}
}

func TestMultiCBEncoder_Finish_FlattensAll(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "flatten-all")

	// 3 passes -> Finish -> HalBufferList() returns 4: the initial recording
	// and the passes
	for i := 0; i < 3; i++ {
		if err := encoder.OpenPass("pass"); err != nil {
			t.Fatalf("OpenPass(%d) failed: %v", i, err)
		}
		if err := encoder.CloseCB(); err != nil {
			t.Fatalf("CloseCB(%d) failed: %v", i, err)
		}
	}

	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatalf("Finish failed: %v", err)
	}
	bufList := cmdBuf.HalBufferList()
	if len(bufList) != 4 {
		t.Errorf("Expected 4 CBs, got %d", len(bufList))
	}
	for i, buf := range bufList {
		if buf == nil {
			t.Errorf("CB at index %d is nil", i)
		}
	}
}

func TestMultiCBEncoder_Finish_WithOpenRecording(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "finish-open")

	// Open a pass but don't close it -> Finish should close it.
	if err := encoder.OpenPass("unclosed"); err != nil {
		t.Fatalf("OpenPass failed: %v", err)
	}

	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatalf("Finish failed: %v", err)
	}
	bufList := cmdBuf.HalBufferList()
	if len(bufList) != 2 {
		t.Errorf("Expected 2 CBs (initial, and the one auto-closed by Finish), got %d", len(bufList))
	}
}

func TestMultiCBEncoder_BackwardCompat_SingleCB(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "backward-compat")

	// Single-CB path (no OpenPass calls) -> halBuffer() returns the one CB.
	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatalf("Finish failed: %v", err)
	}

	// Raw() should return the single CB.
	if cmdBuf.Raw() == nil {
		t.Error("Raw() returned nil for single-CB path")
	}

	// HalBufferList() should return a single-element slice.
	bufList := cmdBuf.HalBufferList()
	if len(bufList) != 1 {
		t.Errorf("Expected 1 CB from HalBufferList, got %d", len(bufList))
	}
	if bufList[0] != cmdBuf.Raw() {
		t.Error("HalBufferList[0] should equal Raw()")
	}
}

func TestMultiCBEncoder_HalBufferList_MultiCB_RawConsistency(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "raw-consistency")

	// Multi-CB path -> Raw() returns first CB, HalBufferList returns all.
	if err := encoder.OpenPass("first"); err != nil {
		t.Fatal(err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatal(err)
	}
	if err := encoder.OpenPass("second"); err != nil {
		t.Fatal(err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatal(err)
	}

	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}

	bufList := cmdBuf.HalBufferList()
	if len(bufList) != 3 {
		t.Fatalf("Expected 3 CBs, got %d", len(bufList))
	}

	// Raw() should return the first CB in the list.
	if cmdBuf.Raw() != bufList[0] {
		t.Error("Raw() should return the first CB in the multi-CB list")
	}
}

// =============================================================================
// Error Cases
// =============================================================================

func TestMultiCBEncoder_CloseCB_NotOpen(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "close-not-open")

	// A second CloseCB, with nothing open any more, should fail.
	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB of the initial recording: %v", err)
	}
	err := encoder.CloseCB()
	if err == nil {
		t.Fatal("Expected error when closing without open pass")
	}
}

func TestMultiCBEncoder_CloseAndSwap_NotOpen(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "swap-not-open")

	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB of the initial recording: %v", err)
	}
	err := encoder.CloseAndSwap()
	if err == nil {
		t.Fatal("Expected error when CloseAndSwap without open pass")
	}
}

func TestMultiCBEncoder_CloseAndSwap_EmptyList(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "swap-empty")

	// The initial recording is open, but no CB is in the list yet ->
	// CloseAndSwap needs at least one.
	err := encoder.CloseAndSwap()
	if err == nil {
		t.Fatal("Expected error when CloseAndSwap with empty CB list")
	}
}

func TestMultiCBEncoder_CloseAndPushFront_NotOpen(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "front-not-open")

	if err := encoder.CloseCB(); err != nil {
		t.Fatalf("CloseCB of the initial recording: %v", err)
	}
	err := encoder.CloseAndPushFront()
	if err == nil {
		t.Fatal("Expected error when CloseAndPushFront without open pass")
	}
}

func TestMultiCBEncoder_OpenPass_WhenNotRecording(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "open-finished")

	// Finish the encoder first.
	_, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}

	// OpenPass should fail because encoder is Finished.
	err = encoder.OpenPass("late")
	if err == nil {
		t.Fatal("Expected error when OpenPass on finished encoder")
	}
}

func TestMultiCBEncoder_OpenPass_AutoClosePrevious(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "auto-close")

	// Open pass1 -> open pass2 (should auto-close pass1).
	if err := encoder.OpenPass("pass1"); err != nil {
		t.Fatal(err)
	}
	if err := encoder.OpenPass("pass2"); err != nil {
		t.Fatal(err)
	}

	// pass1 was auto-closed -> cbList has the initial recording and pass1,
	// pass2 is open.
	if encoder.CBListLen() != 2 {
		t.Errorf("Expected 2 CBs in list (initial, auto-closed pass1), got %d", encoder.CBListLen())
	}
	if !encoder.IsCBOpen() {
		t.Error("Expected pass2 to be open")
	}

	// Close pass2 -> cbList has 2 CBs.
	if err := encoder.CloseCB(); err != nil {
		t.Fatal(err)
	}
	if encoder.CBListLen() != 3 {
		t.Errorf("Expected 3 CBs, got %d", encoder.CBListLen())
	}
}

// =============================================================================
// CoreCommandBuffer Tests
// =============================================================================

func TestCoreCommandBuffer_HalBufferList_Nil(t *testing.T) {
	t.Parallel()

	// A CoreCommandBuffer with nil raw and nil halBuffers.
	cb := &CoreCommandBuffer{}
	list := cb.HalBufferList()
	if list != nil {
		t.Errorf("Expected nil for empty CoreCommandBuffer, got %d items", len(list))
	}
}

func TestCoreCommandBuffer_HalBufferList_SingleRaw(t *testing.T) {
	t.Parallel()

	// A CoreCommandBuffer with only raw set (single-CB path).
	cb := &CoreCommandBuffer{raw: mockCommandBuffer{}}
	list := cb.HalBufferList()
	if len(list) != 1 {
		t.Errorf("Expected 1 for single raw, got %d", len(list))
	}
}

func TestCoreCommandBuffer_HalBufferList_MultiCB(t *testing.T) {
	t.Parallel()

	// A CoreCommandBuffer with halBuffers set (multi-CB path).
	bufs := []hal.CommandBuffer{mockCommandBuffer{}, mockCommandBuffer{}, mockCommandBuffer{}}
	cb := &CoreCommandBuffer{
		raw:        bufs[0],
		halBuffers: bufs,
	}
	list := cb.HalBufferList()
	if len(list) != 3 {
		t.Errorf("Expected 3 for multi-CB, got %d", len(list))
	}
}

func TestCoreCommandBuffer_TextureScope_MultiCB(t *testing.T) {
	t.Parallel()
	encoder, _ := newTestEncoder(t, "scope-multi-cb")

	// Multi-CB recording should still carry the texture scope.
	if err := encoder.OpenPass("pass1"); err != nil {
		t.Fatal(err)
	}
	if err := encoder.CloseCB(); err != nil {
		t.Fatal(err)
	}

	cmdBuf, err := encoder.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if cmdBuf.TextureScope() == nil {
		t.Error("TextureScope should not be nil for multi-CB command buffer")
	}
}
