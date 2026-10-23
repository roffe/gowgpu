//go:build !(js && wasm)

package core

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// =============================================================================
// deviceMapTracker / RegisterPendingMap / PollMaps
// =============================================================================

func TestRegisterPendingMap_Basic(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	buf := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf")
	err := buf.BeginMap(MapModeInternalRead, 0, 256)
	if err != nil {
		t.Fatalf("BeginMap failed: %v", err)
	}

	device.RegisterPendingMap(1, buf)

	if !device.HasPendingMaps() {
		t.Error("HasPendingMaps() should return true after RegisterPendingMap")
	}
	if device.MaxKnownSubmissionIndex() != 1 {
		t.Errorf("MaxKnownSubmissionIndex() = %d, want 1", device.MaxKnownSubmissionIndex())
	}
}

func TestPollMaps_ResolvesCompletedSubmissions(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	buf := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf")
	err := buf.BeginMap(MapModeInternalRead, 0, 256)
	if err != nil {
		t.Fatalf("BeginMap failed: %v", err)
	}

	device.RegisterPendingMap(1, buf)

	// Poll with completedIdx >= 1 should resolve the map.
	didWork := device.PollMaps(1)
	if !didWork {
		t.Error("PollMaps(1) should have resolved pending maps")
	}

	// Buffer should now be Mapped (ResolveMap was called by PollMaps).
	if buf.CurrentMapState() != BufferMapStateMapped {
		t.Errorf("buffer state = %v after PollMaps, want Mapped", buf.CurrentMapState())
	}

	// No more pending maps.
	if device.HasPendingMaps() {
		t.Error("HasPendingMaps() should return false after all maps resolved")
	}
}

func TestPollMaps_SkipsFutureSubmissions(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	buf := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf")
	err := buf.BeginMap(MapModeInternalRead, 0, 256)
	if err != nil {
		t.Fatalf("BeginMap failed: %v", err)
	}

	device.RegisterPendingMap(5, buf)

	// Poll with completedIdx < 5 should NOT resolve the map.
	didWork := device.PollMaps(3)
	if didWork {
		t.Error("PollMaps(3) should not resolve submission 5")
	}
	if buf.CurrentMapState() != BufferMapStatePending {
		t.Errorf("buffer state = %v, want Pending (not resolved yet)", buf.CurrentMapState())
	}
	if !device.HasPendingMaps() {
		t.Error("HasPendingMaps() should still return true")
	}
}

func TestPollMaps_MultipleSubmissions(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	buf1 := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf1")
	buf2 := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf2")
	buf3 := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf3")

	_ = buf1.BeginMap(MapModeInternalRead, 0, 256)
	_ = buf2.BeginMap(MapModeInternalRead, 0, 256)
	_ = buf3.BeginMap(MapModeInternalRead, 0, 256)

	device.RegisterPendingMap(1, buf1)
	device.RegisterPendingMap(2, buf2)
	device.RegisterPendingMap(3, buf3)

	if device.MaxKnownSubmissionIndex() != 3 {
		t.Errorf("MaxKnownSubmissionIndex() = %d, want 3", device.MaxKnownSubmissionIndex())
	}

	// Resolve only up to submission 2.
	device.PollMaps(2)

	if buf1.CurrentMapState() != BufferMapStateMapped {
		t.Error("buf1 should be Mapped (submission 1 completed)")
	}
	if buf2.CurrentMapState() != BufferMapStateMapped {
		t.Error("buf2 should be Mapped (submission 2 completed)")
	}
	if buf3.CurrentMapState() != BufferMapStatePending {
		t.Error("buf3 should still be Pending (submission 3 not completed)")
	}
}

func TestPollMaps_NilTrackerNoWork(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	// No maps registered -- PollMaps should be a no-op.
	didWork := device.PollMaps(100)
	if didWork {
		t.Error("PollMaps on device with no pending maps should return false")
	}
}

func TestHasPendingMaps_NoTracker(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	if device.HasPendingMaps() {
		t.Error("HasPendingMaps() should return false for new device")
	}
}

func TestMaxKnownSubmissionIndex_NoTracker(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	if device.MaxKnownSubmissionIndex() != 0 {
		t.Errorf("MaxKnownSubmissionIndex() = %d, want 0", device.MaxKnownSubmissionIndex())
	}
}

func TestRegisterPendingMap_FreeListRecycling(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	// Cycle 1: register and poll to recycle the slice.
	buf1 := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf1")
	_ = buf1.BeginMap(MapModeInternalRead, 0, 256)
	device.RegisterPendingMap(1, buf1)
	device.PollMaps(1)

	// Cycle 2: register at a different index -- should reuse the recycled slice.
	buf2 := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf2")
	_ = buf2.BeginMap(MapModeInternalRead, 0, 256)
	device.RegisterPendingMap(2, buf2)
	device.PollMaps(2)

	if buf2.CurrentMapState() != BufferMapStateMapped {
		t.Errorf("buf2 state = %v, want Mapped (recycled slice path)", buf2.CurrentMapState())
	}
}

func TestPollMaps_NilBufferSkipped(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	// Manually register a nil buffer -- should not panic.
	device.RegisterPendingMap(1, nil)
	didWork := device.PollMaps(1)
	if !didWork {
		t.Error("PollMaps should report work even for nil buffer (bucket drained)")
	}
}

func TestRegisterPendingMap_SubmissionIndexZero(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")

	buf := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf")
	_ = buf.BeginMap(MapModeInternalRead, 0, 256)

	// Index 0 is the "never submitted" path.
	device.RegisterPendingMap(0, buf)

	didWork := device.PollMaps(0)
	if !didWork {
		t.Error("PollMaps(0) should resolve index=0 entries")
	}
	if buf.CurrentMapState() != BufferMapStateMapped {
		t.Errorf("state = %v, want Mapped", buf.CurrentMapState())
	}
}

func TestHalDeviceHandle(t *testing.T) {
	halDevice := &mockHALDevice{}
	device := NewDevice(halDevice, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "Test")

	hd, ok := device.HalDeviceHandle()
	if !ok {
		t.Fatal("HalDeviceHandle() should return ok=true for valid device")
	}
	if hd == nil {
		t.Error("HalDeviceHandle() returned nil device")
	}

	// Destroy the device.
	device.Destroy()

	_, ok = device.HalDeviceHandle()
	if ok {
		t.Error("HalDeviceHandle() should return ok=false after device destroyed")
	}
}

func TestHalDeviceHandle_NoHAL(t *testing.T) {
	device := &Device{}
	_, ok := device.HalDeviceHandle()
	if ok {
		t.Error("HalDeviceHandle() should return ok=false for device without HAL")
	}
}

// A map request canceled by Unmap leaves its registration behind, and the
// next request's: the stale one neither maps the buffer early nor reports
// the new request canceled.
func TestPollMaps_IgnoresCanceledRequests(t *testing.T) {
	newMapped := func(t *testing.T) (*Device, *Buffer) {
		t.Helper()
		device := NewDevice(&mockHALDevice{}, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "PollTest")
		buf := NewBuffer(mockBuffer{}, device, gputypes.BufferUsageMapRead, 256, "buf")
		if err := buf.BeginMap(MapModeInternalRead, 0, 256); err != nil {
			t.Fatalf("BeginMap: %v", err)
		}
		return device, buf
	}
	remap := func(t *testing.T, device *Device, buf *Buffer, first, second uint64) {
		t.Helper()
		device.RegisterPendingMap(first, buf)
		if err := buf.UnmapBuffer(SnatchGuard{}, nil); err != nil {
			t.Fatalf("UnmapBuffer: %v", err)
		}
		if err := buf.BeginMap(MapModeInternalRead, 0, 256); err != nil {
			t.Fatalf("second BeginMap: %v", err)
		}
		device.RegisterPendingMap(second, buf)
	}

	t.Run("earlier submission", func(t *testing.T) {
		device, buf := newMapped(t)
		remap(t, device, buf, 1, 2)
		device.PollMaps(1)
		if got := buf.CurrentMapState(); got != BufferMapStatePending {
			t.Fatalf("state after the canceled request's submission = %v, want Pending", got)
		}
		if done, _ := buf.Waiter().Status(); done {
			t.Fatal("the new request resolved with the canceled one")
		}
		device.PollMaps(2)
		if got := buf.CurrentMapState(); got != BufferMapStateMapped {
			t.Fatalf("state after its own submission = %v, want Mapped", got)
		}
	})

	t.Run("same submission", func(t *testing.T) {
		device, buf := newMapped(t)
		remap(t, device, buf, 1, 1)
		device.PollMaps(1)
		if got := buf.CurrentMapState(); got != BufferMapStateMapped {
			t.Fatalf("state = %v, want Mapped", got)
		}
		if done, err := buf.Waiter().Status(); !done || err != nil {
			t.Errorf("waiter = %v, %v, want done without error", done, err)
		}
	})
}
