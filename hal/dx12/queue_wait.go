//go:build windows && !(js && wasm)

package dx12

import (
	"fmt"
	"time"

	"golang.org/x/sys/windows"
)

// WaitSubmission waits up to timeout, or with a negative timeout until it
// happens, for the GPU to finish the submission Submit returned index for,
// and reports whether it has. It waits on the device's fence for that value
// alone and holds no queue lock, so submits go on meanwhile.
func (q *Queue) WaitSubmission(index uint64, timeout time.Duration) (bool, error) {
	fence := q.device.fence
	if fence.GetCompletedValue() >= index {
		return true, nil
	}
	event, err := windows.CreateEvent(nil, 0, 0, nil)
	if err != nil {
		return false, fmt.Errorf("dx12: CreateEvent for a submission wait failed: %w", err)
	}
	defer func() { _ = windows.CloseHandle(event) }()
	if err := fence.SetEventOnCompletion(index, uintptr(event)); err != nil {
		return false, fmt.Errorf("dx12: SetEventOnCompletion failed: %w", err)
	}
	timeoutMs := uint32(windows.INFINITE)
	if timeout >= 0 {
		timeoutMs = uint32(min(timeout.Milliseconds(), int64(windows.INFINITE-1)))
		if timeoutMs == 0 && timeout > 0 {
			timeoutMs = 1
		}
	}
	result, err := windows.WaitForSingleObject(event, timeoutMs)
	if err != nil {
		return false, fmt.Errorf("dx12: WaitForSingleObject failed: %w", err)
	}
	switch result {
	case windows.WAIT_OBJECT_0:
		return true, nil
	case uint32(windows.WAIT_TIMEOUT):
		return false, nil
	default:
		return false, fmt.Errorf("dx12: WaitForSingleObject returned unexpected: %d", result)
	}
}
