//go:build !(js && wasm)

package vulkan

import (
	"errors"
	"math"
	"time"
)

// WaitSubmission waits up to timeout, or with a negative timeout until it
// happens, for the GPU to finish the submission Submit returned index for,
// and reports whether it has. It waits on the device's fence for that value
// alone and holds no queue lock, so submits go on meanwhile.
func (q *Queue) WaitSubmission(index uint64, timeout time.Duration) (bool, error) {
	timeoutNs := uint64(math.MaxUint64)
	if timeout >= 0 {
		timeoutNs = uint64(timeout)
	}
	err := q.device.timelineFence.waitForValue(q.device.cmds, q.device.handle, index, timeoutNs)
	if errors.Is(err, errWaitTimeout) {
		return false, nil
	}
	return err == nil, err
}
