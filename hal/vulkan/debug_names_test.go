//go:build !(js && wasm)

package vulkan

import (
	"fmt"
	"sync"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// TestDebugDeviceNamesObjects opens a device on a debug instance, which
// must be able to name objects, and creates labelled buffers from several
// goroutines at once.
func TestDebugDeviceNamesObjects(t *testing.T) {
	d := openTestDevice(t, gputypes.InstanceFlagsDebug)
	if !d.instance.debugEnabled {
		t.Skip("no validation layer, so no VK_EXT_debug_utils")
	}
	if !d.cmds.HasDebugUtils() {
		t.Fatal("the device of a debug instance has no vkSetDebugUtilsObjectNameEXT")
	}
	var wg sync.WaitGroup
	for g := range 4 {
		wg.Go(func() {
			for i := range 50 {
				buf, err := d.CreateBuffer(&hal.BufferDescriptor{Label: fmt.Sprintf("buffer %d of goroutine %d", i, g), Size: 256, Usage: gputypes.BufferUsageCopyDst})
				if err != nil {
					t.Errorf("CreateBuffer: %v", err)
					return
				}
				d.DestroyBuffer(buf)
			}
		})
	}
	wg.Wait()
}
