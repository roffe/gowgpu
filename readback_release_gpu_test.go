//go:build !(js && wasm)

package wgpu

import (
	"fmt"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// A texture released right after its readback, and the next one, which gets
// its tracker index, rendered and read back in turn: the barriers of each
// submit go to that submit's textures, so the released one's image is not
// used by a command buffer still pending when it is destroyed
// (VUID-vkDestroyImage-image-01000).
func TestReleaseRightAfterReadback(t *testing.T) {
	device, h := newValidatedDevice(t)
	st := readbackBuffer(t, device, "readback-release-st")
	for i := range 4 {
		tex, view := pixTexture(t, device, "readback-release", TextureUsageRenderAttachment|TextureUsageCopySrc)
		enc, err := device.CreateCommandEncoder(nil)
		if err != nil {
			t.Fatalf("CreateCommandEncoder: %v", err)
		}
		clearPass(t, enc, view, gputypes.LoadOpClear, gputypes.Color{R: 1, A: 1}, nil)
		copyOut(enc, tex, st)
		finishAndSubmit(t, device, enc)
		expectPixels(t, fmt.Sprintf("texture %d", i), readback(t, st), 255, 0, 0, 255)
		releaseTextureView(tex, view)
	}
	assertNoValidation(t, h)
}
