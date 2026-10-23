//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/noop"
)

// A mappable buffer only goes with the copy usage of its direction, and a
// buffer mapped at creation has a size in whole 4-byte words, as WebGPU
// has them.
func TestCreateBufferChecksMappableUsagesAndMappedSize(t *testing.T) {
	device := newCaptureDevice(&captureDevice{Device: &noop.Device{}})
	for _, tc := range []struct {
		name   string
		desc   BufferDescriptor
		reason core.CreateBufferErrorKind
		valid  bool
	}{
		{"map read for copies in", BufferDescriptor{Size: 8, Usage: BufferUsageMapRead | BufferUsageCopyDst}, 0, true},
		{"map write for copies out", BufferDescriptor{Size: 8, Usage: BufferUsageMapWrite | BufferUsageCopySrc}, 0, true},
		{"map read as storage", BufferDescriptor{Size: 8, Usage: BufferUsageMapRead | gputypes.BufferUsageStorage}, core.CreateBufferErrorMapUsageMismatch, false},
		{"map read for copies out", BufferDescriptor{Size: 8, Usage: BufferUsageMapRead | BufferUsageCopySrc}, core.CreateBufferErrorMapUsageMismatch, false},
		{"map write as vertices", BufferDescriptor{Size: 8, Usage: BufferUsageMapWrite | BufferUsageVertex}, core.CreateBufferErrorMapUsageMismatch, false},
		{"mapped at creation in words", BufferDescriptor{Size: 8, Usage: BufferUsageCopySrc, MappedAtCreation: true}, 0, true},
		{"mapped at creation with 6 bytes", BufferDescriptor{Size: 6, Usage: BufferUsageCopySrc, MappedAtCreation: true}, core.CreateBufferErrorUnalignedMappedSize, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			buf, err := device.CreateBuffer(&tc.desc)
			if buf != nil {
				buf.Release()
			}
			var cbe *core.CreateBufferError
			switch {
			case tc.valid && err != nil:
				t.Errorf("CreateBuffer: %v", err)
			case !tc.valid && (!errors.As(err, &cbe) || cbe.Kind != tc.reason):
				t.Errorf("CreateBuffer: %v, want a CreateBufferError of kind %d", err, tc.reason)
			}
		})
	}
}
