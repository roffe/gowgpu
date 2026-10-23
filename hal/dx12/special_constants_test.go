//go:build windows && !(js && wasm)

package dx12

import (
	"encoding/binary"
	"slices"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// TestSpecialIndirectRecords copies indirect records behind their special
// constants and checks the copies hold [constants | record] for each, the
// layout the layout's command signatures read.
func TestSpecialIndirectRecords(t *testing.T) {
	dev, queue := openTestDevice(t)
	layout, err := dev.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	t.Cleanup(func() { dev.DestroyPipelineLayout(layout) })
	if _, err := layout.(*PipelineLayout).indirectSignatures(); err != nil {
		t.Fatalf("indirectSignatures: %v", err)
	}

	words := []uint32{100, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
	src := newTestBuffer(t, dev, 64, gputypes.BufferUsageCopyDst|gputypes.BufferUsageIndirect)
	data := make([]byte, 0, 64)
	for _, w := range words {
		data = binary.LittleEndian.AppendUint32(data, w)
	}
	if err := queue.WriteBuffer(src, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	for _, tc := range []struct {
		name                       string
		count, size, at, constants uint32
		want                       []uint32
	}{
		{"draws", 2, drawRecordSize, 8, 2, []uint32{3, 4, 1, 2, 3, 4, 7, 8, 5, 6, 7, 8}},
		{"indexed draw", 1, drawIndexedRecordSize, 12, 2, []uint32{4, 5, 1, 2, 3, 4, 5}},
		{"dispatch", 1, dispatchRecordSize, 0, 3, []uint32{1, 2, 3, 1, 2, 3}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			out := newTestBuffer(t, dev, 64, gputypes.BufferUsageCopyDst|gputypes.BufferUsageMapRead)
			n := uint64(len(tc.want) * 4)
			record(t, dev, queue, func(enc *CommandEncoder) {
				args, offset, ok := enc.specialIndirect(src, 4, tc.count, tc.size, tc.at, tc.constants)
				if !ok {
					t.Fatalf("specialIndirect: %v", enc.err)
				}
				enc.CopyBufferToBuffer(args, out, []hal.BufferCopy{{SrcOffset: offset, Size: n}})
			})
			raw := readBuffer(t, dev, out, n)
			got := make([]uint32, len(tc.want))
			for i := range got {
				got[i] = binary.LittleEndian.Uint32(raw[i*4:])
			}
			if !slices.Equal(got, tc.want) {
				t.Fatalf("records = %v, want %v", got, tc.want)
			}
		})
	}
}
