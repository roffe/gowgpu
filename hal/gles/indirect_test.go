//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// Indirect draws record a command for each record, 16 bytes apart for a
// draw and 20 for an indexed one, which keeps the index buffer's offset;
// the counted forms stay unsupported.
func TestRenderPassEncoderIndirectDraws(t *testing.T) {
	enc := &CommandEncoder{drawIndirect: true}
	if err := enc.BeginEncoding("indirect"); err != nil {
		t.Fatal(err)
	}
	pass := enc.BeginRenderPass(&hal.RenderPassDescriptor{ColorAttachments: []hal.RenderPassColorAttachment{}})

	pass.DrawIndirect(&Buffer{id: 7, size: 64}, 4, 2)
	pass.SetIndexBuffer(&Buffer{id: 9, size: 64}, gputypes.IndexFormatUint32, 16)
	pass.DrawIndexedIndirect(&Buffer{id: 8, size: 64}, 0, 2)
	pass.DrawIndirectCount(&Buffer{id: 7, size: 64}, 0, &Buffer{id: 6, size: 4}, 0, 2)
	pass.DrawIndexedIndirectCount(&Buffer{id: 8, size: 64}, 0, &Buffer{id: 6, size: 4}, 0, 2)

	if len(enc.commands) != 5 {
		t.Fatalf("commands = %d, want 2 indirect draws, SetIndexBufferCommand and 2 indexed ones", len(enc.commands))
	}
	for i, want := range []struct {
		offset  uint64
		indexed bool
	}{{4, false}, {20, false}, {}, {0, true}, {20, true}} {
		if i == 2 {
			continue
		}
		c, ok := enc.commands[i].(*DrawIndirectCommand)
		if !ok {
			t.Fatalf("command %d is %T", i, enc.commands[i])
		}
		if c.offset != want.offset || (c.indexed != nil) != want.indexed || !c.native {
			t.Errorf("command %d: offset %d, indexed %v, native %v; want offset %d, indexed %v", i, c.offset, c.indexed != nil, c.native, want.offset, want.indexed)
		}
		if c.indexed != nil && (c.indexed.indexOffset != 16 || c.indexed.indexFormat != gputypes.IndexFormatUint32) {
			t.Errorf("command %d: index buffer at %d as %v", i, c.indexed.indexOffset, c.indexed.indexFormat)
		}
	}
}
