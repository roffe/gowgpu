//go:build (linux || windows) && !(js && wasm)

package gles

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// GL sets the vertex attributes when a vertex buffer is bound, from the
// layout of the pipeline current then, so a buffer drawn under another
// pipeline has to be bound again under that pipeline's layout.
func TestRenderPassSetPipelineRebindsVertexBuffers(t *testing.T) {
	enc := &CommandEncoder{}
	_ = enc.BeginEncoding("test")
	rpe := enc.BeginRenderPass(&hal.RenderPassDescriptor{}).(*RenderPassEncoder)

	glyphs := &RenderPipeline{programID: 1, vertexBuffers: []gputypes.VertexBufferLayout{{ArrayStride: 32}}}
	lines := &RenderPipeline{programID: 2, vertexBuffers: []gputypes.VertexBufferLayout{{ArrayStride: 40}}}
	buf := &Buffer{id: 7}

	rpe.SetPipeline(glyphs)
	rpe.SetVertexBuffer(0, buf, 64)
	rpe.SetPipeline(lines)

	binds := vertexBufferBinds(enc)
	if len(binds) != 2 {
		t.Fatalf("got %d vertex buffer binds, want 2: the bind and the one under the second pipeline", len(binds))
	}
	if last := binds[1]; last.buffer != buf || last.offset != 64 || last.layout == nil || last.layout.ArrayStride != 40 {
		t.Errorf("rebind: buffer %p offset %d layout %+v, want buffer %p at 64 with the 40 byte stride", last.buffer, last.offset, last.layout, buf)
	}

	// A pipeline with no layout for the slot leaves the buffer as it is.
	rpe.SetPipeline(&RenderPipeline{programID: 3})
	if got := len(vertexBufferBinds(enc)); got != 2 {
		t.Errorf("got %d binds after a pipeline without vertex buffers, want 2", got)
	}
}

func vertexBufferBinds(enc *CommandEncoder) []*SetVertexBufferCommand {
	var binds []*SetVertexBufferCommand
	for _, c := range enc.commands {
		if cmd, ok := c.(*SetVertexBufferCommand); ok {
			binds = append(binds, cmd)
		}
	}
	return binds
}
