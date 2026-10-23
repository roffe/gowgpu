// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// Without base instance draws, a draw from instance 2 binds the
// per-instance vertex buffers two records on, draws from instance 0 and
// binds them back; the per-vertex buffers stay where they are.
func TestDrawFromInstanceWithoutBaseInstance(t *testing.T) {
	enc := &CommandEncoder{glCtx: &gl.Context{}} // no base instance draws loaded
	if err := enc.BeginEncoding("first-instance"); err != nil {
		t.Fatal(err)
	}
	pass := enc.BeginRenderPass(&hal.RenderPassDescriptor{})
	pass.SetPipeline(&RenderPipeline{firstInstanceLoc: -1, vertexBuffers: []gputypes.VertexBufferLayout{
		{ArrayStride: 8, StepMode: gputypes.VertexStepModeVertex},
		{ArrayStride: 16, StepMode: gputypes.VertexStepModeInstance},
	}})
	pass.SetVertexBuffer(0, &Buffer{id: 1}, 4)
	pass.SetVertexBuffer(1, &Buffer{id: 2}, 64)
	before := len(enc.commands)
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1, FirstInstance: 2})

	cmds := enc.commands[before:]
	if len(cmds) != 3 {
		t.Fatalf("the draw recorded %d commands, want the instance buffer moved, the draw and the buffer back", len(cmds))
	}
	if vb, ok := cmds[0].(*SetVertexBufferCommand); !ok || vb.slot != 1 || vb.offset != 64+2*16 {
		t.Errorf("first command %#v, want slot 1 bound at 96", cmds[0])
	}
	if d, ok := cmds[1].(*DrawCommand); !ok || d.baseInstance != 0 || d.firstInstance != 2 {
		t.Errorf("second command %#v, want a draw from base instance 0 with instance_index from 2", cmds[1])
	}
	if vb, ok := cmds[2].(*SetVertexBufferCommand); !ok || vb.slot != 1 || vb.offset != 64 {
		t.Errorf("third command %#v, want slot 1 bound back at 64", cmds[2])
	}
}

// A pipeline switch disables the attribute arrays the new pipeline doesn't
// feed, and the end of the pass all of them: nothing disabled them, so a
// pipeline drew with the last one's arrays enabled, bound to buffers it
// doesn't bind or that were deleted.
func TestUnusedVertexAttribsAreDisabled(t *testing.T) {
	enc := &CommandEncoder{glCtx: &gl.Context{}}
	if err := enc.BeginEncoding("attribs"); err != nil {
		t.Fatal(err)
	}
	pass := enc.BeginRenderPass(&hal.RenderPassDescriptor{})
	pass.SetPipeline(&RenderPipeline{firstInstanceLoc: -1, attribs: attribLocations([]gputypes.VertexBufferLayout{
		{ArrayStride: 8, Attributes: []gputypes.VertexAttribute{{ShaderLocation: 0}, {ShaderLocation: 3}}},
	})})
	pass.End()
	var keeps []uint32
	for _, c := range enc.commands {
		if d, ok := c.(*DisableVertexAttribsCommand); ok {
			keeps = append(keeps, d.keep)
		}
	}
	if len(keeps) != 2 || keeps[0] != 1|1<<3 || keeps[1] != 0 {
		t.Errorf("attribute arrays kept %b, want 1001 at the pipeline and none at the end", keeps)
	}
}
