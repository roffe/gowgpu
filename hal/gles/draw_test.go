// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"math"
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

// An integer color attachment clears with its clear value as integers,
// clamped to their range, which glClearBufferfv would leave undefined.
func TestColorClearOfIntegerAttachments(t *testing.T) {
	view := func(f gputypes.TextureFormat) *TextureView {
		return &TextureView{texture: &Texture{format: f, glf: glFormats[f]}}
	}
	v := gputypes.Color{R: 4000, G: -5, B: 1e10, A: 0.5}
	u := colorClear(1, &hal.RenderPassColorAttachment{View: view(gputypes.TextureFormatRGBA32Uint), ClearValue: v})
	if !u.integer || u.signed || u.uints != [4]uint32{4000, 0, math.MaxUint32, 0} || u.drawBuffer != 1 {
		t.Errorf("RGBA32Uint clear = %+v", u)
	}
	s := colorClear(0, &hal.RenderPassColorAttachment{View: view(gputypes.TextureFormatRGBA8Sint), ClearValue: v})
	if !s.integer || !s.signed || s.sints != [4]int32{4000, -5, math.MaxInt32, 0} {
		t.Errorf("RGBA8Sint clear = %+v", s)
	}
	f := colorClear(0, &hal.RenderPassColorAttachment{View: view(gputypes.TextureFormatRGBA16Float), ClearValue: v})
	if f.integer || f.color != [4]float32{4000, -5, 1e10, 0.5} {
		t.Errorf("RGBA16Float clear = %+v", f)
	}
}
