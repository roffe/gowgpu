// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build integration && linux && !(js && wasm)

package gles

import (
	"bytes"
	"runtime"
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// glCalls is how many cgo calls f makes, one per GL call.
func glCalls(f func()) int64 {
	before := runtime.NumCgoCall()
	f()
	return runtime.NumCgoCall() - before
}

// A replayed command whose state the context has already sends no GL call,
// one whose state differs does, and a command that changes that state
// behind the shadow's back makes it be sent again.
func TestReplaySendsStateOnce(t *testing.T) {
	adapterCtx := newTestAdapterContext(t)
	t.Cleanup(adapterCtx.Destroy)
	glCtx := adapterCtx.Lock()
	defer adapterCtx.Unlock()
	glCtx.Shadow = gl.Shadow{}

	sends := func(name string, want bool, cmd Command) {
		t.Helper()
		if n := glCalls(func() { cmd.Execute(glCtx) }); (n > 0) != want {
			t.Errorf("%s sent %d GL calls, want sent=%v", name, n, want)
		}
	}

	blend := gputypes.BlendState{
		Color: gputypes.BlendComponent{SrcFactor: gputypes.BlendFactorSrcAlpha, DstFactor: gputypes.BlendFactorOneMinusSrcAlpha, Operation: gputypes.BlendOperationAdd},
		Alpha: gputypes.BlendComponent{SrcFactor: gputypes.BlendFactorSrcAlpha, DstFactor: gputypes.BlendFactorOneMinusSrcAlpha, Operation: gputypes.BlendOperationAdd},
	}
	state := func(cull gputypes.CullMode) *SetPipelineStateCommand {
		b := blend // its own copy: the comparison is by content, as pipelines each carry their own
		return &SetPipelineStateCommand{cullMode: cull, frontFace: gputypes.FrontFaceCCW,
			colorTargets: []ColorTargetDesc{{Blend: &b, WriteMask: gputypes.ColorWriteMaskAll}}}
	}
	sends("pipeline state", true, state(gputypes.CullModeBack))
	sends("the same pipeline state again", false, state(gputypes.CullModeBack))
	sends("another cull mode", true, state(gputypes.CullModeNone))
	sends("a depth clear", true, &ClearDepthCommand{depth: 1})
	sends("the pipeline state after the clear changed the depth mask", true, state(gputypes.CullModeNone))

	sends("program 0", true, &UseProgramCommand{})
	sends("program 0 again", false, &UseProgramCommand{})

	vao, other := glCtx.GenVertexArrays(1), glCtx.GenVertexArrays(1)
	defer glCtx.DeleteVertexArrays(vao, other)
	buf := glCtx.GenBuffers(1)
	defer glCtx.DeleteBuffers(buf)
	glCtx.BindBuffer(gl.ARRAY_BUFFER, buf)
	glCtx.BufferData(gl.ARRAY_BUFFER, 1024, 0, gl.STATIC_DRAW)
	glCtx.BindBuffer(gl.ARRAY_BUFFER, 0)
	layout := &gputypes.VertexBufferLayout{ArrayStride: 32, Attributes: []gputypes.VertexAttribute{
		{Format: gputypes.VertexFormatFloat32x2, Offset: 0, ShaderLocation: 0},
		{Format: gputypes.VertexFormatFloat32x4, Offset: 16, ShaderLocation: 2},
	}}
	bind := func(offset uint64) *SetVertexBufferCommand {
		return &SetVertexBufferCommand{buffer: &Buffer{id: buf}, offset: offset, layout: layout}
	}
	sends("the VAO", true, &BindVAOCommand{vao: vao})
	sends("the vertex buffer", true, bind(0))
	sends("the vertex buffer again under the same layout", false, bind(0))
	sends("the vertex buffer at another offset", true, bind(64))
	sends("the same VAO again", false, &BindVAOCommand{vao: vao})
	sends("another VAO", true, &BindVAOCommand{vao: other})
	sends("the vertex buffer under the other VAO", true, bind(64))

	tex, sampler := glCtx.GenTextures(1), glCtx.GenSamplers(1)
	defer glCtx.DeleteTextures(tex)
	defer glCtx.DeleteSamplers(sampler)
	view := &TextureView{texture: &Texture{id: tex, target: gl.TEXTURE_2D}, mipCount: 1, layerCount: 1}
	group := &BindGroup{bindings: []groupBinding{
		{binding: 0, class: bindingClassTexture, view: view},
		{binding: 1, class: bindingClassSampler, sampler: sampler},
	}}
	infos := []BindGroupLayoutInfo{{BindingToSlot: []uint8{3, 3}}} // texture and sampler on unit 3
	bg := func() *SetBindGroupCommand {
		return &SetBindGroupCommand{group: group, groupInfos: infos, maxTextureUnits: 16}
	}
	sends("the texture and sampler", true, bg())
	sends("the same bind group again", false, bg())
	glCtx.Shadow.ForgetTextures()
	sends("the bind group after the units were forgotten", true, bg())
}

// WriteBuffer lands exactly its range, keeps the rest, has copied the data
// when it returns and keeps the order of two writes to the same range.
func TestWriteBufferKeepsTheRest(t *testing.T) {
	adapterCtx := newTestAdapterContext(t)
	t.Cleanup(adapterCtx.Destroy)
	glCtx := adapterCtx.Lock()
	const size, from = 1024, 100
	pattern := bytes.Repeat([]byte{0xab}, size)
	id := glCtx.GenBuffers(1)
	glCtx.BindBuffer(gl.COPY_WRITE_BUFFER, id)
	glCtx.BufferData(gl.COPY_WRITE_BUFFER, size, uintptr(unsafe.Pointer(&pattern[0])), gl.DYNAMIC_DRAW)
	glCtx.BindBuffer(gl.COPY_WRITE_BUFFER, 0)
	adapterCtx.Unlock()

	q := &Queue{ctx: adapterCtx}
	buffer := &Buffer{id: id, target: gl.COPY_WRITE_BUFFER}
	data := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	if err := q.WriteBuffer(buffer, from, data); err != nil {
		t.Fatal(err)
	}
	data[0] = 0xff // the caller may change its slice right after the call
	if err := q.WriteBuffer(buffer, from+4, []byte{9, 9, 9, 9}); err != nil {
		t.Fatal(err)
	}

	glCtx = adapterCtx.Lock()
	defer adapterCtx.Unlock()
	defer glCtx.DeleteBuffers(id)
	glCtx.BindBuffer(gl.COPY_READ_BUFFER, id)
	defer glCtx.BindBuffer(gl.COPY_READ_BUFFER, 0)
	glPtr := glCtx.MapBuffer(gl.COPY_READ_BUFFER, gl.READ_ONLY)
	if glPtr == 0 {
		t.Fatal("glMapBuffer failed")
	}
	defer glCtx.UnmapBuffer(gl.COPY_READ_BUFFER)
	got := unsafe.Slice(*(**byte)(unsafe.Pointer(&glPtr)), size)
	want := bytes.Repeat([]byte{0xab}, size)
	copy(want[from:], []byte{1, 2, 3, 4, 9, 9, 9, 9})
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("byte %d is %#x, want %#x", i, got[i], want[i])
		}
	}
}
