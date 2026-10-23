//go:build integration && linux && !(js && wasm)

package gles

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// ClearBufferCommand zeroes its range of a real GL buffer and nothing else,
// in more than one piece of glZeros.
func TestClearBufferCommandZeroesRange(t *testing.T) {
	adapterCtx := newTestAdapterContext(t)
	t.Cleanup(adapterCtx.Destroy)
	glCtx := adapterCtx.Lock()
	defer adapterCtx.Unlock()

	const size, from, n = 320 << 10, 4, 300 << 10
	pattern := bytes.Repeat([]byte{0xab}, size)
	id := glCtx.GenBuffers(1)
	defer glCtx.DeleteBuffers(id)
	glCtx.BindBuffer(gl.COPY_WRITE_BUFFER, id)
	glCtx.BufferData(gl.COPY_WRITE_BUFFER, size, 0, gl.STATIC_DRAW)
	glCtx.BufferSubData(gl.COPY_WRITE_BUFFER, 0, size, unsafe.Pointer(&pattern[0]))
	glCtx.BindBuffer(gl.COPY_WRITE_BUFFER, 0)

	(&ClearBufferCommand{buffer: &Buffer{id: id}, offset: from, size: n}).Execute(glCtx)

	glCtx.BindBuffer(gl.COPY_READ_BUFFER, id)
	defer glCtx.BindBuffer(gl.COPY_READ_BUFFER, 0)
	glPtr := glCtx.MapBuffer(gl.COPY_READ_BUFFER, gl.READ_ONLY)
	if glPtr == 0 {
		t.Fatal("glMapBuffer failed")
	}
	defer glCtx.UnmapBuffer(gl.COPY_READ_BUFFER)
	got := unsafe.Slice(*(**byte)(unsafe.Pointer(&glPtr)), size)
	for i, b := range got {
		want := byte(0)
		if i < from || i >= from+n {
			want = 0xab
		}
		if b != want {
			t.Fatalf("byte %d is %#x, want %#x", i, b, want)
		}
	}
}

// A MapRead buffer starts zeroed and its mappings are read from GL, never
// written back: Unmap uploaded every mapping, so bytes written into a read
// mapping reached the buffer. An empty range at the end maps to a pointer
// rather than indexing past the shadow.
func TestReadMappingIsNotWrittenBack(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	dev := openTestDevice(t, ctx)
	t.Cleanup(dev.Destroy)
	buf, err := dev.CreateBuffer(&hal.BufferDescriptor{Size: 64, Usage: gputypes.BufferUsageMapRead | gputypes.BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer dev.DestroyBuffer(buf)
	mapped := func(offset, size uint64) []byte {
		t.Helper()
		m, err := dev.MapBuffer(buf, offset, size)
		if err != nil || m.Ptr == nil {
			t.Fatalf("MapBuffer(%d, %d): %v, %p", offset, size, err, m.Ptr)
		}
		return unsafe.Slice((*byte)(m.Ptr), size)
	}
	got := mapped(0, 64)
	if !bytes.Equal(got, make([]byte, 64)) {
		t.Errorf("a new MapRead buffer holds % x", got)
	}
	for i := range got {
		got[i] = 0x11
	}
	_ = dev.UnmapBuffer(buf)
	if got := mapped(0, 64); got[0] != 0 {
		t.Errorf("bytes written into a read mapping reached the buffer: % x", got[:8])
	}
	_ = dev.UnmapBuffer(buf)
	mapped(64, 0)
	_ = dev.UnmapBuffer(buf)
}

// DisableVertexAttribsCommand disables what the shadow has enabled outside
// the locations it keeps, without a GL error.
func TestDisableVertexAttribs(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	glCtx := ctx.Lock()
	defer ctx.Unlock()
	vao := glCtx.GenVertexArrays(1)
	defer glCtx.DeleteVertexArrays(vao)
	(&BindVAOCommand{vao: vao}).Execute(glCtx)
	id := glCtx.GenBuffers(1)
	defer glCtx.DeleteBuffers(id)
	glCtx.BindBuffer(gl.ARRAY_BUFFER, id)
	glCtx.BufferData(gl.ARRAY_BUFFER, 64, 0, gl.STATIC_DRAW)
	(&SetVertexBufferCommand{buffer: &Buffer{id: id}, layout: &gputypes.VertexBufferLayout{ArrayStride: 8, Attributes: []gputypes.VertexAttribute{
		{Format: gputypes.VertexFormatFloat32, ShaderLocation: 0}, {Format: gputypes.VertexFormatFloat32, Offset: 4, ShaderLocation: 2},
	}}}).Execute(glCtx)
	(&DisableVertexAttribsCommand{keep: 1}).Execute(glCtx)
	if sh := &glCtx.Shadow; !sh.Attribs[0].Set || sh.Attribs[2].Set {
		t.Errorf("location 0 enabled %v, location 2 %v: want 0 kept and 2 disabled", sh.Attribs[0].Set, sh.Attribs[2].Set)
	}
	if e := glCtx.GetError(); e != 0 {
		t.Errorf("GL error %#x", e)
	}
}
