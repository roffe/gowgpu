//go:build integration && linux && !(js && wasm)

package gles

import (
	"bytes"
	"testing"
	"unsafe"

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
