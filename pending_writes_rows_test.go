//go:build !(js && wasm)

package wgpu

import (
	"bytes"
	"testing"
)

// WebGPU asks a texture write for bytesPerRow times the rows but the last,
// and bytesInLastRow for that one. The padding copy used to drop a last row
// that stopped there.
func TestAlignTextureDataIntoShortLastRow(t *testing.T) {
	const bytesPerRow, aligned, rows, lastRow = 12, 256, 3, 8
	data := make([]byte, bytesPerRow*(rows-1)+lastRow)
	for i := range data {
		data[i] = byte(i + 1)
	}
	var buf []byte
	out := alignTextureDataInto(&buf, data, 0, bytesPerRow, aligned, rows, 1, aligned*rows)
	if len(out) != aligned*rows {
		t.Fatalf("got %d bytes, want %d", len(out), aligned*rows)
	}
	for r := range rows {
		n := bytesPerRow
		if r == rows-1 {
			n = lastRow
		}
		if got, want := out[r*aligned:r*aligned+n], data[r*bytesPerRow:r*bytesPerRow+n]; !bytes.Equal(got, want) {
			t.Errorf("row %d: got %v, want %v", r, got, want)
		}
	}
}
