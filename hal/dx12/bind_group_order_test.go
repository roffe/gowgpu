//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// The descriptor table follows the layout's entry order, so a bind group
// listing its entries in another order has to be put back in that order.
func TestLayoutOrderedEntries(t *testing.T) {
	layout := []BindGroupLayoutEntry{{Binding: 4}, {Binding: 0}, {Binding: 2}}
	entries := []gputypes.BindGroupEntry{{Binding: 0}, {Binding: 2}, {Binding: 4}}
	ordered, err := layoutOrderedEntries(layout, entries)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []uint32{4, 0, 2} {
		if ordered[i].Binding != want {
			t.Errorf("slot %d binds %d, want %d", i, ordered[i].Binding, want)
		}
	}
	if _, err := layoutOrderedEntries(layout, entries[:2]); err == nil {
		t.Error("a layout entry without a bind group entry was accepted")
	}
}
