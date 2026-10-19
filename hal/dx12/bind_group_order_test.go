//go:build windows && !(js && wasm)

package dx12

import (
	"runtime"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// The descriptor table follows the layout's entry order, so a bind group
// listing its entries in another order has to be put back in that order.
func TestLayoutOrderedEntries(t *testing.T) {
	bufs := []*Buffer{{size: 256}, {size: 256}, {size: 256}}
	defer runtime.KeepAlive(bufs)

	layout := &BindGroupLayout{entries: []BindGroupLayoutEntry{
		{Binding: 4, Type: BindingTypeUniformBuffer},
		{Binding: 0, Type: BindingTypeUniformBuffer},
		{Binding: 2, Type: BindingTypeUniformBuffer},
	}}
	entries := []gputypes.BindGroupEntry{bufferEntry(0, bufs[0]), bufferEntry(2, bufs[1]), bufferEntry(4, bufs[2])}
	ordered, _, err := orderBindGroupEntries(layout, entries)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []uint32{4, 0, 2} {
		if ordered[i].entry.Binding != want || ordered[i].layout.Binding != want {
			t.Errorf("slot %d binds %d (layout %d), want %d", i, ordered[i].entry.Binding, ordered[i].layout.Binding, want)
		}
	}
	if _, _, err := orderBindGroupEntries(layout, entries[:2]); err == nil {
		t.Error("a layout entry without a bind group entry was accepted")
	}
}
