//go:build !(js && wasm)

package track

import (
	"errors"
	"slices"
	"testing"

	"github.com/roffe/gowgpu/hal"
)

// A texture of 3 mip levels and 4 array layers.
var full34 = TextureSubresources(3, 4)

func mip(level uint32) TextureSelector {
	return TextureSelector{Mips: Range{level, level + 1}, Layers: full34.Layers}
}

func sub(level, layer uint32) TextureSelector {
	return TextureSelector{Mips: Range{level, level + 1}, Layers: Range{layer, layer + 1}}
}

func TestScopeSubresourcesUsedDifferently(t *testing.T) {
	t.Parallel()

	s := NewTextureUsageScope()
	if err := s.SetSubresourceUsage(1, full34, mip(1), TextureUsesColorTarget); err != nil {
		t.Fatalf("render into mip 1: %v", err)
	}
	if err := s.SetSubresourceUsage(1, full34, mip(0), TextureUsesResource); err != nil {
		t.Fatalf("sample mip 0 beside it: %v", err)
	}
	if err := s.SetSubresourceUsage(1, full34, sub(2, 3), TextureUsesStorageWrite); err != nil {
		t.Fatalf("write layer 3 of mip 2 beside them: %v", err)
	}
	if err := s.SetSubresourceUsage(1, full34, sub(2, 2), TextureUsesResource); err != nil {
		t.Fatalf("sample layer 2 of mip 2: %v", err)
	}
	if got := s.GetUsage(1); got != TextureUsesColorTarget|TextureUsesResource|TextureUsesStorageWrite {
		t.Errorf("usage = %v, want the three", got)
	}

	var conflict *TextureUsageConflictError
	err := s.SetSubresourceUsage(1, full34, sub(1, 2), TextureUsesResource)
	if !errors.As(err, &conflict) {
		t.Fatalf("sampling a layer of the mip it renders into = %v, want a conflict", err)
	}
	if conflict.Selector != sub(1, 2) || conflict.Existing != TextureUsesColorTarget || conflict.New != TextureUsesResource {
		t.Errorf("conflict = %+v, want mip 1 layer 2 ColorTarget/Resource", conflict)
	}
	if err := s.SetUsage(1, TextureUsesCopySrc); err == nil {
		t.Error("a use of the whole texture over its render target is no conflict")
	}
}

func TestScopeSelectionOfWholeTextureIsSimple(t *testing.T) {
	t.Parallel()

	s := NewTextureUsageScope()
	_ = s.SetSubresourceUsage(0, full34, full34, TextureUsesResource)
	if _, c := s.set.get(0); c != nil {
		t.Error("a selection of every subresource made a complex state")
	}
	if err := s.SetSubresourceUsage(0, full34, TextureSelector{Mips: Range{5, 6}, Layers: Range{0, 1}}, TextureUsesCopyDst); err != nil {
		t.Errorf("a selection outside the texture = %v, want nothing", err)
	}
}

func TestEncoderTrackerTransitionsPerSubresource(t *testing.T) {
	t.Parallel()

	tr := NewEncoderTextureTracker()
	// Clear mip 0, then make mip 1 from mip 0 and mip 2 from mip 1.
	tr.SetSubresourceUsage(0, full34, mip(0), TextureUsesColorTarget, nil)
	for level := uint32(1); level < 3; level++ {
		pass := NewTextureUsageScope()
		_ = pass.SetSubresourceUsage(0, full34, mip(level), TextureUsesColorTarget)
		_ = pass.SetSubresourceUsage(0, full34, mip(level-1), TextureUsesResource)
		got := tr.SetFromScope(pass)
		want := []TexturePendingTransition{{Index: 0, Selector: mip(level - 1), Usage: TextureStateTransition{From: TextureUsesColorTarget, To: TextureUsesResource}}}
		if !slices.Equal(got, want) {
			t.Errorf("pass into mip %d: transitions %+v, want %+v", level, got, want)
		}
	}
	if got := tr.StartUsage(0); got != TextureUsesColorTarget {
		t.Errorf("start = %v, want ColorTarget for every level", got)
	}
	if got := tr.EndUsage(0); got != TextureUsesColorTarget|TextureUsesResource {
		t.Errorf("end = %v, want mips 0 and 1 sampled, 2 a target", got)
	}

	// Reading every level back takes each range from its own state.
	got := tr.SetSubresourceUsage(0, full34, TextureSelector{}, TextureUsesCopySrc, nil)
	want := []TexturePendingTransition{
		{Index: 0, Selector: mip(0), Usage: TextureStateTransition{From: TextureUsesResource, To: TextureUsesCopySrc}},
		{Index: 0, Selector: mip(1), Usage: TextureStateTransition{From: TextureUsesResource, To: TextureUsesCopySrc}},
		{Index: 0, Selector: mip(2), Usage: TextureStateTransition{From: TextureUsesColorTarget, To: TextureUsesCopySrc}},
	}
	if !slices.Equal(got, want) {
		t.Errorf("copy out: transitions %+v, want %+v", got, want)
	}
	if _, c := tr.end.get(0); c != nil {
		t.Error("end state is complex with every subresource CopySrc")
	}
}

func TestEncoderTrackerStartsWhatItFirstUses(t *testing.T) {
	t.Parallel()

	tr := NewEncoderTextureTracker()
	tr.SetSubresourceUsage(2, full34, sub(0, 1), TextureUsesCopyDst, nil)
	tr.SetSubresourceUsage(2, full34, sub(0, 1), TextureUsesResource, nil)
	tr.SetSubresourceUsage(2, full34, mip(0), TextureUsesColorTarget, nil)

	// Layer 1 of mip 0 started as a copy destination, the rest of mip 0 as a
	// render target; the other levels are not used.
	var got []TexturePendingTransition
	device := NewTextureTracker()
	device.InsertSingle(2, TextureUsesResource)
	got = device.MergeEncoder(tr)
	want := []TexturePendingTransition{
		{Index: 2, Selector: sub(0, 0), Usage: TextureStateTransition{From: TextureUsesResource, To: TextureUsesColorTarget}},
		{Index: 2, Selector: sub(0, 1), Usage: TextureStateTransition{From: TextureUsesResource, To: TextureUsesCopyDst}},
		{Index: 2, Selector: TextureSelector{Mips: Range{0, 1}, Layers: Range{2, 4}}, Usage: TextureStateTransition{From: TextureUsesResource, To: TextureUsesColorTarget}},
	}
	if !slices.Equal(got, want) {
		t.Errorf("submit barriers %+v, want %+v", got, want)
	}
	if got := device.GetUsage(2); got != TextureUsesColorTarget|TextureUsesResource {
		t.Errorf("device state = %v, want mip 0 a target and the others still sampled", got)
	}

	// A whole-texture use after it takes every level from where it is.
	next := NewEncoderTextureTracker()
	next.SetSubresourceUsage(2, full34, TextureSelector{}, TextureUsesResource, nil)
	got = device.MergeEncoder(next)
	want = []TexturePendingTransition{
		{Index: 2, Selector: mip(0), Usage: TextureStateTransition{From: TextureUsesColorTarget, To: TextureUsesResource}},
	}
	if !slices.Equal(got, want) {
		t.Errorf("second submit barriers %+v, want %+v", got, want)
	}
	if _, c := device.set.get(2); c != nil {
		t.Error("device state stayed complex with every subresource sampled")
	}
}

func TestDeviceSetSingleTakesItsSelection(t *testing.T) {
	t.Parallel()

	device := NewTextureTracker()
	device.InsertSingle(4, TextureUsesUninitialized)
	got := device.SetSubresourceUsage(4, full34, sub(1, 2), TextureUsesCopyDst, nil)
	want := []TexturePendingTransition{{Index: 4, Selector: sub(1, 2), Usage: TextureStateTransition{From: TextureUsesUninitialized, To: TextureUsesCopyDst}}}
	if !slices.Equal(got, want) {
		t.Fatalf("write barriers %+v, want %+v", got, want)
	}
	got = device.SetSubresourceUsage(4, full34, sub(1, 2), TextureUsesCopyDst, nil)
	if len(got) != 1 || got[0].Selector != sub(1, 2) {
		t.Errorf("a second write into it = %+v, want a barrier on it alone", got)
	}
	if got := device.GetUsage(4); got != TextureUsesUninitialized|TextureUsesCopyDst {
		t.Errorf("state = %v, want the written subresource CopyDst and the rest untouched", got)
	}

	snap := device.Save(4)
	device.SetSubresourceUsage(4, full34, TextureSelector{}, TextureUsesColorTarget, nil)
	device.Restore(4, snap)
	if got := device.GetUsage(4); got != TextureUsesUninitialized|TextureUsesCopyDst {
		t.Errorf("restored state = %v, want what was saved", got)
	}
	device.Restore(9, device.Save(9))
	if device.IsTracked(9) {
		t.Error("restoring an untracked texture tracks it")
	}
}

func TestRecordTransitionOfSubresources(t *testing.T) {
	t.Parallel()

	tr := NewEncoderTextureTracker()
	tr.SetSubresourceUsage(0, full34, mip(0), TextureUsesColorTarget, nil)
	tr.RecordTransition(0, full34, TextureSelector{Mips: Range{0, 2}, Layers: full34.Layers}, TextureUsesColorTarget, TextureUsesCopySrc)

	device := NewTextureTracker()
	device.InsertSingle(0, TextureUsesResource)
	got := device.MergeEncoder(tr)
	want := []TexturePendingTransition{
		{Index: 0, Selector: mip(0), Usage: TextureStateTransition{From: TextureUsesResource, To: TextureUsesColorTarget}},
		{Index: 0, Selector: mip(1), Usage: TextureStateTransition{From: TextureUsesResource, To: TextureUsesColorTarget}},
	}
	if !slices.Equal(got, want) {
		t.Errorf("submit barriers %+v, want mip 1 to start where the explicit barrier expects it", got)
	}
}

func TestComplexStateIsolate(t *testing.T) {
	t.Parallel()

	var c complexState
	c.reset(TextureSubresources(1, 8), TextureUsesResource)
	c.set(TextureSelector{Mips: Range{0, 1}, Layers: Range{2, 5}}, TextureUsesCopyDst)
	c.set(TextureSelector{Mips: Range{0, 1}, Layers: Range{4, 6}}, TextureUsesCopySrc)
	want := []layerState{
		{Range{0, 2}, TextureUsesResource},
		{Range{2, 4}, TextureUsesCopyDst},
		{Range{4, 6}, TextureUsesCopySrc},
		{Range{6, 8}, TextureUsesResource},
	}
	if !slices.Equal(c.mips[0], want) {
		t.Fatalf("ranges %+v, want %+v", c.mips[0], want)
	}
	c.set(TextureSelector{Mips: Range{0, 1}, Layers: Range{1, 7}}, TextureUsesResource)
	if !slices.Equal(c.mips[0], []layerState{{Range{0, 8}, TextureUsesResource}}) {
		t.Errorf("ranges %+v, want them joined again", c.mips[0])
	}
	if uses, ok := c.uniform(); !ok || uses != TextureUsesResource {
		t.Errorf("uniform = %v, %v", uses, ok)
	}
}

func TestSelectorRanges(t *testing.T) {
	t.Parallel()

	if got := SelectorFromRange(full34, hal.TextureRange{BaseMipLevel: 1, BaseArrayLayer: 2, ArrayLayerCount: 1}); got != (TextureSelector{Mips: Range{1, 3}, Layers: Range{2, 3}}) {
		t.Errorf("SelectorFromRange = %+v", got)
	}
	if got := sub(1, 2).HALRange(); got != (hal.TextureRange{BaseMipLevel: 1, MipLevelCount: 1, BaseArrayLayer: 2, ArrayLayerCount: 1}) {
		t.Errorf("HALRange = %+v", got)
	}
	if got := (TextureSelector{}).HALRange(); got != (hal.TextureRange{}) {
		t.Errorf("HALRange of every subresource = %+v, want the whole texture", got)
	}
	if sel, ok := full34.within(full34); !ok || sel != (TextureSelector{}) {
		t.Errorf("full34 within itself = %+v, %v, want every subresource", sel, ok)
	}
}
