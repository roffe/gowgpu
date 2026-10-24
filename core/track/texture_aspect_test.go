//go:build !(js && wasm)

package track

import (
	"errors"
	"slices"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// A depth/stencil texture of 3 mip levels and 4 array layers.
var ds34 = TextureSelector{Mips: full34.Mips, Layers: full34.Layers, Aspects: AspectsDepthStencil}

func aspect(a Aspects) TextureSelector {
	return TextureSelector{Aspects: a}
}

func TestScopeAspectsUsedApart(t *testing.T) {
	t.Parallel()

	for _, tc := range []struct {
		name             string
		attached         TextureUses
		sampled, refused Aspects
	}{
		{"depth written, stencil sampled", DepthStencilUses(TextureUsesDepthStencilWrite, TextureUsesDepthStencilRead), AspectStencil, AspectDepth},
		{"stencil written, depth sampled", DepthStencilUses(TextureUsesDepthStencilRead, TextureUsesDepthStencilWrite), AspectDepth, AspectStencil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := NewTextureUsageScope()
			if err := s.SetSubresourceUsage(1, ds34, TextureSelector{}, tc.attached); err != nil {
				t.Fatalf("attach: %v", err)
			}
			if err := s.SetSubresourceUsage(1, ds34, aspect(tc.sampled), TextureUsesResource); err != nil {
				t.Fatalf("sample the read-only aspect: %v", err)
			}
			if got, want := s.GetUsage(1), TextureUsesDepthStencilWrite|TextureUsesDepthStencilRead|TextureUsesResource; got != want {
				t.Errorf("usage = %v, want %v", got, want)
			}

			var conflict *TextureUsageConflictError
			err := s.SetSubresourceUsage(1, ds34, aspect(tc.refused), TextureUsesResource)
			if !errors.As(err, &conflict) {
				t.Fatalf("sampling the written aspect = %v, want a conflict", err)
			}
			if conflict.Selector != aspect(tc.refused) || conflict.Existing != TextureUsesDepthStencilWrite || conflict.New != TextureUsesResource {
				t.Errorf("conflict = %+v, want the written aspect DepthStencilWrite/Resource", conflict)
			}
			if err := s.SetUsage(1, TextureUsesResource); err == nil {
				t.Error("sampling both aspects of a texture with one written is no conflict")
			}
		})
	}
}

func TestScopeAspectsOfSubresources(t *testing.T) {
	t.Parallel()

	s := NewTextureUsageScope()
	stencilMip1 := TextureSelector{Mips: Range{1, 2}, Layers: full34.Layers, Aspects: AspectStencil}
	if err := s.SetSubresourceUsage(0, ds34, stencilMip1, TextureUsesCopyDst); err != nil {
		t.Fatalf("write the stencil of mip 1: %v", err)
	}
	if err := s.SetSubresourceUsage(0, ds34, TextureSelector{Mips: Range{1, 2}, Layers: full34.Layers, Aspects: AspectDepth}, TextureUsesResource); err != nil {
		t.Errorf("sample the depth of mip 1 beside it: %v", err)
	}
	if err := s.SetSubresourceUsage(0, ds34, TextureSelector{Aspects: AspectStencil}, TextureUsesResource); err == nil {
		t.Error("sampling every stencil subresource, the written one too, is no conflict")
	}
}

func TestScopeAspectOfSingleAspectTexture(t *testing.T) {
	t.Parallel()

	s := NewTextureUsageScope()
	if err := s.SetSubresourceUsage(0, full34, TextureSelector{}, TextureUsesDepthStencilWrite); err != nil {
		t.Fatalf("attach: %v", err)
	}
	if err := s.SetSubresourceUsage(0, full34, aspect(AspectDepth), TextureUsesResource); err == nil {
		t.Error("a depth-only view of a texture without stencil is not all of it")
	}
}

func TestEncoderTrackerMovesAspectsTogether(t *testing.T) {
	t.Parallel()

	tr := NewEncoderTextureTracker()
	// A copy of the stencil starts both aspects as copy sources.
	tr.SetSubresourceUsage(3, ds34, aspect(AspectStencil), TextureUsesCopySrc, nil)
	if got := tr.StartUsage(3); got != TextureUsesCopySrc {
		t.Errorf("start = %v, want CopySrc", got)
	}
	if _, c := tr.end.get(3); c != nil {
		t.Error("end state is complex with both aspects copy sources")
	}

	attached := DepthStencilUses(TextureUsesDepthStencilWrite, TextureUsesDepthStencilRead|TextureUsesResource)
	pass := NewTextureUsageScope()
	_ = pass.SetSubresourceUsage(3, ds34, TextureSelector{}, DepthStencilUses(TextureUsesDepthStencilWrite, TextureUsesDepthStencilRead))
	_ = pass.SetSubresourceUsage(3, ds34, aspect(AspectStencil), TextureUsesResource)
	got := tr.SetFromScope(pass)
	want := []TexturePendingTransition{{Index: 3, Usage: TextureStateTransition{From: TextureUsesCopySrc, To: attached}}}
	if !slices.Equal(got, want) {
		t.Fatalf("pass barriers %+v, want %+v", got, want)
	}
	b := got[0].IntoHAL(nil)
	if b.Usage.Old != hal.TextureUsesCopySrc || b.Usage.New != hal.TextureUsesDepthStencilWrite ||
		b.Usage.StencilOld != hal.TextureUsesCopySrc || b.Usage.StencilNew != hal.TextureUsesDepthStencilRead|hal.TextureUsesResource {
		t.Errorf("HAL barrier usage %+v, want depth CopySrc->DepthStencilWrite, stencil CopySrc->DepthStencilRead|Resource", b.Usage)
	}
	if b.Usage.NewUsage != gputypes.TextureUsageRenderAttachment|gputypes.TextureUsageTextureBinding {
		t.Errorf("HAL barrier new usage %v, want the usages of both aspects", b.Usage.NewUsage)
	}

	// Sampling the depth alone takes the stencil along.
	pass.Clear()
	_ = pass.SetSubresourceUsage(3, ds34, aspect(AspectDepth), TextureUsesResource)
	got = tr.SetFromScope(pass)
	want = []TexturePendingTransition{{Index: 3, Usage: TextureStateTransition{From: attached, To: TextureUsesResource}}}
	if !slices.Equal(got, want) {
		t.Fatalf("later pass barriers %+v, want %+v", got, want)
	}
	b = got[0].IntoHAL(nil)
	if b.Usage.StencilOld != hal.TextureUsesDepthStencilRead|hal.TextureUsesResource || b.Usage.StencilNew != hal.TextureUsesResource {
		t.Errorf("HAL barrier usage %+v, want the stencil from its pass state to Resource", b.Usage)
	}
	if b = (TexturePendingTransition{Usage: TextureStateTransition{From: TextureUsesCopySrc, To: TextureUsesResource}}).IntoHAL(nil); b.Usage.StencilOld != 0 || b.Usage.StencilNew != 0 {
		t.Errorf("HAL barrier of aspects in one state %+v, want no stencil states", b.Usage)
	}

	device := NewTextureTracker()
	device.InsertSingle(3, TextureUsesUninitialized)
	got = device.MergeEncoder(tr)
	want = []TexturePendingTransition{{Index: 3, Usage: TextureStateTransition{From: TextureUsesUninitialized, To: TextureUsesCopySrc}}}
	if !slices.Equal(got, want) {
		t.Errorf("submit barriers %+v, want %+v", got, want)
	}
	if got := device.GetUsage(3); got != TextureUsesResource {
		t.Errorf("device state = %v, want Resource", got)
	}
}

func TestDeviceTrackerWriteOfOneAspect(t *testing.T) {
	t.Parallel()

	device := NewTextureTracker()
	device.InsertSingle(1, TextureUsesUninitialized)
	got := device.SetSubresourceUsage(1, ds34, TextureSelector{Mips: Range{0, 1}, Layers: Range{0, 1}, Aspects: AspectStencil}, TextureUsesCopyDst, nil)
	want := []TexturePendingTransition{{Index: 1, Selector: sub(0, 0), Usage: TextureStateTransition{From: TextureUsesUninitialized, To: TextureUsesCopyDst}}}
	if !slices.Equal(got, want) {
		t.Errorf("write barriers %+v, want %+v", got, want)
	}
}

func TestRecordTransitionOfOneAspect(t *testing.T) {
	t.Parallel()

	tr := NewEncoderTextureTracker()
	tr.RecordTransition(0, ds34, SelectorFromRange(ds34, hal.TextureRange{Aspect: gputypes.TextureAspectStencilOnly}), TextureUsesCopyDst, TextureUsesResource)
	if got := tr.StartUsage(0); got != TextureUsesCopyDst {
		t.Errorf("start = %v, want CopyDst on both aspects", got)
	}
	if got := tr.EndUsage(0); got != TextureUsesResource {
		t.Errorf("end = %v, want Resource on both aspects", got)
	}
}

func TestSelectorAspects(t *testing.T) {
	t.Parallel()

	depth, stencil := aspect(AspectDepth), aspect(AspectStencil)
	if depth.Overlaps(stencil) {
		t.Error("the depth and the stencil aspect overlap")
	}
	if !depth.Overlaps(TextureSelector{}) || !stencil.Overlaps(aspect(AspectsDepthStencil)) {
		t.Error("an aspect does not overlap every aspect")
	}
	if r := (TextureSelector{Mips: Range{1, 2}, Layers: Range{0, 4}, Aspects: AspectStencil}).HALRange(); r != (hal.TextureRange{Aspect: gputypes.TextureAspectStencilOnly, BaseMipLevel: 1, MipLevelCount: 1, ArrayLayerCount: 4}) {
		t.Errorf("HAL range %+v, want mip 1 of the stencil", r)
	}
	if got := SelectorFromRange(ds34, hal.TextureRange{Aspect: gputypes.TextureAspectDepthOnly, BaseMipLevel: 2}); got != (TextureSelector{Mips: Range{2, 3}, Layers: full34.Layers, Aspects: AspectDepth}) {
		t.Errorf("selector of a depth range = %+v", got)
	}
	if got := depth.String(); got != "every subresource of the depth aspect" {
		t.Errorf("String = %q", got)
	}
}

func TestSplitUsesSkipBarrier(t *testing.T) {
	t.Parallel()

	read := DepthStencilUses(TextureUsesDepthStencilRead, TextureUsesResource)
	if !SkipBarrier(read, read) {
		t.Error("aspects staying in read states take a barrier")
	}
	written := DepthStencilUses(TextureUsesDepthStencilWrite, TextureUsesResource)
	if !SkipBarrier(written, written) {
		t.Error("a depth attachment staying written beside a sampled stencil takes a barrier")
	}
	if SkipBarrier(written, read) {
		t.Error("a depth going from written to read takes no barrier")
	}
	copyDst := DepthStencilUses(TextureUsesCopyDst, TextureUsesResource)
	if SkipBarrier(copyDst, copyDst) {
		t.Error("a copy destination staying one takes no barrier")
	}
}
