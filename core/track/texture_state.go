//go:build !(js && wasm)

package track

import (
	"fmt"
	"slices"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// Range is the half-open range [Start, End) of the mip levels or the array
// layers of a texture.
type Range struct {
	Start, End uint32
}

func (r Range) intersect(o Range) Range {
	return Range{max(r.Start, o.Start), min(r.End, o.End)}
}

func (r Range) empty() bool {
	return r.Start >= r.End
}

// TextureSelector selects subresources of a texture: the array layers
// Layers of each of the mip levels Mips, of the aspects Aspects. Zero
// ranges select every level and layer, and zero aspects every aspect: the
// zero selector selects all of them. A 3D texture has one layer, its depth
// slices are not subresources.
//
// Reference: wgpu-types TextureSelector
type TextureSelector struct {
	Mips    Range
	Layers  Range
	Aspects Aspects
}

// Aspects selects the depth or the stencil aspect, or both, of a texture.
// Only a texture with both is used per aspect: it selects every subresource
// of the texture with AspectsDepthStencil, and the selection of one aspect
// of any other texture is the selection of all of it.
type Aspects uint8

// Aspects of a depth/stencil texture.
const (
	AspectDepth Aspects = 1 << iota
	AspectStencil

	AspectsDepthStencil = AspectDepth | AspectStencil
)

// AspectsOf is the selection of the aspect a, all of them for
// TextureAspectAll and TextureAspectUndefined.
func AspectsOf(a gputypes.TextureAspect) Aspects {
	switch a {
	case gputypes.TextureAspectDepthOnly:
		return AspectDepth
	case gputypes.TextureAspectStencilOnly:
		return AspectStencil
	}
	return 0
}

// uses is uses on the aspects sel selects of a texture of aspects a, the
// aspects it does not select left unknown.
func (a Aspects) uses(sel Aspects, uses TextureUses) TextureUses {
	if a != AspectsDepthStencil {
		return uses
	}
	switch sel {
	case AspectDepth:
		return joinAspects(uses, usesUnknown)
	case AspectStencil:
		return joinAspects(usesUnknown, uses)
	}
	return uses
}

// TextureSubresources selects every subresource of a texture of mips mip
// levels and layers array layers.
func TextureSubresources(mips, layers uint32) TextureSelector {
	return TextureSelector{Mips: Range{0, max(mips, 1)}, Layers: Range{0, max(layers, 1)}}
}

// SelectorFromRange is the selector of a HAL range of a texture whose
// subresources are full, a count of 0 meaning the levels or layers from
// the base on.
func SelectorFromRange(full TextureSelector, r hal.TextureRange) TextureSelector {
	aspects := AspectsOf(r.Aspect)
	if full.whole() {
		return TextureSelector{Aspects: aspects}
	}
	sel := TextureSelector{
		Mips:    Range{r.BaseMipLevel, full.Mips.End},
		Layers:  Range{r.BaseArrayLayer, full.Layers.End},
		Aspects: aspects,
	}
	if r.MipLevelCount != 0 {
		sel.Mips.End = r.BaseMipLevel + r.MipLevelCount
	}
	if r.ArrayLayerCount != 0 {
		sel.Layers.End = r.BaseArrayLayer + r.ArrayLayerCount
	}
	return sel
}

// whole reports whether s selects every mip level and layer.
func (s TextureSelector) whole() bool {
	return s.Mips == Range{} && s.Layers == Range{}
}

// Overlaps reports whether s and o select a subresource in common.
func (s TextureSelector) Overlaps(o TextureSelector) bool {
	if s.Aspects != 0 && o.Aspects != 0 && s.Aspects&o.Aspects == 0 {
		return false
	}
	if s.whole() || o.whole() {
		return true
	}
	return !s.Mips.intersect(o.Mips).empty() && !s.Layers.intersect(o.Layers).empty()
}

// HALRange is the HAL range of the subresources s selects.
func (s TextureSelector) HALRange() hal.TextureRange {
	var r hal.TextureRange
	switch s.Aspects {
	case AspectDepth:
		r.Aspect = gputypes.TextureAspectDepthOnly
	case AspectStencil:
		r.Aspect = gputypes.TextureAspectStencilOnly
	}
	if s.whole() {
		return r
	}
	r.BaseMipLevel, r.MipLevelCount = s.Mips.Start, s.Mips.End-s.Mips.Start
	r.BaseArrayLayer, r.ArrayLayerCount = s.Layers.Start, s.Layers.End-s.Layers.Start
	return r
}

// String formats s for error messages.
func (s TextureSelector) String() string {
	aspect := ""
	switch s.Aspects {
	case AspectDepth:
		aspect = " of the depth aspect"
	case AspectStencil:
		aspect = " of the stencil aspect"
	}
	if s.whole() {
		return "every subresource" + aspect
	}
	return fmt.Sprintf("mip levels %d..%d, layers %d..%d%s", s.Mips.Start, s.Mips.End, s.Layers.Start, s.Layers.End, aspect)
}

// within is sel clamped to full, the zero selector when it takes in every
// subresource, and false when it selects none. A texture whose subresources
// are not known is used whole. Aspects are left out.
func (s TextureSelector) within(full TextureSelector) (TextureSelector, bool) {
	s.Aspects, full.Aspects = 0, 0
	if s.whole() || full.whole() {
		return TextureSelector{}, true
	}
	s.Mips, s.Layers = s.Mips.intersect(full.Mips), s.Layers.intersect(full.Layers)
	if s.Mips.empty() || s.Layers.empty() {
		return s, false
	}
	if s == full {
		return TextureSelector{}, true
	}
	return s, true
}

// States the trackers keep that are no usage: usesUnknown is the state of
// a subresource a usage scope, or the start of a command buffer, says
// nothing about, which takes no barrier. usesComplex marks a texture whose
// subresources are not all in one state in a stateSet.
//
// A texture with both a depth and a stencil aspect whose aspects are in
// different states is in a state with usesSplit set: the depth aspect's
// state in the bits below it, the stencil aspect's above, from
// stencilShift. Every other state is the one of both aspects.
//
// Reference: wgpu-types TextureUses::UNKNOWN and TextureUses::COMPLEX
const (
	usesUnknown TextureUses = 1 << 14
	usesSplit   TextureUses = 1 << 15
	usesComplex TextureUses = 1 << 31

	stencilShift             = 16
	aspectUses   TextureUses = (1<<stencilShift - 1) &^ usesSplit
)

// aspects is the state of the depth and of the stencil aspect of u.
func (u TextureUses) aspects() (depth, stencil TextureUses) {
	if u&usesSplit == 0 {
		return u, u
	}
	return u & aspectUses, u >> stencilShift & aspectUses
}

// joinAspects is the state of a texture whose depth aspect is in depth and
// stencil aspect in stencil.
func joinAspects(depth, stencil TextureUses) TextureUses {
	if depth == stencil {
		return depth
	}
	return depth | usesSplit | stencil<<stencilShift
}

// DepthStencilUses is the usage of a texture with a depth and a stencil
// aspect that uses its depth aspect with depth and its stencil aspect with
// stencil.
func DepthStencilUses(depth, stencil TextureUses) TextureUses {
	return joinAspects(depth, stencil)
}

// any is every usage of an aspect of u, the unknown ones left out.
func (u TextureUses) any() TextureUses {
	depth, stencil := u.aspects()
	return (depth | stencil) &^ usesUnknown
}

// filled is u with an aspect u says nothing about in the state of the
// other. The trackers that make barriers move both aspects of a texture
// together where a command uses one of them, as Dawn does: Vulkan keeps
// the two aspects of an image in one layout.
func (u TextureUses) filled() TextureUses {
	if u&usesSplit == 0 {
		return u
	}
	depth, stencil := u.aspects()
	switch {
	case depth == usesUnknown:
		return stencil
	case stencil == usesUnknown:
		return depth
	}
	return u
}

// mergeUses is the state of a subresource in usage scope that has been
// used with cur and is used with next, aspect by aspect.
func mergeUses(cur, next TextureUses) TextureUses {
	if (cur|next)&usesSplit == 0 {
		return mergeAspect(cur, next)
	}
	cd, cs := cur.aspects()
	nd, ns := next.aspects()
	return joinAspects(mergeAspect(cd, nd), mergeAspect(cs, ns))
}

func mergeAspect(cur, next TextureUses) TextureUses {
	switch {
	case next == usesUnknown:
		return cur
	case cur == usesUnknown:
		return next
	}
	return cur | next
}

// conflicting reports a state no subresource can be in: an exclusive usage
// with any other on one aspect.
//
// Reference: wgpu-core track/texture.rs ResourceUses::is_invalid
func conflicting(u TextureUses) bool {
	if u&usesSplit != 0 {
		return conflictingAspects(u)
	}
	return conflictingAspect(u)
}

func conflictingAspects(u TextureUses) bool {
	depth, stencil := u.aspects()
	return conflictingAspect(depth) || conflictingAspect(stencil)
}

func conflictingAspect(u TextureUses) bool {
	return u.IsExclusive() && !isPowerOfTwo(uint32(u))
}

// newConflict is the error of the subresources sel selects of the texture
// at i, used with cur, being used with next too, naming the aspect they
// conflict on when the aspects are used apart.
func newConflict(i TrackerIndex, sel TextureSelector, cur, next TextureUses) *TextureUsageConflictError {
	if (cur|next)&usesSplit != 0 {
		cd, cs := cur.aspects()
		nd, ns := next.aspects()
		sel.Aspects, cur, next = AspectDepth, cd, nd
		if !conflictingAspect(mergeAspect(cd, nd)) {
			sel.Aspects, cur, next = AspectStencil, cs, ns
		}
	}
	return &TextureUsageConflictError{Index: i, Selector: sel, Existing: cur &^ usesUnknown, New: next &^ usesUnknown}
}

// layerState is the state of a range of array layers of one mip level.
type layerState struct {
	layers Range
	uses   TextureUses
}

// complexState is the state of a texture whose subresources are not all in
// one state: for each mip level, its array layers in ranges of one state,
// in order and covering every layer.
//
// Reference: wgpu-core track/texture.rs ComplexTextureState, RangedStates
type complexState struct {
	sel  TextureSelector
	mips [][]layerState
}

// reset makes every subresource of full, a texture's, be in uses.
func (c *complexState) reset(full TextureSelector, uses TextureUses) {
	c.grow(int(full.Mips.End))
	c.sel = full
	for i := range c.mips {
		c.mips[i] = append(c.mips[i][:0], layerState{full.Layers, uses})
	}
}

// copyFrom makes c the state src is.
func (c *complexState) copyFrom(src *complexState) {
	c.grow(len(src.mips))
	c.sel = src.sel
	for i := range src.mips {
		c.mips[i] = append(c.mips[i][:0], src.mips[i]...)
	}
}

// grow makes c have n mip levels, keeping the room of the ones it had.
func (c *complexState) grow(n int) {
	if cap(c.mips) < n {
		c.mips = append(c.mips[:cap(c.mips)], make([][]layerState, n-cap(c.mips))...)
	}
	c.mips = c.mips[:n]
}

// full selects every subresource of the texture.
func (c *complexState) full() TextureSelector {
	return c.sel
}

// clamp is sel within the subresources of c, every one of them for the
// zero selector, and false when it selects none of them.
func (c *complexState) clamp(sel TextureSelector) (TextureSelector, bool) {
	full := c.full()
	if sel.whole() {
		return full, true
	}
	sel.Mips, sel.Layers = sel.Mips.intersect(full.Mips), sel.Layers.intersect(full.Layers)
	return sel, !sel.Mips.empty() && !sel.Layers.empty()
}

// isolate splits the layer ranges of mip at the ends of layers and returns
// the ones that make up layers. They are c's: what is set in them is set in
// c.
func (c *complexState) isolate(mip uint32, layers Range) []layerState {
	states := c.mips[mip]
	i := 0
	for states[i].layers.End <= layers.Start {
		i++
	}
	if s := states[i]; s.layers.Start < layers.Start {
		states = slices.Insert(states, i+1, layerState{Range{layers.Start, s.layers.End}, s.uses})
		states[i].layers.End = layers.Start
		i++
	}
	j := i
	for states[j].layers.End < layers.End {
		j++
	}
	if s := states[j]; s.layers.End > layers.End {
		states = slices.Insert(states, j+1, layerState{Range{layers.End, s.layers.End}, s.uses})
		states[j].layers.End = layers.End
	}
	c.mips[mip] = states
	return states[i : j+1]
}

// set puts the subresources sel selects in uses.
func (c *complexState) set(sel TextureSelector, uses TextureUses) {
	sel, ok := c.clamp(sel)
	if !ok {
		return
	}
	for mip := sel.Mips.Start; mip < sel.Mips.End; mip++ {
		states := c.isolate(mip, sel.Layers)
		for i := range states {
			states[i].uses = uses
		}
		c.coalesce(mip)
	}
}

// coalesce joins the neighboring layer ranges of mip that are in one
// state.
func (c *complexState) coalesce(mip uint32) {
	states := c.mips[mip]
	n := 1
	for _, s := range states[1:] {
		if last := &states[n-1]; last.uses == s.uses {
			last.layers.End = s.layers.End
			continue
		}
		states[n] = s
		n++
	}
	c.mips[mip] = states[:n]
}

// uniform returns the state every subresource is in, when they are all in
// one known state.
func (c *complexState) uniform() (TextureUses, bool) {
	uses := c.mips[0][0].uses
	if uses == usesUnknown {
		return 0, false
	}
	for _, states := range c.mips {
		if len(states) != 1 || states[0].uses != uses {
			return 0, false
		}
	}
	return uses, true
}

// union is every state some subresource is in, the unknown ones left out.
func (c *complexState) union() TextureUses {
	var uses TextureUses
	for _, states := range c.mips {
		for _, s := range states {
			uses |= s.uses.any()
		}
	}
	return uses
}

// fill puts the aspect of each range of c it says nothing about in the
// state of the other.
func (c *complexState) fill() {
	for mip, states := range c.mips {
		for i := range states {
			states[i].uses = states[i].uses.filled()
		}
		c.coalesce(uint32(mip))
	}
}

// each calls f with the selector and state of every range of c.
func (c *complexState) each(f func(TextureSelector, TextureUses)) {
	for mip, states := range c.mips {
		for _, s := range states {
			f(TextureSelector{Mips: Range{uint32(mip), uint32(mip) + 1}, Layers: s.layers}, s.uses)
		}
	}
}

// stateSet is the state of every texture a tracker knows, by tracker index:
// one usage for a texture whose subresources are all in one state, or
// usesComplex for one whose are not, whose state is then in complex. The
// complex states of textures it stops tracking are kept in spare for the
// next ones.
//
// Reference: wgpu-core track/texture.rs TextureStateSet
type stateSet struct {
	simple  []TextureUses
	complex map[TrackerIndex]*complexState
	spare   []*complexState
}

func (s *stateSet) ensureSize(n int) {
	for len(s.simple) < n {
		s.simple = append(s.simple, TextureUsesNone)
	}
}

// get returns the state of the texture at i: its one usage, or its complex
// state.
func (s *stateSet) get(i TrackerIndex) (TextureUses, *complexState) {
	if uses := s.simple[i]; uses != usesComplex {
		return uses, nil
	}
	return usesComplex, s.complex[i]
}

// union is every state a subresource of the texture at i is in.
func (s *stateSet) union(i TrackerIndex) TextureUses {
	uses, c := s.get(i)
	if c != nil {
		return c.union()
	}
	return uses.any()
}

func (s *stateSet) setSimple(i TrackerIndex, uses TextureUses) {
	if s.simple[i] == usesComplex {
		s.release(i)
	}
	s.simple[i] = uses
}

func (s *stateSet) newComplex() *complexState {
	if n := len(s.spare); n > 0 {
		c := s.spare[n-1]
		s.spare[n-1] = nil
		s.spare = s.spare[:n-1]
		return c
	}
	return new(complexState)
}

// setComplex makes c the state of the texture at i, or the one usage of
// it when its subresources are all in one.
func (s *stateSet) setComplex(i TrackerIndex, c *complexState) {
	if uses, ok := c.uniform(); ok {
		s.spare = append(s.spare, c)
		s.setSimple(i, uses)
		return
	}
	if s.simple[i] == usesComplex {
		if old := s.complex[i]; old != c {
			s.spare = append(s.spare, old)
		}
	}
	if s.complex == nil {
		s.complex = make(map[TrackerIndex]*complexState)
	}
	s.simple[i] = usesComplex
	s.complex[i] = c
}

func (s *stateSet) release(i TrackerIndex) {
	if c := s.complex[i]; c != nil {
		s.spare = append(s.spare, c)
		delete(s.complex, i)
	}
}

func (s *stateSet) remove(i TrackerIndex) {
	if int(i) < len(s.simple) {
		s.setSimple(i, TextureUsesNone)
	}
}

func (s *stateSet) clear() {
	for i, c := range s.complex {
		s.spare = append(s.spare, c)
		delete(s.complex, i)
	}
	s.simple = s.simple[:0]
}

// stateSource is a state an operation brings to a texture: uses for the
// subresources sel selects of a texture whose subresources are full, or
// another tracker's state of the texture, the complex one when complex is
// set, its ranges filled when fill is.
//
// Reference: wgpu-core track/texture.rs TextureStateProvider
type stateSource struct {
	full    TextureSelector
	sel     TextureSelector
	uses    TextureUses
	complex *complexState
	fill    bool
}

// selection is the source of uses on the subresources sel selects of a
// texture whose subresources are full. It is false when sel selects none.
func selection(full, sel TextureSelector, uses TextureUses) (stateSource, bool) {
	uses = full.Aspects.uses(sel.Aspects, uses)
	full.Aspects = 0
	sel, ok := sel.within(full)
	return stateSource{full: full, sel: sel, uses: uses}, ok
}

// filled is the source with the aspect of a state it says nothing about in
// the state of the other, for a tracker that makes barriers.
func (src stateSource) filled() stateSource {
	if src.complex != nil {
		src.fill = true
	} else {
		src.uses = src.uses.filled()
	}
	return src
}

// sourceOf is the source of the state s has for the texture at i.
func sourceOf(s *stateSet, i TrackerIndex) stateSource {
	uses, c := s.get(i)
	return stateSource{uses: uses, complex: c}
}

// simple reports whether the source sets every subresource to one state,
// uses.
func (src stateSource) simple() bool {
	return src.complex == nil && src.sel.whole()
}

func (src stateSource) fullRange() TextureSelector {
	if src.complex != nil {
		return src.complex.full()
	}
	return src.full
}

// each calls f with every selection of the source and its state.
func (src stateSource) each(f func(TextureSelector, TextureUses)) {
	switch {
	case src.complex != nil && src.fill:
		src.complex.each(func(sel TextureSelector, uses TextureUses) { f(sel, uses.filled()) })
	case src.complex != nil:
		src.complex.each(f)
	default:
		f(src.sel, src.uses)
	}
}

// insert makes src the state of the texture at i, which s does not track:
// subresources src says nothing about are unknown.
//
// Reference: wgpu-core track/texture.rs insert
func (s *stateSet) insert(i TrackerIndex, src stateSource) {
	switch {
	case src.simple():
		s.setSimple(i, src.uses)
	case src.complex != nil:
		c := s.newComplex()
		c.copyFrom(src.complex)
		if src.fill {
			c.fill()
		}
		s.setComplex(i, c)
	default:
		c := s.newComplex()
		c.reset(src.full, usesUnknown)
		c.set(src.sel, src.uses)
		s.setComplex(i, c)
	}
}

// merge adds the usages src brings to those of the texture at i in one
// usage scope, failing on a subresource whose usages conflict.
//
// Reference: wgpu-core track/texture.rs merge
func (s *stateSet) merge(i TrackerIndex, src stateSource) error {
	cur, c := s.get(i)
	switch {
	case c == nil && src.simple():
		merged := mergeUses(cur, src.uses)
		if conflicting(merged) {
			return newConflict(i, TextureSelector{}, cur, src.uses)
		}
		s.simple[i] = merged
		return nil
	case c == nil:
		next := s.newComplex()
		next.reset(src.fullRange(), cur)
		var err error
		src.each(func(sel TextureSelector, uses TextureUses) {
			if err != nil || uses == usesUnknown {
				return
			}
			merged := mergeUses(cur, uses)
			if conflicting(merged) {
				err = newConflict(i, sel, cur, uses)
				return
			}
			next.set(sel, merged)
		})
		if err != nil {
			s.spare = append(s.spare, next)
			return err
		}
		s.setComplex(i, next)
		return nil
	}
	var err error
	src.each(func(sel TextureSelector, uses TextureUses) {
		sel, ok := c.clamp(sel)
		if err != nil || !ok || uses == usesUnknown {
			return
		}
		for mip := sel.Mips.Start; mip < sel.Mips.End; mip++ {
			states := c.isolate(mip, sel.Layers)
			for k := range states {
				merged := mergeUses(states[k].uses, uses)
				if merged == usesUnknown {
					continue
				}
				if conflicting(merged) {
					err = newConflict(i, TextureSelector{Mips: Range{mip, mip + 1}, Layers: states[k].layers}, states[k].uses, uses)
					return
				}
				states[k].uses = merged
			}
			c.coalesce(mip)
		}
	})
	if err == nil {
		s.setComplex(i, c)
	}
	return err
}

// barrier appends to dst the transitions that take the subresources src
// brings a state for from their state in s, for the texture at i, to that
// state. Unknown states on either side take none.
//
// Reference: wgpu-core track/texture.rs barrier
func (s *stateSet) barrier(i TrackerIndex, src stateSource, dst []TexturePendingTransition) []TexturePendingTransition {
	add := func(sel TextureSelector, from, to TextureUses) {
		if from == usesUnknown || to == usesUnknown || SkipBarrier(from, to) {
			return
		}
		dst = append(dst, TexturePendingTransition{Index: i, Selector: sel, Usage: TextureStateTransition{From: from, To: to}})
	}
	cur, c := s.get(i)
	switch {
	case c == nil:
		src.each(func(sel TextureSelector, uses TextureUses) { add(sel, cur, uses) })
	case src.simple():
		c.each(func(sel TextureSelector, uses TextureUses) { add(sel, uses, src.uses) })
	default:
		src.each(func(sel TextureSelector, uses TextureUses) {
			sel, ok := c.clamp(sel)
			if !ok || uses == usesUnknown {
				return
			}
			for mip := sel.Mips.Start; mip < sel.Mips.End; mip++ {
				for _, st := range c.mips[mip] {
					if r := st.layers.intersect(sel.Layers); !r.empty() {
						add(TextureSelector{Mips: Range{mip, mip + 1}, Layers: r}, st.uses, uses)
					}
				}
			}
		})
	}
	return dst
}

// update puts the subresources of the texture at i that src brings a state
// for in that state in s, and leaves the rest as they are. When start is
// given, the subresources whose state s did not know until now take
// startUses there, or src's state when startUses is 0: that is the state a
// command buffer first uses them in.
//
// Reference: wgpu-core track/texture.rs update
func (s *stateSet) update(i TrackerIndex, src stateSource, start *stateSet, startUses TextureUses) {
	cur, c := s.get(i)
	if c == nil && src.simple() {
		s.simple[i] = src.uses
		return
	}
	fillStart := func(mip uint32, layers Range, uses TextureUses) {
		if start == nil {
			return
		}
		if startUses != 0 {
			uses = startUses
		}
		if _, sc := start.get(i); sc != nil {
			sc.set(TextureSelector{Mips: Range{mip, mip + 1}, Layers: layers}, uses)
		}
	}
	if c == nil {
		next := s.newComplex()
		next.reset(src.fullRange(), cur)
		src.each(func(sel TextureSelector, uses TextureUses) {
			if uses == usesUnknown {
				return
			}
			next.set(sel, uses)
		})
		s.setComplex(i, next)
		return
	}
	src.each(func(sel TextureSelector, uses TextureUses) {
		sel, ok := c.clamp(sel)
		if !ok || uses == usesUnknown {
			return
		}
		for mip := sel.Mips.Start; mip < sel.Mips.End; mip++ {
			states := c.isolate(mip, sel.Layers)
			for k := range states {
				if states[k].uses == usesUnknown {
					fillStart(mip, states[k].layers, uses)
				}
				states[k].uses = uses
			}
			c.coalesce(mip)
		}
	})
	if start != nil {
		if _, sc := start.get(i); sc != nil {
			start.setComplex(i, sc)
		}
	}
	s.setComplex(i, c)
}

// snapshot is a copy of the state of one texture of a stateSet.
type snapshot struct {
	uses    TextureUses
	complex *complexState
}

func (s *stateSet) snapshot(i TrackerIndex) snapshot {
	uses, c := s.get(i)
	if c == nil {
		return snapshot{uses: uses}
	}
	cp := new(complexState)
	cp.copyFrom(c)
	return snapshot{uses: usesComplex, complex: cp}
}

func (s *stateSet) restore(i TrackerIndex, snap snapshot) {
	if snap.complex == nil {
		s.setSimple(i, snap.uses)
		return
	}
	c := s.newComplex()
	c.copyFrom(snap.complex)
	s.setComplex(i, c)
}
