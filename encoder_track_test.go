//go:build !(js && wasm)

package wgpu

import (
	"sync/atomic"
	"testing"

	"github.com/roffe/gowgpu/core"
)

func trackTestBuffer() *Buffer {
	return &Buffer{core: &core.Buffer{Ref: core.NewResourceRef("buffer", nil)}}
}

// An encoder clones a resource's ref and lists it for Submit once, however
// often a frame binds it.
func TestEncoderTracksEachResourceOnce(t *testing.T) {
	uniforms := trackTestBuffer()
	vertices := trackTestBuffer()
	group := &BindGroup{ref: core.NewResourceRef("group", nil), boundBuffers: []*Buffer{uniforms}}
	pipeline := &RenderPipeline{ref: core.NewResourceRef("pipeline", nil)}
	tex := &Texture{}

	enc := &CommandEncoder{}
	for range 100 {
		enc.trackPipeline(&pipeline.mark, pipeline.ref)
		enc.trackBindGroup(group)
		enc.trackBuffer(vertices)
		enc.trackBuffer(uniforms)
		enc.trackTexture(tex)
	}

	for name, ref := range map[string]*core.ResourceRef{
		"pipeline": pipeline.ref, "group": group.ref, "uniforms": uniforms.core.Ref, "vertices": vertices.core.Ref,
	} {
		if got := ref.RefCount(); got != 2 {
			t.Errorf("%s ref count = %d, want 2 (owner plus one clone)", name, got)
		}
	}
	if got := len(enc.trackedRefs); got != 4 {
		t.Errorf("tracked refs = %d, want 4", got)
	}
	if len(enc.usedBindGroups) != 1 || len(enc.usedTextures) != 1 {
		t.Errorf("used bind groups = %d, textures = %d, want 1 and 1", len(enc.usedBindGroups), len(enc.usedTextures))
	}
	// The uniform buffer came in through the group, which Submit walks.
	if len(enc.usedBuffers) != 1 || enc.usedBuffers[0] != vertices {
		t.Errorf("used buffers = %v, want just the vertex buffer", enc.usedBuffers)
	}
}

// Encoders track independently: a second encoder using the same resources
// takes its own clones, and the first one tracking them again after that
// repeats an entry rather than missing one.
func TestEncodersTrackIndependently(t *testing.T) {
	buf := trackTestBuffer()
	group := &BindGroup{ref: core.NewResourceRef("group", nil)}

	first, second := &CommandEncoder{}, &CommandEncoder{}
	first.trackBindGroup(group)
	first.trackBuffer(buf)
	second.trackBindGroup(group)
	second.trackBuffer(buf)
	if len(second.trackedRefs) != 2 || len(second.usedBindGroups) != 1 || len(second.usedBuffers) != 1 {
		t.Fatalf("second encoder tracked refs=%d groups=%d buffers=%d, want 2, 1, 1",
			len(second.trackedRefs), len(second.usedBindGroups), len(second.usedBuffers))
	}

	first.trackBindGroup(group)
	first.trackBuffer(buf)
	if len(first.trackedRefs) != 4 || len(first.usedBindGroups) != 2 || len(first.usedBuffers) != 2 {
		t.Fatalf("first encoder tracked refs=%d groups=%d buffers=%d, want 4, 2, 2",
			len(first.trackedRefs), len(first.usedBindGroups), len(first.usedBuffers))
	}
	// Every clone is listed, so dropping the lists balances the counts.
	for _, enc := range []*CommandEncoder{first, second} {
		for _, ref := range enc.trackedRefs {
			ref.Drop()
		}
	}
	if group.ref.RefCount() != 1 || buf.core.Ref.RefCount() != 1 {
		t.Fatalf("ref counts after dropping = %d, %d, want 1, 1", group.ref.RefCount(), buf.core.Ref.RefCount())
	}
}

// A buffer the encoder first met through a bind group is still checked at
// Submit when it is also bound directly: through the group.
func TestSubmitValidatesBufferReachedThroughGroup(t *testing.T) {
	buf := trackTestBuffer()
	buf.released = new(atomic.Bool)
	group := &BindGroup{ref: core.NewResourceRef("group", nil), boundBuffers: []*Buffer{buf}}

	enc := &CommandEncoder{}
	enc.trackBindGroup(group)
	enc.trackBuffer(buf) // e.g. SetVertexBuffer with the same buffer
	buf.released.Store(true)

	cb := &CommandBuffer{usedBuffers: enc.usedBuffers, usedBindGroups: enc.usedBindGroups}
	if err := validateCommandBufferForSubmit(cb, 0); err == nil {
		t.Fatal("Submit accepted a command buffer using a released buffer")
	}
}
