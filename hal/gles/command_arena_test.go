//go:build (windows || linux) && !(js && wasm)

package gles

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

var (
	arenaTestPipeline = &RenderPipeline{programID: 7}
	arenaTestGroup    = &BindGroup{}
)

// recordFrame records n pipeline, bind group and draw triples into one pass.
func recordFrame(t *testing.T, enc *CommandEncoder, n int, vertexCount uint32) hal.CommandBuffer {
	t.Helper()
	if err := enc.BeginEncoding("frame"); err != nil {
		t.Fatal(err)
	}
	pipeline, group := arenaTestPipeline, arenaTestGroup
	pass := enc.BeginRenderPass(&hal.RenderPassDescriptor{})
	for range n {
		pass.SetPipeline(pipeline)
		pass.SetBindGroup(0, group, nil)
		pass.Draw(gputypes.DrawArgs{VertexCount: vertexCount, InstanceCount: 1})
	}
	pass.End()
	cb, err := enc.EndEncoding()
	if err != nil {
		t.Fatal(err)
	}
	return cb
}

func drawCounts(cb hal.CommandBuffer) []uint32 {
	var counts []uint32
	for _, cmd := range cb.(*CommandBuffer).commands {
		if d, ok := cmd.(*DrawCommand); ok {
			counts = append(counts, d.vertexCount)
		}
	}
	return counts
}

// A command buffer keeps its commands until ResetAll hands its storage back,
// even when the encoder records again before that.
func TestCommandArenaKeepsCommandsUntilResetAll(t *testing.T) {
	enc := &CommandEncoder{}
	first := recordFrame(t, enc, 100, 3)
	second := recordFrame(t, enc, 100, 4)

	for _, c := range drawCounts(first) {
		if c != 3 {
			t.Fatalf("first command buffer draws %d vertices, want 3", c)
		}
	}
	if got := len(drawCounts(second)); got != 100 {
		t.Fatalf("second command buffer has %d draws, want 100", got)
	}

	enc.ResetAll([]hal.CommandBuffer{first, second})
	if first.(*CommandBuffer).commands != nil || first.(*CommandBuffer).arena != nil {
		t.Fatal("ResetAll left the storage on the command buffer")
	}
	third := recordFrame(t, enc, 100, 5)
	for _, c := range drawCounts(third) {
		if c != 5 {
			t.Fatalf("third command buffer draws %d vertices, want 5", c)
		}
	}
	for _, c := range drawCounts(second) {
		if c != 4 {
			t.Fatalf("second command buffer draws %d vertices after a reuse, want 4", c)
		}
	}
}

// A steady loop of record, run and ResetAll allocates nothing per command.
func TestCommandArenaSteadyStateAllocations(t *testing.T) {
	enc := &CommandEncoder{}
	enc.ResetAll([]hal.CommandBuffer{recordFrame(t, enc, 200, 3)})
	allocs := testing.AllocsPerRun(20, func() {
		enc.ResetAll([]hal.CommandBuffer{recordFrame(t, enc, 200, 3)})
	})
	// The pass descriptor, the pass encoder and the command buffer, not the
	// 600 commands.
	if allocs > 3 {
		t.Fatalf("a 600 command frame allocates %.0f times, want at most 3", allocs)
	}
}

// Taking storage back clears it, so it stops pinning GL objects.
func TestCommandArenaResetClearsReferences(t *testing.T) {
	enc := &CommandEncoder{}
	cb := recordFrame(t, enc, 10, 3).(*CommandBuffer)
	arena := cb.arena
	commands := cb.commands[:cap(cb.commands)]
	enc.ResetAll([]hal.CommandBuffer{cb})
	for i, c := range commands {
		if c != nil {
			t.Fatalf("command %d still set after ResetAll", i)
		}
	}
	if arena.bindGroups.chunks[0][0].group != nil || arena.bindGroups.used != 0 {
		t.Fatal("bind group command still holds its group after ResetAll")
	}
}
