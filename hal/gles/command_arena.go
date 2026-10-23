//go:build (windows || linux) && !(js && wasm)

package gles

// slabChunk is how many commands of one type a slab chunk holds.
const slabChunk = 64

// slab hands out commands of one type from fixed size chunks. A chunk never
// grows, so a pointer it handed out stays valid until reset, and the chunks
// are kept for the next recording.
type slab[T any] struct {
	chunks [][]T
	used   int // commands handed out since the last reset
}

// put stores v in the next free slot and returns its address.
func (s *slab[T]) put(v T) *T {
	i := s.used / slabChunk
	if i == len(s.chunks) {
		s.chunks = append(s.chunks, make([]T, slabChunk))
	}
	p := &s.chunks[i][s.used%slabChunk]
	*p = v
	s.used++
	return p
}

// reset zeroes the commands handed out, so they stop pinning buffers,
// textures and pipelines for the GC, and makes the slots free again.
func (s *slab[T]) reset() {
	for i := 0; i*slabChunk < s.used; i++ {
		clear(s.chunks[i][:min(slabChunk, s.used-i*slabChunk)])
	}
	s.used = 0
}

// cmdArena holds the commands a frame records over and over: the per draw
// and per dispatch ones and the ones every render pass begins and ends with.
// Recording one of those takes a slot here instead of a heap allocation. The
// arena goes to the command buffer at EndEncoding and comes back to the
// encoder at ResetAll, once the command buffer has run, so a steady stream of
// frames allocates nothing for them. Other commands are rare and still
// allocate.
type cmdArena struct {
	draws          slab[DrawCommand]
	drawsIndexed   slab[DrawIndexedCommand]
	programs       slab[UseProgramCommand]
	pipelineStates slab[SetPipelineStateCommand]
	bindGroups     slab[SetBindGroupCommand]
	vertexBuffers  slab[SetVertexBufferCommand]
	indexBuffers   slab[SetIndexBufferCommand]
	viewports      slab[SetViewportCommand]
	scissors       slab[SetScissorCommand]
	vaos           slab[BindVAOCommand]
	framebuffers   slab[BindFramebufferCommand]
	surfaceFBOs    slab[BindSurfaceFramebufferCommand]
	renderTargets  slab[BindRenderTargetsCommand]
	drawBuffers    slab[SetDrawColorBuffersCommand]
	clears         slab[ClearColorBufferCommand]
	dispatches     slab[DispatchCommand]
	attribs        slab[DisableVertexAttribsCommand]
}

func (a *cmdArena) reset() {
	a.draws.reset()
	a.drawsIndexed.reset()
	a.programs.reset()
	a.pipelineStates.reset()
	a.bindGroups.reset()
	a.vertexBuffers.reset()
	a.indexBuffers.reset()
	a.viewports.reset()
	a.scissors.reset()
	a.vaos.reset()
	a.attribs.reset()
	a.framebuffers.reset()
	a.surfaceFBOs.reset()
	a.renderTargets.reset()
	a.drawBuffers.reset()
	a.clears.reset()
	a.dispatches.reset()
}

// store returns the arena the current recording puts its commands in,
// taking the spare one ResetAll handed back when there is one.
func (e *CommandEncoder) store() *cmdArena {
	if e.arena == nil {
		e.arena, e.spareArena = e.spareArena, nil
		if e.arena == nil {
			e.arena = new(cmdArena)
		}
	}
	return e.arena
}

// reclaim takes back the storage of a command buffer that has run: its arena
// and the backing array of its command list, both cleared. One of each is
// kept as the spare for the next recording.
func (e *CommandEncoder) reclaim(cb *CommandBuffer) {
	if cb.arena != nil {
		cb.arena.reset()
		if e.spareArena == nil {
			e.spareArena = cb.arena
		}
		cb.arena = nil
	}
	if cb.commands != nil {
		clear(cb.commands)
		if cap(cb.commands) > cap(e.spareCommands) {
			e.spareCommands = cb.commands[:0]
		}
		cb.commands = nil
	}
}
