//go:build !(js && wasm)

package core

import (
	"fmt"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/hal"
)

// indirectCountInactive reports whether an indirect-count draw should be skipped.
func (p *CoreRenderPassEncoder) indirectCountInactive(maxDrawCount uint32) bool {
	return p.ended || maxDrawCount == 0
}

type indirectCountHALForwarder func(halIndirect, halCount hal.Buffer, indirectOffset, countOffset uint64, maxDrawCount uint32)

func (p *CoreRenderPassEncoder) forwardIndirectCount(
	indirectBuffer, countBuffer *Buffer,
	indirectOffset, countOffset uint64,
	maxDrawCount uint32,
	forward indirectCountHALForwarder,
) {
	if p.indirectCountInactive(maxDrawCount) || p.raw == nil || indirectBuffer == nil || countBuffer == nil {
		return
	}
	guard := p.device.snatchLock.Read()
	defer guard.Release()
	halIndirect := indirectBuffer.Raw(guard)
	halCount := countBuffer.Raw(guard)
	// Both are read as indirect arguments, so both join the pass's usage
	// scope as such.
	if err := p.recordBufferUsage(indirectBuffer, halIndirect, track.BufferUsesIndirect); err != nil {
		p.encoder.SetError(fmt.Errorf("DrawIndirectCount: indirect buffer: %w", err))
		return
	}
	if err := p.recordBufferUsage(countBuffer, halCount, track.BufferUsesIndirect); err != nil {
		p.encoder.SetError(fmt.Errorf("DrawIndirectCount: count buffer: %w", err))
		return
	}
	if halIndirect != nil && halCount != nil {
		forward(halIndirect, halCount, indirectOffset, countOffset, maxDrawCount)
	}
}
