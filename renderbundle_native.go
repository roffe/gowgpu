//go:build !(js && wasm)

package wgpu

import (
	"fmt"
	"slices"
	"sync/atomic"

	"github.com/roffe/gowgpu/gputypes"
)

// RenderBundleEncoder records reusable render commands.
type RenderBundleEncoder struct {
	device   *Device
	commands []func(*RenderPassEncoder)
	finished atomic.Bool
}

func (e *RenderBundleEncoder) record(command func(*RenderPassEncoder)) {
	if e.finished.Load() {
		return
	}
	e.commands = append(e.commands, command)
}

func (e *RenderBundleEncoder) SetPipeline(pipeline *RenderPipeline) {
	e.record(func(p *RenderPassEncoder) { p.SetPipeline(pipeline) })
}

func (e *RenderBundleEncoder) SetBindGroup(index uint32, group *BindGroup, offsets []uint32) {
	offsets = slices.Clone(offsets)
	e.record(func(p *RenderPassEncoder) { p.SetBindGroup(index, group, offsets) })
}

func (e *RenderBundleEncoder) SetVertexBuffer(slot uint32, buffer *Buffer, offset uint64) {
	e.record(func(p *RenderPassEncoder) { p.SetVertexBuffer(slot, buffer, offset) })
}

func (e *RenderBundleEncoder) SetIndexBuffer(buffer *Buffer, format gputypes.IndexFormat, offset uint64) {
	e.record(func(p *RenderPassEncoder) { p.SetIndexBuffer(buffer, format, offset) })
}

func (e *RenderBundleEncoder) Draw(args gputypes.DrawArgs) {
	e.record(func(p *RenderPassEncoder) { p.Draw(args) })
}

func (e *RenderBundleEncoder) DrawIndexed(args gputypes.DrawIndexedArgs) {
	e.record(func(p *RenderPassEncoder) { p.DrawIndexed(args) })
}

// Finish completes recording and returns the reusable render bundle.
func (e *RenderBundleEncoder) Finish(desc *RenderBundleDescriptor) (_ *RenderBundle, err error) {
	defer e.device.reportError(&err)
	if e == nil || e.finished.Swap(true) {
		return nil, fmt.Errorf("wgpu: RenderBundleEncoder.Finish: %w", ErrReleased)
	}

	label := ""
	if desc != nil {
		label = desc.Label
	}
	commands := e.commands
	e.commands = nil
	return &RenderBundle{commands: commands, device: e.device, label: label}, nil
}

// RenderBundle is a reusable sequence of render commands. A render pass
// executes it through its own encoder, so its commands are validated and
// their resources tracked as if the pass had recorded them.
type RenderBundle struct {
	commands []func(*RenderPassEncoder)
	device   *Device
	label    string
	released atomic.Bool
}

// Release marks the bundle released; render passes skip it after that. It is
// safe to call Release more than once.
func (b *RenderBundle) Release() {
	if b != nil {
		b.released.Store(true)
	}
}
