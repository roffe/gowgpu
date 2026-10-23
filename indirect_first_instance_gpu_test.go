//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// On a device without FeatureIndirectFirstInstance an indirect draw whose
// record has a first instance draws nothing, as WebGPU has it, and one
// without draws: the records are checked on the GPU ahead of the pass, the
// encoder's first pass and a later one alike.
func TestIndirectDrawWithAFirstInstanceIsANoOp(t *testing.T) {
	device, h := newValidatedDevice(t)
	switch {
	case device.Features().Contains(gputypes.FeatureIndirectFirstInstance):
		t.Skip("the device has FeatureIndirectFirstInstance")
	case h.backend == gputypes.BackendGL:
		t.Skip("the GL backend draws nothing indirect")
	}
	skipRootConstantIndirectOnWine(t, h)
	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: solidShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	pipeline, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Vertex: VertexState{Module: shader, EntryPoint: "vs_main"},
		Fragment: &FragmentState{Module: shader, EntryPoint: "fs_main", Targets: []gputypes.ColorTargetState{
			{Format: TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll},
		}},
		Primitive:   gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Multisample: gputypes.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipeline.Release()

	words := func(ws ...uint32) []byte {
		b := make([]byte, 0, 4*len(ws))
		for _, w := range ws {
			b = binary.LittleEndian.AppendUint32(b, w)
		}
		return b
	}
	buffer := func(usage gputypes.BufferUsage, data []byte) *Buffer {
		b, err := device.CreateBuffer(&BufferDescriptor{Size: uint64(len(data)), Usage: usage | BufferUsageCopyDst})
		if err != nil {
			t.Fatalf("CreateBuffer: %v", err)
		}
		t.Cleanup(b.Release)
		if err := device.Queue().WriteBuffer(b, 0, data); err != nil {
			t.Fatalf("WriteBuffer: %v", err)
		}
		return b
	}
	// draw records at 0 (first instance 1) and 16 (none); indexed ones at 32
	// (first instance 1) and 52 (none).
	records := buffer(BufferUsageIndirect, words(3, 1, 0, 1, 3, 1, 0, 0, 3, 1, 0, 0, 1, 3, 1, 0, 0, 0))
	index := buffer(BufferUsageIndex, words(0x00010000, 0x00000002))

	red := gputypes.Color{R: 1, A: 1}
	for _, tc := range []struct {
		name  string
		draws func(p *RenderPassEncoder)
		green bool
	}{
		{"draw with a first instance", func(p *RenderPassEncoder) { p.DrawIndirect(records, 0) }, false},
		{"draw without", func(p *RenderPassEncoder) { p.DrawIndirect(records, 16) }, true},
		{"indexed draw with a first instance", func(p *RenderPassEncoder) { p.DrawIndexedIndirect(records, 32) }, false},
		{"indexed draw without", func(p *RenderPassEncoder) { p.DrawIndexedIndirect(records, 52) }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, passes := range []int{1, 2} {
				tex, view := pixTexture(t, device, "first-instance", TextureUsageRenderAttachment|TextureUsageCopySrc)
				out := readbackBuffer(t, device, "first-instance")
				enc, err := device.CreateCommandEncoder(nil)
				if err != nil {
					t.Fatalf("CreateCommandEncoder: %v", err)
				}
				for i := range passes {
					last := i == passes-1
					clearPass(t, enc, view, gputypes.LoadOpClear, red, func(p *RenderPassEncoder) {
						if !last {
							return
						}
						p.SetPipeline(pipeline)
						p.SetIndexBuffer(index, gputypes.IndexFormatUint16, 0)
						tc.draws(p)
					})
				}
				copyOut(enc, tex, out)
				finishAndSubmit(t, device, enc)
				what := strings.Repeat("later ", passes-1) + "pass"
				if tc.green {
					expectPixels(t, what, readback(t, out), 0, 255, 0, 255)
				} else {
					expectPixels(t, what, readback(t, out), 255, 0, 0, 255)
				}
				releaseTextureView(tex, view)
			}
		})
	}
	assertNoValidation(t, h)
}
