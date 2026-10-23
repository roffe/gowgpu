//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"math"
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
	if device.Features().Contains(gputypes.FeatureIndirectFirstInstance) {
		t.Skip("the device has FeatureIndirectFirstInstance")
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

// instanceColorShader draws the triangle of its vertex buffer in its
// per-instance colour and reads no builtin.
const instanceColorShader = `
struct VSOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec4<f32> }
@vertex fn vs_main(@location(0) pos: vec2<f32>, @location(1) color: vec4<f32>) -> VSOut {
	return VSOut(vec4<f32>(pos, 0.0, 1.0), color);
}
@fragment fn fs_main(in: VSOut) -> @location(0) vec4<f32> { return in.color; }
`

// On a device with FeatureIndirectFirstInstance an indirect draw takes its
// first instance from the record: a per-instance vertex buffer is read from
// that instance on and instance_index counts from it, which DX12 hands the
// shader in its special constants.
func TestIndirectDrawsFromAFirstInstance(t *testing.T) {
	device, h := newValidatedDeviceWith(t, gputypes.Features(gputypes.FeatureIndirectFirstInstance))
	pipeline := func(wgsl string, buffers []gputypes.VertexBufferLayout) *RenderPipeline {
		t.Helper()
		module, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: wgsl})
		if err != nil {
			t.Fatalf("CreateShaderModule: %v", err)
		}
		t.Cleanup(module.Release)
		p, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
			Vertex: VertexState{Module: module, EntryPoint: "vs_main", Buffers: buffers},
			Fragment: &FragmentState{Module: module, EntryPoint: "fs_main", Targets: []gputypes.ColorTargetState{
				{Format: TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll},
			}},
			Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		})
		if err != nil {
			t.Fatalf("CreateRenderPipeline: %v", err)
		}
		t.Cleanup(p.Release)
		return p
	}
	fromBuffers := pipeline(instanceColorShader, []gputypes.VertexBufferLayout{
		{ArrayStride: 8, Attributes: []gputypes.VertexAttribute{{Format: gputypes.VertexFormatFloat32x2, ShaderLocation: 0}}},
		{ArrayStride: 16, StepMode: gputypes.VertexStepModeInstance, Attributes: []gputypes.VertexAttribute{{Format: gputypes.VertexFormatFloat32x4, ShaderLocation: 1}}},
	})
	fromBuiltins := pipeline(vertexIndexShader, nil)

	bytes := func(ws ...float32) []byte {
		b := make([]byte, 0, 4*len(ws))
		for _, w := range ws {
			b = binary.LittleEndian.AppendUint32(b, math.Float32bits(w))
		}
		return b
	}
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
	positions := buffer(BufferUsageVertex, bytes(-1, -1, 3, -1, -1, 3))
	colors := buffer(BufferUsageVertex, bytes(1, 0, 0, 1, 0, 1, 0, 1))
	// draws at 0 (from vertex 0) and 16 (from vertex 3), indexed draws at 32
	// (base vertex 0) and 52 (base vertex 3), all from instance 1.
	records := buffer(BufferUsageIndirect, words(3, 1, 0, 1, 3, 1, 3, 1, 3, 1, 0, 0, 1, 3, 1, 0, 3, 1))
	index := buffer(BufferUsageIndex, words(0x00010000, 0x00000002))

	for _, tc := range []struct {
		name     string
		pipeline *RenderPipeline
		draw     func(p *RenderPassEncoder)
		blue     byte
	}{
		{"draw", fromBuffers, func(p *RenderPassEncoder) { p.DrawIndirect(records, 0) }, 0},
		{"indexed draw", fromBuffers, func(p *RenderPassEncoder) { p.DrawIndexedIndirect(records, 32) }, 0},
		{"draw reading instance_index", fromBuiltins, func(p *RenderPassEncoder) { p.DrawIndirect(records, 16) }, 255},
		{"indexed draw reading instance_index", fromBuiltins, func(p *RenderPassEncoder) { p.DrawIndexedIndirect(records, 52) }, 255},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.pipeline == fromBuiltins {
				skipRootConstantIndirectOnWine(t, h)
			}
			tex, view := pixTexture(t, device, "first-instance", TextureUsageRenderAttachment|TextureUsageCopySrc)
			defer releaseTextureView(tex, view)
			out := readbackBuffer(t, device, "first-instance")
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			clearPass(t, enc, view, gputypes.LoadOpClear, black, func(p *RenderPassEncoder) {
				p.SetPipeline(tc.pipeline)
				if tc.pipeline == fromBuffers {
					p.SetVertexBuffer(0, positions, 0)
					p.SetVertexBuffer(1, colors, 0)
				}
				p.SetIndexBuffer(index, gputypes.IndexFormatUint16, 0)
				tc.draw(p)
			})
			copyOut(enc, tex, out)
			finishAndSubmit(t, device, enc)
			expectPixels(t, tc.name, readback(t, out), 0, 255, tc.blue, 255)
		})
	}
	assertNoValidation(t, h)
}
