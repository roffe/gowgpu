//go:build !(js && wasm)

package wgpu

import (
	"encoding/binary"
	"fmt"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// vertexIndexShader draws a full-screen triangle from any three vertices
// in a row, red from vertex indices below 3 and green from the ones above,
// with the instance index in blue.
const vertexIndexShader = `
struct VOut { @builtin(position) pos: vec4<f32>, @location(0) c: vec4<f32> }
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
	let v = vi % 3u;
	var o: VOut;
	o.pos = vec4<f32>(f32(i32(v & 1u) * 4 - 1), f32(i32(v >> 1u) * 4 - 1), 0.5, 1.0);
	o.c = vec4<f32>(1.0, 0.0, f32(ii), 1.0);
	if (vi >= 3u) {
		o.c = vec4<f32>(0.0, 1.0, f32(ii), 1.0);
	}
	return o;
}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
	return in.c;
}
`

// TestDrawsSeeFirstVertexAndInstance draws with a first vertex, base vertex
// and first instance, directly and indirectly, and checks vertex_index and
// instance_index count from them, as WebGPU has it.
func TestDrawsSeeFirstVertexAndInstance(t *testing.T) {
	device, h := newValidatedDevice(t)
	if h.backend == gputypes.BackendGL {
		t.Skip("the GL backend leaves the first instance and base vertex out of the builtins and records no indirect draws")
	}
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "vertex-index", WGSL: vertexIndexShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Label:     "vertex-index",
		Vertex:    VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipe.Release()

	index, err := device.CreateBuffer(&BufferDescriptor{Label: "indices", Size: 8, Usage: BufferUsageIndex | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer index.Release()
	if err := device.Queue().WriteBuffer(index, 0, []byte{0, 0, 1, 0, 2, 0, 0, 0}); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	args, err := device.CreateBuffer(&BufferDescriptor{Label: "indirect", Size: 64, Usage: BufferUsageIndirect | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer args.Release()
	words := []uint32{
		3, 1, 3, 0, // draw at 0: 3 vertices from vertex 3
		0,             // padding, the indexed record starts at 20
		3, 1, 0, 3, 0, // indexed draw at 20: 3 indices with base vertex 3
	}
	data := make([]byte, 0, 64)
	for _, w := range words {
		data = binary.LittleEndian.AppendUint32(data, w)
	}
	if err := device.Queue().WriteBuffer(args, 0, data); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}

	for _, tc := range []struct {
		name     string
		draw     func(*RenderPassEncoder)
		blue     byte
		indirect bool
	}{
		{"draw", func(p *RenderPassEncoder) {
			p.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1, FirstVertex: 3, FirstInstance: 1})
		}, 255, false},
		{"indexed", func(p *RenderPassEncoder) {
			p.SetIndexBuffer(index, gputypes.IndexFormatUint16, 0)
			p.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 3, InstanceCount: 1, BaseVertex: 3, FirstInstance: 1})
		}, 255, false},
		{"indirect", func(p *RenderPassEncoder) {
			p.DrawIndirect(args, 0)
		}, 0, true},
		{"indexed indirect", func(p *RenderPassEncoder) {
			p.SetIndexBuffer(index, gputypes.IndexFormatUint16, 0)
			p.DrawIndexedIndirect(args, 20)
		}, 0, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.indirect {
				skipRootConstantIndirectOnWine(t, h)
			}
			tex, view := pixTexture(t, device, "vertex-index", TextureUsageRenderAttachment|TextureUsageCopySrc)
			defer releaseTextureView(tex, view)
			staging := readbackBuffer(t, device, "vertex-index")
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			clearPass(t, enc, view, gputypes.LoadOpClear, black, func(pass *RenderPassEncoder) {
				pass.SetPipeline(pipe)
				tc.draw(pass)
			})
			copyOut(enc, tex, staging)
			finishAndSubmit(t, device, enc)
			expectPixels(t, tc.name, readback(t, staging), 0, 255, tc.blue, 255)
		})
	}
	assertNoValidation(t, h)
}

const numWorkgroupsShader = `
@group(0) @binding(0) var<storage, read_write> out: array<vec4<u32>>;
@compute @workgroup_size(1) fn cs_main(@builtin(workgroup_id) id: vec3<u32>, @builtin(num_workgroups) n: vec3<u32>) {
	out[id.x + 4u * (id.y + 2u * id.z)] = vec4<u32>(n, 1u);
}
`

// TestDispatchesSeeNumWorkgroups dispatches 4x2x3 workgroups, directly and
// indirectly, and checks each sees num_workgroups as that.
func TestDispatchesSeeNumWorkgroups(t *testing.T) {
	device, h := newValidatedDevice(t)
	pipe, layout := computePipeline(t, device, "num-workgroups", numWorkgroupsShader, []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
	})
	args, err := device.CreateBuffer(&BufferDescriptor{Label: "dispatch-args", Size: 16, Usage: BufferUsageIndirect | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer args.Release()
	if err := device.Queue().WriteBuffer(args, 4, []byte{4, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0}); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	for _, indirect := range []bool{false, true} {
		name := "direct"
		if indirect {
			name = "indirect"
		}
		t.Run(name, func(t *testing.T) {
			if indirect {
				skipRootConstantIndirectOnWine(t, h)
			}
			out := newStorageBuffer(t, device, 24*16)
			group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: layout, Entries: []BindGroupEntry{{Binding: 0, Buffer: out}}})
			if err != nil {
				t.Fatalf("CreateBindGroup: %v", err)
			}
			defer group.Release()
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := enc.BeginComputePass(nil)
			if err != nil {
				t.Fatalf("BeginComputePass: %v", err)
			}
			pass.SetPipeline(pipe)
			pass.SetBindGroup(0, group, nil)
			if indirect {
				pass.DispatchIndirect(args, 4)
			} else {
				pass.Dispatch(4, 2, 3)
			}
			if err := pass.End(); err != nil {
				t.Fatalf("End: %v", err)
			}
			finishAndSubmit(t, device, enc)
			got := readU32s(t, device, out, 24*4)
			var wrong []string
			for i := 0; i < 24; i++ {
				if v := got[i*4 : i*4+4]; v[0] != 4 || v[1] != 2 || v[2] != 3 || v[3] != 1 {
					wrong = append(wrong, fmt.Sprint(v[:3]))
				}
			}
			if len(wrong) > 0 {
				t.Errorf("%d of 24 workgroups saw num_workgroups other than [4 2 3]: %v", len(wrong), wrong)
			}
		})
	}
	assertNoValidation(t, h)
}

// skipRootConstantIndirectOnWine skips an indirect call whose pipeline reads
// the special constants on DX12 under Wine: DX12 sets them with a root
// constant argument of the command signature, which vkd3d ignores along
// with the whole ExecuteIndirect.
func skipRootConstantIndirectOnWine(t *testing.T, h *captureHandler) {
	t.Helper()
	if h.backend == gputypes.BackendDX12 && underWine() {
		t.Skip("vkd3d ignores ExecuteIndirect with a root constant argument")
	}
}
