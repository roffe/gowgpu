//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

const interfaceShader = `
struct U { color: vec4<f32> }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> s: array<vec4<f32>>;

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return u.color + s[0];
}
`

// CreateRenderPipeline checks the shaders against the layout, the vertex
// buffers and the targets: a binding the layout lacks, hides from the stage
// or declares as another kind, a vertex input without an attribute or of
// another scalar kind, and an output of another kind than its target are
// errors before the backend sees the pipeline.
func TestCreateRenderPipelineChecksTheShaderInterface(t *testing.T) {
	device, h := newValidatedDevice(t)

	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "interface", WGSL: interfaceShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	uniform := gputypes.BindGroupLayoutEntry{Binding: 0, Visibility: gputypes.ShaderStageFragment,
		Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}}
	storage := gputypes.BindGroupLayoutEntry{Binding: 1, Visibility: gputypes.ShaderStageFragment,
		Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}}
	attrs := []gputypes.VertexAttribute{{Format: gputypes.VertexFormatFloat32x2, ShaderLocation: 0}}

	for _, tc := range []struct {
		name    string
		entries []gputypes.BindGroupLayoutEntry
		attrs   []gputypes.VertexAttribute
		target  gputypes.TextureFormat
		wantErr string
	}{
		{name: "matching", entries: []gputypes.BindGroupLayoutEntry{uniform, storage}, attrs: attrs, target: TextureFormatRGBA8Unorm},
		{name: "binding missing", entries: []gputypes.BindGroupLayoutEntry{uniform}, attrs: attrs, target: TextureFormatRGBA8Unorm,
			wantErr: "not in the pipeline layout"},
		{name: "binding invisible", entries: []gputypes.BindGroupLayoutEntry{
			{Binding: 0, Visibility: gputypes.ShaderStageVertex, Buffer: uniform.Buffer}, storage,
		}, attrs: attrs, target: TextureFormatRGBA8Unorm, wantErr: "not visible"},
		{name: "texture for a uniform", entries: []gputypes.BindGroupLayoutEntry{
			{Binding: 0, Visibility: gputypes.ShaderStageFragment, Texture: &gputypes.TextureBindingLayout{
				SampleType: gputypes.TextureSampleTypeFloat, ViewDimension: gputypes.TextureViewDimension2D,
			}}, storage,
		}, attrs: attrs, target: TextureFormatRGBA8Unorm, wantErr: "no buffer"},
		{name: "read-write storage for read-only", entries: []gputypes.BindGroupLayoutEntry{
			uniform, {Binding: 1, Visibility: gputypes.ShaderStageFragment, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
		}, attrs: attrs, target: TextureFormatRGBA8Unorm, wantErr: "buffer"},
		{name: "vertex input without attribute", entries: []gputypes.BindGroupLayoutEntry{uniform, storage},
			attrs: []gputypes.VertexAttribute{{Format: gputypes.VertexFormatFloat32x2, ShaderLocation: 1}}, target: TextureFormatRGBA8Unorm,
			wantErr: "no vertex attribute"},
		{name: "uint attribute for a float input", entries: []gputypes.BindGroupLayoutEntry{uniform, storage},
			attrs: []gputypes.VertexAttribute{{Format: gputypes.VertexFormatUint32x2, ShaderLocation: 0}}, target: TextureFormatRGBA8Unorm,
			wantErr: "vertex input @location(0)"},
		{name: "float output to a uint target", entries: []gputypes.BindGroupLayoutEntry{uniform, storage}, attrs: attrs,
			target: gputypes.TextureFormatRGBA8Uint, wantErr: "fragment output @location(0)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bgl, err := device.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: tc.entries})
			if err != nil {
				t.Fatalf("CreateBindGroupLayout: %v", err)
			}
			defer bgl.Release()
			layout, err := device.CreatePipelineLayout(&PipelineLayoutDescriptor{BindGroupLayouts: []*BindGroupLayout{bgl}})
			if err != nil {
				t.Fatalf("CreatePipelineLayout: %v", err)
			}
			defer layout.Release()
			pipeline, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
				Layout: layout,
				Vertex: VertexState{Module: shader, EntryPoint: "vs_main", Buffers: []gputypes.VertexBufferLayout{
					{ArrayStride: 8, StepMode: gputypes.VertexStepModeVertex, Attributes: tc.attrs},
				}},
				Fragment: &FragmentState{Module: shader, EntryPoint: "fs_main", Targets: []gputypes.ColorTargetState{
					{Format: tc.target, WriteMask: gputypes.ColorWriteMaskAll},
				}},
				Primitive:   gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
				Multisample: gputypes.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
			})
			if pipeline != nil {
				pipeline.Release()
			}
			switch {
			case tc.wantErr == "" && err != nil:
				t.Errorf("CreateRenderPipeline: %v", err)
			case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
				t.Errorf("CreateRenderPipeline: %v, want an error about %q", err, tc.wantErr)
			}
		})
	}
	assertNoValidation(t, h)
}
