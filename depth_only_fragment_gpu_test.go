//go:build !(js && wasm)

package wgpu

import (
	"context"
	"encoding/binary"
	"math"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
)

const fragDepthShader = `
@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return vec4<f32>(p[i], 0.5, 1.0);
}

@fragment
fn fs_main() -> @builtin(frag_depth) f32 {
    return 0.25;
}
`

// A pipeline with a fragment stage and no color targets draws into its
// depth attachment alone, as WebGPU allows: here the depth its fragment
// shader writes.
func TestDepthOnlyPipelineWithAFragmentStage(t *testing.T) {
	device, h := newValidatedDevice(t)
	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: fragDepthShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	pipeline, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Vertex:    VertexState{Module: shader, EntryPoint: "vs_main"},
		Fragment:  &FragmentState{Module: shader, EntryPoint: "fs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		DepthStencil: &DepthStencilState{Format: gputypes.TextureFormatDepth32Float, DepthWriteEnabled: true,
			DepthCompare: gputypes.CompareFunctionAlways},
		Multisample: gputypes.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipeline.Release()

	depth, err := device.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: gputypes.TextureFormatDepth32Float, Usage: TextureUsageRenderAttachment | TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer depth.Release()
	view, err := device.CreateTextureView(depth, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer view.Release()
	out := readbackBuffer(t, device, "depth")

	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{DepthStencilAttachment: &RenderPassDepthStencilAttachment{
		View: view, DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 1,
	}})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	pass.SetPipeline(pipeline)
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	if h.backend != gputypes.BackendGL { // GL reads no depth back
		enc.CopyTextureToBuffer(depth, out, []BufferTextureCopy{{
			BufferLayout: ImageDataLayout{BytesPerRow: pixRow, RowsPerImage: pixSide},
			Size:         Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1},
		}})
	}
	finishAndSubmit(t, device, enc)
	if h.backend != gputypes.BackendGL {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := out.Map(ctx, MapModeRead, 0, pixStaging); err != nil {
			t.Fatalf("Map: %v", err)
		}
		rng, err := out.MappedRange(0, pixStaging)
		if err != nil {
			t.Fatalf("MappedRange: %v", err)
		}
		if got := math.Float32frombits(binary.LittleEndian.Uint32(rng.Bytes())); got != 0.25 {
			t.Errorf("depth = %v, want 0.25 from the fragment shader", got)
		}
		if err := out.Unmap(); err != nil {
			t.Fatalf("Unmap: %v", err)
		}
	}
	assertNoValidation(t, h)
}
