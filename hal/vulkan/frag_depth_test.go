//go:build !(js && wasm)

package vulkan

import (
	"context"
	"encoding/binary"
	"log/slog"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// fullScreenVertex draws a triangle over the whole target.
const fullScreenVertex = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	return vec4<f32>(f32(i32(vi & 1u) * 4 - 1), f32(i32(vi >> 1u) * 4 - 1), 0.0, 1.0);
}
`

// layerMessages collects the validation layer's messages.
type layerMessages struct {
	mu   sync.Mutex
	msgs []string
}

func (h *layerMessages) Enabled(context.Context, slog.Level) bool { return true }
func (h *layerMessages) Handle(_ context.Context, r slog.Record) error {
	if strings.Contains(r.Message, "Validation") || strings.Contains(r.Message, "VUID") {
		h.mu.Lock()
		h.msgs = append(h.msgs, r.Message)
		h.mu.Unlock()
	}
	return nil
}
func (h *layerMessages) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *layerMessages) WithGroup(string) slog.Handler      { return h }

// A pipeline layout with push constants of its own leaves its fragment
// stages no push constant block for the frag_depth clamp: a stage that
// writes frag_depth, from the push constants or not, is made without the
// clamp and draws the depth it writes, and the depth range a pass pushes for
// the clamp stays out of the push constants.
func TestFragDepthBesidePushConstants(t *testing.T) {
	h := &layerMessages{}
	prev := hal.Logger()
	hal.SetLogger(slog.New(h))
	t.Cleanup(func() { hal.SetLogger(prev) })
	d := openTestDevice(t, gputypes.InstanceFlagsDebug)

	for _, tc := range []struct {
		name     string
		fragment string
		want     float32
	}{
		{"depth from the push constants", `
struct Depth { value: f32 }
var<push_constant> depth: Depth;
@fragment fn fs_main() -> @builtin(frag_depth) f32 { return depth.value; }
`, 0.25},
		{"depth of its own", `
@fragment fn fs_main() -> @builtin(frag_depth) f32 { return 0.75; }
`, 0.75},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := drawFragDepth(t, d, fullScreenVertex+tc.fragment, 0.25)
			for i, v := range got {
				if v != tc.want {
					t.Fatalf("depth at texel %d = %v, want %v", i, v, tc.want)
				}
			}
		})
	}

	time.Sleep(250 * time.Millisecond) // the messenger drains every 50ms
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, m := range h.msgs {
		t.Errorf("validation: %s", m)
	}
}

// drawFragDepth draws wgsl into a 4x4 Depth32Float target with a pipeline
// layout that has a 4-byte fragment push constant range, pushing pushed
// there, and returns the depth of each texel.
func drawFragDepth(t *testing.T, d *Device, wgsl string, pushed float32) []float32 {
	t.Helper()
	const side = 4
	module, err := d.CreateShaderModule(&hal.ShaderModuleDescriptor{Label: "frag-depth", Source: hal.ShaderSource{WGSL: wgsl}})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer d.DestroyShaderModule(module)
	layout, err := d.CreatePipelineLayout(&hal.PipelineLayoutDescriptor{
		PushConstantRanges: []hal.PushConstantRange{{Stages: gputypes.ShaderStageFragment, Range: hal.Range{End: 4}}},
	})
	if err != nil {
		t.Fatalf("CreatePipelineLayout: %v", err)
	}
	defer d.DestroyPipelineLayout(layout)
	pipeline, err := d.CreateRenderPipeline(&hal.RenderPipelineDescriptor{
		Layout:    layout,
		Vertex:    hal.VertexState{Module: module, EntryPoint: "vs_main"},
		Primitive: gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
		DepthStencil: &hal.DepthStencilState{Format: gputypes.TextureFormatDepth32Float, DepthWriteEnabled: true,
			DepthCompare: gputypes.CompareFunctionAlways},
		Multisample: gputypes.MultisampleState{Count: 1, Mask: 0xFFFFFFFF},
		Fragment:    &hal.FragmentState{Module: module, EntryPoint: "fs_main"},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer d.DestroyRenderPipeline(pipeline)

	tex, err := d.CreateTexture(&hal.TextureDescriptor{
		Size: hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatDepth32Float,
		Usage: gputypes.TextureUsageRenderAttachment | gputypes.TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer d.DestroyTexture(tex)
	view, err := d.CreateTextureView(tex, &hal.TextureViewDescriptor{
		Format: gputypes.TextureFormatDepth32Float, Dimension: gputypes.TextureViewDimension2D,
		Aspect: gputypes.TextureAspectAll, MipLevelCount: 1, ArrayLayerCount: 1,
	})
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer d.DestroyTextureView(view)
	const rowPitch = 256
	readback, err := d.CreateBuffer(&hal.BufferDescriptor{Label: "frag-depth-readback", Size: rowPitch * side,
		Usage: gputypes.BufferUsageCopyDst | gputypes.BufferUsageMapRead})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer d.DestroyBuffer(readback)

	enc, err := d.CreateCommandEncoder(&hal.CommandEncoderDescriptor{Label: "frag-depth"})
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	if err := enc.BeginEncoding("frag-depth"); err != nil {
		t.Fatalf("BeginEncoding: %v", err)
	}
	enc.TransitionTextures([]hal.TextureBarrier{{Texture: tex, Usage: hal.TextureUsageTransition{
		NewUsage: gputypes.TextureUsageRenderAttachment, New: hal.TextureUsesDepthStencilWrite,
	}}})
	pass := enc.BeginRenderPass(&hal.RenderPassDescriptor{DepthStencilAttachment: &hal.RenderPassDepthStencilAttachment{
		View: view, DepthLoadOp: gputypes.LoadOpClear, DepthStoreOp: gputypes.StoreOpStore, DepthClearValue: 1,
	}})
	pass.SetPipeline(pipeline)
	d.cmds.CmdPushConstants(enc.(*CommandEncoder).active, layout.(*PipelineLayout).handle,
		vk.ShaderStageFlags(vk.ShaderStageFragmentBit), 0, 4, (*uintptr)(unsafe.Pointer(&pushed)))
	pass.SetViewport(gputypes.Viewport{Width: side, Height: side, MinDepth: 0, MaxDepth: 1})
	pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
	pass.End()
	enc.TransitionTextures([]hal.TextureBarrier{{Texture: tex, Usage: hal.TextureUsageTransition{
		OldUsage: gputypes.TextureUsageRenderAttachment, Old: hal.TextureUsesDepthStencilWrite,
		NewUsage: gputypes.TextureUsageCopySrc, New: hal.TextureUsesCopySrc,
	}}})
	enc.CopyTextureToBuffer(tex, readback, []hal.BufferTextureCopy{{
		BufferLayout: hal.ImageDataLayout{BytesPerRow: rowPitch, RowsPerImage: side},
		TextureBase:  hal.ImageCopyTexture{Texture: tex, Aspect: gputypes.TextureAspectDepthOnly},
		Size:         hal.Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
	}})
	enc.TransitionBuffers([]hal.BufferBarrier{{Buffer: readback, Usage: hal.BufferUsageTransition{
		OldUsage: gputypes.BufferUsageCopyDst, NewUsage: gputypes.BufferUsageMapRead,
	}}})
	cb, err := enc.EndEncoding()
	if err != nil {
		t.Fatalf("EndEncoding: %v", err)
	}
	if _, err := d.queue.Submit([]hal.CommandBuffer{cb}); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	if err := d.WaitIdle(); err != nil {
		t.Fatalf("WaitIdle: %v", err)
	}
	enc.ResetAll([]hal.CommandBuffer{cb})
	enc.Destroy()

	mapping, err := d.MapBuffer(readback, 0, rowPitch*side)
	if err != nil {
		t.Fatalf("MapBuffer: %v", err)
	}
	defer func() { _ = d.UnmapBuffer(readback) }()
	raw := unsafe.Slice((*byte)(mapping.Ptr), rowPitch*side)
	depths := make([]float32, 0, side*side)
	for y := range side {
		for x := range side {
			depths = append(depths, math.Float32frombits(binary.LittleEndian.Uint32(raw[y*rowPitch+x*4:])))
		}
	}
	return depths
}
