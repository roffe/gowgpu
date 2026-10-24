//go:build !(js && wasm)

package wgpu

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// TestMSAAResolvesEveryAttachment clears two 4-sample attachments, each to
// its own colour, in a pass that resolves both into textures new to it, and
// reads the resolves. GL resolved attachment 0 alone, and the first resolve
// into a new texture blitted the resolve texture onto itself: making its
// framebuffer bound it for reading too.
func TestMSAAResolvesEveryAttachment(t *testing.T) {
	device, _ := newValidatedDevice(t)
	colors := []gputypes.Color{red, green}
	attachments := make([]RenderPassColorAttachment, len(colors))
	resolves := make([]*Texture, len(colors))
	for i, c := range colors {
		msaa, err := device.CreateTexture(&TextureDescriptor{
			Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}, Format: TextureFormatRGBA8Unorm,
			Usage: TextureUsageRenderAttachment, MipLevelCount: 1, SampleCount: 4, Dimension: TextureDimension2D,
		})
		if err != nil {
			t.Fatalf("CreateTexture(msaa): %v", err)
		}
		defer msaa.Release()
		msaaView, err := device.CreateTextureView(msaa, nil)
		if err != nil {
			t.Fatalf("CreateTextureView: %v", err)
		}
		defer msaaView.Release()
		var resolveView *TextureView
		resolves[i], resolveView = pixTexture(t, device, "resolve", TextureUsageRenderAttachment|TextureUsageCopySrc)
		defer releaseTextureView(resolves[i], resolveView)
		attachments[i] = RenderPassColorAttachment{View: msaaView, ResolveTarget: resolveView,
			LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: c}
	}
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: attachments})
	if err != nil {
		t.Fatalf("BeginRenderPass: %v", err)
	}
	if err := pass.End(); err != nil {
		t.Fatalf("End: %v", err)
	}
	finishAndSubmit(t, device, enc)
	for i, c := range colors {
		pix := readSubresources(t, device, resolves[i], gputypes.TextureAspectAll, []subresource{{width: pixSide, height: pixSide, rowBytes: pixSide * 4, rows: pixSide}})[0]
		expectPixels(t, "resolve of attachment "+string('0'+rune(i)), pix, byte(c.R*255), byte(c.G*255), byte(c.B*255), 255)
	}
}

const coverageShader = `
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
	var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
	return vec4<f32>(pos[vi], 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 0.5); }
`

// TestMultisampleStateWritesItsSamples draws green at alpha 0.5 over red
// into 4 samples with alpha-to-coverage, which covers about half of them,
// and with a sample mask of sample 0, which covers a quarter, and reads the
// resolves. GL ignored both and covered every sample.
func TestMultisampleStateWritesItsSamples(t *testing.T) {
	device, h := newValidatedDevice(t)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{Label: "coverage", WGSL: coverageShader})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	for _, tc := range []struct {
		name       string
		ms         gputypes.MultisampleState
		minR, maxR byte
	}{
		{"alpha-to-coverage", gputypes.MultisampleState{Count: 4, Mask: 0xFFFFFFFF, AlphaToCoverageEnabled: true}, 64, 191},
		{"sample mask", gputypes.MultisampleState{Count: 4, Mask: 1}, 160, 224},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.ms.Mask != 0xFFFFFFFF && h.backend == gputypes.BackendMetal {
				t.Skip("Metal has no pipeline sample mask, see the known issues")
			}
			pipe, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
				Label:       tc.name,
				Vertex:      VertexState{Module: module, EntryPoint: "vs_main"},
				Primitive:   gputypes.PrimitiveState{Topology: gputypes.PrimitiveTopologyTriangleList},
				Multisample: tc.ms,
				Fragment: &FragmentState{Module: module, EntryPoint: "fs_main",
					Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
			})
			if err != nil {
				t.Fatalf("CreateRenderPipeline: %v", err)
			}
			defer pipe.Release()
			msaa, err := device.CreateTexture(&TextureDescriptor{
				Size: Extent3D{Width: pixSide, Height: pixSide, DepthOrArrayLayers: 1}, Format: TextureFormatRGBA8Unorm,
				Usage: TextureUsageRenderAttachment, MipLevelCount: 1, SampleCount: 4, Dimension: TextureDimension2D,
			})
			if err != nil {
				t.Fatalf("CreateTexture(msaa): %v", err)
			}
			defer msaa.Release()
			msaaView, err := device.CreateTextureView(msaa, nil)
			if err != nil {
				t.Fatalf("CreateTextureView: %v", err)
			}
			defer msaaView.Release()
			resolve, resolveView := pixTexture(t, device, "coverage-resolve", TextureUsageRenderAttachment|TextureUsageCopySrc)
			defer releaseTextureView(resolve, resolveView)
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
				View: msaaView, ResolveTarget: resolveView, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: red,
			}}})
			if err != nil {
				t.Fatalf("BeginRenderPass: %v", err)
			}
			pass.SetPipeline(pipe)
			pass.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
			if err := pass.End(); err != nil {
				t.Fatalf("End: %v", err)
			}
			finishAndSubmit(t, device, enc)
			pix := readSubresources(t, device, resolve, gputypes.TextureAspectAll, []subresource{{width: pixSide, height: pixSide, rowBytes: pixSide * 4, rows: pixSide}})[0]
			if r, g := pix[0], pix[1]; r < tc.minR || r > tc.maxR || int(r)+int(g) < 250 || int(r)+int(g) > 260 {
				t.Errorf("resolved pixel %v, want red %d to %d and green the rest", pix[:4], tc.minR, tc.maxR)
			}
			assertNoValidation(t, h)
		})
	}
}
