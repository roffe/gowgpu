//go:build js && wasm

package wgpu

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"testing"
	"time"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/internal/browser"
)

// These run on the browser's real WebGPU, through cmd/wasmchrome:
//
//	GOOS=js GOARCH=wasm go test -exec "env GOOS= GOARCH= go run ./cmd/wasmchrome" -run BrowserGPU -v .
//
// Where there is no WebGPU, as under node, they skip.

// gpuAdapter returns the browser's adapter, and skips without WebGPU.
func gpuAdapter(t *testing.T) *Adapter {
	t.Helper()
	inst, err := CreateInstance(nil)
	if errors.Is(err, browser.ErrWebGPUNotSupported) {
		t.Skip("no WebGPU here")
	}
	if err != nil {
		t.Fatalf("CreateInstance: %v", err)
	}
	adapter, err := inst.RequestAdapter(nil)
	if err != nil {
		t.Fatalf("RequestAdapter: %v", err)
	}
	return adapter
}

// gpuDevice returns a device of the browser's adapter with features, and
// skips without WebGPU or when the adapter lacks them.
func gpuDevice(t *testing.T, features gputypes.Features) *Device {
	t.Helper()
	adapter := gpuAdapter(t)
	if !adapter.Features().ContainsAll(features) {
		t.Skipf("the adapter lacks features %#x", uint64(features&^adapter.Features()))
	}
	device, err := adapter.RequestDevice(&DeviceDescriptor{RequiredFeatures: features})
	if err != nil {
		t.Fatalf("RequestDevice: %v", err)
	}
	t.Cleanup(device.Release)
	return device
}

// gpuRead submits what record encodes, which copies into a mappable
// buffer of size bytes, and returns that buffer's bytes.
func gpuRead(t *testing.T, device *Device, record func(*CommandEncoder, *Buffer), size uint64) []byte {
	t.Helper()
	read, err := device.CreateBuffer(&BufferDescriptor{Label: "read", Size: size, Usage: BufferUsageMapRead | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer(read): %v", err)
	}
	defer read.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	record(enc, read)
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if _, err := device.Queue().Submit(cb); err != nil {
		t.Fatalf("Submit: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := read.Map(ctx, MapModeRead, 0, size); err != nil {
		t.Fatalf("Map: %v", err)
	}
	rng, err := read.MappedRange(0, size)
	if err != nil {
		t.Fatalf("MappedRange: %v", err)
	}
	out := bytes.Clone(rng.Bytes())
	if err := read.Unmap(); err != nil {
		t.Fatalf("Unmap: %v", err)
	}
	return out
}

func TestBrowserGPUAdapter(t *testing.T) {
	adapter := gpuAdapter(t)
	info := adapter.Info()
	t.Logf("%s, %s (%s), features:", info.Name, info.Vendor, info.DeviceType)
	for f := gputypes.Feature(1); f != 0; f <<= 1 {
		if adapter.Features().Contains(f) {
			t.Logf("  %s", f)
		}
	}
	device := gpuDevice(t, 0)
	l := device.Limits()
	t.Logf("maxTextureDimension2D %d, maxBufferSize %d", l.MaxTextureDimension2D, l.MaxBufferSize)
	if l.MaxTextureDimension2D < 8192 || l.MaxBufferSize < 256<<20 {
		t.Errorf("limits below WebGPU's defaults: %+v", l)
	}
}

func TestBrowserGPUBufferRoundTrip(t *testing.T) {
	device := gpuDevice(t, 0)
	src, err := device.CreateBuffer(&BufferDescriptor{Label: "src", Size: 256, Usage: BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer src.Release()
	want := make([]byte, 256)
	for i := range want {
		want[i] = byte(i * 7)
	}
	if err := device.Queue().WriteBuffer(src, 0, want); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	got := gpuRead(t, device, func(enc *CommandEncoder, read *Buffer) { enc.CopyBufferToBuffer(src, 0, read, 0, 256) }, 256)
	if !bytes.Equal(got, want) {
		t.Errorf("read back % x, want % x", got[:16], want[:16])
	}
}

func TestBrowserGPUCompute(t *testing.T) {
	device := gpuDevice(t, 0)
	const n = 1024
	data, err := device.CreateBuffer(&BufferDescriptor{Label: "data", Size: n * 4, Usage: BufferUsageStorage | BufferUsageCopySrc | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer data.Release()
	in := make([]byte, 0, n*4)
	for i := range uint32(n) {
		in = binary.LittleEndian.AppendUint32(in, i)
	}
	if err := device.Queue().WriteBuffer(data, 0, in); err != nil {
		t.Fatalf("WriteBuffer: %v", err)
	}
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    data[id.x] = data[id.x] * 2u + 1u;
}`})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipeline, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Module: module, EntryPoint: "main"})
	if err != nil {
		t.Fatalf("CreateComputePipeline: %v", err)
	}
	defer pipeline.Release()
	group, err := device.CreateBindGroup(&BindGroupDescriptor{Layout: pipeline.GetBindGroupLayout(0), Entries: []BindGroupEntry{{Binding: 0, Buffer: data}}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	got := gpuRead(t, device, func(enc *CommandEncoder, read *Buffer) {
		cp, err := enc.BeginComputePass(nil)
		if err != nil {
			t.Fatalf("BeginComputePass: %v", err)
		}
		cp.SetPipeline(pipeline)
		cp.SetBindGroup(0, group, nil)
		cp.Dispatch(n/64, 1, 1)
		if err := cp.End(); err != nil {
			t.Fatalf("End: %v", err)
		}
		enc.CopyBufferToBuffer(data, 0, read, 0, n*4)
	}, n*4)
	for i := range uint32(n) {
		if v := binary.LittleEndian.Uint32(got[i*4:]); v != i*2+1 {
			t.Fatalf("data[%d] = %d, want %d", i, v, i*2+1)
		}
	}
}

// TestBrowserGPURender draws a red triangle over the middle of a blue
// clear and reads the pixels back.
func TestBrowserGPURender(t *testing.T) {
	device := gpuDevice(t, 0)
	const side, row = 64, 256
	target, err := device.CreateTexture(&TextureDescriptor{
		Label: "target", Size: Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: gputypes.TextureDimension2D, Format: gputypes.TextureFormatRGBA8Unorm,
		Usage: TextureUsageRenderAttachment | TextureUsageCopySrc,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer target.Release()
	view, err := device.CreateTextureView(target, nil)
	if err != nil {
		t.Fatalf("CreateTextureView: %v", err)
	}
	defer view.Release()
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: `
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2<f32>(0.0, 0.6), vec2<f32>(-0.6, -0.6), vec2<f32>(0.6, -0.6));
    return vec4<f32>(p[i], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }`})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer module.Release()
	pipeline, err := device.CreateRenderPipeline(&RenderPipelineDescriptor{
		Vertex: VertexState{Module: module, EntryPoint: "vs"},
		Fragment: &FragmentState{Module: module, EntryPoint: "fs",
			Targets: []gputypes.ColorTargetState{{Format: gputypes.TextureFormatRGBA8Unorm, WriteMask: gputypes.ColorWriteMaskAll}}},
	})
	if err != nil {
		t.Fatalf("CreateRenderPipeline: %v", err)
	}
	defer pipeline.Release()

	got := gpuRead(t, device, func(enc *CommandEncoder, read *Buffer) {
		rp, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: []RenderPassColorAttachment{{
			View: view, LoadOp: gputypes.LoadOpClear, StoreOp: gputypes.StoreOpStore, ClearValue: gputypes.Color{B: 1, A: 1},
		}}})
		if err != nil {
			t.Fatalf("BeginRenderPass: %v", err)
		}
		rp.SetPipeline(pipeline)
		rp.Draw(gputypes.DrawArgs{VertexCount: 3, InstanceCount: 1})
		if err := rp.End(); err != nil {
			t.Fatalf("End: %v", err)
		}
		enc.CopyTextureToBuffer(target, read, []BufferTextureCopy{{
			BufferLayout: ImageDataLayout{BytesPerRow: row, RowsPerImage: side},
			TextureBase:  ImageCopyTexture{Texture: target},
			Size:         Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
		}})
	}, row*side)
	pixel := func(x, y int) []byte { return got[y*row+x*4 : y*row+x*4+4] }
	if p := pixel(side/2, side/2); !bytes.Equal(p, []byte{255, 0, 0, 255}) {
		t.Errorf("middle pixel % x, want the triangle's red", p)
	}
	if p := pixel(1, 1); !bytes.Equal(p, []byte{0, 0, 255, 255}) {
		t.Errorf("corner pixel % x, want the clear's blue", p)
	}
}

func TestBrowserGPUTextureRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name    string
		format  gputypes.TextureFormat
		texel   int
		feature gputypes.Feature
	}{
		{"rgba8unorm", gputypes.TextureFormatRGBA8Unorm, 4, 0},
		{"rgba16float", gputypes.TextureFormatRGBA16Float, 8, 0},
		{"r16unorm", gputypes.TextureFormatR16Unorm, 2, gputypes.FeatureTextureFormatsTier1},
		{"rg16snorm", gputypes.TextureFormatRG16Snorm, 4, gputypes.FeatureTextureFormatsTier1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			device := gpuDevice(t, gputypes.Features(tc.feature))
			const side, row = 4, 256
			tex, err := device.CreateTexture(&TextureDescriptor{
				Size: Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
				Dimension: gputypes.TextureDimension2D, Format: tc.format, Usage: TextureUsageCopyDst | TextureUsageCopySrc,
			})
			if err != nil {
				t.Fatalf("CreateTexture: %v", err)
			}
			defer tex.Release()
			want := make([]byte, side*side*tc.texel)
			for i := range want {
				want[i] = byte(i*13 + 1)
			}
			if tc.format == gputypes.TextureFormatRGBA16Float {
				for i := 1; i < len(want); i += 2 {
					want[i] &= 0x3b // finite halves, which copy unchanged
				}
			}
			if err := device.Queue().WriteTexture(&ImageCopyTexture{Texture: tex}, want,
				&ImageDataLayout{BytesPerRow: side * uint32(tc.texel), RowsPerImage: side}, &Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1}); err != nil {
				t.Fatalf("WriteTexture: %v", err)
			}
			got := gpuRead(t, device, func(enc *CommandEncoder, read *Buffer) {
				enc.CopyTextureToBuffer(tex, read, []BufferTextureCopy{{
					BufferLayout: ImageDataLayout{BytesPerRow: row, RowsPerImage: side},
					TextureBase:  ImageCopyTexture{Texture: tex},
					Size:         Extent3D{Width: side, Height: side, DepthOrArrayLayers: 1},
				}})
			}, row*side)
			for y := range side {
				w := want[y*side*tc.texel : (y+1)*side*tc.texel]
				if g := got[y*row : y*row+len(w)]; !bytes.Equal(g, w) {
					t.Fatalf("row %d read back % x, want % x", y, g, w)
				}
			}
		})
	}
}

// TestBrowserGPUErrorScope checks that an error the browser finds, in a
// shader it compiles, reaches the scope around it, and only that.
func TestBrowserGPUErrorScope(t *testing.T) {
	device := gpuDevice(t, 0)
	device.PushErrorScope(ErrorFilterValidation)
	module, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: "fn broken( {"})
	if err == nil {
		defer module.Release()
	}
	if gpuErr := device.PopErrorScope(); gpuErr == nil {
		t.Error("PopErrorScope: no error for WGSL that does not parse")
	} else {
		t.Logf("caught: %s", gpuErr.Message)
	}

	device.PushErrorScope(ErrorFilterValidation)
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: 16, Usage: BufferUsageCopyDst})
	if err == nil {
		defer buf.Release()
	}
	if gpuErr := device.PopErrorScope(); gpuErr != nil {
		t.Errorf("PopErrorScope: %v for a valid buffer", gpuErr)
	}
}
