//go:build js && wasm

// Command browser-compute doubles four floats with a WebGPU compute shader
// in the browser and reads them back. Build it next to index.html and serve
// the directory:
//
//	GOOS=js GOARCH=wasm go build -o compute.wasm .
//	cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" .
//	python3 -m http.server
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"syscall/js"
	"time"

	"github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/gputypes"
)

const computeShader = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    if (i < arrayLength(&input)) {
        output[i] = input[i] * 2.0;
    }
}
`

var (
	input    = []float32{1, 2, 3, 4}
	expected = []float32{2, 4, 6, 8}
	bufSize  = uint64(len(input) * 4)
)

// log prints msg and adds it to the page.
func log(msg string) {
	fmt.Println(msg)
	el := js.Global().Get("document").Call("getElementById", "log")
	el.Set("innerHTML", el.Get("innerHTML").String()+msg+"<br>")
}

func main() {
	log("browser-compute starting...")
	if err := run(); err != nil {
		log("FAIL: " + err.Error())
		return
	}
	log("PASS: the compute shader doubled every value")
}

func run() error {
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		return fmt.Errorf("CreateInstance: %w", err)
	}
	defer instance.Release()
	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		return fmt.Errorf("RequestAdapter: %w", err)
	}
	defer adapter.Release()
	log("adapter: " + adapter.Info().Name)
	device, err := adapter.RequestDevice(nil)
	if err != nil {
		return fmt.Errorf("RequestDevice: %w", err)
	}
	defer device.Release()

	ps, err := createPipeline(device)
	if err != nil {
		return err
	}
	defer ps.release()
	bufs, err := createBuffers(device, ps.bgl)
	if err != nil {
		return err
	}
	defer bufs.release()
	if err := dispatch(device, ps.pipeline, bufs); err != nil {
		return err
	}
	results, err := readBack(bufs.readback)
	if err != nil {
		return err
	}
	log(fmt.Sprintf("%v * 2 = %v", input, results))
	for i, want := range expected {
		if results[i] != want {
			return fmt.Errorf("results[%d] = %v, want %v", i, results[i], want)
		}
	}
	return nil
}

type pipelineSet struct {
	shader   *wgpu.ShaderModule
	bgl      *wgpu.BindGroupLayout
	layout   *wgpu.PipelineLayout
	pipeline *wgpu.ComputePipeline
}

func (p *pipelineSet) release() {
	p.pipeline.Release()
	p.layout.Release()
	p.bgl.Release()
	p.shader.Release()
}

func createPipeline(device *wgpu.Device) (*pipelineSet, error) {
	shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{Label: "double-shader", WGSL: computeShader})
	if err != nil {
		return nil, fmt.Errorf("create shader: %w", err)
	}
	bgl, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
		Label: "double-bgl",
		Entries: []gputypes.BindGroupLayoutEntry{
			{Binding: 0, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage}},
			{Binding: 1, Visibility: wgpu.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage}},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("create bind group layout: %w", err)
	}
	layout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{Label: "double-pl", BindGroupLayouts: []*wgpu.BindGroupLayout{bgl}})
	if err != nil {
		return nil, fmt.Errorf("create pipeline layout: %w", err)
	}
	pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
		Label: "double-pipeline", Layout: layout, Module: shader, EntryPoint: "main",
	})
	if err != nil {
		return nil, fmt.Errorf("create compute pipeline: %w", err)
	}
	return &pipelineSet{shader: shader, bgl: bgl, layout: layout, pipeline: pipeline}, nil
}

type bufferSet struct {
	input, output, readback *wgpu.Buffer
	group                   *wgpu.BindGroup
}

func (b *bufferSet) release() {
	b.group.Release()
	b.readback.Release()
	b.output.Release()
	b.input.Release()
}

func createBuffers(device *wgpu.Device, bgl *wgpu.BindGroupLayout) (*bufferSet, error) {
	buffer := func(label string, usage gputypes.BufferUsage) (*wgpu.Buffer, error) {
		b, err := device.CreateBuffer(&wgpu.BufferDescriptor{Label: label, Size: bufSize, Usage: usage})
		if err != nil {
			return nil, fmt.Errorf("create %s buffer: %w", label, err)
		}
		return b, nil
	}
	var bufs bufferSet
	var err error
	if bufs.input, err = buffer("input", wgpu.BufferUsageStorage|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	if bufs.output, err = buffer("output", wgpu.BufferUsageStorage|wgpu.BufferUsageCopySrc); err != nil {
		return nil, err
	}
	if bufs.readback, err = buffer("readback", wgpu.BufferUsageMapRead|wgpu.BufferUsageCopyDst); err != nil {
		return nil, err
	}
	data := make([]byte, bufSize)
	for i, v := range input {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}
	if err := device.Queue().WriteBuffer(bufs.input, 0, data); err != nil {
		return nil, fmt.Errorf("write input buffer: %w", err)
	}
	bufs.group, err = device.CreateBindGroup(&wgpu.BindGroupDescriptor{
		Label: "double-bg", Layout: bgl,
		Entries: []wgpu.BindGroupEntry{
			{Binding: 0, Buffer: bufs.input, Size: bufSize},
			{Binding: 1, Buffer: bufs.output, Size: bufSize},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("create bind group: %w", err)
	}
	return &bufs, nil
}

func dispatch(device *wgpu.Device, pipeline *wgpu.ComputePipeline, bufs *bufferSet) error {
	encoder, err := device.CreateCommandEncoder(nil)
	if err != nil {
		return fmt.Errorf("create encoder: %w", err)
	}
	pass, err := encoder.BeginComputePass(nil)
	if err != nil {
		return fmt.Errorf("begin compute pass: %w", err)
	}
	pass.SetPipeline(pipeline)
	pass.SetBindGroup(0, bufs.group, nil)
	pass.Dispatch(1, 1, 1)
	if err := pass.End(); err != nil {
		return fmt.Errorf("end compute pass: %w", err)
	}
	encoder.CopyBufferToBuffer(bufs.output, 0, bufs.readback, 0, bufSize)
	cmdBuf, err := encoder.Finish()
	if err != nil {
		return fmt.Errorf("finish encoder: %w", err)
	}
	if _, err := device.Queue().Submit(cmdBuf); err != nil {
		return fmt.Errorf("submit: %w", err)
	}
	return nil
}

// readBack maps buf and returns the floats in it. Map blocks until the
// browser maps the buffer, which is fine on the main goroutine: the event
// loop runs while it waits.
func readBack(buf *wgpu.Buffer) ([]float32, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := buf.Map(ctx, wgpu.MapModeRead, 0, bufSize); err != nil {
		return nil, fmt.Errorf("map readback buffer: %w", err)
	}
	rng, err := buf.MappedRange(0, bufSize)
	if err != nil {
		_ = buf.Unmap()
		return nil, fmt.Errorf("readback MappedRange: %w", err)
	}
	results := make([]float32, len(input))
	for i := range results {
		results[i] = math.Float32frombits(binary.LittleEndian.Uint32(rng.Bytes()[i*4:]))
	}
	if err := buf.Unmap(); err != nil {
		return nil, fmt.Errorf("unmap readback buffer: %w", err)
	}
	return results, nil
}
