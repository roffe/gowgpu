# Compute shaders

How to run a compute shader with gowgpu: WGSL in, numbers back on the CPU. The code below is one program cut into pieces.

For complete programs look at `examples/compute-copy`, `examples/compute-sum` and `examples/compute-particles`. They're headless, and compute-copy and compute-sum check the GPU result against the CPU, so `go run ./examples/compute-copy` is also a quick way to see if compute works on a machine.

What each backend does differently is in [COMPUTE-BACKENDS.md](COMPUTE-BACKENDS.md).

## The shader

This one doubles every element of a buffer:

```wgsl
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i < arrayLength(&input)) {
        output[i] = input[i] * 2.0;
    }
}
```

- `@compute` marks the entry point and `@workgroup_size(64)` gives each workgroup 64 invocations.
- `global_invocation_id` is the invocation's index across all workgroups. The bounds check is there because the last workgroup can stick out past the end of the data.
- `var<storage, read>` is a read-only storage buffer, `var<storage, read_write>` is one the shader can write.
- `arrayLength(&input)` is the number of elements in the bound buffer.

### Workgroup size

64 is a safe default. WebGPU only promises 256 invocations per workgroup (that's `gputypes.DefaultLimits()`). The real numbers for your device are in `device.Limits().MaxComputeInvocationsPerWorkgroup` and `MaxComputeWorkgroupSizeX/Y/Z`, and a device from `RequestDevice(nil)` gets the adapter's own limits (DX12 and Metal report a fixed 1024, Vulkan and GL ask the driver).

`CreateComputePipeline` checks `@workgroup_size` against those limits and returns a `*core.CreateComputePipelineError` when it doesn't fit.

## Getting a device

```go
import (
	wgpu "github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/gputypes"

	// Registers the native backends.
	_ "github.com/roffe/gowgpu/hal/allbackends"
)
```

```go
instance, err := wgpu.CreateInstance(nil)
if err != nil {
	return err
}
defer instance.Release()

adapter, err := instance.RequestAdapter(nil)
if err != nil {
	return err
}
defer adapter.Release()

device, err := adapter.RequestDevice(nil)
if err != nil {
	return err
}
defer device.Release()
```

`CreateInstance(nil)` picks from Vulkan, Metal and DX12. GL is registered too, but you only get it by asking for it with `&wgpu.InstanceDescriptor{Backends: gputypes.BackendsGL}`.

A machine without a GPU gets compute through a CPU driver: Mesa's lavapipe on Vulkan, or WARP on DX12 (gowgpu only picks WARP when there's no hardware adapter).

`adapter.DownlevelCapabilities().Flags` has `gputypes.DownlevelFlagsComputeShaders` when the adapter can do compute at all. Vulkan, DX12, Metal and the browser always set it. On GL it depends on the context the driver gives you, and on desktop Linux there's no compute (see [COMPUTE-BACKENDS.md](COMPUTE-BACKENDS.md#gl)).

## Shader module

```go
shader, err := device.CreateShaderModule(&wgpu.ShaderModuleDescriptor{
	Label: "double",
	WGSL:  shaderWGSL,
})
if err != nil {
	return err
}
defer shader.Release()
```

naga turns the WGSL into whatever the backend wants:

| Backend | Path |
|---------|------|
| Vulkan | WGSL -> SPIR-V |
| DX12 | WGSL -> HLSL -> DXBC with FXC (`d3dcompiler_47.dll`), or WGSL -> DXIL straight from naga with `GOGPU_DX12_DXIL=1` |
| Metal | WGSL -> MSL |
| GL | WGSL -> GLSL |
| Browser | no naga, the WGSL string goes to the browser as is |

DX12 and GL compile when you create the pipeline, so on those a broken shader shows up as an error from `CreateComputePipeline`, not from `CreateShaderModule`.

The descriptor also has a `SPIRV []uint32` field. Only Vulkan takes SPIR-V as is, so stick to WGSL.

## Layout and pipeline

```go
bgl, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{
	Label: "double",
	Entries: []gputypes.BindGroupLayoutEntry{
		{
			Binding:    0,
			Visibility: wgpu.ShaderStageCompute,
			Buffer:     &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeReadOnlyStorage},
		},
		{
			Binding:    1,
			Visibility: wgpu.ShaderStageCompute,
			Buffer:     &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeStorage},
		},
	},
})
if err != nil {
	return err
}
defer bgl.Release()

layout, err := device.CreatePipelineLayout(&wgpu.PipelineLayoutDescriptor{
	Label:            "double",
	BindGroupLayouts: []*wgpu.BindGroupLayout{bgl},
})
if err != nil {
	return err
}
defer layout.Release()

pipeline, err := device.CreateComputePipeline(&wgpu.ComputePipelineDescriptor{
	Label:      "double",
	Layout:     layout,
	Module:     shader,
	EntryPoint: "main",
})
if err != nil {
	return err
}
defer pipeline.Release()
```

Always pass a `Layout`. The native backends have no automatic layouts: Vulkan and GL refuse a nil `Layout` (Vulkan says "compute pipeline layout is required (automatic layouts are not supported)"), and there's no `GetBindGroupLayout` to get one back. The browser build does turn a nil `Layout` into `"auto"` and has `ComputePipeline.GetBindGroupLayout`, but that's browser only.

`ComputePipelineDescriptor` also has `Constants` (values for WGSL `override` declarations) and `ZeroInitializeWorkgroupMemory`. No backend passes either of them on yet (there's a TODO for it in each hal backend), so don't build on override constants.

## Buffers

```go
const n = 1024
size := uint64(n * 4) // n f32 values

input, err := device.CreateBuffer(&wgpu.BufferDescriptor{
	Label: "input",
	Size:  size,
	Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
})
if err != nil {
	return err
}
defer input.Release()

output, err := device.CreateBuffer(&wgpu.BufferDescriptor{
	Label: "output",
	Size:  size,
	Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopySrc,
})
if err != nil {
	return err
}
defer output.Release()

staging, err := device.CreateBuffer(&wgpu.BufferDescriptor{
	Label: "staging",
	Size:  size,
	Usage: wgpu.BufferUsageMapRead | wgpu.BufferUsageCopyDst,
})
if err != nil {
	return err
}
defer staging.Release()
```

WebGPU only allows `MapRead` together with `CopyDst`, so the result takes a detour: the shader writes `output`, a copy moves it to `staging`, and the CPU maps `staging`.

Fill the input with `Queue.WriteBuffer`:

```go
data := make([]byte, size)
for i := 0; i < n; i++ {
	binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(float32(i)))
}
if err := device.Queue().WriteBuffer(input, 0, data); err != nil {
	return err
}
```

`WriteBuffer` wants a buffer with `CopyDst`, an offset and length that are multiples of 4, and a buffer that isn't mapped. On Vulkan, DX12 and Metal it doesn't touch the GPU right away: the data goes into a staging encoder that is sent ahead of your command buffers on the next `Submit`.

For small parameters like counts and scale factors use a uniform buffer: `wgpu.BufferUsageUniform | wgpu.BufferUsageCopyDst` on the buffer and `gputypes.BufferBindingTypeUniform` in the layout. compute-copy passes its element count and scale that way.

## Bind group

```go
bg, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{
	Label:  "double",
	Layout: bgl,
	Entries: []wgpu.BindGroupEntry{
		{Binding: 0, Buffer: input, Size: size},
		{Binding: 1, Buffer: output, Size: size},
	},
})
if err != nil {
	return err
}
defer bg.Release()
```

`Size: 0` means the rest of the buffer. An `Offset` has to be a multiple of `Limits.MinStorageBufferOffsetAlignment` (`MinUniformBufferOffsetAlignment` for uniforms), which is 256 in the WebGPU defaults.

## Dispatch

```go
encoder, err := device.CreateCommandEncoder(nil)
if err != nil {
	return err
}
pass, err := encoder.BeginComputePass(nil)
if err != nil {
	return err
}
pass.SetPipeline(pipeline)
pass.SetBindGroup(0, bg, nil)
pass.Dispatch((n+63)/64, 1, 1) // 1024 elements, 64 per workgroup: 16 workgroups
if err := pass.End(); err != nil {
	return err
}
encoder.CopyBufferToBuffer(output, 0, staging, 0, size)

cmd, err := encoder.Finish()
if err != nil {
	return err
}
if _, err := device.Queue().Submit(cmd); err != nil {
	return err
}
```

The pass methods don't return errors. A missing pipeline, a missing or wrong bind group or too many workgroups gets recorded and comes back from `Finish()`, wrapping sentinels like `wgpu.ErrDispatchMissingPipeline` or `wgpu.ErrDispatchWorkgroupCountExceeded`, so use `errors.Is`.

You don't need barriers between dispatches or before the copy. The native backends put one after every dispatch, and in the browser that's the browser's job.

gowgpu doesn't check `CopyBufferToBuffer` yet, so stick to the WebGPU rules yourself: offsets and size multiples of 4, `CopySrc` on the source and `CopyDst` on the destination.

`Submit` doesn't block. It returns a submission index, and `device.Queue().Poll()` returns the last one the GPU has finished.

### Indirect dispatch

```go
// args holds three u32 workgroup counts: x, y, z.
pass.DispatchIndirect(args, 0)
```

The buffer needs `wgpu.BufferUsageIndirect`, the offset has to be a multiple of 4 and the 12 bytes have to fit in the buffer.

If the device managed to build its internal check pipeline when it was created, gowgpu first runs a tiny shader that turns the dispatch into 0, 0, 0 when any count is over `MaxComputeWorkgroupsPerDimension`. A bad args buffer then does nothing instead of hanging the GPU.

## Reading the result

```go
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

if err := staging.Map(ctx, wgpu.MapModeRead, 0, size); err != nil {
	return err
}
rng, err := staging.MappedRange(0, size)
if err != nil {
	_ = staging.Unmap()
	return err
}
b := rng.Bytes()
result := make([]float32, n)
for i := range result {
	result[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
}
if err := staging.Unmap(); err != nil {
	return err
}
```

`Map` blocks until the GPU is done with the buffer or `ctx` runs out, and it polls the device itself. The offset has to be a multiple of 8 and the size a multiple of 4.

On the native backends `rng.Bytes()` is the mapped memory itself, no copy, and it's only valid until `Unmap` (after that it returns nil). Copy out what you need first.

### WriteBuffer, then Map: submit in between

`Map` doesn't flush pending writes. On Vulkan, DX12 and Metal a `WriteBuffer` waits in the staging encoder for the next `Submit`, so a `WriteBuffer` straight into a `MapRead | CopyDst` buffer followed by `Map` gives you the old contents (zeros for a new buffer). That's what `TestIntegrationQueueReadBuffer` and `TestBufferMapRoundTrip` trip over on Vulkan.

Submit first. An empty submit is enough:

```go
if _, err := device.Queue().Submit(); err != nil {
	return err
}
```

### Without blocking

A render loop that can't wait starts the map and checks on it later:

```go
pending, err := staging.MapAsync(wgpu.MapModeRead, 0, size)
if err != nil {
	return err
}

// Once per frame:
device.Poll(wgpu.PollPoll) // Queue.Submit does this too
if ready, err := pending.Status(); ready {
	if err != nil {
		return err
	}
	rng, err := staging.MappedRange(0, size)
	if err != nil {
		_ = staging.Unmap()
		return err
	}
	process(rng.Bytes())
	_ = staging.Unmap()
}
```

Call `MapAsync` after the `Submit` that fills the buffer: it waits for the submissions that exist when you call it, not for later ones.

`Status` doesn't poll anything by itself. `device.Poll(wgpu.PollPoll)` or the next `Queue.Submit` moves the map along. `pending.Wait(ctx)` blocks like `Map` does.

## Timing a compute pass

Can't be done through the wgpu package yet. `ComputePassDescriptor` only has `Label`, and core drops compute-pass timestamp writes anyway. The HAL has them (`hal.ComputePassTimestampWrites`), but a `wgpu.Device` doesn't give you its HAL queue, so you'd have to drive the HAL yourself.

What the wgpu package does have is `Device.CreateQuerySet` with `wgpu.QueryTypeTimestamp`, `RenderPassDescriptor.TimestampWrites` and `CommandEncoder.ResolveQuerySet`. A timestamp query set needs `gputypes.FeatureTimestampQuery` on the device, and the Vulkan and DX12 adapters don't report it right now. There's also no way to get the timestamp period from a `wgpu.Queue`. Details are in [COMPUTE-BACKENDS.md](COMPUTE-BACKENDS.md#timestamp-queries).

## Errors

Every create call returns an error, check it. Validation errors from core are typed, so `errors.As` works: `*core.CreateComputePipelineError` (workgroup size over the limit, empty entry point), `*core.CreateBindGroupError`, `*core.CreateBufferError` and `*core.FeatureError` (import `github.com/roffe/gowgpu/core`).

Pass and encoder errors come back from `Finish()` as described above. Map errors are sentinels: `wgpu.ErrMapAlignment`, `wgpu.ErrMapInvalidMode`, `wgpu.ErrMapCanceled` and friends.

`wgpu.ErrOutOfMemory` exists, but only the Vulkan swapchain code returns it today, so `errors.Is(err, wgpu.ErrOutOfMemory)` won't catch a failed buffer or pipeline.

## Further reading

- [COMPUTE-BACKENDS.md](COMPUTE-BACKENDS.md): what each backend does and doesn't do
- [WebGPU compute passes](https://www.w3.org/TR/webgpu/#compute-passes)
- [WGSL](https://www.w3.org/TR/WGSL/)
