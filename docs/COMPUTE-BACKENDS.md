# Compute on each backend

What compute looks like on each gowgpu backend and where they differ. How to write and run a compute shader is in [COMPUTE-SHADERS.md](COMPUTE-SHADERS.md).

## Support matrix

| | Vulkan | DX12 | Metal | GL | Browser | Noop |
|---|---|---|---|---|---|---|
| Platforms | Windows, Linux, the BSDs, Android, macOS (MoltenVK) | Windows | macOS, iOS | Linux, Windows, Android | `GOOS=js GOARCH=wasm` | anywhere, tests only |
| Compute pipelines | Yes | Yes | Yes | Depends on the context, not on desktop Linux (see below) | Yes | Accepted, nothing runs |
| Storage buffers | Yes | Yes | Yes | Same as compute | Yes | Plain Go memory |
| Indirect dispatch | Yes | Yes | Yes | Same as compute | Yes | No-op |
| Buffer mapping | Yes | Yes | Yes | Yes | Yes | Plain Go memory |
| Dynamic offsets | Yes | Ignored | Yes | Yes | Yes | n/a |
| Timestamp queries | Partly, see below | Partly, see below | No | Windows only | No | No |
| Workgroup limits | From the driver | Fixed, 1024 | Fixed, 1024 | From the driver | From the browser | n/a |

There's no software backend any more. A machine without a GPU gets compute through a CPU driver instead: Mesa's lavapipe on Vulkan, or WARP on DX12 (gowgpu only picks WARP when there's no hardware adapter).

Compute has been run on Vulkan (NVIDIA RTX 4080 SUPER on Linux): compute-copy, compute-sum and compute-particles all pass. DX12, Metal and GL have only run fyne so far, which doesn't use compute, so compute on those is untested.

Things that are the same everywhere:

- Every native backend puts a barrier after each dispatch, so later dispatches in the pass and copies after it see the writes.
- There are no automatic pipeline layouts on the native backends, so pass one. Vulkan and GL fail without it.
- `ComputePipelineDescriptor.Constants` and `ZeroInitializeWorkgroupMemory` aren't passed on by any backend yet.
- `DispatchIndirect` first runs a small shader that checks the workgroup counts and turns a dispatch over the device limit into zero workgroups. It destroys its bind group before the GPU runs it, and every indirect dispatch recorded before one `Submit` gets the params of the last one, so submit after each indirect dispatch until that's fixed.

## Vulkan

- naga compiles the WGSL to SPIR-V in `CreateShaderModule`. SPIR-V in `ShaderModuleDescriptor.SPIRV` goes to the driver as is.
- A nil pipeline layout fails with "compute pipeline layout is required (automatic layouts are not supported)".
- After every `Dispatch` and `DispatchIndirect` there's a compute-to-compute memory barrier. `End` adds a global barrier from compute writes to later compute, transfer and host access, so a `CopyBufferToBuffer` after the pass sees the result.
- Limits come from `VkPhysicalDeviceLimits` (`maxComputeWorkGroupInvocations`, `maxComputeWorkGroupSize`, `maxComputeSharedMemorySize`, `maxComputeWorkGroupCount`).
- `Queue.WriteBuffer` is staged and only goes out with the next `Submit`, and `Buffer.Map` doesn't flush it. See [WriteBuffer, then Map](COMPUTE-SHADERS.md#writebuffer-then-map-submit-in-between).

## DX12

- Shaders are compiled when the pipeline is created: WGSL -> HLSL -> DXBC with FXC (`d3dcompiler_47.dll`) by default, or WGSL -> DXIL straight from naga with `GOGPU_DX12_DXIL=1`. `GOGPU_DX12_DXIL_VALIDATE=1` also runs the DXC validator on every DXIL blob so real errors reach the log.
- The root signature comes from the pipeline layout. A nil layout gets an empty root signature, which only works for a shader without bindings.
- Descriptor heaps get bound the first time `SetBindGroup` (or `SetPipeline` for a pipeline with samplers) needs them.
- Dynamic bind group offsets are ignored ([gogpu/wgpu#343](https://github.com/gogpu/wgpu/issues/343)). Don't use `HasDynamicOffset` bindings on DX12.
- Before each dispatch the bound storage buffers are moved to `UNORDERED_ACCESS`, and after it there's a global UAV barrier.
- `DispatchIndirect` is `ExecuteIndirect` with the device's dispatch command signature (12 bytes of `D3D12_DISPATCH_ARGUMENTS`).
- Limits are fixed: 1024 invocations per workgroup, 1024 for X and Y, 64 for Z, 32 KB workgroup storage, 65535 workgroups per dimension.
- `WriteBuffer` is staged until the next `Submit` like on Vulkan.

## Metal

- naga compiles the WGSL to MSL.
- The compute encoder is an `MTLComputeCommandEncoder` from the command buffer, retained so it survives the autorelease pool drain.
- Bind groups go in with `setBuffer:offset:atIndex:`, `setTexture:atIndex:` and `setSamplerState:atIndex:`. Slots are counted per resource type across all groups, and dynamic offsets are added to the buffer offset.
- `arrayLength()` needs the buffer sizes in MSL, so before each dispatch the backend binds naga's `_buffer_sizes` argument with `setBytes`.
- `Dispatch` is `dispatchThreadgroups:threadsPerThreadgroup:` with the shader's workgroup size, `DispatchIndirect` is `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`. Each one is followed by `memoryBarrierWithScope:` for buffers and textures.
- Limits are fixed: 1024 per dimension and 1024 invocations per workgroup, 32 KB workgroup storage.
- `WriteBuffer` is staged until the next `Submit` like on Vulkan.

## GL

- Whether you get compute depends on the context the driver gives you. The adapter reports `DownlevelFlagsComputeShaders` for GLES 3.1+, desktop GL 4.3+ or `GL_ARB_compute_shader`, and without it `CreateComputePipeline` fails with "missing required downlevel flags: ComputeShaders". naga only writes real storage buffers for GLSL 4.30 and GLSL ES 3.10 and up, so in practice compute needs a GL 4.3 or GLES 3.1 context.
- On desktop Linux gowgpu asks EGL for exactly GL 3.3 core (GLES 3.0 is the fallback for a Wayland surface), and NVIDIA gives you exactly 3.3.0. So no compute there:
  - Without `RequestAdapterOptions.CompatibleSurface` the GL adapter has no real capabilities (a placeholder on Wayland, default limits and no downlevel flags on X11 or headless), and `CreateComputePipeline` fails with "missing required downlevel flags: ComputeShaders".
  - With a surface the adapter does report compute, because NVIDIA lists `GL_ARB_compute_shader` on that context. But naga then writes GLSL 3.30 with the storage buffers as uniform blocks, and the driver refuses the shader (NVIDIA: "assignment to uniform").
- Android asks for GLES 3.0 and Windows uses a plain `wglCreateContext` context. Whether those come with compute is up to the driver. Neither has been tried for compute.
- GL isn't in the default instance. Ask for it with `wgpu.InstanceDescriptor{Backends: gputypes.BackendsGL}`. It's not available on the BSDs.
- naga compiles the WGSL to GLSL when the pipeline is created. Don't name WGSL things `gl_*`: naga keeps the name and strict drivers (Mali) refuse it.
- Commands are recorded and replayed in `Queue.Submit`. A dispatch is `glDispatchCompute` (`glDispatchComputeIndirect` for indirect) followed by `glMemoryBarrier`.
- `WriteBuffer` goes straight to GL with no staging, so the "WriteBuffer, then Map" problem doesn't exist here.
- Limits come from GL (`GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS` and friends).

## Browser

- Built with `GOOS=js GOARCH=wasm`. The root package talks to `navigator.gpu` through `syscall/js`: no hal, no naga, no cgo.
- The WGSL string goes to `createShaderModule` as is.
- A nil pipeline layout becomes `"auto"`, and `ComputePipeline.GetBindGroupLayout` exists here (browser build only).
- `Map` awaits the `mapAsync` promise on a goroutine. `MappedRange.Bytes()` is a Go copy made with `getMappedRange` and `js.CopyBytesToGo`, not a view.
- `DispatchIndirect` is `dispatchWorkgroupsIndirect`.
- There are no query sets in the browser build.
- Limits and features are whatever the browser reports.
- Not run yet. `examples/browser-compute` doesn't build (it was written against an older API and was already broken in gogpu v0.34.5). `examples/browser-test` builds.

## Noop

- `hal/noop` is for tests. `hal/allbackends` doesn't register it, importing `hal/noop` does. `wgpu.Instance.RequestAdapter` skips it, so you only get it through the hal.
- Buffers are plain Go byte slices: `WriteBuffer` copies into them and mapping returns a pointer into them. Copies and dispatches do nothing, and a compute pipeline is an empty stand-in.
- `CreateQuerySet` returns `hal.ErrTimestampsNotSupported`.

## Timestamp queries

This is messy right now, so here it is in one place.

The wgpu package has `Device.CreateQuerySet` with `wgpu.QueryTypeTimestamp`, `RenderPassDescriptor.TimestampWrites` and `CommandEncoder.ResolveQuerySet`. A timestamp query set needs `gputypes.FeatureTimestampQuery` on the device. Compute passes can't be timed through the wgpu package: `ComputePassDescriptor` has no timestamp writes and core drops them.

Below that, in the HAL:

| Backend | Timestamp writes | Adapter reports `FeatureTimestampQuery` |
|---------|------------------|------------------------------------------|
| Vulkan | Compute passes only, render pass timestamp writes are ignored. Each write resets its query and runs `vkCmdWriteTimestamp` (top of pipe at the start of the pass, bottom of pipe at the end). `ResolveQuerySet` puts in a barrier and copies with `vkCmdCopyQueryPoolResults` (64-bit, wait). | No |
| DX12 | Render and compute passes: timestamp query heap, `EndQuery` at the start and end of the pass, `ResolveQueryData`. | No, and DX12 won't open a device with a feature its adapter doesn't report |
| Metal | Not implemented, `CreateQuerySet` returns `hal.ErrTimestampsNotSupported`. | Yes on Metal 3 GPUs, so don't trust the flag there |
| GL | Windows: render and compute passes, when the context has timestamp queries. Linux: `CreateQuerySet` returns `hal.ErrTimestampsNotSupported`. | When `GL_ARB_timer_query` is there |
| Browser | None | n/a |
| Noop | `CreateQuerySet` returns `hal.ErrTimestampsNotSupported`. | No |

So through the wgpu package the only combination that can work today is render-pass timestamps on GL on Windows, and that hasn't been run.

The tick length is `hal.Queue.GetTimestampPeriod()` (Vulkan reads `timestampPeriod`, DX12 converts the queue's timestamp frequency, Metal returns 1.0). A `wgpu.Queue` doesn't expose it yet.

## Workgroup size

64 works everywhere. WebGPU only promises 256 invocations per workgroup, so check `device.Limits()` before going above that, and measure: the best size depends on the GPU and the shader.

## Alignment

| What | Rule | Checked by gowgpu |
|------|------|-------------------|
| Bind group buffer offset | Multiple of `MinUniformBufferOffsetAlignment` or `MinStorageBufferOffsetAlignment` (256 in the WebGPU defaults, the driver's value on Vulkan) | Yes, `CreateBindGroup` |
| `Queue.WriteBuffer` | Offset and size multiples of 4 | Yes |
| `Buffer.Map`, `Buffer.MappedRange` | Offset multiple of 8, size multiple of 4 | Yes |
| `CommandEncoder.CopyBufferToBuffer` | Offsets and size multiples of 4 | No, keep to it yourself |
| `CommandEncoder.CopyTextureToBuffer`, `CopyBufferToTexture` | `bytesPerRow` a multiple of 256 | Yes, `Finish` fails |
| `DispatchIndirect` | Offset multiple of 4, 12 bytes from the offset | Yes |
