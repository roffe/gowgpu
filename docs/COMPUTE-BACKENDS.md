# Compute on each backend

What compute looks like on each gowgpu backend and where they differ. How to write and run a compute shader is in [COMPUTE-SHADERS.md](COMPUTE-SHADERS.md).

## Support matrix

| | Vulkan | DX12 | Metal | GL | Browser | Noop |
|---|---|---|---|---|---|---|
| Platforms | Windows, Linux, the BSDs, Android, macOS (MoltenVK) | Windows | macOS, iOS | Linux, Windows, Android | `GOOS=js GOARCH=wasm` | anywhere, tests only |
| Compute pipelines | Yes | Yes | Yes | GL 4.3 or GLES 3.1 (see below) | Yes | Accepted, nothing runs |
| Storage buffers | Yes | Yes | Yes | Same as compute | Yes | Plain Go memory |
| Indirect dispatch | Yes | Yes | Yes | Same as compute | Yes | No-op |
| Buffer mapping | Yes | Yes | Yes | Yes | Yes | Plain Go memory |
| Dynamic offsets | Yes | Yes | Yes | Yes | Yes | n/a |
| Pipeline constants | Yes | Yes | Yes | Yes | Yes | n/a |
| Timestamp queries | No, see below | No, see below | No | Desktop GL | No | No |
| Workgroup limits | From the driver | Fixed, 1024 | From the GPU family | From the driver | From the browser | n/a |

There's no software backend any more. A machine without a GPU gets compute through a CPU driver instead: Mesa's lavapipe on Vulkan, or WARP on DX12 (gowgpu only picks WARP when there's no hardware adapter).

The compute tests in the root package pass on Vulkan (NVIDIA RTX 4080 SUPER and lavapipe), on GL (NVIDIA's and Mesa llvmpipe's 4.6 contexts) and on DX12 under wine, and they passed on Metal on an M2 Pro before the October 2026 vet, whose Metal changes haven't run on a Mac yet.

Things that are the same everywhere:

- Every native backend puts a barrier after each dispatch, so later dispatches in the pass and copies after it see the writes.
- A pipeline without a `Layout` gets the one its shader implies, and `GetBindGroupLayout` hands out its groups.
- `ComputePipelineDescriptor.Constants` go to the shader: each native backend runs the stage through naga's `ProcessOverrides` with them before it writes the shader, and the browser passes them to `createComputePipeline`. `ZeroInitializeWorkgroupMemory` isn't passed on by any backend yet.
- `DispatchIndirect` first runs a small shader that checks the workgroup counts and turns a dispatch over the device limit into zero workgroups.

## Vulkan

- naga compiles the WGSL to SPIR-V in `CreateShaderModule`, with bounds checks for indices, buffers and image loads and with bounded loops, as wgpu-hal does. A module with overrides is compiled again for each pipeline, with its constants. SPIR-V in `ShaderModuleDescriptor.SPIRV` goes to the driver as is.
- After every `Dispatch` and `DispatchIndirect` there's a compute-to-compute memory barrier. `End` adds a global barrier from compute writes to later compute, transfer and host access, so a `CopyBufferToBuffer` after the pass sees the result, and every command buffer ends with a barrier that makes writes visible to the host.
- Limits come from `VkPhysicalDeviceLimits` (`maxComputeWorkGroupInvocations`, `maxComputeWorkGroupSize`, `maxComputeSharedMemorySize`, `maxComputeWorkGroupCount`).
- `Queue.WriteBuffer` is staged and only goes out with the next `Submit`. `MapAsync` submits first when a staged write goes to the buffer it maps.

## DX12

- Shaders are compiled when the pipeline is created: WGSL -> HLSL -> DXBC with FXC (`d3dcompiler_47.dll`) by default, or WGSL -> DXIL straight from naga with `GOGPU_DX12_DXIL=1`. `GOGPU_DX12_DXIL_VALIDATE=1` also runs the DXC validator on every DXIL blob so real errors reach the log.
- The root signature comes from the pipeline layout, and each pipeline compiles its entry points with its own layout's registers. The root signature ends with three root constants, which hold `num_workgroups` for a dispatch, set from the indirect arguments for `DispatchIndirect`.
- Descriptor heaps get bound the first time `SetBindGroup` (or `SetPipeline` for a pipeline with samplers) needs them.
- A uniform buffer with a dynamic offset gets a root CBV instead of a descriptor in its group's table, and a storage buffer with one gets its offset in root constants that naga's HLSL and DXIL add to every address, as in Rust's wgpu-hal. `arrayLength()` on such a storage buffer counts from the bind group's offset to the end of the buffer.
- Before each dispatch the resources of the groups the pipeline uses go to the states their bindings use, read-write storage to `UNORDERED_ACCESS` and read-only storage to the shader resource states, and after it there's a global UAV barrier.
- `DispatchIndirect` is `ExecuteIndirect` with the device's dispatch command signature (12 bytes of `D3D12_DISPATCH_ARGUMENTS`).
- Limits are fixed: 1024 invocations per workgroup, 1024 for X and Y, 64 for Z, 32 KB workgroup storage, 65535 workgroups per dimension.
- `WriteBuffer` is staged until the next `Submit` like on Vulkan.

## Metal

- naga compiles the WGSL to MSL when the pipeline is created, for its entry point alone.
- The compute encoder is an `MTLComputeCommandEncoder` from the command buffer, retained so it survives the autorelease pool drain.
- Bind groups go in with `setBuffer:offset:atIndex:`, `setTexture:atIndex:` and `setSamplerState:atIndex:`. The pipeline layout counts the slots per resource type across its groups, each pipeline compiles its entry point's MSL with that numbering, and dynamic offsets are added to the buffer offset.
- `arrayLength()` needs the buffer sizes in MSL, so before each dispatch the backend binds naga's `_buffer_sizes` argument with `setBytes`.
- `Dispatch` is `dispatchThreadgroups:threadsPerThreadgroup:` with the shader's workgroup size, `DispatchIndirect` is `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`. Each one is followed by `memoryBarrierWithScope:` for buffers and textures.
- Limits come from the GPU family's feature set table: 1024 invocations per workgroup on most, 512 on Apple2 and in the iOS Simulator. A pipeline whose workgroup is past what its pipeline state takes, `maxTotalThreadsPerThreadgroup`, which a register-heavy kernel can bring under 1024, fails to be made.
- `WriteBuffer` is staged until the next `Submit` like on Vulkan.

## GL

- Compute needs the GLSL the device writes to have compute shaders and storage buffers, GLSL 4.30 or GLSL ES 3.10, so a GL 4.3 or GLES 3.1 context. Below that the adapter doesn't report `DownlevelFlagsComputeShaders`, and `CreateComputePipeline` fails with "missing required downlevel flags: ComputeShaders".
- On desktop Linux gowgpu asks EGL for the newest desktop GL core from 4.6 down to 4.3 before 3.3, since NVIDIA makes exactly the version it's asked for. NVIDIA and Mesa give you 4.6, with compute. GLES 3.0 is the fallback where EGL makes no desktop context.
- On Wayland the GL instance has no context until there's a surface, so pass it as `RequestAdapterOptions.CompatibleSurface`. On X11 and headless an adapter made without a surface is probed like any other.
- Android asks for GLES 3.0 and Windows uses a plain `wglCreateContext` context, so whether those come with compute is up to the driver. WGL on NVIDIA gives 4.6, and the compute tests pass on it under wine; Android hasn't been tried for compute.
- GL isn't in the default instance. Ask for it with `wgpu.InstanceDescriptor{Backends: gputypes.BackendsGL}`. It's not available on the BSDs.
- naga compiles the WGSL to GLSL when the pipeline is created. Don't name WGSL things `gl_*`: naga keeps the name and strict drivers (Mali) refuse it.
- Commands are recorded and replayed in `Queue.Submit`. A dispatch is `glDispatchCompute` (`glDispatchComputeIndirect` for indirect) followed by `glMemoryBarrier`.
- `WriteBuffer` goes straight to GL with no staging.
- Limits come from GL (`GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS` and friends).

## Browser

- Built with `GOOS=js GOARCH=wasm`. The root package talks to `navigator.gpu` through `syscall/js`: no hal, no naga, no cgo.
- The WGSL string goes to `createShaderModule` as is.
- A nil pipeline layout becomes `"auto"`, which the browser derives.
- `MapAsync` calls `mapAsync` before it returns and settles in the promise's callbacks, and `Map` waits for that. Don't block in a `js.FuncOf` callback: the promise only settles after the callback returns. `MappedRange.Bytes()` is a Go copy made with `getMappedRange` and `js.CopyBytesToGo`, not a view.
- `DispatchIndirect` is `dispatchWorkgroupsIndirect`.
- There are no query sets in the browser build.
- Limits and features are whatever the browser reports. `RequestDevice(nil)` asks for the adapter's limits, as natively.
- It has run in Firefox 158 and Chrome 156. `examples/browser-compute` and `examples/browser-test` build.

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
| Metal | Not implemented: no timestamp query sets yet. Occlusion query sets work. | No |
| GL | Render and compute passes, on Linux and Windows, when the context has timestamp queries. | On desktop GL where `glQueryCounter` is there (GL 3.3, `GL_ARB_timer_query`), not on GLES |
| Browser | None | n/a |
| Noop | `CreateQuerySet` returns `hal.ErrTimestampsNotSupported`. | No |

So through the wgpu package the combination that works today is render-pass timestamps on desktop GL. `TestTimestampWritesResolve` runs it on Linux.

The tick length is `hal.Queue.GetTimestampPeriod()` (Vulkan reads `timestampPeriod`, DX12 converts the queue's timestamp frequency, Metal returns 1.0). A `wgpu.Queue` doesn't expose it yet.

## Workgroup size

64 works everywhere. WebGPU only promises 256 invocations per workgroup, so check `device.Limits()` before going above that, and measure: the best size depends on the GPU and the shader.

## Alignment

| What | Rule | Checked by gowgpu |
|------|------|-------------------|
| Bind group buffer offset | Multiple of `MinUniformBufferOffsetAlignment` or `MinStorageBufferOffsetAlignment` (256 in the WebGPU defaults, the driver's value on Vulkan) | Yes, `CreateBindGroup` |
| `Queue.WriteBuffer` | Offset and size multiples of 4 | Yes |
| `Buffer.Map`, `Buffer.MappedRange` | Offset multiple of 8, size multiple of 4 | Yes |
| `CommandEncoder.CopyBufferToBuffer` | Offsets and size multiples of 4 | Yes, `Finish` fails |
| `CommandEncoder.CopyTextureToBuffer`, `CopyBufferToTexture` | `bytesPerRow` a multiple of 256 | Yes, `Finish` fails |
| `DispatchIndirect` | Offset multiple of 4, 12 bytes from the offset | Yes |
