# Ray tracing (experimental)

Short version: the HAL has acceleration structure code for Vulkan, DX12 and Metal, and there's a `gputypes.FeatureRayQuery` flag for it, but nothing above the HAL uses it yet and it hasn't been run in gowgpu. There's no CPU fallback (that went with the software backend) and no example program.

Ray tracing isn't in the W3C WebGPU spec, so all of this can still change.

The target is inline ray queries, meaning rays traced from compute and fragment shaders (`FeatureRayQuery`). Ray tracing pipelines aren't there, `FeatureRayTracingPipelines` is only a flag.

## Backends

| Backend | What the HAL calls |
|---------|--------------------|
| Vulkan | `VK_KHR_acceleration_structure` (`vkCreateAccelerationStructureKHR`, `vkCmdBuildAccelerationStructuresKHR` and friends) |
| DX12 | DXR through `ID3D12Device5` and `ID3D12GraphicsCommandList4` |
| Metal | `MTLAccelerationStructure` (`newAccelerationStructureWithSize:` on the device) |
| GL, noop | Stubs, `CreateAccelerationStructure` returns an error |
| Browser | Nothing, WebGPU has no ray tracing |

## The HAL interface

The types are in `hal/raytracing.go`. On `hal.Device`:

- `CreateAccelerationStructure`, `DestroyAccelerationStructure`
- `GetAccelerationStructureBuildSizes`
- `GetAccelerationStructureDeviceAddress`
- `TlasInstanceToBytes`, which packs a `hal.TlasInstance` into the backend's 64-byte instance format

On `hal.CommandEncoder`:

- `BuildAccelerationStructures` (batched, one call for several builds)
- `PlaceAccelerationStructureBarrier`
- `CopyAccelerationStructure` (clone or compact)
- `ReadAccelerationStructureCompactSize`

The descriptors follow Rust wgpu-hal: `AccelerationStructureDescriptor`, `AccelerationStructureEntries` (triangles, AABBs or instances), `BuildAccelerationStructureDescriptor` and `GetAccelerationStructureBuildSizesDescriptor`.

## internal/raytracing

BLAS and TLAS bookkeeping and validation, following Rust wgpu-core:

- compaction state per BLAS: Idle -> Waiting -> Ready -> Compacted
- build ordering, so a TLAS only references BLASes that have been built
- scratch buffer alignment
- checks for `FeatureRayQuery`, geometry and instance counts, update flags, AABB strides and triangle transforms

It talks to the device through a small `DeviceContext` interface. core has an adapter for it in `core/ray_tracing.go`, but nothing calls it yet.

## What's missing

- No ray tracing API in the wgpu package: no BLAS or TLAS types, nothing on `Device` or `CommandEncoder`. `Device.HalDevice()` gives you the `hal.Device`, but you can't record into the HAL encoder behind a `wgpu.CommandEncoder` or reach the HAL queue from wgpu, so in practice you'd drive the HAL yourself.
- No backend reports `FeatureRayQuery` on its adapter.
- The Vulkan device is created with only `VK_KHR_swapchain` (plus `VK_KHR_incremental_present` when the driver has it). `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_deferred_host_operations` and the `bufferDeviceAddress` feature aren't enabled, so the Vulkan path can't work as it is.
- A shader can't see an acceleration structure yet. `gputypes.BindGroupLayoutEntry` has an `AccelerationStructure` field, but no backend turns it into a descriptor and there's no binding resource type to put a TLAS in a bind group.
- It hasn't been run on any hardware in gowgpu. The tests are unit tests in `internal/raytracing` (with a mock `DeviceContext`) and a few benchmarks in `hal/bench_raytracing_test.go`.
