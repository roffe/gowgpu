# Architecture

How gowgpu is put together, from the API you call down to the driver.

The root package (`wgpu`) is the API. In a native build it checks your calls in `core` and then runs them on a backend through the `hal` interfaces. In the browser it skips both and talks to `navigator.gpu`. The backends reach their drivers through cgo (Vulkan, Metal, EGL and GL) or plain `syscall` (DX12 and WGL).

```
                         your code
             wgpu "github.com/roffe/gowgpu"
               |                          |
       native (*_native.go)        js/wasm (*_browser.go)
               |                          |
             core/                 internal/browser/
   validation, state, lifetimes    syscall/js -> navigator.gpu
               |
             hal/   interfaces and registry
               |
    +----------+----------+----------+----------+
    |          |          |          |          |
  vulkan/    metal/     dx12/      gles/      noop/
   cgo        cgo      syscall   EGL: cgo   for tests
                                 WGL: syscall
```

## Packages

| Path | What's in it |
|---|---|
| root (`*.go`) | the public API. `*_native.go` is the core/hal path, `*_browser.go` the js/wasm one |
| `core/` | validation, state tracking, error scopes, deferred destruction |
| `hal/` | the backend interfaces and registry, with the backends below it |
| `hal/allbackends` | a blank import that registers what the platform has |
| `gputypes/` | the WebGPU enums and structs (gogpu/gputypes v0.8.0, MIT, carried in this module) |
| `internal/browser` | `navigator.gpu` through `syscall/js` |
| `internal/dl` | `dlopen` and `dlsym` for EGL, X11 and Wayland |
| `internal/pipelinecache` | the pipeline cache files on disk |
| `internal/raytracing` | acceleration structure builds and their validation |
| `internal/indirect` | overflow-safe math for indirect draw spans |
| `internal/thread` | a render thread helper the `cmd/` demos use |
| `cmd/` | `vk-gen` and `vk-cgo` for the Vulkan bindings, `cgocheck`, small test programs |
| `examples/` | compute and headless rendering examples |

`core` and `hal` are ordinary importable packages. gogpu planned to move them under `internal/`, that never happened here. An app normally needs the root package, `gputypes` and a backend import, nothing more.

## The root package

This is what you use: `Instance`, `Adapter`, `Device`, `Queue`, `Buffer`, `Texture`, `TextureView`, `Sampler`, `ShaderModule`, `BindGroupLayout`, `PipelineLayout`, `BindGroup`, `RenderPipeline`, `ComputePipeline`, `CommandEncoder`, `CommandBuffer`, `RenderPassEncoder`, `ComputePassEncoder`, `RenderBundle`, `QuerySet`, `Surface` and `SurfaceTexture`.

Everything that can fail returns an error, and everything you create has a `Release()`. The common `gputypes` constants are re-exported (`wgpu.BufferUsageStorage`, `wgpu.BackendsGL` and so on), for the rest import `gputypes`. Descriptors turn into hal descriptors through their `toHAL()` methods.

Draw-time checks live here too, in the pass encoders: a pipeline is set, the bind groups match its layout, enough vertex buffers are bound, the index format matches. A failed check is recorded on the encoder and comes back from `End()` or `Finish()`, with sentinels like `ErrDrawMissingPipeline` for `errors.Is`.

Logging goes through `log/slog` and stays silent until you call `wgpu.SetLogger`.

## core

core sits between the API and hal. It does the validating so hal doesn't have to.

- `core/validate.go` checks descriptors for textures, samplers, shader modules, pipeline layouts, pipelines, bind group layouts, bind groups and render passes against the device's limits and features. Feature gates are in `feature_validate.go`, texture usage rules in `texture_usage_validate.go`. How they're tested: [VALIDATION-TESTING.md](VALIDATION-TESTING.md).
- Typed errors in `core/error.go` (`CreateTextureError`, `CreateBindGroupError` and friends) work with `errors.As`.
- Errors while encoding are recorded with `SetError` and returned from `End()` or `Finish()`, the WebGPU way. Error scopes are `Device.PushErrorScope` and `PopErrorScope`.
- A surface goes Unconfigured, Configured, Acquired and back. You can't acquire twice or present without an acquire. `Surface.SetPrepareFrame` hooks in a size or DPI check that runs before every acquire.
- A command encoder is Recording, InRenderPass, InComputePass, Finished or Error. A pass has to end before the next one begins, and after a recorded error the encoder stays in Error.
- Resource usage is tracked the way Rust wgpu-core's `track/mod.rs` does it. Each render pass, each compute dispatch and each copy is a usage scope (`core/track` `TextureUsageScope`, `BufferUsageScope`): the attachments, the textures and buffers its bind groups bind, the vertex, index and indirect buffers, the copy's endpoints. Conflicts, the same texture as colour and depth attachment or sampled by the pass that draws into it, a buffer a dispatch writes and reads, are only checked inside one scope. The command buffer keeps an `EncoderTextureTracker` and `EncoderBufferTracker` with the first and last usage of every resource; when a scope is applied to them, the transitions from the resource's last usage come out and go in as HAL barriers: a copy's and a dispatch's right ahead of it, a render pass's in a command buffer of its own inserted ahead of the pass's (a pass can't take barriers inside, and its bind groups are only known when it ends; the encoder's first pass is never split, and nothing is reopened after a split pass until the next command needs it). At `Queue.Submit` the device tracker (`core/device_tracker.go`) merges each command buffer in turn: barriers from the device state into its first usages go in a barrier command buffer right ahead of it, and the device adopts its last usages. The trackers and scope live in the pooled encoder state, so a frame allocates nothing for them.
- HAL handles sit in `Snatchable` wrappers, so destroying one can't race whoever is using it.

`Queue.Submit` refuses a command buffer that was already submitted, or that uses a buffer, texture or bind group you released.

## hal

The interfaces every backend implements are in `hal/api.go`, `hal/command.go` and `hal/resource.go`: `Backend`, `Instance`, `Adapter`, `Device`, `Queue`, `Surface`, `CommandEncoder`, `RenderPassEncoder`, `ComputePassEncoder` and the resource types.

hal trusts core. What's left in the backends is nil checks, and those return errors starting with `BUG:`, so a hole in core's validation shows up as one.

### Vulkan

`hal/vulkan`, first pick on Linux, the BSDs, Windows and Android.

- `vk/` holds the bindings, generated from `vk.xml`, see [Native calls](#native-calls).
- `memory/` is a buddy allocator, and it refuses anything bigger than `maxMemoryAllocationSize`.
- Command buffers come from a pool 16 at a time and go back with one `vkResetCommandPool`.
- Reconfiguring a swapchain is transactional. After a sync or present failure the swapchain is marked broken and has to be made again.
- With a `CompatibleSurface`, adapter selection asks `vkGetPhysicalDeviceSurfaceSupportKHR` for every queue family.
- Surfaces: Xlib and Wayland on Linux and the BSDs, Win32, `CAMetalLayer` through MoltenVK on macOS, and `ANativeWindow` on Android (API 29 or newer, see [ANDROID.md](ANDROID.md)).
- `PresentWithDamage` passes the rects on with `VK_KHR_incremental_present` when the device has it.
- `InstanceFlagsDebug` turns on `VK_LAYER_KHRONOS_validation` if it's installed. Android refuses the flag.

### Metal

`hal/metal`, macOS and iOS.

- `objc.go` is the Objective-C runtime through cgo, see [Native calls](#native-calls).
- An autorelease pool locks the goroutine to its OS thread from push to pop. `RunInFramePool` wraps a frame in one, so drawables don't pile up during a live resize.
- Indexed multi-draws of 1024 to 52428 draws go through an Indirect Command Buffer (`icb_indexed.go`).
- Texture copies get split into array layers or 3D depth slices (`texture_copy.go`).
- `Surface.SetPresentsWithTransaction` presents inside Core Animation's transaction, for live resize on macOS.

### DX12

`hal/dx12`, Windows. No cgo, it all goes through `syscall`.

- `d3d12/`, `dxgi/` and `d3dcompile/` are the COM interfaces and the DLLs behind them.
- Shaders: naga turns WGSL into HLSL, and FXC from `d3dcompiler_47.dll` compiles that for shader model 5.1. With `GOGPU_DX12_DXIL=1` naga writes DXIL for shader model 6.0 directly. `GOGPU_DX12_DXIL_VALIDATE=1` runs the DXC validator on every blob, so the real error ends up in the log. Compiled shaders are cached in memory by SHA-256.
- Resource states are tracked per command list, and the barriers they need go in front of them at submit (`state_tracker.go`).
- Copies get split into 2D array layers or 3D slices, with D3D12's 512 byte placement alignment (`copy_plan.go`).
- At most two frames in flight.
- The swapchain is made for the HWND, or through DirectComposition when you configure premultiplied alpha. `GOGPU_DX12_FORCE_HWND=1` skips DirectComposition so RenderDoc can capture.
- WARP, Microsoft's CPU renderer, only shows up when there's no hardware adapter, so VMs and CI still render.
- `InstanceFlagsDebug` turns on the debug layer and DRED, and a removed device logs its breadcrumbs. Add `InstanceFlagsValidation` for GPU based validation.
- A uniform buffer with a dynamic offset gets a root CBV after its group's descriptor table, and a storage buffer with one gets its offset in root constants that naga's HLSL adds to its addresses, as in Rust's wgpu-hal. A render pass keeps its vertex buffer views and sets the ones that changed right before a draw, with the strides of the pipeline that draws.

### GL

`hal/gles` is OpenGL and OpenGL ES, through EGL on Linux and Android and through WGL on Windows. It only has Linux and Windows files, so there's no GL on the BSDs: `hal/allbackends` imports it there, but the package is empty.

Where the context lives:

| Platform | Context |
|---|---|
| Windows (WGL) | on a hidden 1x1 window owned by the Instance, made on the first `Lock()` so it's born on the render thread |
| Linux, X11 or no display | on the Instance, surfaceless or on a 16x16 pbuffer, desktop GL 3.3 core. Surfaces on the same window system share it |
| Linux, Wayland | on each Surface, made on the app's `wl_display`. Desktop GL 3.3 core, or GLES 3.0 when that's all EGL has |
| Android | on the Instance, GLES 3.0 on EGL's default display, so the device survives the new `ANativeWindow` you get on every resume |

So on Wayland the GL instance has no context until there's a surface. Make the surface first and pass it as `CompatibleSurface` to the first `RequestAdapter` on that instance. GL adapters are enumerated once per instance, and without a surface you get a placeholder that can't open a device.

`AdapterContext` guards the context with a mutex and `LockOSThread`. `Lock()` makes it current for resource and command work, `LockForDC` and `LockForSurface` for presenting, `Unlock()` makes nothing current again.

Shaders go from WGSL to GLSL through naga, at the GLSL version the driver reports. Separate textures and samplers become combined samplers (`SamplerBindMap`). Through EGL, `PresentWithDamage` uses `eglSwapBuffersWithDamageKHR` when it's there. WGL ignores the rects.

Compute runs with `glDispatchCompute`, but only on GL 4.3 or GLES 3.1, or with `GL_ARB_compute_shader`. NVIDIA gives you exactly the 3.3 core context gowgpu asks for, and there the adapter reports no compute shaders: `CreateComputePipeline` returns `missing required downlevel flags: ComputeShaders`. Check `Adapter.DownlevelCapabilities()` first, see [DownlevelCapabilities](#downlevelcapabilities).

`CopyTextureToBuffer` reads the texture with `glReadPixels` and writes each row into the buffer's GL storage at WebGPU's row pitch. Textures are stored top row first, like WebGPU's; only the blit to the window flips.

Known issue: naga keeps GLSL identifiers that start with `gl_` and Mali refuses them, so don't name anything `gl_*` in WGSL.

### noop

`hal/noop` takes any surface target and draws nothing. It's for tests. `hal/allbackends` doesn't register it, and `CreateInstance` skips it even when it is registered. Tests use its types directly, or wrap a noop device with `wgpu.NewDeviceFromHAL`.

### Browser

`GOOS=js GOARCH=wasm`. The root `*_browser.go` files call `internal/browser`, which talks to `navigator.gpu` through `syscall/js`. No core, no hal and no cgo: the browser does its own validation.

- A promise blocks the calling goroutine until it settles (`AwaitPromise`).
- Data moves with `js.CopyBytesToGo` and `js.CopyBytesToJS`.
- WGSL goes to `createShaderModule` as it is, naga isn't involved.
- A surface is a canvas with a `GPUCanvasContext`. `Present` does nothing since the browser presents by itself, damage rects are ignored, and `Configure` refuses `PresentModeMailbox` and `PresentModeImmediate`.
- JS methods on hot paths are bound once up front (`method.Call("bind", obj)`).

It builds (`examples/browser-test`), but it hasn't been run yet.

### Ray tracing

Experimental. Vulkan (`VK_KHR_acceleration_structure`), DX12 (DXR) and Metal build acceleration structures, GLES and noop return an error. The interface is in `hal/raytracing.go`. Build ordering, compaction, alignment and the `FeatureRayQuery` gate are checked in `internal/raytracing`. More in [RAY-TRACING.md](RAY-TRACING.md).

## Native calls

gogpu called the drivers through goffi, which allocated on every call. gowgpu uses cgo instead. A call through one of its trampolines into a no-op C function is about 23 ns and zero allocations (`go test -bench . ./hal/vulkan/vk/`, on my Ryzen 7 7800X3D). In fyne's widgets-50 scene (NVIDIA, Wayland) a frame went from 1.43 to 1.25 ms and from 311 to 193 KB allocated.

### Trampolines

A trampoline is a small static C function, one per C signature, that takes the function pointer as its first argument and calls it. The preambles include no SDK headers, they declare what they need by hand. Vulkan handles cross as `uint64_t` and pointer sized integers as `uintptr_t`, so 32-bit builds come out right. Pointers to Go memory cross as real C pointers (`const uint8_t *`), not as integers. The Go GC and stack copying keep seeing them, and since cgo finds no pointers inside a `uint8_t`, cgocheck doesn't walk the Go memory a create info points at.

Trampolines are marked `#cgo noescape` and `#cgo nocallback` wherever that's safe, all 61 Vulkan ones included. Stack pointees stay on the stack and nothing allocates. nocallback is a promise that C never calls back into Go during that call, so the Vulkan debug messenger stays in C: it copies each message into a ring buffer, and a goroutine drains that into the logger every 50 ms.

### Vulkan bindings

Generated from `vk.xml`. `cmd/vk-gen` writes the wrappers, then `cmd/vk-cgo` rewrites them onto the trampolines and writes `trampolines.h`. Regenerating gives the same files byte for byte:

```bash
(cd cmd/vk-gen && go run . -spec vk.xml -out ../../hal/vulkan/vk/) && go run ./cmd/vk-cgo hal/vulkan/vk
```

### Libraries

Nothing links against the GPU libraries. They're opened at run time, when a backend first needs them:

| Library | Opened by | Names |
|---|---|---|
| Vulkan loader | `hal/vulkan/vk`, `dlopen` (`LoadLibrary` on Windows) | `libvulkan.so.1`, `libvulkan.so` on Android, `libvulkan.dylib` (MoltenVK), `vulkan-1.dll` |
| EGL | `internal/dl` | `libEGL.so.1` |
| Wayland | `internal/dl` | `libwayland-client.so.0`, `libwayland-egl.so.1` |
| X11 | `internal/dl` | `libX11.so.6` |

`internal/dl` falls back to the unversioned `.so` name for each of them. GL functions come from `eglGetProcAddress`.

So a build needs no SDK or headers for any of these, and the binary starts fine on a machine that doesn't have them. That backend just fails to make its instance and isn't used.

### Metal

Objective-C through cgo. `objc_msgSend` is cast to the exact C type of each call shape: integer and pointer arguments by count, and separate trampolines for floats and for structs passed by value (`MTLViewport`, `MTLRegion`, `MTLClearColor` and so on), so the C compiler applies the ABI rules. It links `-lobjc -framework Foundation -framework Metal -framework QuartzCore`, so building needs the Apple SDK, like any cgo app there.

### DX12 and WGL

`syscall`, no cgo. `d3d12.dll`, `dxgi.dll`, `dcomp.dll` and `d3dcompiler_47.dll` through `syscall.NewLazyDLL`, `opengl32.dll`, `gdi32.dll` and `user32.dll` through `syscall.LoadDLL`.

On Windows only Vulkan needs cgo (mingw-w64 when you cross-compile). The root package with `hal/dx12` and `hal/gles` builds with `CGO_ENABLED=0`. `hal/allbackends` pulls in Vulkan, and without cgo it stops with `hal/vulkan needs cgo: build with CGO_ENABLED=1 and a C compiler`.

### Browser

`syscall/js`, nothing else.

### Checking cgo for another platform

`cmd/cgocheck` type-checks a cgo package for a GOOS/GOARCH this machine has no SDK for. It fakes package `C` and writes the preambles out, so `clang -ffreestanding` can compile the C side on its own. One package at a time, its imports go through the real cgo:

```bash
go run ./cmd/cgocheck -goos ios -goarch arm64 -out /tmp/pre ./hal/metal
```

## Backend registration

Every backend package registers itself from `init()` with `hal.RegisterBackend`. Import `hal/allbackends` to get what the platform has, or a backend package directly for just that one:

```go
import _ "github.com/roffe/gowgpu/hal/allbackends"
```

| Platform | `hal/allbackends` registers |
|---|---|
| Linux | Vulkan, GL (EGL) |
| FreeBSD, OpenBSD, NetBSD | Vulkan |
| Windows | Vulkan, DX12, GL (WGL) |
| macOS | Metal, Vulkan (MoltenVK, if installed) |
| iOS | Metal |
| Android arm64, amd64 | Vulkan, GLES 3.0 (EGL) |

The BSDs only get Vulkan, with X11 and Wayland surfaces through `hal/vulkan/api_xdg.go`, and they have never been built or run. 32-bit Android registers nothing.

The browser build doesn't use hal at all. `hal/allbackends` has no files under js/wasm, so importing it there fails with `build constraints exclude all Go files`. Keep that import in a file the wasm build skips.

`CreateInstance(nil)` uses `gputypes.BackendsPrimary`: Vulkan, Metal and DX12, plus the browser's WebGPU in a wasm build. GL is registered, but only used when you ask for it:

```go
instance, err := wgpu.CreateInstance(&wgpu.InstanceDescriptor{Backends: gputypes.BackendsGL})
```

It's meant as a fallback. fyne makes the default instance first, and only goes for a GL one when nothing else can draw its window.

If you pass a descriptor, set `Backends`. A zero mask means no backends at all, and `RequestAdapter` comes back with `no adapters available`.

The backends are tried in this order: Vulkan, Metal, DX12, GL. `RequestAdapter` takes the first adapter that isn't a CPU one, and `PowerPreference` is only a hint. CPU adapters only win when there's no GPU, or when you set `ForceFallbackAdapter`. That's how a machine without a GPU still renders: WARP on Windows, and on Linux lavapipe, Mesa's CPU Vulkan driver, when it's installed.

core copies the hal registry once, on the first `CreateInstance` (a `sync.Once`), so a backend registered after that is never seen. Probing the backends is serialized by a mutex, because concurrent driver init raced on Windows CI.

## Surfaces

A surface is made from a typed target, so a backend always knows what kind of handle it got:

```go
// Raw handles. You keep them valid until Surface.Release returns.
surface, err := instance.CreateSurfaceUnsafe(wgpu.SurfaceTargetFromWaylandSurface(display, wlSurface))
```

```go
// A Go value that owns the window and implements wgpu.SurfaceTarget.
surface, err := instance.CreateSurfaceFromTarget(window)
```

The old `instance.CreateSurface(displayHandle, windowHandle)` still works and maps onto the same targets.

Every backend checks the kind with `hal.SurfaceTarget.RequireKind` before it touches a handle. A surface is made on every backend in the instance that accepts the target, and `RequestAdapter`, `GetSurfaceCapabilities` and `Configure` each use the one that belongs to the adapter's backend. The full mapping is in [SURFACE-TARGETS.md](SURFACE-TARGETS.md).

## Pending writes

`Queue.WriteBuffer` and `Queue.WriteTexture` don't submit anything by themselves. On Vulkan, Metal and DX12 they copy your data into staging memory and record the copy into one shared command encoder (`pending_writes.go`). `Queue.Submit` puts those copies in front of your command buffers. Same idea as wgpu-core's PendingWrites.

```
WriteBuffer(buf, data)  ---+
WriteBuffer(buf2, data) ---+--> copies recorded into one encoder
WriteTexture(tex, data) ---+
                           |
Queue.Submit(cmds)         |
  flush <------------------+
  hal Submit([pending copies, texture barriers, cmds...])
  staging, encoder and destinations stay alive until the GPU is done
```

The staging memory comes from a staging belt (`staging_belt.go`): 256 KB chunks handed out with a bump pointer and reused once the GPU is done with them, so the steady state allocates nothing. A bigger write gets its own staging buffers, at most 64 MB each (Rust wgpu's cap, and on Vulkan never more than `maxMemoryAllocationSize`), with one copy per piece.

GL writes go straight to the driver instead (its `SupportsCommandBufferCopies()` is false): no staging, no encoder.

Nothing but `Submit` flushes pending writes. Before you `Map` a buffer you just wrote, submit. An empty `Queue.Submit()` is enough.

## Resource lifecycle

```go
// Error checks left out.
instance, _ := wgpu.CreateInstance(nil)
defer instance.Release()

adapter, _ := instance.RequestAdapter(nil)
defer adapter.Release()

device, _ := adapter.RequestDevice(nil)
defer device.Release()

buffer, _ := device.CreateBuffer(&wgpu.BufferDescriptor{
	Size:  256,
	Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
})
defer buffer.Release()

encoder, _ := device.CreateCommandEncoder(nil)
pass, _ := encoder.BeginComputePass(nil)
// SetPipeline, SetBindGroup, Dispatch...
pass.End()
commands, _ := encoder.Finish()
device.Queue().Submit(commands) // returns the submission index, doesn't wait for the GPU
```

Release what you make. `Release()` doesn't pull a resource away from the GPU: the destroy is queued against the last submission (`core.DestroyQueue.Defer`) and runs once the GPU has finished it, which gets checked after every `Submit` and when the device waits for idle.

Buffers, bind groups and pipelines are refcounted on top of that. Passes and copies take a reference (`ResourceRef.Clone`) on each one they use. The encoder collects those, they move to the command buffer at `Finish()` and to the destroy queue at `Submit`, and they're dropped when that submission completes. The resource goes when its last reference does, so releasing it before `Submit` is fine.

`Buffer` and `BindGroup` also register a `runtime.AddCleanup`. If the GC collects one you never released, it warns through the default `slog` logger and takes the normal destroy path. That's a safety net, not a replacement for `Release()`.

`Instance.Release` releases the devices and then the surfaces it made, before the native instance.

## DownlevelCapabilities

WebGPU assumes every adapter has compute shaders, indirect draws, independent blending and so on, and a browser just doesn't offer an adapter that hasn't. gowgpu runs on drivers below that line, like a GLES 3.0 context without compute, so it reports what an adapter can't do instead of refusing it.

That's Rust wgpu's `DownlevelCapabilities`, not something from the WebGPU spec: 27 flags in `gputypes/downlevel.go`, at the same bit positions as Rust's.

| Backend | Where the flags come from |
|---|---|
| Vulkan | `VkPhysicalDeviceFeatures` |
| Metal, DX12, noop, browser | `DefaultDownlevelCapabilities()`, everything set |
| GL | queried from the context: version and extensions |

The flags go from the hal adapter into `core.Adapter`, are copied into the device, and you read them with `Adapter.DownlevelCapabilities()`:

```go
caps := adapter.DownlevelCapabilities()
if !caps.Flags.Contains(gputypes.DownlevelFlagsComputeShaders) {
	// no compute here, take the render pass path
}
```

`CreateComputePipeline` checks `DownlevelFlagsComputeShaders` itself (`core.Device.RequireDownlevelFlags`), so a GL context without compute gets an error and not a crash in the driver.

## Pipeline cache

Vulkan and DX12 keep what the driver compiled on disk, so the next launch starts faster.

- Vulkan: one `VkPipelineCache` per device, passed to every `vkCreateGraphicsPipelines` and `vkCreateComputePipelines`, written to disk with `vkGetPipelineCacheData` when the device is destroyed.
- DX12: every PSO's `GetCachedBlob` goes into its own `.pso` file and is handed back through `D3D12_CACHED_PIPELINE_STATE` next time. A stale blob gives `E_INVALIDARG`, then the blob is deleted and the PSO made without it.

The files live in `os.UserCacheDir()/gowgpu/<backend>/<adapter key>/` and are written atomically, temp file and rename, by `internal/pipelinecache`. If the cache can't be set up you get a warning in the log and pipelines are made uncached.

## Dependencies

go.mod has two:

- `github.com/gogpu/naga` v0.19.0, the shader compiler: WGSL to SPIR-V for Vulkan, MSL for Metal, HLSL or DXIL for DX12 and GLSL for GL.
- `golang.org/x/sys`.

`gputypes` used to be its own module (gogpu/gputypes v0.8.0). It's carried in this one now, with its MIT LICENSE in `gputypes/`.

To build you need Go 1.25 or newer (naga needs it) and a C compiler for cgo, except for a Windows build without Vulkan and the browser build. macOS and iOS need the Apple SDK. The GPU libraries themselves only have to be there at run time.
