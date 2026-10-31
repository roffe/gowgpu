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

Draw-time checks live here too, in the pass encoders: a pipeline is set and fits the pass's attachments, the bind groups match its layout, every vertex buffer it reads is bound and holds what the draw reads, the index buffer and format match. A failed check is recorded on the encoder and comes back from `End()` or `Finish()`, with sentinels like `ErrDrawMissingPipeline` for `errors.Is`.

Bind groups reach the backend lazily. `SetBindGroup` takes any slot below `MaxBindGroups` and the binder keeps the group, with its dynamic offsets, until the current pipeline's layout takes it at that slot, and a pipeline change hands the groups on again from the first slot whose layout changes, as wgpu-core's binder does. So a group may be set before the pipeline, on every backend.

Render bundles are recorded here as well. A `RenderBundleEncoder` keeps its commands in Go, and `RenderPassEncoder.ExecuteBundles` replays them through the pass, as wgpu-core and Dawn do: they're validated and tracked as if the pass had recorded them, and the backend draws them inline. Each bundle starts with no pipeline, bind groups or vertex and index buffers, and the pass has none of them after the bundles. No backend has bundles of its own.

Logging goes through `log/slog` and stays silent until you call `wgpu.SetLogger`.

## core

core sits between the API and hal. It does the validating so hal doesn't have to.

- `core/validate.go` checks descriptors for textures, texture views, samplers, shader modules, pipeline layouts, pipelines, bind group layouts, bind groups and render passes against the device's limits and features. Copies are checked in `copy_validate.go`, feature gates in `feature_validate.go`, texture usage rules in `texture_usage_validate.go`, and the root package checks a pipeline against the naga IR of its shaders (`pipeline_interface.go`), as wgpu-core's `check_stage` does. How they're tested: [VALIDATION-TESTING.md](VALIDATION-TESTING.md).
- Typed errors in `core/error.go` (`CreateTextureError`, `CreateBindGroupError` and friends) work with `errors.As`.
- Errors while encoding are recorded with `SetError` and returned from `End()` or `Finish()`, the WebGPU way. The first one is kept, since it's the one that explains the rest.
- Error scopes. An error a device call returns (the create calls, `Finish`, `Queue.Submit`, `WriteBuffer`, `WriteTexture`, `Surface.Configure`) also goes to the innermost scope `PushErrorScope` made that captures its kind, out of memory as such and the rest as validation, and `PopErrorScope` returns it. An error no scope captures goes to the function `Device.SetUncapturedErrorHandler` set, on the goroutine of the call that made it and before that call returns it, or without one to the logger at debug level. A lost or released device reports nothing, and `PopErrorScope` panics without a pushed scope, as in the browser.
- On a device made without `FeatureIndirectFirstInstance`, every indirect draw of a render pass reads a copy of its records that a compute pass ahead of the render pass makes, with the instance count of a record that has a first instance zeroed (`core/indirect_draw_validation.go`), as wgpu-core's indirect draw validation does. `DispatchIndirect` gets a small check of its workgroup counts ahead of it on every device, which turns a dispatch past the device's limit into none.
- A surface goes Unconfigured, Configured, Acquired and back. You can't acquire twice or present without an acquire. `Surface.SetPrepareFrame` hooks in a size or DPI check that runs before every acquire. The hook and the backend's acquire run without the surface's lock, so the hook may call back into the surface, and `Configure`, `Unconfigure` and `Destroy` wait for an acquire in progress. Configuring onto another device unconfigures on the old one first.
- A command encoder is Recording, InRenderPass, InComputePass, Finished or Error. A pass has to end before the next one begins, and after a recorded error the encoder stays in Error.
- Resource usage is tracked the way Rust wgpu-core's `track/mod.rs` does it. Each render pass, each compute dispatch and each copy is a usage scope (`core/track` `TextureUsageScope`, `BufferUsageScope`): the attachments, the textures and buffers its bind groups bind, the vertex, index and indirect buffers, the copy's endpoints. Conflicts, the same texture as colour and depth attachment or sampled by the pass that draws into it, a buffer a dispatch writes and reads, are only checked inside one scope. The command buffer keeps an `EncoderTextureTracker` and `EncoderBufferTracker` with the first and last usage of every resource; when a scope is applied to them, the transitions from the resource's last usage come out and go in as HAL barriers: a copy's and a dispatch's right ahead of it, a render pass's in a command buffer of its own inserted ahead of the pass's (a pass can't take barriers inside, and its bind groups are only known when it ends; the encoder's first pass is never split, and nothing is reopened after a split pass until the next command needs it). At `Queue.Submit` the device tracker (`core/device_tracker.go`) merges each command buffer in turn: barriers from the device state into its first usages go in a barrier command buffer right ahead of it, and the device adopts its last usages. The trackers and scope live in the pooled encoder state, so a frame allocates nothing for them.
- Textures are tracked per subresource, as wgpu-core's texture tracker does. A `track.TextureSelector` picks mip levels and array layers (the zero selector is the whole texture, and a 3D texture's slices are one layer). A texture whose subresources are all in one state is kept as that one state, on a direct path that costs what whole-texture tracking did, and one whose aren't as ranges of layers per mip level, which collapse back once they agree again. Attachments and bound views put the subresources they take in the scope, copies their regions and queue writes what they write. Conflicts are per subresource, and a transition carries the range that changes state: a subresource range on Vulkan, the subresource indices of the range on DX12. The aspects of a depth/stencil texture are tracked together, as in wgpu-core.
- Surface textures are tracked like any other, in the order their work runs, as wgpu-core does. An acquired image starts in no state and on the zero list, and the submit that first uses it zeroes it ahead of its first user, unless that user clears it, a queue write sets it or an earlier submit did. Every submit that uses one ends with a barrier into the present state, and a present of an image nothing set zeroes it first, with a clear pass in the batch the present submits, on GL too. The views of one acquisition share one texture, and the Vulkan backend records the barriers core gives it and adds none of its own.
- A texture a pass discarded (`StoreOpDiscard`) reads as zeros. A later pass of the same encoder that loads it clears it instead, a pass, dispatch or copy that reads it otherwise gets a clear pass ahead of it, and a later command buffer of the submit that uses it gets the clear in its barrier command buffer. One still discarded when the submit ends goes back on the queue's zero list.
- HAL handles sit in `Snatchable` wrappers, so destroying one can't race whoever is using it.

`Queue.Submit` refuses a command buffer that was already submitted, that's in the list twice, that belongs to another device, or that uses a buffer, texture or bind group you released. When the backend fails a submit the device is lost, as in wgpu-core: its pending maps fail with `ErrMapDeviceLost`, and the queue's calls return errors that wrap `ErrDeviceLost` from then on.

## hal

The interfaces every backend implements are in `hal/api.go`, `hal/command.go` and `hal/resource.go`: `Backend`, `Instance`, `Adapter`, `Device`, `Queue`, `Surface`, `CommandEncoder`, `RenderPassEncoder`, `ComputePassEncoder` and the resource types.

hal trusts core. What's left in the backends is nil checks, and those return errors starting with `BUG:`, so a hole in core's validation shows up as one.

A backend's `Queue.Submit` that fails after the GPU got the command buffers returns the submission index with its error, so core keeps the submission and its resources until that index completes. An error with index 0 means nothing ran.

### Vulkan

`hal/vulkan`, first pick on Linux, the BSDs, Windows and Android.

- `vk/` holds the bindings, generated from `vk.xml`, see [Native calls](#native-calls).
- `memory/` is a buddy allocator, and it refuses anything bigger than `maxMemoryAllocationSize`.
- Command buffers come from a pool 16 at a time and go back with one `vkResetCommandPool`.
- WGSL is compiled with wgpu-hal's options: bounds checks for indices, buffers and image loads, bounded loops, a point size of 1, and `frag_depth` clamped to the viewport's depth range, which a render pass pushes as push constants for pipelines whose layout has none of its own. A module with overrides is compiled again for each pipeline, with its constants.
- A reconfigure destroys the old swapchain right after the `vkCreateSwapchainKHR` that retired it, whatever the result, as wgpu-hal and Dawn do, so one that fails leaves the surface without a swapchain until the next `Configure`. After a sync or present failure the swapchain is marked broken and has to be made again. A surface configured with view formats gets a swapchain made with `VK_SWAPCHAIN_CREATE_MUTABLE_FORMAT_BIT_KHR`, which needs `VK_KHR_swapchain_mutable_format`.
- With a `CompatibleSurface`, adapter selection asks `vkGetPhysicalDeviceSurfaceSupportKHR` for every queue family, and `Configure` asks again for the surface it configures.
- Surfaces: Xlib and Wayland on Linux and the BSDs, Win32, `CAMetalLayer` through MoltenVK on macOS, and `ANativeWindow` on Android (API 29 or newer, see [ANDROID.md](ANDROID.md)).
- `PresentWithDamage` passes the rects on with `VK_KHR_incremental_present` when the device has it.
- `InstanceFlagsDebug` turns on `VK_LAYER_KHRONOS_validation` if it's installed. Android refuses the flag.

### Metal

`hal/metal`, macOS and iOS.

- `objc.go` is the Objective-C runtime through cgo, see [Native calls](#native-calls).
- An autorelease pool locks the goroutine to its OS thread from push to pop. `RunInFramePool` wraps a frame in one, so drawables don't pile up during a live resize.
- Indexed multi-draws of 1024 to 52428 draws go through an Indirect Command Buffer (`icb_indexed.go`).
- Each pipeline stage is compiled to MSL for its entry point alone, after naga's `ProcessOverrides`, with the bindings numbered the way the pipeline layout binds them (`mslmap`). The vertex stage reads its own attributes, naga's vertex pulling, with vertex buffer i at buffer slot 30-i, and a stage that reads runtime-sized arrays or the vertex buffers gets their sizes in naga's `_buffer_sizes`.
- Limits come from the GPU family's feature set table, as Dawn has them. A command buffer that ends in an error marks the queue lost, and `Submit` and `WaitIdle` return `ErrDeviceLost` from then on.
- Texture copies get split into array layers or 3D depth slices (`texture_copy.go`), and buffer-texture blits the way Metal's validation takes them, as Dawn's copy splitter does.
- `Surface.SetPresentsWithTransaction` presents inside Core Animation's transaction, for live resize on macOS.

### DX12

`hal/dx12`, Windows. No cgo, it all goes through `syscall`.

- `d3d12/`, `dxgi/` and `d3dcompile/` are the COM interfaces and the DLLs behind them.
- Shaders: naga turns WGSL into HLSL, and FXC from `d3dcompiler_47.dll` compiles that for shader model 5.1. With `GOGPU_DX12_DXIL=1` naga writes DXIL for shader model 6.0 directly, on an adapter that has it. `GOGPU_DX12_DXIL_VALIDATE=1` runs the DXC validator on every blob, so the real error ends up in the log. Each pipeline compiles its stages for its own layout, after naga's `ProcessOverrides`, and the vertex stage with the fragment stage it feeds. Compiled shaders are cached in memory by SHA-256.
- Every root signature ends with three root constants that naga reads `first_vertex`, `first_instance` and `num_workgroups` from, as in Rust's wgpu-hal. Indirect draws and dispatches get them from per-layout command signatures that read a copy of the records with the constants ahead of each.
- Resource states are tracked per command list, and the barriers they need go in front of them at submit (`state_tracker.go`).
- Copies get split into 2D array layers or 3D slices, with D3D12's 512 byte placement alignment (`copy_plan.go`).
- At most two frames in flight. A Mailbox swap chain has three buffers, and every swap chain is made able to tear where the system supports it and tears for Immediate only.
- The swapchain is made for the HWND, or through DirectComposition when you configure premultiplied alpha. `GOGPU_DX12_FORCE_HWND=1` skips DirectComposition so RenderDoc can capture.
- A removed device's errors wrap `hal.ErrDeviceLost`, with the reason.
- WARP, Microsoft's CPU renderer, only shows up when there's no hardware adapter, so VMs and CI still render.
- `InstanceFlagsDebug` turns on the debug layer and DRED, and a removed device logs its breadcrumbs. Add `InstanceFlagsValidation` for GPU based validation.
- A uniform buffer with a dynamic offset gets a root CBV after its group's descriptor table, and a storage buffer with one gets its offset in root constants that naga's HLSL and DXIL add to its addresses, as in Rust's wgpu-hal. A render pass keeps its vertex buffer views and sets the ones that changed right before a draw, with the strides of the pipeline that draws.

### GL

`hal/gles` is OpenGL and OpenGL ES, through EGL on Linux and Android and through WGL on Windows. It only has Linux and Windows files, so there's no GL on the BSDs: `hal/allbackends` imports it there, but the package is empty.

Where the context lives:

| Platform | Context |
|---|---|
| Windows (WGL) | on a hidden 1x1 window owned by the Instance, which a thread of its own makes and destroys. The context is made on the first `Lock()`, so it's born on the render thread |
| Linux, X11 or no display | on the Instance, surfaceless or on a 16x16 pbuffer, the newest desktop GL core context EGL makes from 4.6 down to 3.3, or GLES 3.0 when EGL makes no desktop context of a version. Surfaces on the same window system share it |
| Linux, Wayland | on each Surface, made on the app's `wl_display`. Desktop GL from 4.6 down to 3.3 core, or GLES 3.0 when that's all EGL has |
| Android | on the Instance, GLES 3.0 on EGL's default display, so the device survives the new `ANativeWindow` you get on every resume |

So on Wayland the GL instance has no context until there's a surface. Make the surface first and pass it as `CompatibleSurface` to the first `RequestAdapter` on that instance. GL adapters are enumerated once per instance, and without a surface you get a placeholder that can't open a device.

The desktop versions are asked for newest first because NVIDIA makes exactly the version it's asked for, and compute needs 4.3. Every EGL context asks for robust access, EGL 1.5's or `EGL_EXT_create_context_robustness`'s, and goes without where EGL refuses both, as wgpu-hal asks. An EGL display is terminated only with the last context gowgpu made on it.

`AdapterContext` guards the context with a mutex and `LockOSThread`. `Lock()` makes it current for resource and command work, `LockForDC` and `LockForSurface` for presenting, `Unlock()` makes nothing current again.

Shaders go from WGSL to GLSL through naga, at the GLSL version the driver reports, each stage after naga's `ProcessOverrides`. Indices into arrays and buffers are clamped (naga's Restrict policies, as Dawn does on GL), image loads read zero out of bounds from GLSL 4.30, and `frag_depth` is clamped to a uniform the render pass sets to the viewport's depth range. Separate textures and samplers become combined samplers (`SamplerBindMap`), and a texture sampled with two samplers is refused, since a texture unit holds one. Through EGL, `PresentWithDamage` uses the swap with damage entry point the display lists. WGL ignores the rects.

Compute runs with `glDispatchCompute` where the GLSL the device writes has compute shaders and storage buffers, GLSL 4.30 or GLSL ES 3.10. Below that the adapter reports no compute shaders and `CreateComputePipeline` returns `missing required downlevel flags: ComputeShaders`. Check `Adapter.DownlevelCapabilities()` first, see [DownlevelCapabilities](#downlevelcapabilities).

Where the context has texture views (GL 4.3, `ARB_texture_view`, `OES_` or `EXT_texture_view`), textures are made with immutable storage, and a view the shaders bind that has only some of its texture's mip levels or layers, or another dimension, gets a `glTextureView` of its own, as Dawn makes them. Without texture views a view's mip range is set on the texture itself, and a bind group refuses a view of some layers, or of another dimension, to sample or to store to more than one layer. GL has no views in another format either way.

`CopyTextureToBuffer` reads the texture with `glReadPixels` and writes each row into the buffer's GL storage at WebGPU's row pitch. On GLES it reads in the pairs every ES driver takes and repacks the texels. Compressed textures can't be read back, and on GLES neither can depth or stencil. Textures are stored top row first, like WebGPU's; only the blit to the window flips.

Known issue: naga keeps GLSL identifiers that start with `gl_` and Mali refuses them, so don't name anything `gl_*` in WGSL.

### noop

`hal/noop` takes any surface target and draws nothing. It's for tests. `hal/allbackends` doesn't register it, and `CreateInstance` skips it even when it is registered. Tests use its types directly, or wrap a noop device with `wgpu.NewDeviceFromHAL`.

### Browser

`GOOS=js GOARCH=wasm`. The root `*_browser.go` files call `internal/browser`, which talks to `navigator.gpu` through `syscall/js`. No core, no hal and no cgo: the browser does its own validation.

- A promise blocks the calling goroutine until it settles (`AwaitPromise`). That's fine on the main goroutine and any other, but not in a `js.FuncOf` callback such as a `requestAnimationFrame` one: the browser settles the promise only after the callback returns, so blocking there deadlocks. Callback-driven code maps with `MapAsync` and checks `MapPending.Status`.
- `MapAsync` calls `mapAsync` before it returns and settles the map in the promise's callbacks. `Poll(PollWait)` and `WaitIdle` wait for `onSubmittedWorkDone`.
- Data moves with `js.CopyBytesToGo` and `js.CopyBytesToJS`.
- WGSL goes to `createShaderModule` as it is, naga isn't involved.
- A surface is a canvas with a `GPUCanvasContext`. The browser presents by itself, so `Present` only notes the texture, and `GetCurrentTexture` waits for the next animation frame when the canvas still holds that texture and the event loop hasn't run since, so a native-style render loop shows its frames. It never waits in a `js.FuncOf` callback. Damage rects are ignored, and `Configure` refuses `PresentModeMailbox` and `PresentModeImmediate`.
- A browser exception comes back as an error: from the call that threw, or from `Finish` for a pass method. The device records `GPUDevice.lost`, and calls return `ErrDeviceLost` after it. Error scopes are the browser's own, and `SetUncapturedErrorHandler` is a listener for the device's `uncapturederror` event.
- JS methods on hot paths are bound once up front (`method.Call("bind", obj)`).

It has run in Firefox 158 and Chrome 156. `examples/browser-test` and `examples/browser-compute` build.

### Ray tracing

Experimental. Vulkan (`VK_KHR_acceleration_structure`), DX12 (DXR) and Metal build acceleration structures, GLES and noop return an error. The interface is in `hal/raytracing.go`. Build ordering, compaction, alignment and the `FeatureRayQuery` gate are checked in `internal/raytracing`. More in [RAY-TRACING.md](RAY-TRACING.md).

## Native calls

gogpu called the drivers through goffi, which allocated on every call. gowgpu uses cgo instead. A call through one of its trampolines into a no-op C function is about 23 ns and zero allocations (`go test -bench . ./hal/vulkan/vk/`, on my Ryzen 7 7800X3D). In fyne's widgets-50 scene (NVIDIA, Wayland) a frame went from 1.43 to 1.25 ms and from 311 to 193 KB allocated.

### Trampolines

A trampoline is a small static C function, one per C signature, that takes the function pointer as its first argument and calls it. The preambles include no SDK headers, they declare what they need by hand. Pointer sized integers cross as `uintptr_t`, and so do Vulkan's dispatchable handles (instance, physical device, device, queue, command buffer), which are pointers. The non-dispatchable ones are 64 bits on every target, so they're `uint64` in Go and `uint64_t` in C. Structs are laid out as C lays them out on 32-bit targets too: arm, mips and windows/386 align 64-bit members to 8 bytes where Go aligns them to 4, so the generator writes their padding out for those (`types_align8_gen.go`). `TestStructLayoutsMatchC` checks every struct against `vulkan_core.h` through clang for amd64, arm64, 386, arm and windows/386.

Pointers to Go memory cross as real C pointers, not as integers, so the Go GC and stack copying keep seeing them. A pointer to memory without pointers crosses as `const uint8_t *`. A pointer to memory that holds pointers, a create or submit info, crosses as `const void *` built from `unsafe.Slice(p, n)`, the form in which cgocheck checks every element with its Go type, `n` being the length `vk.xml` gives the parameter. What such memory points at has to be pinned for the call (`vk.Pin`, a `runtime.Pinner`) or be a local the compiler keeps on the stack, or cgocheck panics with `argument of cgo function has Go pointer to unpinned Go pointer`. That costs the create calls a few tens of ns. The draw, bind and dynamic state commands take no such arguments, and `Submit` copies its arrays to its stack when they fit, as they nearly always do, so it allocates nothing.

Trampolines are marked `#cgo noescape` and `#cgo nocallback` wherever that's safe, all 161 Vulkan ones included, which carry the 690 generated wrappers. Stack pointees stay on the stack and nothing allocates. nocallback is a promise that C never calls back into Go during that call, so the Vulkan debug messenger stays in C: it copies each message into a ring buffer, and a goroutine drains that into the logger every 50 ms.

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

`syscall`, no cgo. `d3d12.dll`, `dxgi.dll` and `dcomp.dll` from System32 through `windows.NewLazySystemDLL`, so a DLL of the same name next to the executable isn't taken, `d3dcompiler_47.dll` through `syscall.NewLazyDLL`, which looks next to the executable first, so shipping one picks the FXC version, and `opengl32.dll`, `gdi32.dll` and `user32.dll` through `syscall.LoadDLL`.

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

The BSDs only get Vulkan, with X11 and Wayland surfaces through `hal/vulkan/api_xdg.go`, and they have never been built or run. 32-bit Android registers nothing, though `hal/vulkan` and `hal/gles` build there.

32-bit builds: linux/386, windows/386, android/arm and android/386 build. The root suite passes as a linux/386 binary on NVIDIA's 32-bit Vulkan driver and as windows/386 on Vulkan under wine; 32-bit Android hasn't run, and linux/arm hasn't been built here, though `TestStructLayoutsMatchC` checks its struct layouts. DX12 refuses a 32-bit build at `CreateInstance`, as it passes 64-bit values to COM methods as one `uintptr`.

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

GL writes go straight to the driver instead (its `SupportsCommandBufferCopies()` is false), with no staging. GL's pending writes record only the passes that zero textures, and submit them before a direct write goes out.

`Submit` flushes pending writes, and so does `Surface.Present`, which submits them before it hands the frame over, as wgpu-core does. `MapAsync` submits first when one of them goes to the buffer it maps, so a map sees the write.

New buffers are zeroed in the batch on Vulkan, ahead of every command buffer that can use them, unless a write covers them whole, and a mappable one through a mapping; on GL every new buffer is written with zeros straight away, and DX12 and Metal hand them out zeroed. New textures are zeroed by copies or a clear pass in the batch. On GL, which has no copies in the batch, the ones a pass can't clear are written with zeros from the CPU, and compressed ones are zeroed by the GL backend when it makes them.

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

Buffers, bind groups, pipelines, texture views, samplers and pipeline layouts are refcounted on top of that. Passes and copies take a reference (`ResourceRef.Clone`) on each one they use, a bind group holds its views and samplers, a pipeline its layout, and an encoder the views of its render pass attachments. The encoder's references move to the command buffer at `Finish()` and to the destroy queue at `Submit`, and they're dropped when that submission completes. The resource goes when its last reference does, so releasing it before `Submit` is fine. `Texture.Release` and `QuerySet.Release` take the lock a `Submit` holds, so a release on another goroutine either comes first and fails the `Submit`, or waits for it and defers the destroy past it.

`Buffer` and `BindGroup` also register a `runtime.AddCleanup`. If the GC collects one you never released, it warns through the default `slog` logger and takes the normal destroy path, but a buffer that's still mapped is kept, mapping and all. That's a safety net, not a replacement for `Release()`.

After `Device.Release`, pending maps fail with `ErrMapDeviceLost`, `Poll` returns false, `MapAsync` returns `ErrReleased`, and an encoder still recording records nothing more and fails `Finish`.

`Instance.Release` releases the devices and then the surfaces it made, before the native instance.

## DownlevelCapabilities

WebGPU assumes every adapter has compute shaders, indirect draws, independent blending and so on, and a browser just doesn't offer an adapter that hasn't. gowgpu runs on drivers below that line, like a GLES 3.0 context without compute, so it reports what an adapter can't do instead of refusing it.

That's Rust wgpu's `DownlevelCapabilities`, not something from the WebGPU spec: 27 flags in `gputypes/downlevel.go`, at the same bit positions as Rust's.

| Backend | Where the flags come from |
|---|---|
| Vulkan | `VkPhysicalDeviceFeatures`, and the device's extensions for surface view formats |
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
- DX12: every PSO's `GetCachedBlob` goes into its own `.pso` file and is handed back through `D3D12_CACHED_PIPELINE_STATE` next time. A blob D3D12 refuses, with `E_INVALIDARG` or `D3D12_ERROR_DRIVER_VERSION_MISMATCH`, is deleted and the PSO made without it. The directory is named by the adapter's PCI ids, subsystem included, and the user-mode driver version, not the LUID, which only holds until a reboot.

The files live in `os.UserCacheDir()/gowgpu/<backend>/<adapter key>/`, written by `internal/pipelinecache`. Each starts with a header of a magic, the blob's length and its SHA-256, and goes to disk through a temp file of its own that's synced before the rename, so two processes saving at once or a crash can't leave a torn file for the driver to load. A file that doesn't check out is rebuilt. If the cache can't be set up you get a warning in the log and pipelines are made uncached.

## Dependencies

go.mod has two:

- `github.com/gogpu/naga`, the shader compiler: WGSL to SPIR-V for Vulkan, MSL for Metal, HLSL or DXIL for DX12 and GLSL for GL. Pinned to upstream main until a release after v0.20.5 has gogpu/naga#131 in it.
- `golang.org/x/sys`.

`gputypes` used to be its own module (gogpu/gputypes v0.8.0). It's carried in this one now, with its MIT LICENSE in `gputypes/`.

To build you need Go 1.25 or newer (naga needs it) and a C compiler for cgo, except for a Windows build without Vulkan and the browser build. macOS and iOS need the Apple SDK. The GPU libraries themselves only have to be there at run time.
