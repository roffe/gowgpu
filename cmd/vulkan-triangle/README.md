# vulkan-triangle

Draws a red triangle on a dark blue background in an 800x600 window, straight on `hal/vulkan`. There's no `wgpu` package or `core` in between, so if this draws and your app doesn't, look above the HAL.

Windows only. It makes a Win32 window and a `VK_KHR_win32_surface` surface for it.

## What it does

Instance, surface, adapters, device, then a BGRA8Unorm swapchain with FIFO. It always opens the first adapter. The shaders are WGSL that naga compiles to SPIR-V when the modules get created. Then a pipeline with an empty layout, and a loop that acquires, draws 3 vertices, submits and presents until you close the window.

The Win32 message pump runs on the main thread and all GPU work on a render thread from `internal/thread`. Resizing configures the swapchain again. While you drag the border it waits until you let go. Every 180 frames it waits for the device to go idle and resets the command pool.

The instance gets `InstanceFlagsDebug`, so the Khronos validation layer is switched on when it's installed and skipped when it's not. Its messages go to `hal.Logger()`, and this command never calls `hal.SetLogger`.

## Building

Vulkan needs cgo. From Linux with mingw-w64:

```bash
CGO_ENABLED=1 GOOS=windows CC=x86_64-w64-mingw32-gcc go build ./cmd/vulkan-triangle
```

On Windows itself it's `go build ./cmd/vulkan-triangle` with `CGO_ENABLED=1` and a mingw-w64 gcc in `PATH`.

With cgo off the build stops at `hal/vulkan needs cgo`. On anything but Windows the build constraints exclude all its files.

## Running

It needs a Vulkan driver, `vulkan-1.dll` comes with it. Then run `vulkan-triangle.exe`.

It prints each step as it goes, and every 60 frames the frame count and the average FPS since the loop started:

```
=== Vulkan Triangle Integration Test (Multi-Thread) ===

1. Creating window... OK
2. Creating render thread... OK
3. Creating Vulkan backend... OK
4. Creating Vulkan instance... OK
5. Creating surface... OK
6. Enumerating adapters... OK (found 1)
   - Adapter 0: <name> (<vendor> <driver info>)
7. Opening device... OK
8. Configuring surface... OK
9. Creating shader modules... OK
10. Creating pipeline layout... OK
11. Creating render pipeline... OK

=== Starting Render Loop ===
Close the window to exit

Rendered 60 frames (60.0 FPS)
Rendered 120 frames (60.0 FPS)
...
```

Close the window and it prints `=== Test Complete ===` with the total frames and the average FPS.

When a step fails it prints `FATAL:` and the error on that same line, and exits with 1. Without `vulkan-1.dll` that's step 4. Errors inside the loop print `Frame error:` and it carries on.

## Files

| File | What's in it |
|---|---|
| `main.go` | setup, render loop and cleanup |
| `window_windows.go` | the Win32 window and message pump, through `golang.org/x/sys/windows` |
| `shaders.go` | the WGSL shaders, plus SPIR-V from glslc that nothing uses any more (the comment there says why) |
| `shaders/` | the GLSL that SPIR-V came from |

## See also

These are Windows only too:

- `cmd/vk-test`: Vulkan library, instance, surface and device, no drawing
- `cmd/wgpu-triangle`: a triangle through the public `wgpu` API, and `cmd/wgpu-triangle-mt` the same with a render thread like this one
- `cmd/gles-test`: a triangle on the GL backend through WGL
