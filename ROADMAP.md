# Roadmap

Where gowgpu is and what comes next. No dates, just the order i'd take things in.

## Where it is

gowgpu runs fyne's WebGPU painter on Linux, Windows, macOS, iOS and Android, through Vulkan, DX12, Metal and GL. The [README](README.md) has the table of what has run on what hardware, and the known issues.

## Next

1. **The failing root tests.** Automatic pipeline layouts on Vulkan and DX12 (Metal takes pipelines without one), and `Map` flushing a `WriteBuffer` that wasn't submitted yet. CI skips them by name until then.
2. **DX12.** Honour dynamic bind group offsets, and take the vertex stride from the pipeline that draws instead of the one set at `SetVertexBuffer`.
3. **`gl_` names in GLSL.** naga keeps WGSL identifiers that start with `gl_`, which GLSL reserves. Rename them on the way out, either in naga or before the GLSL reaches the driver.
4. **Vulkan image layouts.** The validation layer reports layout errors from the state tracking gogpu wrote, and every queue write starts from `UNDEFINED`, which allows a driver to throw away what's in the texture.
5. **What the vet left open.** `DispatchIndirect` validation has to keep its bind group alive until the GPU is done with it and give every dispatch its own params. Depth, multisampled and compressed textures need zeroing on the HAL side. Render bundles crash on Vulkan.
6. **Real Apple hardware.** Metal has only run on GitHub's macOS runners and in the iOS simulator.
7. **The browser backend.** Get `examples/browser-compute` building against the current API again and run it in a browser.
8. **The lint backlog**, then a first tagged release.

## Not planned

- A CPU rasterizer. lavapipe and WARP do that job, and they run the real shaders.
- The Rust wgpu-native backend.
- Going back to an FFI written in Go: cgo is faster here, and every app that uses gowgpu so far needs cgo anyway.
