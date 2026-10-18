# Roadmap

Where gowgpu is and what comes next. No dates, just the order i'd take things in.

## Where it is

gowgpu runs fyne's WebGPU painter on Linux, Windows, macOS, iOS and Android, through Vulkan, DX12, Metal and GL. The [README](README.md) has the table of what has run on what hardware, and the known issues.

## Next

1. **CI that works.** The workflow is still gogpu's: it builds with cgo off and cross compiles a Rust backend that's gone. It should build with cgo for Linux, Windows, macOS, iOS, Android and the browser, and run the tests on lavapipe (Linux), WARP (Windows) and the paravirtual GPU GitHub's macOS runners have.
2. **The 10 failing root tests.** Automatic pipeline layouts on Vulkan, and `Map` flushing a `WriteBuffer` that wasn't submitted yet.
3. **Dead API out.** `HeadlessSurfaceTarget`, `Surface.ReadPixels` and `WritePixels` and the HAL framebuffer extensions behind them only ever worked on the software backend.
4. **GL readback.** Reading a texture back swaps red and blue.
5. **DX12.** Honour dynamic bind group offsets, and take the vertex stride from the pipeline that draws instead of the one set at `SetVertexBuffer`.
6. **`gl_` names in GLSL.** naga keeps WGSL identifiers that start with `gl_`, which GLSL reserves. Rename them on the way out, either in naga or before the GLSL reaches the driver.
7. **Vulkan image layouts.** The validation layer reports layout errors from the state tracking gogpu wrote.
8. **Real Apple hardware.** Metal has only run on GitHub's macOS runners and in the iOS simulator.
9. **The browser backend.** Get `examples/browser-compute` building against the current API again and run it in a browser.
10. **The lint backlog**, then a first tagged release.

## Not planned

- A CPU rasterizer. lavapipe and WARP do that job, and they run the real shaders.
- The Rust wgpu-native backend.
- Going back to an FFI written in Go: cgo is faster here, and every app that uses gowgpu so far needs cgo anyway.
