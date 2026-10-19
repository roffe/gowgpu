# Roadmap

Where gowgpu is and what comes next. No dates, just the order i'd take things in.

## Where it is

gowgpu runs fyne's WebGPU painter on Linux, Windows, macOS, iOS and Android, through Vulkan, DX12, Metal and GL. The [README](README.md) has the table of what has run on what hardware, and the known issues.

## Next

1. **`gl_` names in GLSL.** naga keeps WGSL identifiers that start with `gl_`, which GLSL reserves. Rename them on the way out, either in naga or before the GLSL reaches the driver.
2. **Vulkan image layouts.** The validation layer reports layout errors from the state tracking gogpu wrote, and every queue write starts from `UNDEFINED`, which allows a driver to throw away what's in the texture.
3. **What the vet left open.** GL still needs to zero depth, multisampled and compressed textures. Render bundles crash on Vulkan.
4. **A real iOS device.** Every test passes on Metal on an M2 Pro Mac, but iOS has only run in the simulator.
5. **The browser backend.** Get `examples/browser-compute` building against the current API again and run it in a browser.
6. **The lint backlog**, then a first tagged release.

## Not planned

- A CPU rasterizer. lavapipe and WARP do that job, and they run the real shaders.
- The Rust wgpu-native backend.
- Going back to an FFI written in Go: cgo is faster here, and every app that uses gowgpu so far needs cgo anyway.
