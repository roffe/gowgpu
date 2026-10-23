# Roadmap

Where gowgpu is and what comes next. No dates, just the order i'd take things in.

## Where it is

gowgpu runs fyne's WebGPU painter on Linux, Windows, macOS, iOS and Android, through Vulkan, DX12, Metal and GL, and its browser backend has run in Firefox and Chrome. The [README](README.md) has the table of what has run on what hardware, and the known issues.

## Next

1. **Run the vet fixes where they haven't run.** Everything from 2026-10-23 has run on Linux (NVIDIA, lavapipe, llvmpipe), under wine and in Firefox and Chrome, and nowhere else.
   - Mac: `go test ./hal/metal/... .` with the new Metal tests, once with `MTL_DEBUG_LAYER=1`, then fyne: drawing, startup time now that each stage compiles its own MSL, frame pacing now that Mailbox keeps vsync, a minimised or covered window with an animation running, live resize, and fmw's zero-copy video. Pipeline constants, the `frag_depth` clamp, vertex pulling and `FeatureIndirectFirstInstance` have only been type checked. Then the iOS simulator and a real iOS device.
   - Windows: the root suite and `./hal/dx12` on real DX12 with the debug layer, through FXC and with `GOGPU_DX12_DXIL=1`, then fyne for resize, Mailbox pacing with three buffers, vsync and tearing. Indirect draws that read `vertex_index` or `instance_index` and stencil copies have only run under vkd3d-proton, GL through WGL only under wine, and windows/arm64 not at all.
   - Android: the tablet on Vulkan and GLES with fyne and fmw, and the root suite built with the NDK. On GLES: readback, sRGB window surfaces, multisampled renderbuffers, ETC2 and ASTC, texture views on Mali, zeroing at creation, and the base vertex and indirect draw paths.
2. **The naga fixes upstream.** The first ones are open as gogpu/naga#118 to #127, the later rounds follow, and the fork goes once they're all in.
3. **What the vet left open.** Timestamp queries on Vulkan, DX12 and Metal, BC formats on Metal, and tracking the aspects of a depth/stencil texture apart. The rest of the known issues are in the [README](README.md).
4. **The lint backlog**, then a first tagged release.

## Not planned

- A CPU rasterizer. lavapipe and WARP do that job, and they run the real shaders.
- The Rust wgpu-native backend.
- Going back to an FFI written in Go: cgo is faster here, and every app that uses gowgpu so far needs cgo anyway.
