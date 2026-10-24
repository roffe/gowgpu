# TODO

What's still open after the 2026-10-23 vet and the perf check on 2026-10-24. The bigger picture is in [ROADMAP.md](ROADMAP.md).

## Perf

- [ ] **fyne widgets 50 is ~7% slower on windowed Vulkan** (0.97 vs 1.05 ms p50 in fynevsbellui, widgets 500 ~2% with a worse p99). It came in somewhere in 668fe0e..ecc2af7, before b27bb72. The SetPipeline check is not it. Offscreen the painter is just as fast as before, so it's in the surface/submit/present path. A 4 kHz CPU profile shows no single hot spot: draw-time validation plus bind groups bound at draw time is ~4 µs a frame, and the Vulkan driver calls (bind pipeline, submit, present, acquire) take 10-40% longer for the same calls. Suspects to revert one at a time on top of ecc2af7 and race against it: ba1678f (WGSL bounds checks and bounded loops), b27370d (semaphores per acquired swapchain), aa350be (bind groups before each draw). Needs an idle box, the difference is the size of the noise from a busy desktop.
- [ ] **~0.5 KB more allocated per frame** on every scene, cause not found yet.

## naga (the fork, then upstream)

- [x] HLSL writes `@builtin(primitive_index)` as `SV_PrimitiveID`, sorted ahead of the other system values as Rust naga does for FXC. Upstream as gogpu/naga#130.
- [x] MSL folds a pipeline's sample mask into the fragment stage (`msl.Options.FixedSampleMask`), and the Metal HAL passes it. Passes on the Mac, upstream as gogpu/naga#131.

## Backends

- [x] Vulkan: clamp `frag_depth` for pipelines whose layout has push constants of its own (HAL users only).
- [x] Track the depth and stencil aspects of a texture apart, as Dawn does.
- [x] DX12: `arrayLength()` on a storage buffer with a dynamic offset counts the binding.
- [x] Timestamp queries on Vulkan and DX12, compute passes included, with `Queue.TimestampPeriod` for the tick length.
- [ ] Timestamp queries on Metal: counter sample buffers, which differ between Apple silicon and Intel/AMD Macs.
- [ ] BC formats on Metal.
- [ ] Browser: the 16-bit normalized formats, needs a `texture-formats-tier1` feature in gputypes.

## Hardware runs

- [x] fmw on Windows (Vulkan, DX12, GL), Linux (Vulkan, GL), macOS and Android, 2026-10-24, all fine.
- [ ] The test suites themselves on a Mac, real DX12 with the debug layer, and Android, as ROADMAP step 1 lists.
