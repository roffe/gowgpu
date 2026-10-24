# gowgpu

WebGPU for Go. One API over Vulkan, Metal, DirectX 12 and OpenGL, and over the browser's own WebGPU when you build for wasm.

It started out as a fork of [gogpu/wgpu](https://github.com/gogpu/wgpu) v0.34.5 and went its own way. Native calls go through cgo instead of goffi, the Rust and software backends are gone, and the only dependencies left are [naga](https://github.com/roffe/naga), the shader compiler (my fork of gogpu/naga), and `golang.org/x/sys`.

It's what draws everything in the WebGPU painter of [fyne](https://github.com/roffe/fyne/tree/wgpu), so that's where it gets its mileage.

[![Go Reference](https://pkg.go.dev/badge/github.com/roffe/gowgpu.svg)](https://pkg.go.dev/github.com/roffe/gowgpu)

## Getting it

```bash
go get github.com/roffe/gowgpu@main
```

You need Go 1.25 or newer and cgo, more on that under [Building](#building). There are no tagged releases yet, so you get a pseudo-version.

The package is called `wgpu`:

```go
import wgpu "github.com/roffe/gowgpu"
```

## Quick start

```go
package main

import (
	"fmt"
	"log"

	wgpu "github.com/roffe/gowgpu"
	_ "github.com/roffe/gowgpu/hal/allbackends" // the backends this platform has
)

func main() {
	instance, err := wgpu.CreateInstance(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer instance.Release()

	adapter, err := instance.RequestAdapter(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer adapter.Release()
	info := adapter.Info()
	fmt.Printf("%s on %s\n", info.Backend, info.Name)

	device, err := adapter.RequestDevice(nil)
	if err != nil {
		log.Fatal(err)
	}
	defer device.Release()

	buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{
		Label: "data",
		Size:  1024,
		Usage: wgpu.BufferUsageStorage | wgpu.BufferUsageCopyDst,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer buf.Release()

	if err := device.Queue().WriteBuffer(buf, 0, []byte{1, 2, 3, 4}); err != nil {
		log.Fatal(err)
	}
}
```

`hal/allbackends` registers what the platform has, `CreateInstance(nil)` picks between Vulkan, Metal and DX12, and `RequestAdapter` hands you the best GPU it found. On a machine without any of them you get `no adapters available`.

The examples are a good next step: [compute-copy](examples/compute-copy), [compute-sum](examples/compute-sum) and [compute-particles](examples/compute-particles) for compute, and [triangle-headless](examples/triangle-headless), which renders to a texture and writes a PNG. The ones that open a window live in [examples/desktop](examples/desktop), a module of its own because they need GLFW: [particles](examples/desktop/particles) runs a compute shader and a render pipeline on 200,000 particles every frame, and [fluid](examples/desktop/fluid) is water, 16,384 SPH particles binned into a grid with atomics, that the mouse pulls and pushes, and [raytracing](examples/desktop/raytracing) path traces the Ray Tracing in One Weekend scene in a compute shader, a sample a pixel every frame, while you orbit it with the mouse. [reaction](examples/desktop/reaction) is Gray-Scott reaction-diffusion stepped between two storage textures, which you paint into, and [fractal](examples/desktop/fractal) ray marches the Mandelbulb in a fragment shader, with an endless ride into the Mandelbrot set on M that keeps picking new spots to head for, perturbation against a reference orbit worked out in `math/big`. `cd examples/desktop && go run ./fluid`.

## Backends

| Platform | Backends | Build needs |
|---|---|---|
| Linux | Vulkan, OpenGL (EGL) | cgo |
| FreeBSD, OpenBSD, NetBSD | Vulkan | cgo |
| Windows | Vulkan, DX12, OpenGL (WGL) | cgo for Vulkan |
| macOS | Metal, Vulkan (with MoltenVK installed) | cgo, Xcode |
| iOS | Metal | cgo, Xcode |
| Android (arm64, amd64) | Vulkan, OpenGL ES 3.0 | cgo, NDK |
| Browser (`js/wasm`) | WebGPU | nothing |

OpenGL is registered but not part of the default instance. Ask for it when nothing else works:

```go
instance, err := wgpu.CreateInstance(&wgpu.InstanceDescriptor{Backends: gputypes.BackendsGL})
```

fyne does exactly that: it tries Vulkan, Metal and DX12 first, and goes for GL only when none of them can draw the window. On Linux that's the newest desktop GL the driver makes, from 4.6 down to 3.3, or GLES 3.0 when that's all EGL offers. Compute needs GL 4.3 or GLES 3.1.

No GPU at all? DX12 picks WARP, Microsoft's CPU renderer, when there's no hardware adapter, so Windows VMs and CI runners just work. On Linux install Mesa's lavapipe (`vulkan-swrast` on Arch, `mesa-vulkan-drivers` on Debian and Ubuntu) and it shows up as a CPU Vulkan adapter.

## Building

Every backend except DX12 talks to the driver through cgo. The libraries themselves (libvulkan, libEGL, X11, Wayland) are opened at run time, so building needs no SDK or headers for them, and the binary still starts on a machine that lacks one. That backend just isn't there.

- **Linux:** any gcc or clang.
- **Windows:** only Vulkan needs cgo. From Linux: `CGO_ENABLED=1 GOOS=windows CC=x86_64-w64-mingw32-gcc go build`. DX12 and GL go through `syscall`, so an app that imports only `hal/dx12` builds with `CGO_ENABLED=0`.
- **macOS and iOS:** Xcode. Metal and QuartzCore get linked, like in any cgo app there.
- **Android:** the NDK's clang as `CC`, for example `$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang`. See [docs/ANDROID.md](docs/ANDROID.md).
- **Browser:** `GOOS=js GOARCH=wasm go build`, no cgo.

Building `hal/allbackends` with cgo off stops with `hal/vulkan needs cgo`.

## Why cgo and not goffi

gogpu calls the drivers through goffi, an FFI written in Go. That's handy for cross compiling, but it boxes every argument: a Vulkan call cost 85 to 128 ns and 3 to 8 heap allocations, an Objective-C message to Metal about 210 ns and 9 allocations. A GPU API makes a lot of those every frame.

gowgpu calls through tiny C trampolines instead, one per C signature, with the function pointer as the first argument and `#cgo noescape` and `nocallback` where that's safe. That's about 20 ns and no allocations per call. In fyne's widget benchmark (NVIDIA on Wayland) a frame went from 1.43 to 1.25 ms and from 311 to 193 KB allocated.

fyne needs cgo anyway for GLFW and its mobile drivers, so there it costs nothing extra.

## What has run where

| Platform | Backend | On |
|---|---|---|
| Linux | Vulkan | NVIDIA RTX 4080 SUPER, Mesa lavapipe, and a linux/386 build on NVIDIA's 32-bit driver |
| Linux | GL | NVIDIA through EGL (Wayland, X11 and headless), Mesa llvmpipe |
| Windows | DX12 | a Windows VM, WARP on GitHub Actions under `-race`, and wine with its own vkd3d and with GE-Proton's vkd3d-proton |
| Windows | Vulkan | wine (winevulkan), amd64 and 386 |
| Windows | GL | wine (WGL on NVIDIA) |
| macOS | Metal | Apple M2 Pro, GitHub Actions macos-15 |
| iOS | Metal | the simulator |
| Android 12 | Vulkan, GLES | Lenovo Tab M10, Mali-G52 |
| Browser | WebGPU | Firefox 158 and Chrome 156 on Linux |

Not run yet: the BSDs, GL on Windows itself, a real iOS device and 32-bit Android. If you try one of those, open an issue and tell me how it went.

The fixes from the vet on 2026-10-23 (see the [CHANGELOG](CHANGELOG.md)) have only run on Linux, under wine and in the two browsers. Metal, Android and Windows itself haven't run them yet, the [roadmap](ROADMAP.md) has what to check there.

## Known issues

- GL without texture views (desktop GL below 4.3 without `GL_ARB_texture_view`, GLES without `OES_` or `EXT_texture_view`): a bind group refuses a view of some of a texture's layers, or of another dimension, to sample or to store to more than one layer, and a dispatch that samples one mip level of a texture and stores to another stores nothing.
- GL has no views in another format, so it offers neither `DownlevelFlagsViewFormats` nor surface view formats, and it reads no compressed texture back: `CopyTextureToBuffer` of one logs an error and copies nothing, as in wgpu-hal. GLES reads no depth or stencil back either, and refuses a write into the stencil of a combined depth/stencil texture.
- Timestamp queries only work on desktop GL. Vulkan, DX12 and Metal don't offer `FeatureTimestampQuery` yet.
- Metal has no compressed texture formats yet, so it doesn't offer `TextureCompressionBC`.
- Metal applies a pipeline's sample mask in its fragment stage, so a pipeline without one ignores the mask. Dawn adds a fragment stage for it.
- In the browser the 16-bit normalized formats can't be used, as gputypes has no `texture-formats-tier1` feature to ask for.

What's next is in [ROADMAP.md](ROADMAP.md).

## Environment variables

| Variable | What it does |
|---|---|
| `GOGPU_DX12_DXIL=1` | DX12 compiles shaders to DXIL with naga instead of going HLSL through FXC (`d3dcompiler_47.dll`). An adapter without shader model 6.0 stays on FXC, with a warning |
| `GOGPU_DX12_DXIL_VALIDATE=1` | runs the DXC validator on every DXIL blob, so the real error ends up in the log, and writes a rejected blob to the temp directory |
| `GOGPU_DX12_FORCE_HWND=1` | skips DirectComposition so RenderDoc can capture, at the cost of premultiplied alpha |
| `DISABLE_RTSS_LAYER` | gowgpu sets it to 1 on Windows before creating a Vulkan instance, unless it's already set. It switches off RivaTuner's Vulkan layer, which crashed `vkCreateDevice` |

Picking a backend is up to the app. fyne reads `FYNE_WGPU_BACKEND=vulkan|dx12|metal|gl`.

## Changes from gogpu/wgpu

- cgo instead of goffi everywhere: Vulkan, Metal, EGL and GL, X11 and Wayland.
- Gone: the `-tags rust` wgpu-native backend, the software backend and the gpucontext helpers.
- `gputypes` lives in this module now, as `github.com/roffe/gowgpu/gputypes`.
- OpenGL is back in `hal/allbackends` as an opt-in fallback, and it works on Android too.
- DX12 falls back to WARP when there's no GPU.
- naga is my fork, [github.com/roffe/naga](https://github.com/roffe/naga): gogpu/naga v0.20.3 with shader translator fixes, which are on their way upstream (gogpu/naga#118 to #126 so far, more to come).
- GPU handles are `uint64`, so 32-bit builds keep Vulkan's 64-bit handles whole, and the Vulkan bindings are laid out as C lays them out on 386 and arm.
- A correctness vet against Rust's wgpu, Dawn and the API specs confirmed 239 defects in core and the backends, and the fixes are in: render bundles work on every backend, textures are tracked per mip level and layer, native error scopes capture errors, much more of WebGPU's validation runs before a call reaches the driver, and a lot of smaller things per backend. What's left of it is under known issues.
- Fixes along the way: GL presenting into the window and uploading textures at an offset, the EGL config for X11 windows on NVIDIA, Vulkan surface formats after an Android app comes back to the foreground, Metal triangle strips, depth bias and iOS display sync, draws with a non-zero first instance, and DX12 under `-race`.

All of it is in the [CHANGELOG](CHANGELOG.md).

## Docs

- [Architecture](docs/ARCHITECTURE.md)
- [Surface targets](docs/SURFACE-TARGETS.md)
- [Android](docs/ANDROID.md)
- [Compute shaders](docs/COMPUTE-SHADERS.md) and the [backend differences](docs/COMPUTE-BACKENDS.md)
- [Ray tracing](docs/RAY-TRACING.md), experimental
- [Validation tests](docs/VALIDATION-TESTING.md)
- [Contributing](CONTRIBUTING.md)

## License

MIT, see [LICENSE](LICENSE). gowgpu is built on [gogpu/wgpu](https://github.com/gogpu/wgpu) and [gogpu/gputypes](https://github.com/gogpu/gputypes) by Andrey Kolkov and the GoGPU contributors. Big thanks to them, none of this would exist without their work.
