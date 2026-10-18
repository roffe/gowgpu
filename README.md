# gowgpu

WebGPU for Go. One API over Vulkan, Metal, DirectX 12 and OpenGL, and over the browser's own WebGPU when you build for wasm.

It started out as a fork of [gogpu/wgpu](https://github.com/gogpu/wgpu) v0.34.5 and went its own way. Native calls go through cgo instead of goffi, the Rust and software backends are gone, and the only dependencies left are [naga](https://github.com/gogpu/naga), the shader compiler, and `golang.org/x/sys`.

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

The examples are a good next step: [compute-copy](examples/compute-copy), [compute-sum](examples/compute-sum) and [compute-particles](examples/compute-particles) for compute, and [triangle-headless](examples/triangle-headless), which renders to a texture and writes a PNG.

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

fyne does exactly that: it tries Vulkan, Metal and DX12 first, and goes for GL only when none of them can draw the window. On Linux that's desktop GL 3.3, or GLES 3.0 when that's all EGL offers.

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
| Linux | Vulkan | NVIDIA RTX 4080 SUPER, Mesa lavapipe |
| Linux | GL | NVIDIA through EGL, Wayland and X11 |
| Windows | DX12 | a Windows VM, and WARP on GitHub Actions under `-race` |
| Windows | Vulkan | wine (winevulkan) |
| macOS | Metal | GitHub Actions macos-15 |
| iOS | Metal | the simulator |
| Android 12 | Vulkan, GLES | Lenovo Tab M10, Mali-G52 |

Not run yet: the browser backend, the BSDs, GL on Windows and real Apple hardware. If you try one of those, open an issue and tell me how it went.

## Known issues

- On Vulkan, pipelines need an explicit layout (there are no automatic layouts). And `Map` doesn't see a `WriteBuffer` until something is submitted, an empty `Queue.Submit()` is enough. 10 tests in the root package fail because of those two, same as on gogpu v0.34.5.
- GL: reading a texture back swaps red and blue. Presenting is fine.
- DX12 ignores dynamic bind group offsets, and takes the vertex stride from whatever pipeline was set when `SetVertexBuffer` ran.
- naga keeps identifiers that start with `gl_` when it writes GLSL, and strict drivers like Mali refuse them. Don't name things `gl_*` in WGSL.
- `HeadlessSurfaceTarget` and `Surface.ReadPixels` and `WritePixels` are still in the API, but nothing implements them since the software backend went. They'll go.
- `examples/browser-compute` doesn't build, it was written against an older API.
- CI is still gogpu's and fails. It needs a rewrite for cgo.

What's next is in [ROADMAP.md](ROADMAP.md).

## Environment variables

| Variable | What it does |
|---|---|
| `GOGPU_DX12_DXIL=1` | DX12 compiles shaders to DXIL with naga instead of going HLSL through FXC (`d3dcompiler_47.dll`) |
| `GOGPU_DX12_DXIL_VALIDATE=1` | runs the DXC validator on every DXIL blob, so the real error ends up in the log |
| `GOGPU_DX12_FORCE_HWND=1` | skips DirectComposition so RenderDoc can capture, at the cost of premultiplied alpha |

Picking a backend is up to the app. fyne reads `FYNE_WGPU_BACKEND=vulkan|dx12|metal|gl`.

## Changes from gogpu/wgpu

- cgo instead of goffi everywhere: Vulkan, Metal, EGL and GL, X11 and Wayland.
- Gone: the `-tags rust` wgpu-native backend, the software backend and the gpucontext helpers.
- `gputypes` lives in this module now, as `github.com/roffe/gowgpu/gputypes`.
- OpenGL is back in `hal/allbackends` as an opt-in fallback, and it works on Android too.
- DX12 falls back to WARP when there's no GPU.
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
