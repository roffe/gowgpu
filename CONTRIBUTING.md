# Contributing

Bug reports, fixes and test runs on hardware i don't have are all welcome. The last one especially: see the "what has run where" table in the [README](README.md).

## Setup

You need Go 1.25 or newer, a C compiler (cgo) and preferably a GPU.

```bash
git clone https://github.com/roffe/gowgpu
cd gowgpu
go build ./...
go test ./...
```

`go test ./...` runs the root package tests on whatever GPU your machine has, and skips them when there's no adapter. None of them fail today, keep it that way.

No GPU? Install Mesa's lavapipe and Vulkan runs on the CPU. You don't even need root for it, see [Vulkan without a GPU or root](#vulkan-without-a-gpu-or-root).

## Before you send a PR

```bash
gofmt -l .
go vet ./...
go build ./...
go test ./...
golangci-lint run
```

golangci-lint has a backlog of old findings, mostly from gogpu. Don't add new ones.

Build for the platforms your change touches. Most of them cross compile from Linux:

```bash
# Windows, Vulkan needs cgo
CGO_ENABLED=1 GOOS=windows CC=x86_64-w64-mingw32-gcc go build ./...

# Android, with the NDK's clang
CGO_ENABLED=1 GOOS=android GOARCH=arm64 \
  CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang go build ./...

# browser
GOOS=js GOARCH=wasm go build .
```

macOS and iOS need Apple's SDK to link. Without a Mac, `cmd/cgocheck` still type checks a cgo package for those targets, with a faked `C` package, and writes out the C side so clang can compile it on its own:

```bash
go run ./cmd/cgocheck -goos ios -goarch arm64 -out /tmp/pre ./hal/metal
```

It checks one package at a time, because the packages it imports still go through the real cgo.

In the PR, say what you ran it on: OS, GPU, driver and backend.

## The Vulkan bindings

`hal/vulkan/vk` is generated. `cmd/vk-gen` writes Go bindings from `vk.xml`, then `cmd/vk-cgo` moves the wrappers onto the C trampolines in `trampolines.h`. Don't edit the generated files by hand, change the generator and run:

```bash
(cd cmd/vk-gen && go run . -spec vk.xml -out ../../hal/vulkan/vk/) && go run ./cmd/vk-cgo hal/vulkan/vk
```

It reproduces the checked in files byte for byte, so `git diff` after it shows exactly what your change did.

Every command in `vk.xml` has a wrapper. To call one gowgpu doesn't use yet, load its pointer in `commands.go` (`LoadInstance` or `LoadDevice`) and call the method; there's no wrapper to write by hand.

## Calling into C

All native calls go through small header-free C trampolines, one per C signature, with the function pointer as the first argument. When you add one:

- Pass Go memory as a typed pointer (`*C.uint8_t` and friends), never as `uintptr`. A `uintptr` doesn't keep the memory alive and doesn't follow a stack that moves.
- Go memory that a C argument points at and that holds pointers itself, a create info with an array behind it, needs what those pointers point at pinned for the call (`vk.Pin`, or a `runtime.Pinner` of your own), or kept in a local the compiler leaves on the stack. cgocheck checks it, it's on by default, and the call panics with `argument of cgo function has Go pointer to unpinned Go pointer` otherwise. The generated Vulkan wrappers hand such memory to C as `unsafe.Slice(p, n)`, so cgocheck sees all of it.
- Use `uintptr_t` on the C side for pointer-sized integers and Vulkan's dispatchable handles, and `uint64_t` for the non-dispatchable ones, which are 64 bits on 32-bit targets too.
- Mark it `#cgo noescape` and `#cgo nocallback` only when C really neither keeps the pointer nor calls back into Go. Anything that dispatches events (Wayland roundtrips, debug messengers) must not be `nocallback`.
- Open libraries with `internal/dl` at run time. Don't link against them: a missing library should mean a missing backend, not a binary that won't start.

## Vulkan without a GPU or root

Arch packages can be unpacked anywhere. `pacman -Sp vulkan-swrast vulkan-validation-layers` prints the package URLs, download and `tar -xf` them into a directory, and point the JSON manifests' `library_path` at the unpacked `.so` files. Then:

```bash
# lavapipe, Mesa's CPU Vulkan driver
VK_ICD_FILENAMES=<dir>/usr/share/vulkan/icd.d/lvp_icd.json go test ./...

# the Khronos validation layer
VK_LAYER_PATH=<dir>/usr/share/vulkan/explicit_layer.d \
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation go test ./...
```

To check what happens on a machine with no Vulkan at all, point `VK_ICD_FILENAMES` at a JSON whose `library_path` doesn't exist.

### The validation layer

Synchronization validation and best practices go on through the layer's environment:

```bash
VK_LAYER_PATH=<dir>/usr/share/vulkan/explicit_layer.d \
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
VK_LAYER_VALIDATE_SYNC=1 VK_LAYER_VALIDATE_BEST_PRACTICES=1 \
go test ./...
```

The root, core and Vulkan suites run without a validation error under it. With best practices on, a few validated tests fail anyway: their device counts every message the layer sends, best-practices warnings included, and one of those quotes a VUID too. So judge such a run by grepping the output for `Validation Error`, `VUID` and `SYNC-HAZARD` and reading what those lines say, not by the tests that fail.

## Testing other backends

The GPU tests in package `wgpu`, most of the root package's, run on the backend `GOWGPU_TEST_BACKEND` names, `gl`, `dx12` or `vulkan`. Without it they run on the first adapter found, Vulkan, Metal or DX12 where there's one, and the tests in package `wgpu_test` always take the default instance.

```bash
# GL through EGL, NVIDIA's where it's installed, without a window system
DISPLAY= WAYLAND_DISPLAY= GOWGPU_TEST_BACKEND=gl go test ./...

# GL on Mesa's llvmpipe
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
DISPLAY= WAYLAND_DISPLAY= GOWGPU_TEST_BACKEND=gl go test ./...
```

`-tags integration` adds the `hal/gles` tests that need a real EGL context.

### DX12 under wine and Proton

DX12 runs under wine well enough for a first check. Build the package's test binary for Windows and run it there:

```bash
CGO_ENABLED=1 GOOS=windows CC=x86_64-w64-mingw32-gcc go test -c -o dx12.exe .
WINEPREFIX=$HOME/.wine-gowgpu GOWGPU_TEST_BACKEND=dx12 wine dx12.exe
```

The prefix needs Microsoft's `d3dcompiler_47.dll` in its system32 (`winetricks d3dcompiler_47` gets it): Wine's own compiles HLSL with vkd3d-shader, which lacks the shader model 5.1 resource arrays naga's HLSL uses. An exit status of 137 after `PASS` is wine tearing down, not a failure.

Wine's own `d3d12.dll` is vkd3d, which leaves out ExecuteIndirect with root constant arguments (indirect draws that read `vertex_index` or `instance_index`), copies and barriers of a stencil plane and `BeginRenderPass`. The tests that need those skip under it, and only under it: they look for the "Wine builtin DLL" mark of the `d3d12.dll` that got loaded. It also refuses naga's DXIL with scalar allocas (E8012), so the root suite with `GOGPU_DX12_DXIL=1` needs vkd3d-proton.

vkd3d-proton has all of that. GE-Proton ships it, usually under `~/.local/share/Steam/compatibilitytools.d/`. Let Proton's script make the prefix once: `STEAM_COMPAT_DATA_PATH=<dir> STEAM_COMPAT_CLIENT_INSTALL_PATH=$HOME/.local/share/Steam "$P/proton" run` with any program puts vkd3d-proton's `d3d12.dll` and `d3d12core.dll` and DXVK's `dxgi.dll` into `<dir>/pfx`, and Microsoft's `d3dcompiler_47.dll` goes in next to them. `proton run` drops a program's output and hangs without a display, so the tests run under Proton's own wine:

```bash
P=$HOME/.local/share/Steam/compatibilitytools.d/GE-Proton11-7-x86_64
WINEPREFIX=<dir>/pfx \
WINEDLLPATH=$P/files/lib/vkd3d:$P/files/lib/wine \
LD_LIBRARY_PATH=$P/files/lib/x86_64-linux-gnu:$P/files/lib/i386-linux-gnu \
WINEDLLOVERRIDES='d3d12=n;d3d12core=n;dxgi=n;d3dcompiler_47=n' \
WINEESYNC=1 WINEFSYNC=1 GOWGPU_TEST_BACKEND=dx12 \
"$P/files/bin/wine" dx12.exe
"$P/files/bin/wineserver" -k
```

The last line stops the prefix's wineserver, or its `winedevice.exe` services outlive the run. vkd3d-proton's `d3d12.dll` and `d3d12core.dll` from elsewhere (Lutris ships builds of it) in a plain wine prefix's system32, with the same overrides but `dxgi`, work as well. `VKD3D_DEBUG=info` shows which build got loaded. Under vkd3d-proton the root suite and `./hal/dx12` run with none of the vkd3d skips, through FXC and through naga's DXIL.

It's still not Windows: the debug layer, the real runtime's DXIL signing and windows/arm64 need a Windows box.

## 32-bit builds

linux/386, windows/386, android/arm and android/386 build:

```bash
# linux/386, with a 32-bit C toolchain (gcc-multilib on Debian and Ubuntu, lib32-glibc and lib32-gcc-libs on Arch)
CGO_ENABLED=1 GOARCH=386 go build ./...

# windows/386
CGO_ENABLED=1 GOOS=windows GOARCH=386 CC=i686-w64-mingw32-gcc go build ./...

# android/arm, and i686-linux-android31-clang for android/386
CGO_ENABLED=1 GOOS=android GOARCH=arm \
  CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi31-clang go build ./...
```

A linux/386 test binary runs on Vulkan with the 32-bit loader and driver installed (`lib32-vulkan-icd-loader` and `lib32-nvidia-utils` on Arch), and a windows/386 one under wine. The root suite passes both ways. 32-bit Android hasn't run: `hal/allbackends` registers nothing there. DX12 refuses a 32-bit build at `CreateInstance`.

`TestStructLayoutsMatchC` in `hal/vulkan/vk` checks the size and every field offset of the generated structs against `vulkan_core.h` through clang, for amd64, arm64, 386, arm and windows/386. It needs clang and the Vulkan headers, and skips without them. Run it after changing `cmd/vk-gen`.

## Commit messages

[Conventional Commits](https://www.conventionalcommits.org/): `type(scope): summary`, for example

```
fix(gles): write textures at their origin and row pitch
feat(dx12): fall back to WARP when no hardware adapter works
refactor!: drop the software backend
```

Scopes are the package or backend: `core`, `hal`, `vulkan`, `metal`, `dx12`, `gles`, `browser`, `gputypes`, `allbackends`, `cmd`, `docs`, `ci`. Use the body for why, not what.

## AI tools

Use whatever tools you like. You own the result: know what the code does, test it on real hardware where it matters, and be ready to change it after review.

## Reporting bugs

Open an [issue](https://github.com/roffe/gowgpu/issues) with the Go version, OS, GPU and driver, the backend (`adapter.Info()` tells you), and the smallest program that shows the problem. Logs help: the HAL logs through `log/slog`.
