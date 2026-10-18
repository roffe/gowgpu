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

`go test ./...` runs the root package tests on whatever GPU your machine has, and skips them when there's no adapter. On Vulkan 10 of them fail today, same as on gogpu v0.34.5: 7 because Vulkan has no automatic pipeline layouts and 3 because `Map` doesn't see a `WriteBuffer` that was never submitted. Don't add to that list.

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

## Calling into C

All native calls go through small header-free C trampolines, one per C signature, with the function pointer as the first argument. When you add one:

- Pass Go memory as a typed pointer (`*C.uint8_t` and friends), never as `uintptr`. A `uintptr` doesn't keep the memory alive and doesn't follow a stack that moves.
- Use `uintptr_t` on the C side for pointer-sized integers so 32-bit builds stay right.
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
