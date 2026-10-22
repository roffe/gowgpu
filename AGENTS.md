# AGENTS.md

Notes for AI coding agents working in or with gowgpu.

## What it is

WebGPU for Go: one API over Vulkan, Metal, DX12 and OpenGL, and over the browser's WebGPU on `js/wasm`. Module `github.com/roffe/gowgpu`, package name `wgpu`, so import it as `wgpu "github.com/roffe/gowgpu"`. It started as a fork of gogpu/wgpu v0.34.5 and is standalone now: don't assume gogpu's docs, APIs or issue numbers apply.

```go
import (
	wgpu "github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/gputypes"         // enums and descriptors
	_ "github.com/roffe/gowgpu/hal/allbackends" // register the platform's backends
)
```

`CreateInstance(nil)` uses Vulkan, Metal and DX12. GL is registered but only used when asked for (`&wgpu.InstanceDescriptor{Backends: gputypes.BackendsGL}`).

## Layout

- root `*.go`: the public API. `*_native.go` goes through `core` and `hal`, `*_browser.go` through `syscall/js`.
- `core/`: validation, state tracking, deferred destruction.
- `hal/`: the backend interfaces, `hal/vulkan`, `hal/metal`, `hal/dx12`, `hal/gles` (with `egl`, `gl`, `wgl`), `hal/noop`, `hal/allbackends`.
- `gputypes/`: WebGPU types, from gogpu/gputypes v0.8.0.
- `internal/dl`: dlopen/dlsym. `cmd/`: generators and backend test programs. `examples/`: compute and headless rendering, and `examples/desktop`, the windowed demos, a module of its own because they need GLFW.

There is no software backend, no `-tags rust` backend, no goffi and no gpucontext any more.

## Rules

- Native calls go through cgo: header-free C trampolines, one per C signature, function pointer first. Pass Go memory as typed pointers, never `uintptr`. Use `#cgo noescape`/`nocallback` only when C neither keeps the pointer nor calls back into Go. Load libraries at run time with `internal/dl`, don't link them. DX12 and WGL use `syscall` instead.
- `hal/vulkan/vk` is generated. Change `cmd/vk-gen` or `cmd/vk-cgo` and regenerate: `(cd cmd/vk-gen && go run . -spec vk.xml -out ../../hal/vulkan/vk/) && go run ./cmd/vk-cgo hal/vulkan/vk`.
- Build every platform a change touches. See [CONTRIBUTING.md](CONTRIBUTING.md) for the cross compile commands and `cmd/cgocheck` for Apple targets.
- Root tests run on the machine's real GPU and skip without one. None of them fail today; keep it that way.
- Commit messages are Conventional Commits: `type(scope): summary`.
- Docs are written in the maintainer's voice: plain and direct, short paragraphs, no em dashes, no marketing words. Only claim what the code does today.
- Don't ask users to star or sponsor anything.

## Build and test

```bash
go build ./...
go test ./...
golangci-lint run
CGO_ENABLED=1 GOOS=windows CC=x86_64-w64-mingw32-gcc go build ./...
GOOS=js GOARCH=wasm go build .
```
