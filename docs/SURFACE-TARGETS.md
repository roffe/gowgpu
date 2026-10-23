# Surface targets

A surface is made from a typed target. The target says what kind of handle it carries (HWND, Xlib window, Wayland surface, `ANativeWindow`, `CAMetalLayer` or a canvas), so a backend never has to guess Xlib from Wayland out of two integers, and it refuses a kind it can't use before it touches any handle.

The safe/unsafe split follows Rust wgpu v29.0.3, pinned at [`4cbe6232`](https://github.com/gfx-rs/wgpu/tree/4cbe6232b2d7c289b6e1a38416a6ae1461a22e81). Rust has an enum of raw-window-handle values and `unsafe` methods. Go has neither, so here it's an opaque value with named constructors, checked at the API boundary.

## Making a surface

| Method | Takes | Ownership |
|---|---|---|
| `CreateSurfaceUnsafe(SurfaceTargetUnsafe)` | raw handles from a `SurfaceTargetFrom*` constructor | nothing is retained, the handles have to stay valid until `Surface.Release` returns |
| `CreateSurfaceFromTarget(SurfaceTarget)` | a provider, any value with `SurfaceTarget() (SurfaceTargetUnsafe, error)` | the provider is called once and kept until the surface is released |
| `CreateSurface(displayHandle, windowHandle)` | the old two `uintptr`s | nothing is retained, mapped onto a typed target by GOOS, see [The old CreateSurface](#the-old-createsurface) |
| `CreateSurfaceFromCanvas(js.Value)` | an `HTMLCanvasElement` or `OffscreenCanvas` | browser only |

### Constructors

| Constructor | Handles | Taken by |
|---|---|---|
| `SurfaceTargetFromWindowsHWND(hinstance, hwnd)` | `HWND` required, `HINSTANCE` may be 0 | Vulkan, DX12, GL (WGL) |
| `SurfaceTargetFromXlibWindow(display, window)` | `Display*` and `Window`, both required | Vulkan on Linux and the BSDs, GL on Linux |
| `SurfaceTargetFromWaylandSurface(display, surface)` | `wl_display*` and `wl_surface*`, both required | Vulkan on Linux and the BSDs, GL on Linux |
| `SurfaceTargetFromAndroidNativeWindow(window)` | `ANativeWindow*`, there's no display | Vulkan and GLES on Android |
| `SurfaceTargetFromMetalLayer(layer)` | `CAMetalLayer*` | Metal, and Vulkan through MoltenVK on macOS |
| `SurfaceTargetFromWebCanvasID(id)` | the canvas's `data-raw-handle`, 0 picks the first canvas | the browser only |

There's no GL on the BSDs, so there only Vulkan takes Xlib and Wayland targets. The BSDs have never been built or run.

### Provider

```go
type AndroidWindow struct {
	nativeWindow uintptr
	// whatever owns or retains the native window goes here
}

func (w *AndroidWindow) SurfaceTarget() (wgpu.SurfaceTargetUnsafe, error) {
	return wgpu.SurfaceTargetFromAndroidNativeWindow(w.nativeWindow), nil
}

window := &AndroidWindow{nativeWindow: nativeWindow}
surface, err := instance.CreateSurfaceFromTarget(window)
```

`CreateSurfaceFromTarget` calls `SurfaceTarget()` exactly once. On success it keeps the provider until `Surface.Release` has destroyed the backend surfaces, so the provider's Go objects stay reachable that long. The native objects behind it still have to stay valid for that whole time. A finalizer is no substitute for that.

### Raw handles

```go
surface, err := instance.CreateSurfaceUnsafe(
	wgpu.SurfaceTargetFromAndroidNativeWindow(nativeWindow),
)
```

Nothing is retained. Every display, window, surface or layer you pass has to stay valid until `Surface.Release` returns. A zero handle where one is required is refused before any backend sees it.

Go can't mark a method unsafe, so the name, the opaque constructors and the validation carry the contract.

## Errors

- `ErrReleased`: the instance was already released. This is checked first, so a provider or target is never looked at after teardown.
- `ErrInvalidSurfaceTarget`: an empty target, a nil or typed-nil provider, an unknown kind, or a required handle that is zero. Checked before any backend call.
- `ErrUnsupportedSurfaceTarget`: no backend in the instance takes the kind, like a canvas ID on native or a native target in the browser. Each backend's own error is joined inside.
- When every backend failed and at least one of them did take the kind, you get `wgpu: failed to create surface for every enabled backend`, with every backend's error joined.
- A provider's error is wrapped with `%w`, so `errors.Is` and `errors.As` still find it.

## One surface per backend

On native, `CreateSurface*` tries every backend in the instance, in priority order, and keeps one surface for each backend that takes the target. It succeeds when at least one does, like Rust wgpu's `surface_per_backend`.

- `RequestAdapter` with `CompatibleSurface` checks each adapter against the surface of its own backend. A backend without a surface can't be picked. Vulkan asks `vkGetPhysicalDeviceSurfaceSupportKHR`, GL makes its adapter from the surface's context.
- `Adapter.GetSurfaceCapabilities` uses the surface of the adapter's backend, and returns nil when there isn't one. It never falls back to another backend's surface.
- `Surface.Configure` makes the surface of the device's backend the active one, and creates it then if the first try failed. The others stay around unused until release.
- `Surface.Release` destroys all of them, then lets go of the provider. Calling it twice is fine.
- `Instance.Release` releases its devices and then its surfaces. The browser's doesn't, release browser surfaces yourself.

## The old CreateSurface

`Instance.CreateSurface(displayHandle, windowHandle)` still works. It goes down the same typed path and retains nothing. On native the target comes from `runtime.GOOS`:

| GOOS | Target |
|---|---|
| windows | `SurfaceTargetFromWindowsHWND(displayHandle, windowHandle)`, so pass 0 or the HINSTANCE |
| darwin | `SurfaceTargetFromMetalLayer(windowHandle)` |
| linux | Wayland when `WAYLAND_DISPLAY` is set, Xlib otherwise |
| android | `SurfaceTargetFromAndroidNativeWindow(windowHandle)`, `displayHandle` is ignored |
| anything else, like the BSDs and ios | no mapping, you get `ErrInvalidSurfaceTarget` |

So on the BSDs and iOS use a typed target. On Linux a typed target also gets rid of the `WAYLAND_DISPLAY` guess: an XWayland window in a Wayland session can go in as an Xlib target.

In the browser `windowHandle` is the canvas ID and `displayHandle` is ignored. If you already have the canvas as a `js.Value`, `CreateSurfaceFromCanvas` is the better call.

## Without a window

There's no headless surface. Render into a texture and `CopyTextureToBuffer` it, the normal WebGPU way. `examples/triangle-headless` does that and writes a PNG.

## API mapping

Every exported symbol of the typed surface path, with its Rust wgpu v29.0.3 counterpart where there is one. The Rust sources: [`Instance::create_surface`](https://github.com/gfx-rs/wgpu/blob/4cbe6232b2d7c289b6e1a38416a6ae1461a22e81/wgpu/src/api/instance.rs#L175-L280), [`SurfaceTarget` and `SurfaceTargetUnsafe`](https://github.com/gfx-rs/wgpu/blob/4cbe6232b2d7c289b6e1a38416a6ae1461a22e81/wgpu/src/api/surface.rs#L254-L458), [core surface creation](https://github.com/gfx-rs/wgpu/blob/4cbe6232b2d7c289b6e1a38416a6ae1461a22e81/wgpu-core/src/instance.rs#L218-L290) and [`wgpu_hal::Instance::create_surface`](https://github.com/gfx-rs/wgpu/blob/4cbe6232b2d7c289b6e1a38416a6ae1461a22e81/wgpu-hal/src/lib.rs#L648-L662).

### Root package

| Go symbol | Rust | Contract |
|---|---|---|
| `ErrInvalidSurfaceTarget` | `CreateSurfaceErrorKind::RawHandle` | malformed input, caught before any backend call |
| `ErrUnsupportedSurfaceTarget` | `CreateSurfaceError::FailedToCreateSurfaceForAnyBackend` | no backend takes the kind, their errors wrapped inside |
| `SurfaceTarget` | safe `SurfaceTarget<'window>` | called once, kept until the surface is released |
| `SurfaceTargetUnsafe` | `SurfaceTargetUnsafe::RawHandle` | opaque, so kind and handles can't be mixed up; retains nothing |
| `SurfaceTargetFrom*` | the `RawWindowHandle` variants | see [Constructors](#constructors) |
| `(*Instance).CreateSurfaceFromTarget` | `Instance::create_surface` | checks the instance, calls the provider once, keeps it on success |
| `(*Instance).CreateSurfaceUnsafe` | `unsafe Instance::create_surface_unsafe` | validates, retains nothing |
| `(*Instance).CreateSurface` | none | the GOOS mapping above, retains nothing |
| `(*Instance).RequestAdapter` with `CompatibleSurface` | `RequestAdapterOptions::compatible_surface` | each adapter checked against its own backend's surface |
| `(*Adapter).GetSurfaceCapabilities` | `Surface::get_capabilities` | the same backend's surface only, nil without one |
| `(*Surface).Configure` | core's `surface_per_backend` lookup | activates the surface of the device's backend |
| `(*Surface).Release` | `Surface` drop | destroys every backend surface, then drops the provider; idempotent |
| `(*Instance).Release` | none, Rust surfaces have their own lifetime | native: devices, then surfaces; browser: release surfaces yourself |

### hal

| Go symbol | Rust | Contract |
|---|---|---|
| `hal.ErrUnsupportedSurfaceTarget` | `wgpu_hal::InstanceError` for a foreign handle | returned before a handle is read |
| `hal.SurfaceTargetKind` | the `RawWindowHandle` discriminant | `SurfaceTargetWindowsHWND`, `SurfaceTargetXlibWindow`, `SurfaceTargetWaylandSurface`, `SurfaceTargetAndroidNativeWindow`, `SurfaceTargetMetalLayer`. The root package never sends `SurfaceTargetInvalid` |
| `hal.SurfaceTarget` (`Kind`, `DisplayHandle`, `WindowHandle`) | the raw display and window handle pair | borrowed, hal never owns the handles. `DisplayHandle` is unused for Android and Metal |
| `hal.SurfaceTarget.RequireKind` | a `match` arm | wraps `hal.ErrUnsupportedSurfaceTarget`, touches no handle |
| `hal.SurfaceTargetKind.String` | `Debug` | stable names for logs, unknown values print as a number |
| `hal.Instance.CreateSurface(hal.SurfaceTarget)` | `wgpu_hal::Instance::create_surface` | one typed target instead of two integers |

### What each backend takes

| `CreateSurface` in | Takes | Notes |
|---|---|---|
| `hal/vulkan`, Linux and the BSDs | Xlib, Wayland | the instance enables whichever of `VK_KHR_xlib_surface` and `VK_KHR_wayland_surface` the loader has, both at once is fine |
| `hal/vulkan`, Windows | Windows HWND | a zero HINSTANCE becomes `GetModuleHandleW(NULL)` |
| `hal/vulkan`, macOS | Metal layer | through `VK_EXT_metal_surface` |
| `hal/vulkan`, Android | Android native window | display ignored. gowgpu never acquires or releases the window, see [ANDROID.md](ANDROID.md) |
| `hal/metal` | Metal layer | refuses a zero layer, retains it, and releases it when the surface is destroyed |
| `hal/dx12` | Windows HWND | keeps the HWND, the swapchain gets made in `Configure` |
| `hal/gles`, Windows | Windows HWND | gives the window's DC the hidden window's pixel format when it has none, so the context can present to it, and refuses a window that has another one |
| `hal/gles`, Linux and Android | Xlib, Wayland, Android native window | picks the EGL display for that kind explicitly, no guessing from the environment |
| `hal/noop` | anything | no native access, for tests |

### core

| Go symbol | Rust | Contract |
|---|---|---|
| `core.HALInstanceEntry` (`Backend`, `Instance`) | one entry of core's `instance_per_backend` | core owns the instance |
| `core.(*Instance).HALInstanceEntries` | iterating `instance_per_backend` | a copy, in backend priority order. Ownership stays with core |
| `core.(*Instance).RequestAdapterWithSurfaces` | adapter selection against `surface_per_backend` | each adapter only sees its own backend's surface. An empty map is a plain `RequestAdapter` |
| `core.(*Instance).RequestAdapterWithSurface` | none | one surface for every backend, for callers that drive a single hal |

### EGL

| Go symbol | Contract |
|---|---|
| `egl.ContextConfig.WindowKind` | a `*WindowKind`. nil detects the window system from the environment, set forces X11, Wayland, Android or surfaceless |
| `egl.DefaultContextConfig` | desktop GL 3.3 core, `WindowKind` nil. Where EGL takes a version (EGL 1.5 or `EGL_KHR_create_context`), `NewContext` asks for 4.6 down to 4.3 before it, robust where it can, and it refuses a desktop context where EGL takes none |
| `egl.NewContext` | uses an explicit `WindowKind` without looking at the environment, otherwise detects. Owns any display connection it opened itself |
| `egl.GetEGLDisplay` | the old auto-detecting entry point |
| `egl.DetectWindowKind` | Android on Android. Otherwise surfaceless when neither `DISPLAY` nor `WAYLAND_DISPLAY` is set and EGL can do it, then Wayland, then X11, and surfaceless when neither works |
