# Android

gowgpu runs on Android arm64 and amd64. You get Vulkan by default, and GLES 3.0 through EGL as a fallback when you ask for it. Both go through cgo, so you build with the NDK's clang.

## Backends

`hal/allbackends` registers both on android/arm64 and android/amd64, see `register_android.go`. On 32-bit Android (arm, 386) it registers nothing. `hal/vulkan` and `hal/gles` build there, but neither has run.

Vulkan is what `wgpu.CreateInstance(nil)` gives you. `libvulkan.so` gets opened with dlopen when the instance is created, nothing links against it. On a device without Vulkan the app still starts, the backend just isn't there.

GLES is registered but not part of the default instance. Ask for it:

```go
instance, err := wgpu.CreateInstance(&wgpu.InstanceDescriptor{Backends: gputypes.BackendsGL})
```

The GLES context is a GLES 3.0 context on EGL's default display, and it lives on the Instance, not on a surface. Android hands the app a new `ANativeWindow` every time it comes back to the foreground. Since the context isn't tied to a window, everything the device made survives that. A surface only makes an EGL window surface on its `ANativeWindow`, when you configure it.

## Building

```bash
CGO_ENABLED=1 GOOS=android GOARCH=arm64 \
  CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android31-clang \
  go build ./...
```

`$NDK` is where your NDK is, i use r27c. For amd64 it's `GOARCH=amd64` and `x86_64-linux-android31-clang`. The 31 is API 31, Android 12. With cgo off it doesn't build.

A fyne app:

```bash
ANDROID_NDK_HOME=$NDK fyne package --target android/arm64 --app-id com.example.myapp
```

fyne's WebGPU painter is the default on android/arm64 and android/amd64. `--target android` builds every ABI, and on the 32-bit ones fyne uses its GL painter since gowgpu has nothing there.

## Surfaces

```go
surface, err := instance.CreateSurfaceUnsafe(wgpu.SurfaceTargetFromAndroidNativeWindow(window))
```

`window` is the raw `ANativeWindow*` as a `uintptr`. There's no display handle on Android. The old `instance.CreateSurface(display, window)` works too and ignores `display`. If a Go object owns the window, implement `wgpu.SurfaceTarget` on it and use `CreateSurfaceFromTarget`, more in [Surface targets](SURFACE-TARGETS.md).

gowgpu never calls `ANativeWindow_acquire` or `ANativeWindow_release`. The window has to stay valid until `Surface.Release` returns, so with NativeActivity release the surface before you return from `onNativeWindowDestroyed`.

When the app comes back you get a new window. Make a new surface for it on the same instance and configure it with the device you already have. The device and everything on it carries on, on Vulkan and on GLES.

Take the format from `adapter.GetSurfaceCapabilities(surface).Formats`. Mali's Vulkan driver doesn't offer BGRA8, so a hard-coded `TextureFormatBGRA8Unorm` won't configure.

## Vulkan specifics

- Needs API 29 (Android 10) or newer. The instance reads `ro.build.version.sdk` and refuses anything older.
- `InstanceFlagsDebug` is refused, there are no debug callbacks on Android.
- The swapchain always uses the identity pre-transform and leaves rotation to the compositor. Android then reports `VK_SUBOPTIMAL_KHR` whenever the display is rotated away from its natural orientation, so gowgpu doesn't report SUBOPTIMAL on Android at all.
- Android 10 can't do finite acquire timeouts, so on API 29 acquire waits for as long as it takes. API 30 and newer get the timeout you ask for.
- The swapchain is made at the surface's `currentExtent`. When you rotate, Android's `currentExtent` can lag the new window size for a few frames, and the swapchain comes out at the old size. Compare `Surface.ActualExtent()` with the size you configured and configure again on the next frame until they match. fyne does exactly that.

## What has run

Android 12 on a Lenovo Tab M10 3rd gen (Unisoc T610, Mali-G52), with fyne: Vulkan and GLES both draw, come back fine after the app was in the background, and handle rotation.

That's the only device so far. Nothing older than Android 12, no emulator, no amd64, and the fixes from the vet on 2026-10-23 haven't run on Android yet.

## Known issues

- Mali refuses GLSL identifiers that start with `gl_`, and naga keeps them when it writes GLSL. So on GLES, don't name anything `gl_*` in WGSL.
- 32-bit Android isn't registered.

## Tests

The Android parts of the Vulkan backend that don't need a device are plain Go with host tests: the API level and debug flag checks, pre-transform, SUBOPTIMAL, the acquire timeout and the `VkAndroidSurfaceCreateInfoKHR` layout.

```bash
go test -run 'Android|SwapchainPlatformPolicy' ./hal/vulkan/ ./hal/vulkan/vk/
```

Everything else needs a device. In fyne `FYNE_WGPU_BACKEND=gl` forces GLES, but an Android app doesn't get your shell's environment, so set it with `os.Setenv` in an `init()` of a test build.
