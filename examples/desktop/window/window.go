// Package window opens a GLFW window with a gowgpu surface on it and runs
// frames on it, for the examples that draw on screen.
package window

import (
	"errors"
	"fmt"
	"log"
	"runtime"
	"slices"
	"time"

	"github.com/go-gl/glfw/v3.4/glfw"
	wgpu "github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/gputypes"

	_ "github.com/roffe/gowgpu/hal/allbackends"
)

func init() {
	runtime.LockOSThread() // GLFW's window and events belong to the main thread
}

// Window is a GLFW window and the gowgpu device that draws to it.
type Window struct {
	Win    *glfw.Window
	Device *wgpu.Device
	Info   gputypes.AdapterInfo
	Format gputypes.TextureFormat // of the surface, so what pipelines target

	title    string
	instance *wgpu.Instance
	surface  *wgpu.Surface
	adapter  *wgpu.Adapter
	config   wgpu.SurfaceConfiguration
}

// Frame draws one frame into view, the window's width by height pixels, dt
// seconds after the last one.
type Frame func(view *wgpu.TextureView, width, height int, dt float32) error

// Open makes a window and the instance, surface, adapter and device that
// draw to it. backend is vulkan, dx12, metal or gl, or "" for whichever of
// Vulkan, Metal and DX12 the machine has.
func Open(title, backend string) (*Window, error) {
	var desc *wgpu.InstanceDescriptor
	if backend != "" {
		b := map[string]gputypes.Backends{
			"vulkan": wgpu.BackendsVulkan, "dx12": wgpu.BackendsDX12, "metal": wgpu.BackendsMetal, "gl": wgpu.BackendsGL,
		}[backend]
		if b == 0 {
			return nil, fmt.Errorf("unknown backend %q", backend)
		}
		desc = &wgpu.InstanceDescriptor{Backends: b}
	}
	if err := glfw.Init(); err != nil {
		return nil, err
	}
	glfw.WindowHint(glfw.ClientAPI, glfw.NoAPI) // gowgpu presents to the window, GLFW makes no GL context
	win, err := glfw.CreateWindow(1280, 800, title, nil, nil)
	if err != nil {
		glfw.Terminate()
		return nil, err
	}
	w := &Window{Win: win, title: title}
	if err := w.open(desc); err != nil {
		w.Close()
		return nil, err
	}
	return w, nil
}

// open makes the gowgpu side; Close releases whatever it got to.
func (w *Window) open(desc *wgpu.InstanceDescriptor) (err error) {
	if w.instance, err = wgpu.CreateInstance(desc); err != nil {
		return fmt.Errorf("create instance: %w", err)
	}
	if w.surface, err = w.instance.CreateSurfaceUnsafe(surfaceTarget(w.Win)); err != nil {
		return fmt.Errorf("create surface: %w", err)
	}
	if w.adapter, err = w.instance.RequestAdapter(&wgpu.RequestAdapterOptions{CompatibleSurface: w.surface}); err != nil {
		return fmt.Errorf("request adapter: %w", err)
	}
	w.Info = w.adapter.Info()
	log.Printf("%s on %s (%s)", w.Info.Backend, w.Info.Name, w.Info.DeviceType)
	if w.Device, err = w.adapter.RequestDevice(nil); err != nil {
		return fmt.Errorf("request device: %w", err)
	}
	w.config = surfaceConfig(w.adapter, w.surface)
	w.Format = w.config.Format
	return nil
}

// Close releases everything Open made, the surface before the device and
// the window.
func (w *Window) Close() {
	if w.surface != nil {
		w.surface.Release()
	}
	if w.Device != nil {
		w.Device.Release()
	}
	if w.adapter != nil {
		w.adapter.Release()
	}
	if w.instance != nil {
		w.instance.Release()
	}
	w.Win.Destroy()
	glfw.Terminate()
}

// surfaceConfig picks a format the shaders write straight into, Mailbox when
// the surface has it so the frame rate is not tied to the display, and an
// opaque surface.
func surfaceConfig(adapter *wgpu.Adapter, surface *wgpu.Surface) wgpu.SurfaceConfiguration {
	cfg := wgpu.SurfaceConfiguration{
		Format:      gputypes.TextureFormatBGRA8Unorm,
		Usage:       gputypes.TextureUsageRenderAttachment,
		PresentMode: gputypes.PresentModeFifo,
		AlphaMode:   gputypes.CompositeAlphaModeAuto,
	}
	caps := adapter.GetSurfaceCapabilities(surface)
	if caps == nil {
		return cfg
	}
	if len(caps.Formats) > 0 {
		cfg.Format = caps.Formats[0]
		for _, f := range []gputypes.TextureFormat{gputypes.TextureFormatBGRA8Unorm, gputypes.TextureFormatRGBA8Unorm} {
			if slices.Contains(caps.Formats, f) {
				cfg.Format = f
				break
			}
		}
	}
	if slices.Contains(caps.PresentModes, gputypes.PresentModeMailbox) {
		cfg.PresentMode = gputypes.PresentModeMailbox
	}
	if slices.Contains(caps.AlphaModes, gputypes.CompositeAlphaModeOpaque) {
		cfg.AlphaMode = gputypes.CompositeAlphaModeOpaque
	}
	return cfg
}

// Run calls frame for every frame until the window closes, or after limit
// frames when that is set. The surface follows the window's size, dt is
// capped so a stall does not jump a simulation, and the title shows the
// adapter and the frame rate.
func (w *Window) Run(limit int, frame Frame) error {
	var (
		last   = time.Now()
		second = last
		frames int
	)
	for total := 0; !w.Win.ShouldClose() && (limit == 0 || total < limit); total++ {
		glfw.PollEvents()
		width, height := w.Win.GetFramebufferSize()
		if width == 0 || height == 0 { // minimized
			time.Sleep(50 * time.Millisecond)
			continue
		}
		if uint32(width) != w.config.Width || uint32(height) != w.config.Height { //nolint:gosec // GLFW sizes are positive
			w.config.Width, w.config.Height = uint32(width), uint32(height) //nolint:gosec // GLFW sizes are positive
			if err := w.surface.Configure(w.Device, &w.config); err != nil {
				return fmt.Errorf("configure surface: %w", err)
			}
		}

		now := time.Now()
		dt := min(float32(now.Sub(last).Seconds()), 1.0/30)
		last = now
		err := w.present(width, height, dt, frame)
		if errors.Is(err, wgpu.ErrSurfaceOutdated) || errors.Is(err, wgpu.ErrSurfaceLost) {
			w.config.Width = 0 // configure again next frame
			continue
		}
		if err != nil {
			return err
		}

		frames++
		if now.Sub(second) >= time.Second {
			w.Win.SetTitle(fmt.Sprintf("%s on %s (%s), %d fps", w.title, w.Info.Name, w.Info.Backend, frames))
			frames, second = 0, now
		}
	}
	return nil
}

// present acquires the frame, has frame draw into it and shows it.
func (w *Window) present(width, height int, dt float32, frame Frame) error {
	tex, _, err := w.surface.GetCurrentTexture()
	if err != nil {
		return err
	}
	view, err := tex.CreateView(nil)
	if err != nil {
		w.surface.DiscardTexture()
		return err
	}
	defer view.Release()
	if err := frame(view, width, height, dt); err != nil {
		w.surface.DiscardTexture()
		return err
	}
	if err := w.surface.Present(tex); err != nil {
		w.surface.DiscardTexture()
		return err
	}
	return nil
}
