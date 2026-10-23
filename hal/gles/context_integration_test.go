//go:build integration && linux && !(js && wasm)

package gles

import (
	"runtime"
	"strings"
	"testing"

	"github.com/roffe/gowgpu/hal/gles/egl"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// GL_CONTEXT_FLAGS and its GL_CONTEXT_FLAG_ROBUST_ACCESS_BIT, desktop GL
// 4.5 and ES 3.2, and GL_CONTEXT_ROBUST_ACCESS of the robustness
// extensions before them.
const (
	glContextFlags        = 0x821E
	glContextFlagRobust   = 0x4
	glContextRobustAccess = 0x90F3
)

func robustAccess(g *gl.Context) bool {
	var flags, robust int32
	g.GetIntegerv(glContextFlags, &flags)
	g.GetIntegerv(glContextRobustAccess, &robust)
	_ = g.GetError()
	return flags&glContextFlagRobust != 0 || robust != 0
}

// Where EGL makes robust contexts the context is one, and so is the helper
// context that shares its objects.
func TestContextsAreRobust(t *testing.T) {
	for _, tc := range []struct {
		name string
		ctx  func(*testing.T) *AdapterContext
	}{{"desktop", newTestAdapterContext}, {"es", newTestESAdapterContext}} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := tc.ctx(t)
			t.Cleanup(ctx.Destroy)
			display := ctx.EGL().Display()
			if !strings.HasPrefix(egl.QueryString(display, egl.Version), "1.5") && !ctx.EGL().HasExtension("EGL_EXT_create_context_robustness") {
				t.Skip("the display makes no robust contexts")
			}
			ctx.keepCurrent = true
			g := ctx.Lock()
			robust := robustAccess(g)
			ctx.Unlock()
			if !robust {
				t.Error("the context has no robust access")
			}

			done := make(chan string, 1)
			go func() {
				runtime.LockOSThread()
				defer runtime.UnlockOSThread()
				g := ctx.Lock()
				defer ctx.Unlock()
				switch {
				case !ctx.OnHelper():
					done <- "the lock from another thread did not run on the helper context"
				case !robustAccess(g):
					done <- "the helper context has no robust access"
				default:
					done <- ""
				}
			}()
			if msg := <-done; msg != "" {
				t.Error(msg)
			}
		})
	}
}

// A desktop context is the newest the driver makes from 4.6 down: NVIDIA
// makes the version asked for, and compute shaders need 4.3, which a
// driver that makes a 4.3 context has.
func TestDesktopContextIsTheNewest(t *testing.T) {
	ctx := newTestAdapterContext(t)
	t.Cleanup(ctx.Destroy)
	g := ctx.Lock()
	version := g.GetString(gl.VERSION)
	ctx.Unlock()

	probe := egl.DefaultContextConfig()
	probe.GLVersionMajor, probe.GLVersionMinor = 4, 3
	p, err := egl.NewContext(probe)
	if err != nil {
		t.Skipf("no OpenGL 4.3 context: %v", err)
	}
	p.Destroy()
	if major, minor, es := parseGLVersion(version); es || major < 4 || major == 4 && minor < 3 {
		t.Fatalf("the context is %q where the driver makes OpenGL 4.3", version)
	}
}
