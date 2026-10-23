//go:build linux && !(js && wasm)

package egl

import (
	"slices"
	"testing"
	"unsafe"
)

func TestSelectWindowKindHonorsExplicitTarget(t *testing.T) {
	x11 := WindowKindX11
	wayland := WindowKindWayland
	tests := []struct {
		name      string
		requested *WindowKind
		detected  WindowKind
		want      WindowKind
		wantCalls int
	}{
		{name: "explicit X11 under Wayland", requested: &x11, detected: WindowKindWayland, want: WindowKindX11, wantCalls: 0},
		{name: "explicit Wayland under X11", requested: &wayland, detected: WindowKindX11, want: WindowKindWayland, wantCalls: 0},
		{name: "nil uses automatic selection", detected: WindowKindSurfaceless, want: WindowKindSurfaceless, wantCalls: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			calls := 0
			got := selectWindowKind(test.requested, func() WindowKind {
				calls++
				return test.detected
			})
			if got != test.want {
				t.Fatalf("selectWindowKind() = %v, want %v", got, test.want)
			}
			if calls != test.wantCalls {
				t.Fatalf("detector calls = %d, want %d", calls, test.wantCalls)
			}
		})
	}
}

// EGL 1.4 takes the OpenGL ES major version alone; the rest needs EGL 1.5
// or EGL_KHR_create_context, which name the debug flag differently.
func TestContextAttribs(t *testing.T) {
	desktop := DefaultContextConfig()
	desktop.Debug = true
	es := ContextConfig{GLVersionMajor: 3, GLVersionMinor: 0, GLES: true, Debug: true}
	tests := []struct {
		name       string
		cfg        ContextConfig
		egl15, khr bool
		want       []EGLInt // nil: refused
	}{
		{"ES on EGL 1.4", es, false, false, []EGLInt{ContextMajorVersion, 3, None}},
		{"desktop on EGL 1.4", desktop, false, false, nil},
		{"ES with KHR_create_context", es, false, true, []EGLInt{ContextMajorVersion, 3, ContextMinorVersion, 0, ContextFlagsKHR, ContextOpenGLDebugBitKHR, None}},
		{"desktop with KHR_create_context", desktop, false, true, []EGLInt{ContextMajorVersion, 3, ContextMinorVersion, 3,
			ContextOpenGLProfileMask, ContextOpenGLCoreProfileBit, ContextFlagsKHR, ContextOpenGLDebugBitKHR, None}},
		{"desktop on EGL 1.5", desktop, true, false, []EGLInt{ContextMajorVersion, 3, ContextMinorVersion, 3,
			ContextOpenGLProfileMask, ContextOpenGLCoreProfileBit, ContextOpenGLDebug, EGLInt(True), None}},
	}
	for _, test := range tests {
		got, err := contextAttribs(test.cfg, test.egl15, test.khr, robustNone)
		if test.want == nil {
			if err == nil {
				t.Errorf("%s: %v, want it refused", test.name, got)
			}
			continue
		}
		if err != nil || !slices.Equal(got, test.want) {
			t.Errorf("%s: %v, %v, want %v", test.name, got, err, test.want)
		}
	}
}

// Robust access is asked for with EGL 1.5's attribute, else with
// EGL_EXT_create_context_robustness's, and last not at all.
func TestRobustContextRequests(t *testing.T) {
	es := ContextConfig{GLVersionMajor: 3, GLES: true}
	tests := []struct {
		egl15, ext bool
		want       []robustness
		attrib     EGLInt // the first request's robust access attribute
	}{
		{true, true, []robustness{robustCore, robustExt, robustNone}, ContextOpenGLRobustAccess},
		{true, false, []robustness{robustCore, robustNone}, ContextOpenGLRobustAccess},
		{false, true, []robustness{robustExt, robustNone}, ContextOpenGLRobustAccessExt},
		{false, false, []robustness{robustNone}, None},
	}
	for _, test := range tests {
		reqs := contextRequests(es, test.egl15, test.ext)
		var got []robustness
		for _, r := range reqs {
			got = append(got, r.robust)
		}
		if !slices.Equal(got, test.want) {
			t.Errorf("EGL 1.5 %v, EXT %v: %v, want %v", test.egl15, test.ext, got, test.want)
		}
		attribs, err := contextAttribs(reqs[0].cfg, test.egl15, false, reqs[0].robust)
		if err != nil {
			t.Fatal(err)
		}
		for _, a := range []EGLInt{ContextOpenGLRobustAccess, ContextOpenGLRobustAccessExt} {
			if slices.Contains(attribs, a) != (a == test.attrib) {
				t.Errorf("EGL 1.5 %v, EXT %v: attributes %#x, want robust access %#x", test.egl15, test.ext, attribs, test.attrib)
			}
		}
	}
}

// eglGetProcAddress gives a swap with damage also for a display without
// the extension, so the display's extensions pick it, KHR before EXT.
func TestDamageSwapFollowsTheDisplay(t *testing.T) {
	var k, e byte
	khr, ext := unsafe.Pointer(&k), unsafe.Pointer(&e)
	tests := []struct {
		exts string
		want unsafe.Pointer
	}{
		{"EGL_KHR_surfaceless_context EGL_KHR_gl_colorspace", nil},
		{"EGL_KHR_swap_buffers_with_damage", khr},
		{"EGL_EXT_swap_buffers_with_damage", ext},
		{"EGL_EXT_swap_buffers_with_damage EGL_KHR_swap_buffers_with_damage", khr},
		{"EGL_KHR_swap_buffers_with_damagefoo", nil},
	}
	for _, test := range tests {
		if got := damageSwap(test.exts, khr, ext); got != test.want {
			t.Errorf("%q: %p, want %p", test.exts, got, test.want)
		}
	}
	if got := damageSwap("EGL_KHR_swap_buffers_with_damage EGL_EXT_swap_buffers_with_damage", nil, ext); got != ext {
		t.Errorf("without the KHR entry point: %p, want the EXT one %p", got, ext)
	}
}
