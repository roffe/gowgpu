package gputypes

import (
	"testing"
	"unsafe"
)

func TestViewportZeroValue(t *testing.T) {
	t.Parallel()

	var vp Viewport

	if vp.X != 0 || vp.Y != 0 {
		t.Errorf("zero Viewport origin = (%v, %v), want (0, 0)", vp.X, vp.Y)
	}
	if vp.Width != 0 || vp.Height != 0 {
		t.Errorf("zero Viewport size = (%v, %v), want (0, 0)", vp.Width, vp.Height)
	}
	if vp.MinDepth != 0 || vp.MaxDepth != 0 {
		t.Errorf("zero Viewport depth = (%v, %v), want (0, 0)", vp.MinDepth, vp.MaxDepth)
	}
}

func TestViewportTypicalValues(t *testing.T) {
	t.Parallel()

	vp := Viewport{
		X: 0, Y: 0,
		Width: 800, Height: 600,
		MinDepth: 0, MaxDepth: 1,
	}

	if vp.Width != 800 {
		t.Errorf("Width = %v, want 800", vp.Width)
	}
	if vp.Height != 600 {
		t.Errorf("Height = %v, want 600", vp.Height)
	}
	if vp.MaxDepth != 1 {
		t.Errorf("MaxDepth = %v, want 1", vp.MaxDepth)
	}
}

func TestViewportFieldLayout(t *testing.T) {
	t.Parallel()

	// VkViewport layout: x, y, width, height, minDepth, maxDepth — 6 x float32 = 24 bytes.
	var vp Viewport

	if got := unsafe.Sizeof(vp); got != 24 {
		t.Errorf("sizeof(Viewport) = %d, want 24 (6 x float32)", got)
	}
	if got := unsafe.Offsetof(vp.X); got != 0 {
		t.Errorf("offset(X) = %d, want 0", got)
	}
	if got := unsafe.Offsetof(vp.MaxDepth); got != 20 {
		t.Errorf("offset(MaxDepth) = %d, want 20", got)
	}
}

func TestScissorRectZeroValue(t *testing.T) {
	t.Parallel()

	var sr ScissorRect

	if sr.X != 0 || sr.Y != 0 {
		t.Errorf("zero ScissorRect origin = (%d, %d), want (0, 0)", sr.X, sr.Y)
	}
	if sr.Width != 0 || sr.Height != 0 {
		t.Errorf("zero ScissorRect size = (%d, %d), want (0, 0)", sr.Width, sr.Height)
	}
}

func TestScissorRectTypicalValues(t *testing.T) {
	t.Parallel()

	sr := ScissorRect{X: 0, Y: 0, Width: 800, Height: 600}

	if sr.Width != 800 {
		t.Errorf("Width = %d, want 800", sr.Width)
	}
	if sr.Height != 600 {
		t.Errorf("Height = %d, want 600", sr.Height)
	}
}

func TestScissorRectFieldLayout(t *testing.T) {
	t.Parallel()

	// 4 x uint32 = 16 bytes.
	var sr ScissorRect

	if got := unsafe.Sizeof(sr); got != 16 {
		t.Errorf("sizeof(ScissorRect) = %d, want 16 (4 x uint32)", got)
	}
}

func TestExtent3DNewExtent2D(t *testing.T) {
	t.Parallel()

	e := NewExtent2D(1920, 1080)

	if e.Width != 1920 {
		t.Errorf("Width = %d, want 1920", e.Width)
	}
	if e.Height != 1080 {
		t.Errorf("Height = %d, want 1080", e.Height)
	}
	if e.DepthOrArrayLayers != 1 {
		t.Errorf("DepthOrArrayLayers = %d, want 1", e.DepthOrArrayLayers)
	}
}

func TestOriginZero(t *testing.T) {
	t.Parallel()

	if OriginZero.X != 0 || OriginZero.Y != 0 || OriginZero.Z != 0 {
		t.Errorf("OriginZero = %+v, want {0, 0, 0}", OriginZero)
	}
}
