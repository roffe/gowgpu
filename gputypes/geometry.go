package gputypes

// Extent3D describes a 3D size.
//
// It is used for texture dimensions and copy operations.
// For 2D textures, DepthOrArrayLayers represents the array layer count.
// For 3D textures, it represents the depth.
type Extent3D struct {
	// Width is the size in the X dimension (must be > 0).
	Width uint32
	// Height is the size in the Y dimension (must be > 0).
	Height uint32
	// DepthOrArrayLayers is the size in Z or array layer count (must be > 0).
	DepthOrArrayLayers uint32
}

// NewExtent2D creates an Extent3D for a 2D texture with 1 layer.
func NewExtent2D(width, height uint32) Extent3D {
	return Extent3D{
		Width:              width,
		Height:             height,
		DepthOrArrayLayers: 1,
	}
}

// NewExtent3D creates an Extent3D for a 3D texture.
func NewExtent3D(width, height, depth uint32) Extent3D {
	return Extent3D{
		Width:              width,
		Height:             height,
		DepthOrArrayLayers: depth,
	}
}

// Origin3D describes a 3D origin point.
//
// It is used to specify the starting point for texture copy operations.
type Origin3D struct {
	// X is the X coordinate.
	X uint32
	// Y is the Y coordinate.
	Y uint32
	// Z is the Z coordinate (or array layer for 2D array textures).
	Z uint32
}

// OriginZero is the origin at (0, 0, 0).
var OriginZero = Origin3D{X: 0, Y: 0, Z: 0}

// Viewport describes viewport transformation parameters.
//
// Maps 1:1 to VkViewport, MTLViewport, D3D12_VIEWPORT.
// All native GPU APIs use a struct for viewport; the WebGPU JS spec uses
// positional params because JavaScript lacks cheap value types.
// Go has value types, so a struct is the Go-idiomatic choice.
type Viewport struct {
	// X is the left edge of the viewport in pixels.
	X float32
	// Y is the top edge of the viewport in pixels.
	Y float32
	// Width is the viewport width in pixels.
	Width float32
	// Height is the viewport height in pixels.
	Height float32
	// MinDepth is the minimum depth value (typically 0).
	MinDepth float32
	// MaxDepth is the maximum depth value (typically 1).
	MaxDepth float32
}

// ScissorRect describes a scissor clipping rectangle.
//
// Maps 1:1 to MTLScissorRect and SDL3 SDL_Rect.
// DX12 D3D12_RECT uses min/max corners; the backend converts.
type ScissorRect struct {
	// X is the left edge of the scissor rectangle in pixels.
	X uint32
	// Y is the top edge of the scissor rectangle in pixels.
	Y uint32
	// Width is the scissor rectangle width in pixels.
	Width uint32
	// Height is the scissor rectangle height in pixels.
	Height uint32
}
