// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

package gles

import (
	"context"
	"errors"
	"fmt"
	"image"
	"log/slog"
	"unsafe"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/gles/egl"
	"github.com/roffe/gowgpu/hal/gles/gl"
)

// errOtherThread is the answer of Submit, Present, Surface.Configure and the
// CreateCommandEncoder that makes the VAO from a goroutine other than the one
// holding the context under InstanceFlagsGLPinnedThread: the helper context
// they would run on shares no VAO or framebuffer, so nothing would be drawn.
var errOtherThread = errors.New("gles: InstanceFlagsGLPinnedThread keeps the GL context on the goroutine that used it first; this call comes from another goroutine")

// Queue implements hal.Queue for OpenGL on Linux.
// Holds a shared *AdapterContext (owned by Instance or Surface).
type Queue struct {
	ctx             *AdapterContext
	submissionIndex uint64
	fence           *Fence // signaled at each submit for GPU completion tracking
}

// Submit submits command buffers to the GPU.
// Acquires the AdapterContext lock, makes context current on pbuffer/surfaceless,
// executes all GL commands, signals the fence, and flushes.
func (q *Queue) Submit(commandBuffers []hal.CommandBuffer) (uint64, error) {
	glCtx := q.ctx.Lock()
	defer q.ctx.Unlock()
	if q.ctx.OnHelper() {
		return 0, errOtherThread
	}
	glCtx.Shadow = gl.Shadow{} // what ran between submissions is not followed

	// A GL error is only reported through the logger, so the per command
	// glGetError runs only when a logger would show the warning. Otherwise
	// one call at the end clears the error flag, as the check after the last
	// command did, so it does not reach whatever checks glGetError next.
	checkEach := hal.Logger().Enabled(context.Background(), slog.LevelWarn)
	for _, cb := range commandBuffers {
		cmdBuf, ok := cb.(*CommandBuffer)
		if !ok {
			return 0, fmt.Errorf("gles: invalid command buffer type")
		}

		// Execute recorded commands with GL error checking.
		for i, cmd := range cmdBuf.commands {
			cmd.Execute(glCtx)
			if !checkEach {
				continue
			}
			if glErr := glCtx.GetError(); glErr != 0 {
				detail := fmt.Sprintf("%T", cmd)
				if vaoCmd, ok := cmd.(*BindVAOCommand); ok {
					detail = fmt.Sprintf("%T{vao=%d}", cmd, vaoCmd.vao)
				}
				hal.Logger().Warn("gles: GL error after command", "error", fmt.Sprintf("0x%x", glErr), "index", i, "command", detail)
			}
		}
	}
	if !checkEach {
		_ = glCtx.GetError()
	}

	q.submissionIndex++

	// Rust wgpu-hal queue.rs:1915-1921: fence.maintain → fence.signal → gl.flush.
	// FenceSync must be inserted BEFORE Flush so the sync object tracks the
	// commands being flushed. Flushing first would leave the fence un-flushed.
	if q.fence != nil {
		q.fence.Maintain()
		if err := q.fence.Signal(q.submissionIndex); err != nil {
			return 0, err
		}
	}

	glCtx.Flush()

	return q.submissionIndex, nil
}

// PollCompleted returns the highest submission index known to be completed.
// Polls pending GL sync objects via glGetSynciv (non-blocking, no flush).
// Safe because Submit() always flushes after inserting the fence — the fence
// is guaranteed to be in the GPU command queue by the time we poll it.
// Maintenance (cleanup of completed sync objects) happens in Submit(), not here
// (matches Rust wgpu-hal device.rs:1564 get_fence_value).
func (q *Queue) PollCompleted() uint64 {
	if q.fence == nil {
		return q.submissionIndex
	}
	// glGetSynciv needs the context current on this thread, like every
	// other GL call; without it the query answered nothing and completion
	// lagged a submit behind.
	q.ctx.Lock()
	defer q.ctx.Unlock()
	return q.fence.GetLatest()
}

// WriteBuffer writes data to a buffer immediately.
func (q *Queue) WriteBuffer(buffer hal.Buffer, offset uint64, data []byte) error {
	buf, ok := buffer.(*Buffer)
	if !ok {
		return fmt.Errorf("gles: WriteBuffer: invalid buffer type")
	}
	if len(data) == 0 {
		return nil
	}

	glCtx := q.ctx.Lock()
	defer q.ctx.Unlock()

	glCtx.BindBuffer(buf.target, buf.id)
	glCtx.BufferSubData(buf.target, int(offset), len(data), unsafe.Pointer(&data[0]))
	glCtx.BindBuffer(buf.target, 0)
	return nil
}

// WriteTexture writes data to a texture immediately.
func (q *Queue) WriteTexture(dst *hal.ImageCopyTexture, data []byte, layout *hal.ImageDataLayout, size *hal.Extent3D) error {
	tex, ok := dst.Texture.(*Texture)
	if !ok {
		return fmt.Errorf("gles: invalid texture type for WriteTexture")
	}

	glCtx := q.ctx.Lock()
	defer q.ctx.Unlock()
	writeTexture(glCtx, tex, dst, data, layout, size)
	return nil
}

// Present presents a surface texture to the screen.
//
// Makes the GL context current on the window EGLSurface (via LockForSurface),
// blits the Surface's swapchain offscreen FBO to the default framebuffer with
// an explicit Y-flip, then SwapBuffers. Mirrors Rust wgpu-hal
// src/gles/egl.rs Surface::present (1280-1308) and Windows LockForDC present.
//
// damageRects is an optional list of rectangles (physical pixels, top-left
// origin) indicating which surface regions changed this frame. When non-empty
// and EGL_KHR_swap_buffers_with_damage is available, the rects are passed to
// eglSwapBuffersWithDamageKHR as compositor hints. EGL uses bottom-left
// origin, so Y coordinates are flipped here. When the extension is unavailable
// or no rects are provided, the standard eglSwapBuffers path is used.
func (q *Queue) Present(surface hal.Surface, _ hal.SurfaceTexture, damageRects []image.Rectangle) error {
	surf, ok := surface.(*Surface)
	if !ok {
		return fmt.Errorf("gles: invalid surface type")
	}

	glCtx := q.ctx.LockForSurface(surf.eglSurface)
	defer q.ctx.Unlock()
	if q.ctx.OnHelper() {
		return errOtherThread
	}

	// eglSwapInterval applies to the current draw surface, so it is set
	// here, once per configuration: Immediate and Mailbox swap without
	// waiting for vblank, FIFO waits. Before 2026-10-18 nothing set it and
	// every mode swapped at the driver's default.
	if !surf.swapIntervalSet && surf.config != nil {
		interval := egl.EGLInt(1)
		switch surf.config.PresentMode {
		case hal.PresentModeImmediate, hal.PresentModeMailbox:
			interval = 0
		}
		egl.SwapInterval(surf.eglDisplay, interval)
		surf.swapIntervalSet = true
	}

	surf.blitSwapchainToDefaultWith(glCtx)

	// Use damage-aware swap when the extension is available and rects provided.
	if len(damageRects) > 0 && egl.HasSwapBuffersWithDamage() {
		// Convert image.Rectangle (top-left origin) to EGL packed int32 array
		// (bottom-left origin). Each rect is {x, y, width, height}.
		// Stack-allocate for up to 8 rects (8 * 4 = 32 ints).
		var stackInts [32]int32
		ints := stackInts[:0]
		surfaceHeight := int32(surf.fboHeight)
		for _, r := range damageRects {
			// Y-flip: EGL uses bottom-left origin.
			// egl_y = surface_height - rect.Max.Y
			ints = append(ints,
				int32(r.Min.X),
				surfaceHeight-int32(r.Max.Y),
				int32(r.Dx()),
				int32(r.Dy()),
			)
		}
		result := egl.SwapBuffersWithDamage(surf.eglDisplay, surf.eglSurface, &ints[0], int32(len(damageRects)))
		if result == egl.False {
			return fmt.Errorf("gles: eglSwapBuffersWithDamageKHR failed: error 0x%x", egl.GetError())
		}
		return nil
	}

	// Standard full-surface swap.
	result := egl.SwapBuffers(surf.eglDisplay, surf.eglSurface)
	if result == egl.False {
		return fmt.Errorf("gles: eglSwapBuffers failed: error 0x%x", egl.GetError())
	}

	return nil
}

// GetTimestampPeriod returns the timestamp period in nanoseconds.
func (q *Queue) GetTimestampPeriod() float32 {
	// OpenGL doesn't have a standard way to query this
	// Return 1.0 to indicate nanoseconds
	return 1.0
}

// SupportsCommandBufferCopies returns false for GLES on Linux.
// GLES uses direct GL calls for writes, not command buffer copy operations.
func (q *Queue) SupportsCommandBufferCopies() bool {
	return false
}

// SetSwapchainSuppressed is a no-op on GLES.
// GLES uses eglSwapBuffers for presentation, which is not affected by command
// submission ordering. See BUG-WGPU-VK-005 (Vulkan-specific).
func (q *Queue) SetSwapchainSuppressed(_ bool) {}
