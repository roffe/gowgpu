//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

import (
	"fmt"
	"runtime"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
	"github.com/roffe/gowgpu/internal/indirect"
)

// CommandBuffer holds a recorded Vulkan command buffer.
// Standalone encoders take it from cmdBufferResultPool (VK-PERF-004);
// pool-managed ones reuse their own (see CommandEncoder.result).
type CommandBuffer struct {
	handle vk.CommandBuffer
	pool   vk.CommandPool
}

// Destroy releases the command buffer resources.
// Returns the struct to the pool for reuse (VK-PERF-004).
func (c *CommandBuffer) Destroy() {
	// Command buffers are freed when the pool is destroyed or reset.
	// We cannot call vkFreeCommandBuffers here because:
	// 1. GPU may still be using this command buffer (async submission)
	// 2. Proper solution requires fence-based tracking or pool reset after WaitIdle
	c.handle = 0
	c.pool = 0
	cmdBufferResultPool.Put(c)
}

// allocationGranularity is the number of VkCommandBuffers to batch-allocate
// when the free list is empty. Matches Rust wgpu-hal ALLOCATION_GRANULARITY.
// 16 amortizes the cost of vkAllocateCommandBuffers across many BeginEncoding
// calls and aligns with NVIDIA/ARM best practices for command buffer pooling.
const allocationGranularity = 16

// CommandEncoder implements hal.CommandEncoder for Vulkan.
//
// Uses the free list pattern (Rust wgpu-hal parity): the encoder owns a
// VkCommandPool and manages a pool of VkCommandBuffer handles internally.
// BeginEncoding pops from the free list (batch-allocating if empty),
// EndEncoding detaches the active handle, and ResetAll recycles completed
// and discarded handles back to the free list with a single vkResetCommandPool.
//
// Two ownership modes:
//   - Standalone (poolManaged=false): After EndEncoding, the encoder detaches its
//     pool to the CommandBuffer result and returns itself to encoderPool
//     (sync.Pool for struct reuse). FreeCommandBuffer recycles the Vulkan resources.
//     This is the default mode for user-created command encoders.
//   - Pool-managed (poolManaged=true): After EndEncoding, the encoder retains
//     ownership of its VkCommandPool. After GPU completion, ResetAll resets the
//     pool so the encoder can be reused via the wgpu-level encoder pool.
//     Used by pendingWrites.
//
// Pooled via encoderPool to avoid per-frame heap allocation (VK-PERF-003).
//
// Reference: Rust wgpu-hal vulkan/mod.rs:939-965 (struct),
// vulkan/command.rs:7-192 (begin/end/discard/reset_all).
type CommandEncoder struct {
	device *Device
	pool   vk.CommandPool

	// active is the current recording VkCommandBuffer. Zero means not recording.
	// Replaces the old isRecording bool — matches Rust wgpu-hal's self.active
	// null check pattern (vulkan/command.rs:153).
	active vk.CommandBuffer

	// free holds available VkCommandBuffers in Vulkan "initial" state.
	// BeginEncoding pops from here; batch-allocates allocationGranularity
	// handles via vkAllocateCommandBuffers when empty.
	free []vk.CommandBuffer

	// discarded holds used VkCommandBuffers that were abandoned via
	// DiscardEncoding. They could be in any Vulkan state except "pending".
	// ResetAll moves them back to free after vkResetCommandPool.
	discarded []vk.CommandBuffer

	// results are the CommandBuffer structs a pool-managed encoder handed
	// out; the first usedResults are live until the next ResetAll.
	results     []*CommandBuffer
	usedResults int

	label       string
	poolManaged bool // true when managed by wgpu-level encoder pool

	// ADR-060: Inline present barrier optimization.
	// Tracks the swapchain image targeted by BeginRenderPass so that
	// EndEncoding can inject a pipeline barrier (COLOR_ATTACHMENT_OPTIMAL
	// -> PRESENT_SRC_KHR) into the SAME command buffer, eliminating the
	// separate vkQueueSubmit that ensurePresentLayout would otherwise need.
	// Set by BeginRenderPass when a swapchain view is used; cleared by
	// EndEncoding after the barrier is injected (or if no swapchain was used).
	//
	// For multi-submit frames (e.g. g3d + gg), each Submit's EndEncoding
	// injects the barrier, and the next Submit's BeginRenderPass inserts a
	// reverse barrier (PRESENT_SRC -> COLOR_ATTACHMENT) when LoadOp::Load
	// needs the image back in COLOR_ATTACHMENT_OPTIMAL.
	//
	// Reference: Rust wgpu-core queue.rs:1284 — "Transition surface textures
	// into Present state" is appended to the last baked encoder inline.
	swapchainImage  vk.Image       // VkImage of the swapchain target (0 = none)
	swapchainLayout vk.ImageLayout // layout the render pass leaves the image in
	swapchain       *Swapchain     // back-pointer for layout tracking updates

	// presented is the swapchain whose image presentedImage an inline
	// present barrier of this encoder marked PRESENT_SRC_KHR, kept until
	// ResetAll: the fields above are per command buffer (BeginEncoding
	// clears them), and a swapchain pass split off into a command buffer
	// of its own (core recordPassTransitions) is closed, barrier and all,
	// while the encoder goes on recording. A DiscardEncoding after that
	// drops the closed command buffer too, so the image never reaches
	// PRESENT_SRC_KHR and its tracking goes back to UNDEFINED.
	presented      *Swapchain
	presentedImage uint32
}

// BeginEncoding begins command recording.
//
// Pops a VkCommandBuffer from the free list. If the free list is empty,
// batch-allocates allocationGranularity (16) command buffers from the pool.
// This matches Rust wgpu-hal begin_encoding (vulkan/command.rs:122-151).
//
// Returns an error if the device is nil or if Vulkan allocation/begin fails.
func (e *CommandEncoder) BeginEncoding(label string) error {
	e.label = label

	if e.device == nil {
		return fmt.Errorf("vulkan: BeginEncoding called with nil device")
	}

	// Batch-allocate command buffers when free list is empty.
	if len(e.free) == 0 {
		allocInfo := vk.CommandBufferAllocateInfo{
			SType:              vk.StructureTypeCommandBufferAllocateInfo,
			CommandPool:        e.pool,
			Level:              vk.CommandBufferLevelPrimary,
			CommandBufferCount: allocationGranularity,
		}
		buffers := make([]vk.CommandBuffer, allocationGranularity)
		result := e.device.cmds.AllocateCommandBuffers(e.device.handle, &allocInfo, &buffers[0])
		if result != vk.Success {
			return fmt.Errorf("vulkan: vkAllocateCommandBuffers failed: %d", result)
		}
		e.free = append(e.free, buffers...)
	}

	// Pop from free list.
	raw := e.free[len(e.free)-1]
	e.free = e.free[:len(e.free)-1]

	// VK-001: Validate handle after allocation. goffi returns zeros on nil
	// function pointer (no crash, no error), so vkAllocateCommandBuffers
	// could "succeed" with a null handle (gogpu#119).
	if raw == 0 {
		return fmt.Errorf("vulkan: allocated command buffer has null handle")
	}

	// Begin command buffer with ONE_TIME_SUBMIT for per-frame recording.
	beginInfo := vk.CommandBufferBeginInfo{
		SType: vk.StructureTypeCommandBufferBeginInfo,
		Flags: vk.CommandBufferUsageFlags(vk.CommandBufferUsageOneTimeSubmitBit),
	}

	result := vkBeginCommandBuffer(e.device.cmds, raw, &beginInfo)
	if result != vk.Success {
		// Return the buffer to the free list on failure — it is still in
		// initial state and can be reused.
		e.free = append(e.free, raw)
		return fmt.Errorf("vulkan: vkBeginCommandBuffer failed: %d", result)
	}

	e.active = raw

	// ADR-060: Reset inline present barrier tracking for this new recording.
	e.swapchainImage = 0
	e.swapchainLayout = 0
	e.swapchain = nil

	return nil
}

// EndEncoding finishes command recording and returns a command buffer.
// Reuses CommandBuffer structs, see result (VK-PERF-004).
//
// In standalone mode (default): detaches pool to the result and
// returns the encoder struct to encoderPool for reuse.
// In pool-managed mode: the encoder retains ownership of its VkCommandPool.
// After GPU completion, call ResetAll to prepare for the next BeginEncoding cycle.
//
// ADR-060: Before closing the command buffer, injects an inline pipeline
// barrier to transition the swapchain image from COLOR_ATTACHMENT_OPTIMAL
// to PRESENT_SRC_KHR. This eliminates the separate vkQueueSubmit that
// ensurePresentLayout would otherwise need, recovering ~30 FPS on Intel
// Iris Xe. The barrier is recorded into the SAME command buffer as the
// user's work — zero extra CB, zero extra submit.
//
// Reference: Rust wgpu-hal end_encoding (vulkan/command.rs:153-163).
// Reference: Rust wgpu-core queue.rs:1284 — present barrier appended inline.
func (e *CommandEncoder) EndEncoding() (hal.CommandBuffer, error) {
	if e.active == 0 {
		return nil, fmt.Errorf("vulkan: command encoder is not recording")
	}

	// ADR-060: Inject inline present barrier before closing the command buffer.
	// If a render pass targeted a swapchain image and left it in
	// COLOR_ATTACHMENT_OPTIMAL, append a pipeline barrier to transition it to
	// PRESENT_SRC_KHR. This is the same barrier ensurePresentLayout would
	// record in a separate submit — but here it is inside the user's CB.
	//
	// For multi-submit frames, the next BeginRenderPass inserts a reverse
	// barrier (PRESENT_SRC -> COLOR_ATTACHMENT) when LoadOp::Load is used,
	// so the inline barrier in the previous submit does not cause a layout
	// mismatch.
	if e.swapchainImage != 0 && e.swapchainLayout != vk.ImageLayoutPresentSrcKhr {
		e.injectInlinePresentBarrier()
	}

	result := vkEndCommandBuffer(e.device.cmds, e.active)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkEndCommandBuffer failed: %d", result)
	}

	cb := e.result()
	cb.handle = e.active
	e.active = 0

	if e.poolManaged {
		// Pool-managed mode: encoder retains VkCommandPool ownership.
		// CommandBuffer must NOT carry pool — otherwise FreeCommandBuffer
		// would return it to freeAllocators, causing double-free when
		// encoder.Destroy() also destroys the same pool at shutdown.
		cb.pool = 0
	} else {
		// Standalone mode: CommandBuffer takes ownership of pool.
		cb.pool = e.pool

		// Detach resources to CommandBuffer, return encoder struct.
		// The pool goes with the CommandBuffer; free/discarded buffers
		// are abandoned (they belong to the pool and will be reset when
		// the pool is recycled via FreeCommandBuffer).
		e.device = nil
		e.pool = 0
		e.free = e.free[:0]
		e.discarded = e.discarded[:0]
		e.label = ""
		e.swapchainImage = 0
		e.swapchainLayout = 0
		e.swapchain = nil
		e.presented = nil
		encoderPool.Put(e)
	}

	return cb, nil
}

// result returns the struct for the command buffer EndEncoding hands out.
// A pool-managed encoder reuses its own: ResetAll takes back every buffer of
// the pool, so they are free again after it. They never leave the encoder,
// so their pool stays 0 and a late FreeCommandBuffer on one stays a no-op,
// where a struct from cmdBufferResultPool could already belong to a
// standalone encoder by then.
func (e *CommandEncoder) result() *CommandBuffer {
	if !e.poolManaged {
		return cmdBufferResultPool.Get().(*CommandBuffer)
	}
	if e.usedResults == len(e.results) {
		e.results = append(e.results, new(CommandBuffer))
	}
	cb := e.results[e.usedResults]
	e.usedResults++
	return cb
}

// injectInlinePresentBarrier records a pipeline barrier inside the active
// command buffer, transitioning the swapchain image from its current tracked
// layout to PRESENT_SRC_KHR. Called from EndEncoding to avoid a separate
// vkQueueSubmit in ensurePresentLayout.
//
// The source access mask and pipeline stage are determined by the layout the
// render pass left the image in, matching ensurePresentLayout's switch logic.
func (e *CommandEncoder) injectInlinePresentBarrier() {
	// Determine source access mask and pipeline stage based on the tracked layout.
	var srcAccess vk.AccessFlags
	var srcStage vk.PipelineStageFlags
	switch e.swapchainLayout {
	case vk.ImageLayoutColorAttachmentOptimal:
		srcAccess = vk.AccessFlags(vk.AccessColorAttachmentWriteBit)
		srcStage = vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)
	case vk.ImageLayoutTransferDstOptimal:
		srcAccess = vk.AccessFlags(vk.AccessTransferWriteBit)
		srcStage = vk.PipelineStageFlags(vk.PipelineStageTransferBit)
	case vk.ImageLayoutTransferSrcOptimal:
		srcAccess = vk.AccessFlags(vk.AccessTransferReadBit)
		srcStage = vk.PipelineStageFlags(vk.PipelineStageTransferBit)
	default:
		srcAccess = 0
		srcStage = vk.PipelineStageFlags(vk.PipelineStageTopOfPipeBit)
	}

	barrier := vk.ImageMemoryBarrier{
		SType:               vk.StructureTypeImageMemoryBarrier,
		SrcAccessMask:       srcAccess,
		DstAccessMask:       0, // Present engine does not need explicit access
		OldLayout:           e.swapchainLayout,
		NewLayout:           vk.ImageLayoutPresentSrcKhr,
		SrcQueueFamilyIndex: vk.QueueFamilyIgnored,
		DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		Image:               e.swapchainImage,
		SubresourceRange: vk.ImageSubresourceRange{
			AspectMask:     vk.ImageAspectFlags(vk.ImageAspectColorBit),
			BaseMipLevel:   0,
			LevelCount:     1,
			BaseArrayLayer: 0,
			LayerCount:     1,
		},
	}

	e.device.cmds.CmdPipelineBarrier(
		e.active,
		srcStage,
		vk.PipelineStageFlags(vk.PipelineStageBottomOfPipeBit),
		0,      // dependencyFlags
		0, nil, // memory barriers
		0, nil, // buffer barriers
		1, &barrier,
	)

	// Update the swapchain's layout tracking so ensurePresentLayout in
	// present() sees PRESENT_SRC_KHR and returns early (zero-cost path).
	if e.swapchain != nil {
		e.swapchain.SetImageLayout(e.swapchain.currentImage, vk.ImageLayoutPresentSrcKhr)
		e.presented, e.presentedImage = e.swapchain, e.swapchain.currentImage
	}

	// Prevent double-injection if EndEncoding is called again (should not
	// happen, but defensive).
	e.swapchainImage = 0
}

// setupInlinePresentBarrier prepares the encoder for inline present barrier
// injection in EndEncoding. Called from BeginRenderPass when a swapchain
// image is targeted.
//
// Two responsibilities:
//  1. Record the swapchain image/layout so EndEncoding knows what to barrier.
//  2. If the image is currently in PRESENT_SRC_KHR (from a previous submit's
//     inline barrier) and LoadOp::Load requires COLOR_ATTACHMENT_OPTIMAL,
//     insert a reverse barrier before vkCmdBeginRenderPass to prevent a
//     Vulkan layout mismatch VUID violation.
func (e *CommandEncoder) setupInlinePresentBarrier(
	view, resolveView *TextureView,
	hasMSAAResolve bool,
	colorFinalLayout vk.ImageLayout,
	loadOp gputypes.LoadOp,
) {
	swapView := swapchainTargetView(view, resolveView, hasMSAAResolve)
	if swapView == nil || swapView.swapchain == nil {
		return
	}
	sc := swapView.swapchain
	idx := sc.currentImage

	// Reverse barrier for multi-submit LoadOp::Load: if a previous submit's
	// EndEncoding transitioned the image to PRESENT_SRC_KHR, we must
	// transition back to COLOR_ATTACHMENT_OPTIMAL before the render pass.
	if int(idx) < len(sc.imageLayouts) &&
		sc.imageLayouts[idx] == vk.ImageLayoutPresentSrcKhr &&
		loadOpToVk(loadOp) == vk.AttachmentLoadOpLoad {
		e.injectReverseBarrier(sc, idx)
	}

	// Record the swapchain image for EndEncoding's forward barrier.
	e.swapchainImage = sc.images[idx]
	e.swapchainLayout = colorFinalLayout
	e.swapchain = sc
}

// injectReverseBarrier transitions a swapchain image from PRESENT_SRC_KHR
// back to COLOR_ATTACHMENT_OPTIMAL inside the active command buffer. This is
// needed in multi-submit frames when a previous submit's EndEncoding injected
// an inline forward barrier, but the current submit uses LoadOp::Load which
// requires the image in COLOR_ATTACHMENT_OPTIMAL.
func (e *CommandEncoder) injectReverseBarrier(sc *Swapchain, idx uint32) {
	barrier := vk.ImageMemoryBarrier{
		SType:               vk.StructureTypeImageMemoryBarrier,
		SrcAccessMask:       0, // Present engine has no pending writes
		DstAccessMask:       vk.AccessFlags(vk.AccessColorAttachmentWriteBit | vk.AccessColorAttachmentReadBit),
		OldLayout:           vk.ImageLayoutPresentSrcKhr,
		NewLayout:           vk.ImageLayoutColorAttachmentOptimal,
		SrcQueueFamilyIndex: vk.QueueFamilyIgnored,
		DstQueueFamilyIndex: vk.QueueFamilyIgnored,
		Image:               sc.images[idx],
		SubresourceRange: vk.ImageSubresourceRange{
			AspectMask:     vk.ImageAspectFlags(vk.ImageAspectColorBit),
			BaseMipLevel:   0,
			LevelCount:     1,
			BaseArrayLayer: 0,
			LayerCount:     1,
		},
	}
	e.device.cmds.CmdPipelineBarrier(
		e.active,
		vk.PipelineStageFlags(vk.PipelineStageBottomOfPipeBit),
		vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit),
		0,      // dependencyFlags
		0, nil, // memory barriers
		0, nil, // buffer barriers
		1, &barrier,
	)
	sc.SetImageLayout(idx, vk.ImageLayoutColorAttachmentOptimal)
}

// DiscardEncoding discards the current recording without creating a command buffer.
// The active command buffer is moved to the discarded list for later pool reset.
// Rust wgpu-hal does NOT call vkEndCommandBuffer on discarded buffers —
// vkResetCommandPool handles them in any state (vulkan/command.rs:165-173).
//
// In standalone mode, recycles the pool and returns the encoder struct.
// In pool-managed mode, retains resources for ResetAll+reuse.
func (e *CommandEncoder) DiscardEncoding() {
	if e.active != 0 {
		e.discarded = append(e.discarded, e.active)
		e.active = 0
	}

	// ADR-060: Clear inline present barrier tracking — discarded work
	// should not trigger a layout transition. Reset the swapchain image
	// layout to UNDEFINED so ensurePresentLayout sees the real GPU state
	// (the discarded render pass never executed).
	if e.swapchain != nil {
		e.swapchain.SetImageLayout(e.swapchain.currentImage, vk.ImageLayoutUndefined)
	}
	e.swapchainImage = 0
	e.swapchainLayout = 0
	e.swapchain = nil
	// The same for a command buffer this encoder closed with a present
	// barrier inside: it goes with the discard, so its image is not
	// presentable either.
	if e.presented != nil {
		e.presented.SetImageLayout(e.presentedImage, vk.ImageLayoutUndefined)
		e.presented = nil
	}

	if !e.poolManaged {
		// Standalone mode: recycle pool and encoder struct (VK-POOL-001).
		// The pool contains all free+discarded buffers; they will be reset
		// when the pool is recycled via FreeCommandBuffer/acquireAllocator.
		if e.device != nil && e.pool != 0 {
			e.device.recyclePool(e.pool)
		}
		e.device = nil
		e.pool = 0
		e.free = e.free[:0]
		e.discarded = e.discarded[:0]
		e.label = ""
		encoderPool.Put(e)
	}
	// Pool-managed mode: encoder retains resources for ResetAll+reuse.
}

// ResetAll recycles completed command buffers and discarded buffers back to the
// free list, then resets the entire command pool. After this call, all buffers
// in the free list are in Vulkan "initial" state, ready for vkBeginCommandBuffer.
//
// This is much cheaper than individual vkResetCommandBuffer calls (NVIDIA, ARM).
// Reference: Rust wgpu-hal reset_all (vulkan/command.rs:175-192).
func (e *CommandEncoder) ResetAll(commandBuffers []hal.CommandBuffer) {
	// Recycle completed command buffers back to the free list.
	for _, cb := range commandBuffers {
		if vcb, ok := cb.(*CommandBuffer); ok && vcb.handle != 0 {
			e.free = append(e.free, vcb.handle)
		}
	}

	// Recycle discarded buffers.
	e.free = append(e.free, e.discarded...)
	e.discarded = e.discarded[:0]
	e.usedResults = 0
	// The command buffers are spent, so is the present barrier one of them
	// carried.
	e.presented = nil

	// Batch reset entire pool — one call resets all command buffers to
	// initial state. Much cheaper than N individual resets (NVIDIA, ARM docs).
	if e.device != nil && e.pool != 0 {
		vkResetCommandPool(e.device.cmds, e.device.handle, e.pool, 0)
	}
}

// SetPoolManaged marks this encoder as managed by the wgpu-level encoder pool.
// When true, EndEncoding retains ownership of the VkCommandPool instead of
// detaching it to the CommandBuffer result. This enables encoder reuse after
// ResetAll without creating new Vulkan resources.
func (e *CommandEncoder) SetPoolManaged(managed bool) {
	e.poolManaged = managed
}

// Destroy releases the VkCommandPool owned by this encoder.
// Destroying the pool implicitly frees all command buffers allocated from it.
// Must be called when the encoder is permanently retired (e.g., device shutdown).
func (e *CommandEncoder) Destroy() {
	if e.device != nil && e.pool != 0 {
		vkDestroyCommandPool(e.device.cmds, e.device.handle, e.pool, nil)
	}
	e.pool = 0
	e.active = 0
	e.free = nil
	e.discarded = nil
	e.results, e.usedResults = nil, 0
	e.device = nil
}

// TransitionBuffers transitions buffer states for synchronization.
func (e *CommandEncoder) TransitionBuffers(barriers []hal.BufferBarrier) {
	if e.active == 0 || len(barriers) == 0 {
		return
	}

	// Up to eight barriers, the usual case, stay on the stack:
	// vkCmdPipelineBarrier does not keep them.
	var stack [8]vk.BufferMemoryBarrier
	bufferBarriers := stack[:0]
	if len(barriers) > len(stack) {
		bufferBarriers = make([]vk.BufferMemoryBarrier, 0, len(barriers))
	}
	// The barrier waits for the stages the old usages ran in and holds back
	// the stages the new ones run in, not the whole pipeline.
	var srcStages, dstStages vk.PipelineStageFlags
	for _, b := range barriers {
		buf, ok := b.Buffer.(*Buffer)
		if !ok || buf.handle == 0 {
			hal.Logger().Warn("TransitionBuffers: skipping invalid buffer (nil or destroyed)")
			continue
		}

		srcAccess, srcStage := bufferUsageToAccessAndStage(b.Usage.OldUsage)
		dstAccess, dstStage := bufferUsageToAccessAndStage(b.Usage.NewUsage)
		srcStages |= srcStage
		dstStages |= dstStage

		bufferBarriers = append(bufferBarriers, vk.BufferMemoryBarrier{
			SType:               vk.StructureTypeBufferMemoryBarrier,
			SrcAccessMask:       srcAccess,
			DstAccessMask:       dstAccess,
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored,
			DstQueueFamilyIndex: vk.QueueFamilyIgnored,
			Buffer:              buf.handle,
			Offset:              0,
			Size:                vk.DeviceSize(vk.WholeSize),
		})
	}

	if len(bufferBarriers) == 0 {
		return
	}

	vkCmdPipelineBarrier(
		e.device.cmds,
		e.active,
		srcStages,
		barrierDstStages(dstStages),
		0,      // dependencyFlags
		0, nil, // memory barriers
		uint32(len(bufferBarriers)), &bufferBarriers[0],
		0, nil, // image barriers
	)
}

// barrierDstStages is the destination stage mask of a barrier whose new
// usages ran in stages: nothing known (TOP_OF_PIPE alone, from a zero usage)
// means every later command has to wait.
func barrierDstStages(stages vk.PipelineStageFlags) vk.PipelineStageFlags {
	if stages == 0 || stages == vk.PipelineStageFlags(vk.PipelineStageTopOfPipeBit) {
		return vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit)
	}
	return stages
}

// TransitionTextures transitions texture states for synchronization.
func (e *CommandEncoder) TransitionTextures(barriers []hal.TextureBarrier) {
	if e.active == 0 || len(barriers) == 0 {
		return
	}

	// Up to eight barriers, the usual case, stay on the stack:
	// vkCmdPipelineBarrier does not keep them.
	var stack [8]vk.ImageMemoryBarrier
	imageBarriers := stack[:0]
	if len(barriers) > len(stack) {
		imageBarriers = make([]vk.ImageMemoryBarrier, 0, len(barriers))
	}
	var srcStages, dstStages vk.PipelineStageFlags
	for _, b := range barriers {
		tex, ok := b.Texture.(*Texture)
		if !ok || tex.handle == 0 {
			hal.Logger().Warn("TransitionTextures: skipping invalid texture (nil or destroyed)")
			continue
		}

		srcAccess, srcStage, oldLayout := textureBarrierState(tex, textureUses(tex, b.Usage.Old, b.Usage.OldUsage))
		dstAccess, dstStage, newLayout := textureBarrierState(tex, textureUses(tex, b.Usage.New, b.Usage.NewUsage))
		srcStages |= srcStage
		dstStages |= dstStage

		imageBarriers = append(imageBarriers, vk.ImageMemoryBarrier{
			SType:               vk.StructureTypeImageMemoryBarrier,
			SrcAccessMask:       srcAccess,
			DstAccessMask:       dstAccess,
			OldLayout:           oldLayout,
			NewLayout:           newLayout,
			SrcQueueFamilyIndex: vk.QueueFamilyIgnored,
			DstQueueFamilyIndex: vk.QueueFamilyIgnored,
			Image:               tex.handle,
			SubresourceRange: vk.ImageSubresourceRange{
				AspectMask:     textureAspectToVk(b.Range.Aspect, tex.format),
				BaseMipLevel:   b.Range.BaseMipLevel,
				LevelCount:     mipLevelCountOrRemaining(b.Range.MipLevelCount),
				BaseArrayLayer: b.Range.BaseArrayLayer,
				LayerCount:     arrayLayerCountOrRemaining(b.Range.ArrayLayerCount),
			},
		})
	}

	if len(imageBarriers) == 0 {
		return
	}

	vkCmdPipelineBarrier(
		e.device.cmds,
		e.active,
		srcStages,
		barrierDstStages(dstStages),
		0,
		0, nil,
		0, nil,
		uint32(len(imageBarriers)), &imageBarriers[0],
	)
}

// ClearBuffer clears a buffer region to zero.
func (e *CommandEncoder) ClearBuffer(buffer hal.Buffer, offset, size uint64) {
	if e.active == 0 {
		return
	}

	buf, ok := buffer.(*Buffer)
	if !ok {
		return
	}

	// vkCmdFillBuffer fills with a 32-bit value (0 for zero fill)
	vkCmdFillBuffer(e.device.cmds, e.active, buf.handle, vk.DeviceSize(offset), vk.DeviceSize(size), 0)
}

// CopyBufferToBuffer copies data between buffers.
func (e *CommandEncoder) CopyBufferToBuffer(src, dst hal.Buffer, regions []hal.BufferCopy) {
	if e.active == 0 {
		return
	}

	srcBuf, srcOk := src.(*Buffer)
	dstBuf, dstOk := dst.(*Buffer)
	if !srcOk || !dstOk {
		return
	}

	vkRegions := make([]vk.BufferCopy, len(regions))
	for i, r := range regions {
		vkRegions[i] = vk.BufferCopy{
			SrcOffset: vk.DeviceSize(r.SrcOffset),
			DstOffset: vk.DeviceSize(r.DstOffset),
			Size:      vk.DeviceSize(r.Size),
		}
	}

	vkCmdCopyBuffer(e.device.cmds, e.active, srcBuf.handle, dstBuf.handle, uint32(len(vkRegions)), &vkRegions[0])
}

// convertBufferImageCopyRegions converts HAL BufferTextureCopy regions to Vulkan
// BufferImageCopy, into dst. The format parameter is the texture format, used
// to determine block copy size for correct bytes-to-texels conversion of
// bufferRowLength.
func convertBufferImageCopyRegions(dst []vk.BufferImageCopy, regions []hal.BufferTextureCopy, format gputypes.TextureFormat, dimension gputypes.TextureDimension) []vk.BufferImageCopy {
	vkRegions := dst[:0]
	if cap(vkRegions) < len(regions) {
		vkRegions = make([]vk.BufferImageCopy, 0, len(regions))
	}
	vkRegions = vkRegions[:len(regions)]
	blockSize := format.BlockCopySize()
	if blockSize == 0 {
		blockSize = 4
	}
	for i, r := range regions {
		// Vulkan bufferRowLength is in TEXELS, not bytes.
		// Convert from WebGPU's BytesPerRow (bytes) to Vulkan's bufferRowLength (texels)
		// using the format's known block size — NOT inference from BytesPerRow/Width,
		// which gives wrong results when BytesPerRow is padded to alignment.
		bufferRowLength := uint32(0)
		if r.BufferLayout.BytesPerRow > 0 {
			bufferRowLength = r.BufferLayout.BytesPerRow / blockSize
		}
		baseArrayLayer := uint32(0)
		layerCount := uint32(1)
		imageOffsetZ := int32(r.TextureBase.Origin.Z)
		imageExtentDepth := r.Size.DepthOrArrayLayers
		if dimension != gputypes.TextureDimension3D {
			baseArrayLayer = r.TextureBase.Origin.Z
			layerCount = r.Size.DepthOrArrayLayers
			if layerCount == 0 {
				layerCount = 1
			}
			imageOffsetZ = 0
			imageExtentDepth = 1
		}

		vkRegions[i] = vk.BufferImageCopy{
			BufferOffset:      vk.DeviceSize(r.BufferLayout.Offset),
			BufferRowLength:   bufferRowLength,
			BufferImageHeight: r.BufferLayout.RowsPerImage,
			ImageSubresource: vk.ImageSubresourceLayers{
				AspectMask:     textureAspectToVkSimple(r.TextureBase.Aspect),
				MipLevel:       r.TextureBase.MipLevel,
				BaseArrayLayer: baseArrayLayer,
				LayerCount:     layerCount,
			},
			ImageOffset: vk.Offset3D{
				X: int32(r.TextureBase.Origin.X),
				Y: int32(r.TextureBase.Origin.Y),
				Z: imageOffsetZ,
			},
			ImageExtent: vk.Extent3D{
				Width:  r.Size.Width,
				Height: r.Size.Height,
				Depth:  imageExtentDepth,
			},
		}
	}
	return vkRegions
}

// CopyBufferToTexture copies data from a buffer to a texture.
func (e *CommandEncoder) CopyBufferToTexture(src hal.Buffer, dst hal.Texture, regions []hal.BufferTextureCopy) {
	if e.active == 0 {
		return
	}

	srcBuf, srcOk := src.(*Buffer)
	dstTex, dstOk := dst.(*Texture)
	if !srcOk || !dstOk || len(regions) == 0 {
		return
	}

	// The regions stay on the stack for the usual one-region copy: the
	// command does not keep them.
	var stack [4]vk.BufferImageCopy
	vkRegions := convertBufferImageCopyRegions(stack[:0], regions, dstTex.format, dstTex.dimension)
	vkCmdCopyBufferToImage(
		e.device.cmds,
		e.active,
		srcBuf.handle,
		dstTex.handle,
		vk.ImageLayoutTransferDstOptimal,
		uint32(len(vkRegions)),
		&vkRegions[0],
	)
}

// CopyTextureToBuffer copies data from a texture to a buffer.
func (e *CommandEncoder) CopyTextureToBuffer(src hal.Texture, dst hal.Buffer, regions []hal.BufferTextureCopy) {
	if e.active == 0 {
		return
	}

	srcTex, srcOk := src.(*Texture)
	dstBuf, dstOk := dst.(*Buffer)
	if !srcOk || !dstOk || len(regions) == 0 {
		return
	}

	var stack [4]vk.BufferImageCopy
	vkRegions := convertBufferImageCopyRegions(stack[:0], regions, srcTex.format, srcTex.dimension)
	vkCmdCopyImageToBuffer(
		e.device.cmds,
		e.active,
		srcTex.handle,
		vk.ImageLayoutTransferSrcOptimal,
		dstBuf.handle,
		uint32(len(vkRegions)),
		&vkRegions[0],
	)
}

// CopyTextureToTexture copies data between textures.
func (e *CommandEncoder) CopyTextureToTexture(src, dst hal.Texture, regions []hal.TextureCopy) {
	if e.active == 0 {
		return
	}

	srcTex, srcOk := src.(*Texture)
	dstTex, dstOk := dst.(*Texture)
	if !srcOk || !dstOk || len(regions) == 0 {
		return
	}

	vkRegions := make([]vk.ImageCopy, len(regions))
	for i, r := range regions {
		vkRegions[i] = vk.ImageCopy{
			SrcSubresource: vk.ImageSubresourceLayers{
				AspectMask:     textureAspectToVk(r.SrcBase.Aspect, srcTex.format),
				MipLevel:       r.SrcBase.MipLevel,
				BaseArrayLayer: 0,
				LayerCount:     1,
			},
			SrcOffset: vk.Offset3D{
				X: int32(r.SrcBase.Origin.X),
				Y: int32(r.SrcBase.Origin.Y),
				Z: int32(r.SrcBase.Origin.Z),
			},
			DstSubresource: vk.ImageSubresourceLayers{
				AspectMask:     textureAspectToVk(r.DstBase.Aspect, dstTex.format),
				MipLevel:       r.DstBase.MipLevel,
				BaseArrayLayer: 0,
				LayerCount:     1,
			},
			DstOffset: vk.Offset3D{
				X: int32(r.DstBase.Origin.X),
				Y: int32(r.DstBase.Origin.Y),
				Z: int32(r.DstBase.Origin.Z),
			},
			Extent: vk.Extent3D{
				Width:  r.Size.Width,
				Height: r.Size.Height,
				Depth:  r.Size.DepthOrArrayLayers,
			},
		}
	}

	vkCmdCopyImage(
		e.device.cmds,
		e.active,
		srcTex.handle,
		vk.ImageLayoutTransferSrcOptimal,
		dstTex.handle,
		vk.ImageLayoutTransferDstOptimal,
		uint32(len(vkRegions)),
		&vkRegions[0],
	)
}

// ResolveQuerySet copies query results from a query set into a destination buffer.
// For timestamp queries, each result is a uint64 (8 bytes).
// This uses vkCmdCopyQueryPoolResults under the hood.
func (e *CommandEncoder) ResolveQuerySet(querySet hal.QuerySet, firstQuery, queryCount uint32, destination hal.Buffer, destinationOffset uint64) {
	qs, ok := querySet.(*QuerySet)
	if !ok || qs.pool == 0 || e.active == 0 {
		return
	}
	buf, ok := destination.(*Buffer)
	if !ok || buf.handle == 0 {
		return
	}

	// Pipeline barrier: ensure timestamps are written before copy.
	memBarrier := vk.MemoryBarrier{
		SType:         vk.StructureTypeMemoryBarrier,
		SrcAccessMask: vk.AccessFlags(vk.AccessTransferWriteBit),
		DstAccessMask: vk.AccessFlags(vk.AccessTransferReadBit),
	}
	vkCmdPipelineBarrier(
		e.device.cmds,
		e.active,
		vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit),
		vk.PipelineStageFlags(vk.PipelineStageTransferBit),
		0,
		1, &memBarrier,
		0, nil,
		0, nil,
	)

	// Use vkCmdCopyQueryPoolResults to copy timestamp values to the buffer.
	// Stride is 8 bytes per timestamp (uint64).
	// Flags: VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT.
	vkCmdCopyQueryPoolResults(
		e.device.cmds,
		e.active,
		qs.pool,
		firstQuery,
		queryCount,
		buf.handle,
		destinationOffset,
		8, // stride: sizeof(uint64)
		vk.QueryResultFlags(vk.QueryResult64Bit|vk.QueryResultWaitBit),
	)
}

// renderAreaView returns the attachment view that defines a pass's render
// area and sample count: the first color attachment carrying one, or the
// depth/stencil attachment when the pass declares no usable color attachment
// at all.
//
// A pass with no color attachments is legal in WebGPU -- it is how a shadow map
// is drawn -- and Vulkan still needs an extent and a sample count to build the
// framebuffer, which only the attachments can supply.
//
// Returns nil when nothing resolves to a view of this backend, which means
// there is nothing to render into and no pass worth beginning.
//
// Rust wgpu-hal has no equivalent: its RenderPassDescriptor carries extent
// and sample_count as fields the caller supplies, so begin_render_pass reads
// them straight off the descriptor (wgpu-hal/src/vulkan/command.rs). This
// HAL's descriptor carries neither, so a pass has to derive both from its
// attachments -- and a depth-only pass has only the depth attachment to
// derive them from.
func renderAreaView(desc *hal.RenderPassDescriptor) *TextureView {
	if desc == nil {
		return nil
	}
	for _, ca := range desc.ColorAttachments {
		if view := asTextureView(ca.View); view != nil {
			return view
		}
	}
	if desc.DepthStencilAttachment != nil {
		return asTextureView(desc.DepthStencilAttachment.View)
	}
	return nil
}

// asTextureView narrows a HAL view to this backend's, treating an absent view,
// a typed-nil pointer and a view belonging to another backend alike: each comes
// back as nil. The comma-ok assertion is what makes that true -- a failed
// assertion yields a nil *TextureView -- so callers get a pointer they can
// compare against nil rather than one they must not dereference.
func asTextureView(view hal.TextureView) *TextureView {
	v, _ := view.(*TextureView)
	return v
}

// attachmentSampleCount reports the sample count a pass's attachments share.
// Swapchain views carry no texture and are always single-sampled.
func attachmentSampleCount(view *TextureView) vk.SampleCountFlagBits {
	if view != nil && view.texture != nil && view.texture.samples > 1 {
		return vk.SampleCountFlagBits(view.texture.samples)
	}
	return vk.SampleCountFlagBits(1)
}

// BeginRenderPass begins a render pass using VkRenderPass (classic Vulkan approach).
// This is compatible with Intel drivers that don't properly support dynamic rendering.
// Supports up to MaxColorAttachments (8) color attachments with optional MSAA resolve,
// plus an optional depth/stencil attachment. Uses sync.Pool for RenderPassEncoder reuse.
//
// Reference: Rust wgpu-hal begin_render_pass (vulkan/command.rs:803-933).
func (e *CommandEncoder) BeginRenderPass(desc *hal.RenderPassDescriptor) hal.RenderPassEncoder {
	rpe := renderPassPool.Get().(*RenderPassEncoder)
	rpe.encoder = e
	rpe.desc = desc
	rpe.pipeline = nil
	rpe.indexFormat = 0
	rpe.renderPass = 0
	rpe.framebuffer = 0
	rpe.begun = false

	if e.active == 0 {
		return rpe
	}

	// Render area and sample count both come from the attachment that defines
	// the pass -- the first color attachment, or the depth attachment when the
	// pass has no color attachments at all. Every early return below leaves
	// rpe.begun false, so End knows not to end a pass that was never begun.
	areaView := renderAreaView(desc)
	if areaView == nil {
		return rpe
	}
	renderWidth, renderHeight := areaView.size.Width, areaView.size.Height
	sampleCount := attachmentSampleCount(areaView)

	// Build render pass key and framebuffer key from ALL color attachments.
	rpKey := RenderPassKey{
		ColorCount:  len(desc.ColorAttachments),
		SampleCount: sampleCount,
	}
	fbKey := FramebufferKey{
		Width:  renderWidth,
		Height: renderHeight,
	}

	// Clear values array: one per VkAttachment (color + resolve + depth).
	// Using a fixed-size array avoids heap allocation on this per-frame path.
	var clearValuesArr [hal.MaxTotalAttachments]vk.ClearValue
	clearValues := clearValuesArr[:0]

	for i, ca := range desc.ColorAttachments {
		if i >= hal.MaxColorAttachments {
			break
		}

		view, ok := ca.View.(*TextureView)
		if !ok || view == nil {
			// Absent slot — leave the key entry zero-valued (FormatUndefined).
			continue
		}

		// Determine color format from the view.
		var colorFormat vk.Format
		if view.texture != nil {
			colorFormat = textureFormatToVk(view.texture.format)
		} else if view.isSwapchain {
			colorFormat = view.vkFormat
		}

		// Check for MSAA resolve target.
		var resolveView *TextureView
		if ca.ResolveTarget != nil {
			resolveView, _ = ca.ResolveTarget.(*TextureView)
		}
		hasMSAAResolve := resolveView != nil && sampleCount > vk.SampleCountFlagBits(1)

		// Determine the final layout for the "output" attachment.
		// BUG-WGPU-VK-007: offscreen textures with TextureBinding must end in
		// ImageLayoutGeneral to prevent Intel CCS stale-pixel artifacts.
		colorFinalLayout := vk.ImageLayoutColorAttachmentOptimal
		if !view.isSwapchain {
			colorFinalLayout = offscreenFinalLayout(view)
		}
		if hasMSAAResolve {
			if resolveView.isSwapchain {
				colorFinalLayout = vk.ImageLayoutColorAttachmentOptimal
			} else {
				colorFinalLayout = offscreenFinalLayout(resolveView)
			}
		}

		// ADR-060: Inline present barrier for swapchain targets.
		// Only the FIRST swapchain attachment drives the barrier; additional
		// swapchain attachments are unusual and would need separate tracking.
		if e.swapchainImage == 0 {
			e.setupInlinePresentBarrier(view, resolveView, hasMSAAResolve, colorFinalLayout, ca.LoadOp)
		}

		// BUG-WGPU-VK-006: Update swapchain image layout tracking.
		updateSwapchainLayout(view, resolveView, hasMSAAResolve, colorFinalLayout)

		rpKey.Colors[i] = ColorAttachmentKeyEntry{
			Format:      colorFormat,
			LoadOp:      loadOpToVk(ca.LoadOp),
			StoreOp:     storeOpToVk(ca.StoreOp),
			FinalLayout: colorFinalLayout,
			HasResolve:  hasMSAAResolve,
		}

		// Framebuffer: color view
		fbKey.Views[fbKey.ViewCount] = view.handle
		fbKey.ViewCount++

		// Clear value for color attachment
		clearValues = append(clearValues, vk.ClearValueColor(
			float32(ca.ClearValue.R),
			float32(ca.ClearValue.G),
			float32(ca.ClearValue.B),
			float32(ca.ClearValue.A),
		))

		if hasMSAAResolve {
			// Framebuffer: resolve view (immediately after color)
			fbKey.Views[fbKey.ViewCount] = resolveView.handle
			fbKey.ViewCount++

			// Clear value for resolve attachment.
			clearValues = append(clearValues, vk.ClearValueColor(
				float32(ca.ClearValue.R),
				float32(ca.ClearValue.G),
				float32(ca.ClearValue.B),
				float32(ca.ClearValue.A),
			))
		}
	}

	// Handle depth/stencil attachment
	if desc.DepthStencilAttachment != nil {
		dsa := desc.DepthStencilAttachment
		if dsView, ok := dsa.View.(*TextureView); ok && dsView.texture != nil {
			rpKey.DepthFormat = textureFormatToVk(dsView.texture.format)
			rpKey.DepthReadOnly = dsa.DepthReadOnly && dsa.StencilReadOnly
			rpKey.DepthLoadOp = loadOpToVk(dsa.DepthLoadOp)
			rpKey.DepthStoreOp = storeOpToVk(dsa.DepthStoreOp)
			rpKey.StencilLoadOp = loadOpToVk(dsa.StencilLoadOp)
			rpKey.StencilStoreOp = storeOpToVk(dsa.StencilStoreOp)

			fbKey.Views[fbKey.ViewCount] = dsView.handle
			fbKey.ViewCount++
		}
		clearValues = append(clearValues, vk.ClearValueDepthStencil(dsa.DepthClearValue, dsa.StencilClearValue))
	}

	// Get or create render pass from cache
	cache := e.device.GetRenderPassCache()
	renderPass, err := cache.GetOrCreateRenderPass(rpKey)
	if err != nil {
		return rpe
	}
	rpe.renderPass = renderPass

	fbKey.RenderPass = renderPass

	// Get or create framebuffer from cache
	framebuffer, err := cache.GetOrCreateFramebuffer(fbKey)
	if err != nil {
		return rpe
	}
	rpe.framebuffer = framebuffer

	// Begin render pass
	renderPassBegin := vk.RenderPassBeginInfo{
		SType:       vk.StructureTypeRenderPassBeginInfo,
		RenderPass:  renderPass,
		Framebuffer: framebuffer,
		RenderArea: vk.Rect2D{
			Offset: vk.Offset2D{X: 0, Y: 0},
			Extent: vk.Extent2D{Width: renderWidth, Height: renderHeight},
		},
		ClearValueCount: uint32(len(clearValues)),
	}
	if len(clearValues) > 0 {
		renderPassBegin.PClearValues = &clearValues[0]
	}

	vkCmdBeginRenderPass(e.device.cmds, e.active, &renderPassBegin, vk.SubpassContentsInline)
	runtime.KeepAlive(clearValues)
	rpe.begun = true

	// Set default viewport and scissor for the render area.
	// Always set viewport/scissor — the pipeline declares them as dynamic state,
	// so they must be initialized before any draw call regardless of dimensions.
	viewW := max(float32(renderWidth), 1.0)
	viewH := max(float32(renderHeight), 1.0)

	// Y-flip for WebGPU compatibility: Vulkan Y points down, WebGPU Y points up.
	viewport := vk.Viewport{
		X:        0,
		Y:        viewH, // Start at bottom
		Width:    viewW,
		Height:   -viewH, // Negative height for Y-flip
		MinDepth: 0.0,
		MaxDepth: 1.0,
	}
	vkCmdSetViewport(e.device.cmds, e.active, 0, 1, &viewport)

	scissor := vk.Rect2D{
		Offset: vk.Offset2D{X: 0, Y: 0},
		Extent: vk.Extent2D{Width: max(renderWidth, 1), Height: max(renderHeight, 1)},
	}
	vkCmdSetScissor(e.device.cmds, e.active, 0, 1, &scissor)

	// Set default blend constants and stencil reference.
	// All pipelines declare these as dynamic state (matching Rust wgpu),
	// so they must be initialized before any draw call (VK-PIPE-001).
	vkCmdSetBlendConstants(e.device.cmds, e.active, &[4]float32{0, 0, 0, 0})
	vkCmdSetStencilReference(e.device.cmds, e.active,
		vk.StencilFaceFlags(vk.StencilFaceFrontAndBack), 0)

	return rpe
}

// BeginComputePass begins a compute pass.
// Uses sync.Pool for ComputePassEncoder reuse (VK-PERF-005).
func (e *CommandEncoder) BeginComputePass(desc *hal.ComputePassDescriptor) hal.ComputePassEncoder {
	cpe := computePassPool.Get().(*ComputePassEncoder)
	cpe.encoder = e
	cpe.pipeline = nil
	cpe.timestampWrites = nil

	// Write beginning-of-pass timestamp if requested.
	// active != 0 check prevents SIGSEGV from null dispatch table
	// dereference (VK-001, gogpu#119).
	if desc != nil && desc.TimestampWrites != nil {
		cpe.timestampWrites = desc.TimestampWrites
		if qs, ok := desc.TimestampWrites.QuerySet.(*QuerySet); ok && qs.pool != 0 {
			if desc.TimestampWrites.BeginningOfPassWriteIndex != nil && e.active != 0 {
				idx := *desc.TimestampWrites.BeginningOfPassWriteIndex
				e.device.cmds.CmdResetQueryPool(e.active, qs.pool, idx, 1)
				e.device.cmds.CmdWriteTimestamp(
					e.active,
					vk.PipelineStageTopOfPipeBit,
					qs.pool,
					idx,
				)
			}
		}
	}

	return cpe
}

// RenderPassEncoder implements hal.RenderPassEncoder for Vulkan.
type RenderPassEncoder struct {
	encoder     *CommandEncoder
	desc        *hal.RenderPassDescriptor
	pipeline    *RenderPipeline
	indexFormat gputypes.IndexFormat
	// For VkRenderPass-based rendering (not dynamic rendering)
	renderPass  vk.RenderPass
	framebuffer vk.Framebuffer
	// begun records whether vkCmdBeginRenderPass actually ran, which End needs
	// because BeginRenderPass returns an encoder either way.
	begun bool
	// pending holds the bind groups set before any pipeline, which WebGPU
	// allows and Vulkan cannot bind without a pipeline layout: SetPipeline
	// binds them.
	pending [maxBindGroupSlots]pendingBindGroup
}

// maxBindGroupSlots is how many bind group slots a pass has (WebGPU's
// maxBindGroups).
const maxBindGroupSlots = 8

// pendingBindGroup is a bind group set before the pipeline.
type pendingBindGroup struct {
	group   *BindGroup
	offsets []uint32
}

const (
	drawIndirectStride        = uint32(16)
	drawIndexedIndirectStride = uint32(20)
	indexedIndirectStride     = drawIndexedIndirectStride
)

// End finishes the render pass.
// Returns the encoder to the pool for reuse (VK-PERF-006).
func (e *RenderPassEncoder) End() {
	// BeginRenderPass returns an encoder even when it declined to begin a pass:
	// an inactive command buffer, attachments that do not resolve, or a render
	// pass or framebuffer that could not be created. Ending a pass that was
	// never begun is undefined behavior -- the driver faults inside
	// vkCmdEndRenderPass rather than reporting a validation error.
	//
	// Use vkCmdEndRenderPass (VkRenderPass handles layout transitions automatically
	// via FinalLayout in AttachmentDescription)
	if e.begun && e.encoder.active != 0 {
		vkCmdEndRenderPass(e.encoder.device.cmds, e.encoder.active)
	}

	// Return to pool for reuse.
	e.encoder = nil
	e.desc = nil
	e.pipeline = nil
	e.renderPass = 0
	e.framebuffer = 0
	e.begun = false
	e.pending = [maxBindGroupSlots]pendingBindGroup{}
	renderPassPool.Put(e)
}

// SetPipeline sets the render pipeline, and binds the bind groups set
// before it.
func (e *RenderPassEncoder) SetPipeline(pipeline hal.RenderPipeline) {
	p, ok := pipeline.(*RenderPipeline)
	if !ok || e.encoder.active == 0 {
		return
	}
	e.pipeline = p
	vkCmdBindPipeline(e.encoder.device.cmds, e.encoder.active, vk.PipelineBindPointGraphics, p.handle)
	for i := range e.pending {
		if pending := e.pending[i]; pending.group != nil {
			e.pending[i] = pendingBindGroup{}
			e.SetBindGroup(uint32(i), pending.group, pending.offsets) //nolint:gosec // slot index
		}
	}
}

// SetBindGroup sets a bind group. Before a pipeline is set there is no
// layout to bind it with, so it waits for SetPipeline.
func (e *RenderPassEncoder) SetBindGroup(index uint32, group hal.BindGroup, offsets []uint32) {
	bg, ok := group.(*BindGroup)
	if !ok || e.encoder.active == 0 {
		return
	}
	if e.pipeline == nil {
		if index < maxBindGroupSlots {
			e.pending[index] = pendingBindGroup{group: bg, offsets: append([]uint32(nil), offsets...)}
		}
		return
	}

	var pOffsets *uint32
	if len(offsets) > 0 {
		pOffsets = &offsets[0]
	}

	vkCmdBindDescriptorSets(
		e.encoder.device.cmds,
		e.encoder.active,
		vk.PipelineBindPointGraphics,
		e.pipeline.layout,
		index,
		1,
		&bg.handle,
		uint32(len(offsets)),
		pOffsets,
	)
}

// SetVertexBuffer sets a vertex buffer.
// Uses stack variables instead of slice allocations (VK-PERF-007).
func (e *RenderPassEncoder) SetVertexBuffer(slot uint32, buffer hal.Buffer, offset uint64) {
	buf, ok := buffer.(*Buffer)
	if !ok || e.encoder.active == 0 {
		return
	}

	// Stack-allocated single values avoid heap allocation (VK-PERF-007).
	vkOffset := vk.DeviceSize(offset)
	vkBuffer := buf.handle

	vkCmdBindVertexBuffers(e.encoder.device.cmds, e.encoder.active, slot, 1, &vkBuffer, &vkOffset)
}

// SetIndexBuffer sets the index buffer.
func (e *RenderPassEncoder) SetIndexBuffer(buffer hal.Buffer, format gputypes.IndexFormat, offset uint64) {
	buf, ok := buffer.(*Buffer)
	if !ok || e.encoder.active == 0 {
		return
	}

	e.indexFormat = format
	indexType := vk.IndexTypeUint16
	if format == gputypes.IndexFormatUint32 {
		indexType = vk.IndexTypeUint32
	}

	vkCmdBindIndexBuffer(e.encoder.device.cmds, e.encoder.active, buf.handle, vk.DeviceSize(offset), indexType)
}

// SetViewport sets the viewport.
// NOTE: Applies Y-flip for WebGPU/OpenGL coordinate system compatibility (matches Rust wgpu).
func (e *RenderPassEncoder) SetViewport(vp gputypes.Viewport) {
	if e.encoder.active == 0 {
		return
	}

	// Y-flip: Start Y at y+height, use negative height
	viewport := vk.Viewport{
		X:        vp.X,
		Y:        vp.Y + vp.Height, // Y-flip: start at bottom
		Width:    vp.Width,
		Height:   -vp.Height, // Y-flip: negative height
		MinDepth: vp.MinDepth,
		MaxDepth: vp.MaxDepth,
	}

	vkCmdSetViewport(e.encoder.device.cmds, e.encoder.active, 0, 1, &viewport)
}

// SetScissorRect sets the scissor rectangle.
func (e *RenderPassEncoder) SetScissorRect(rect gputypes.ScissorRect) {
	if e.encoder.active == 0 {
		return
	}

	scissor := vk.Rect2D{
		Offset: vk.Offset2D{X: int32(rect.X), Y: int32(rect.Y)},
		Extent: vk.Extent2D{Width: rect.Width, Height: rect.Height},
	}

	vkCmdSetScissor(e.encoder.device.cmds, e.encoder.active, 0, 1, &scissor)
}

// SetBlendConstant sets the blend constant.
func (e *RenderPassEncoder) SetBlendConstant(color *gputypes.Color) {
	if e.encoder.active == 0 || color == nil {
		return
	}

	blendConstants := [4]float32{
		float32(color.R),
		float32(color.G),
		float32(color.B),
		float32(color.A),
	}

	vkCmdSetBlendConstants(e.encoder.device.cmds, e.encoder.active, &blendConstants)
}

// SetStencilReference sets the stencil reference value.
func (e *RenderPassEncoder) SetStencilReference(ref uint32) {
	if e.encoder.active == 0 {
		return
	}

	// Set for both front and back faces
	vkCmdSetStencilReference(e.encoder.device.cmds, e.encoder.active, vk.StencilFaceFlags(vk.StencilFaceFrontAndBack), ref)
}

// Draw draws primitives.
func (e *RenderPassEncoder) Draw(args gputypes.DrawArgs) {
	if e.encoder.active == 0 {
		return
	}
	vkCmdDraw(e.encoder.device.cmds, e.encoder.active, args.VertexCount, args.InstanceCount, args.FirstVertex, args.FirstInstance)
}

// DrawIndexed draws indexed primitives.
func (e *RenderPassEncoder) DrawIndexed(args gputypes.DrawIndexedArgs) {
	if e.encoder.active == 0 {
		return
	}

	vkCmdDrawIndexed(e.encoder.device.cmds, e.encoder.active, args.IndexCount, args.InstanceCount, args.FirstIndex, args.BaseVertex, args.FirstInstance)
}

// DrawIndirect draws primitives with GPU-generated parameters.
func (e *RenderPassEncoder) DrawIndirect(buffer hal.Buffer, offset uint64, drawCount uint32) {
	buf, ok := buffer.(*Buffer)
	if !ok || e.encoder.active == 0 || drawCount == 0 {
		return
	}
	if !indirect.RangeFits(buf.size, offset, uint64(drawIndirectStride), drawCount) {
		return
	}
	call, batched, ok := indirectCallPlan(e.encoder.device.supportsMultiDrawIndirect,
		e.encoder.device.maxDrawIndirectCount, offset, drawCount, drawIndirectStride)
	if !ok {
		return
	}
	if batched {
		vkCmdDrawIndirect(e.encoder.device.cmds, e.encoder.active, buf.handle, vk.DeviceSize(call.offset), call.count, call.stride)
		return
	}
	for i := uint32(0); i < drawCount; i++ {
		recordOffset, _ := indirect.RecordOffset(offset, uint64(drawIndirectStride), i)
		vkCmdDrawIndirect(e.encoder.device.cmds, e.encoder.active, buf.handle, vk.DeviceSize(recordOffset), 1, drawIndirectStride)
	}
}

// DrawIndexedIndirect draws indexed primitives with GPU-generated parameters.
// Vulkan can encode a span in one command only when the optional
// multiDrawIndirect feature is enabled and the count fits the device limit;
// otherwise emit the exact single-record loop.
func (e *RenderPassEncoder) DrawIndexedIndirect(buffer hal.Buffer, offset uint64, drawCount uint32) {
	buf, ok := buffer.(*Buffer)
	if !ok || e.encoder.active == 0 || drawCount == 0 {
		return
	}
	if !indirect.RangeFits(buf.size, offset, uint64(drawIndexedIndirectStride), drawCount) {
		return
	}
	call, batched, ok := indirectCallPlan(
		e.encoder.device.supportsMultiDrawIndirect,
		e.encoder.device.maxDrawIndirectCount,
		offset,
		drawCount,
		drawIndexedIndirectStride,
	)
	if !ok {
		return
	}

	if batched {
		vkCmdDrawIndexedIndirect(e.encoder.device.cmds, e.encoder.active, buf.handle, vk.DeviceSize(call.offset), call.count, call.stride)
		return
	}
	for i := uint32(0); i < drawCount; i++ {
		recordOffset, ok := indirect.RecordOffset(offset, uint64(drawIndexedIndirectStride), i)
		if !ok {
			return
		}
		vkCmdDrawIndexedIndirect(e.encoder.device.cmds, e.encoder.active, buf.handle, vk.DeviceSize(recordOffset), call.count, call.stride)
	}
}

// DrawIndirectCount draws primitives using a GPU-provided draw count.
func (e *RenderPassEncoder) DrawIndirectCount(
	buffer hal.Buffer, offset uint64,
	countBuffer hal.Buffer, countOffset uint64,
	maxDrawCount uint32,
) {
	var native vulkanIndirectCountNativeCmd
	if dev := e.encoder.device; dev.supportsDrawIndirectCount && dev.cmds.HasDrawIndirectCount() {
		native = func(cmdBuf vk.CommandBuffer, indirect vk.Buffer, off vk.DeviceSize, count vk.Buffer, countOff vk.DeviceSize, maxCount uint32) {
			dev.cmds.CmdDrawIndirectCount(cmdBuf, indirect, off, count, countOff, maxCount, drawIndirectStride)
		}
	}
	e.drawIndirectCount(buffer, offset, countBuffer, countOffset, maxDrawCount, drawIndirectStride, native, vkCmdDrawIndirect)
}

// DrawIndexedIndirectCount draws indexed primitives using a GPU-provided draw count.
func (e *RenderPassEncoder) DrawIndexedIndirectCount(
	buffer hal.Buffer, offset uint64,
	countBuffer hal.Buffer, countOffset uint64,
	maxDrawCount uint32,
) {
	var native vulkanIndirectCountNativeCmd
	if dev := e.encoder.device; dev.supportsDrawIndirectCount && dev.cmds.HasDrawIndirectCount() {
		native = func(cmdBuf vk.CommandBuffer, indirect vk.Buffer, off vk.DeviceSize, count vk.Buffer, countOff vk.DeviceSize, maxCount uint32) {
			dev.cmds.CmdDrawIndexedIndirectCount(cmdBuf, indirect, off, count, countOff, maxCount, drawIndexedIndirectStride)
		}
	}
	e.drawIndirectCount(buffer, offset, countBuffer, countOffset, maxDrawCount, drawIndexedIndirectStride, native, vkCmdDrawIndexedIndirect)
}

type indexedIndirectCall struct {
	offset uint64
	count  uint32
	stride uint32
}

// indexedIndirectCallPlan returns the first native call shape for an indexed
// indirect draw. The plan is pure so count/stride policy can be tested without
// constructing a Vulkan command buffer or invoking FFI.
func indirectCallPlan(supportsMultiDraw bool, maxDrawCount uint32, offset uint64, drawCount, stride uint32) (indexedIndirectCall, bool, bool) {
	if drawCount == 0 {
		return indexedIndirectCall{}, false, false
	}
	if _, ok := indirect.RecordOffset(offset, uint64(stride), drawCount-1); !ok {
		return indexedIndirectCall{}, false, false
	}
	if supportsMultiDraw && drawCount <= maxDrawCount {
		return indexedIndirectCall{offset: offset, count: drawCount, stride: stride}, true, true
	}
	return indexedIndirectCall{offset: offset, count: 1, stride: stride}, false, true
}

func indexedIndirectCallPlan(supportsMultiDraw bool, maxDrawCount uint32, offset uint64, drawCount uint32) (indexedIndirectCall, bool, bool) {
	return indirectCallPlan(supportsMultiDraw, maxDrawCount, offset, drawCount, drawIndexedIndirectStride)
}

func indexedIndirectRecordOffset(offset uint64, index uint32) (uint64, bool) {
	return indirect.RecordOffset(offset, uint64(drawIndexedIndirectStride), index)
}

// ExecuteBundle executes a pre-recorded render bundle.
func (e *RenderPassEncoder) ExecuteBundle(bundle hal.RenderBundle) {
	vkBundle, ok := bundle.(*RenderBundle)
	if !ok || vkBundle == nil || e.encoder.active == 0 {
		return
	}

	// Execute the secondary command buffer
	e.encoder.device.cmds.CmdExecuteCommands(
		e.encoder.active,
		1,
		&vkBundle.commandBuffer,
	)
}

// ComputePassEncoder implements hal.ComputePassEncoder for Vulkan.
type ComputePassEncoder struct {
	encoder         *CommandEncoder
	pipeline        *ComputePipeline
	timestampWrites *hal.ComputePassTimestampWrites
}

// End finishes the compute pass.
// Writes end-of-pass timestamp if requested, then inserts a global memory
// barrier so compute shader writes are visible to subsequent commands
// (transfers, other dispatches, etc.). Without this barrier the GPU may
// reorder a CopyBufferToBuffer before the compute shader has finished
// writing, causing stale/zero reads.
// Returns the encoder to the pool for reuse (VK-PERF-005).
func (e *ComputePassEncoder) End() {
	// VK-001: Defense-in-depth — active == 0 check prevents SIGSEGV from
	// null dispatch table dereference (gogpu#119).
	if e.encoder == nil || e.encoder.active == 0 {
		return
	}

	// Write end-of-pass timestamp if requested.
	if e.timestampWrites != nil {
		if qs, ok := e.timestampWrites.QuerySet.(*QuerySet); ok && qs.pool != 0 {
			if e.timestampWrites.EndOfPassWriteIndex != nil {
				idx := *e.timestampWrites.EndOfPassWriteIndex
				e.encoder.device.cmds.CmdResetQueryPool(e.encoder.active, qs.pool, idx, 1)
				e.encoder.device.cmds.CmdWriteTimestamp(
					e.encoder.active,
					vk.PipelineStageBottomOfPipeBit,
					qs.pool,
					idx,
				)
			}
		}
	}

	// Global memory barrier: compute writes → everything after.
	memBarrier := vk.MemoryBarrier{
		SType:         vk.StructureTypeMemoryBarrier,
		SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit),
		DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessTransferReadBit | vk.AccessTransferWriteBit | vk.AccessHostReadBit),
	}
	vkCmdPipelineBarrier(
		e.encoder.device.cmds,
		e.encoder.active,
		vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit),
		vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit|vk.PipelineStageTransferBit|vk.PipelineStageHostBit),
		0,
		1, &memBarrier,
		0, nil,
		0, nil,
	)

	// Return to pool for reuse.
	e.encoder = nil
	e.pipeline = nil
	e.timestampWrites = nil
	computePassPool.Put(e)
}

// SetPipeline sets the compute pipeline.
func (e *ComputePassEncoder) SetPipeline(pipeline hal.ComputePipeline) {
	p, ok := pipeline.(*ComputePipeline)
	if !ok || e.encoder.active == 0 {
		return
	}
	e.pipeline = p

	vkCmdBindPipeline(e.encoder.device.cmds, e.encoder.active, vk.PipelineBindPointCompute, p.handle)
}

// SetBindGroup sets a bind group.
func (e *ComputePassEncoder) SetBindGroup(index uint32, group hal.BindGroup, offsets []uint32) {
	bg, ok := group.(*BindGroup)
	if !ok || e.encoder.active == 0 || e.pipeline == nil {
		return
	}

	var pOffsets *uint32
	if len(offsets) > 0 {
		pOffsets = &offsets[0]
	}

	vkCmdBindDescriptorSets(
		e.encoder.device.cmds,
		e.encoder.active,
		vk.PipelineBindPointCompute,
		e.pipeline.layout,
		index,
		1,
		&bg.handle,
		uint32(len(offsets)),
		pOffsets,
	)
}

// Dispatch dispatches compute work. Without a pipeline bound there is
// nothing to run and the driver faults, so the dispatch is dropped.
func (e *ComputePassEncoder) Dispatch(x, y, z uint32) {
	if e.encoder.active == 0 || e.pipeline == nil {
		return
	}

	vkCmdDispatch(e.encoder.device.cmds, e.encoder.active, x, y, z)
	e.insertComputeBarrier()
}

// DispatchIndirect dispatches compute work with GPU-generated parameters.
func (e *ComputePassEncoder) DispatchIndirect(buffer hal.Buffer, offset uint64) {
	buf, ok := buffer.(*Buffer)
	if !ok || e.encoder.active == 0 || e.pipeline == nil {
		return
	}

	vkCmdDispatchIndirect(e.encoder.device.cmds, e.encoder.active, buf.handle, vk.DeviceSize(offset))
	e.insertComputeBarrier()
}

// insertComputeBarrier inserts a compute→compute memory barrier after a dispatch.
// VAL-008: ensures storage buffer writes from one dispatch are visible to subsequent
// dispatches. This is the "global barrier" approach (Option B) — always correct,
// slightly over-synchronizes but avoids the complexity of per-resource usage tracking.
func (e *ComputePassEncoder) insertComputeBarrier() {
	memBarrier := vk.MemoryBarrier{
		SType:         vk.StructureTypeMemoryBarrier,
		SrcAccessMask: vk.AccessFlags(vk.AccessShaderWriteBit),
		DstAccessMask: vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit),
	}
	vkCmdPipelineBarrier(
		e.encoder.device.cmds,
		e.encoder.active,
		vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit),
		vk.PipelineStageFlags(vk.PipelineStageComputeShaderBit),
		0,
		1, &memBarrier,
		0, nil,
		0, nil,
	)
}

// Ray tracing methods (BuildAccelerationStructures, PlaceAccelerationStructureBarrier,
// CopyAccelerationStructure, ReadAccelerationStructureCompactSize) are in raytracing.go.

// --- Helper functions ---

// offscreenFinalLayout returns the Vulkan image layout that an offscreen
// (non-swapchain) texture view should be left in at the end of a render pass.
//
// If the underlying texture also has TextureBinding usage (i.e., it will be
// sampled by fragment shaders in a later render pass), the layout must be
// ImageLayoutGeneral so that reads see coherent data. Without this, Intel
// CCS-compressed color-attachment data would not be decompressed on the read
// path, producing stale "trail" artifacts (BUG-WGPU-VK-007).
//
// Textures without TextureBinding usage (pure render targets) can stay in
// ColorAttachmentOptimal, which enables maximal CCS compression.
func offscreenFinalLayout(view *TextureView) vk.ImageLayout {
	if view.texture != nil && view.texture.usage&gputypes.TextureUsageTextureBinding != 0 {
		return vk.ImageLayoutGeneral
	}
	return vk.ImageLayoutColorAttachmentOptimal
}

// updateSwapchainLayout updates swapchain image layout tracking for
// BUG-WGPU-VK-006. When a render pass targets a swapchain image (directly
// or via MSAA resolve), the Vulkan render pass finalLayout transitions the
// image automatically. Recording the expected layout lets present() skip
// the defensive barrier when it's not needed (zero overhead common case).
func updateSwapchainLayout(view *TextureView, resolveView *TextureView, hasMSAAResolve bool, finalLayout vk.ImageLayout) {
	if !hasMSAAResolve && view.isSwapchain && view.swapchain != nil {
		// Non-MSAA: the color attachment IS the swapchain image.
		view.swapchain.SetImageLayout(view.swapchain.currentImage, finalLayout)
	} else if hasMSAAResolve && resolveView != nil && resolveView.isSwapchain && resolveView.swapchain != nil {
		// MSAA: the resolve target IS the swapchain image.
		resolveView.swapchain.SetImageLayout(resolveView.swapchain.currentImage, finalLayout)
	}
}

// swapchainTargetView returns the TextureView that represents the swapchain
// image in a render pass, or nil if the render pass does not target a swapchain.
// With MSAA, the resolve target is the swapchain image; without MSAA, it is
// the color attachment directly.
func swapchainTargetView(view *TextureView, resolveView *TextureView, hasMSAAResolve bool) *TextureView {
	if !hasMSAAResolve && view.isSwapchain && view.swapchain != nil {
		return view
	}
	if hasMSAAResolve && resolveView != nil && resolveView.isSwapchain && resolveView.swapchain != nil {
		return resolveView
	}
	return nil
}

// shaderStages are the stages a shader binding is read or written in. Which
// shader a WebGPU binding is visible to is not known here, so all of them.
const shaderStages = vk.PipelineStageFlags(vk.PipelineStageVertexShaderBit | vk.PipelineStageFragmentShaderBit | vk.PipelineStageComputeShaderBit)

func bufferUsageToAccessAndStage(usage gputypes.BufferUsage) (vk.AccessFlags, vk.PipelineStageFlags) {
	var access vk.AccessFlags
	var stage vk.PipelineStageFlags

	if usage&gputypes.BufferUsageCopySrc != 0 {
		access |= vk.AccessFlags(vk.AccessTransferReadBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageTransferBit)
	}
	if usage&gputypes.BufferUsageCopyDst != 0 {
		access |= vk.AccessFlags(vk.AccessTransferWriteBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageTransferBit)
	}
	if usage&gputypes.BufferUsageVertex != 0 {
		access |= vk.AccessFlags(vk.AccessVertexAttributeReadBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageVertexInputBit)
	}
	if usage&gputypes.BufferUsageIndex != 0 {
		access |= vk.AccessFlags(vk.AccessIndexReadBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageVertexInputBit)
	}
	if usage&gputypes.BufferUsageUniform != 0 {
		access |= vk.AccessFlags(vk.AccessUniformReadBit)
		stage |= shaderStages
	}
	if usage&gputypes.BufferUsageStorage != 0 {
		access |= vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)
		stage |= shaderStages
	}
	if usage&gputypes.BufferUsageIndirect != 0 {
		access |= vk.AccessFlags(vk.AccessIndirectCommandReadBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageDrawIndirectBit)
	}

	if stage == 0 {
		stage = vk.PipelineStageFlags(vk.PipelineStageTopOfPipeBit)
	}

	return access, stage
}

// textureUses is the fine-grained state one end of a barrier names: the one
// the caller gave, else the one its coarse usage stands for on tex's format.
func textureUses(tex *Texture, uses hal.TextureUses, usage gputypes.TextureUsage) hal.TextureUses {
	if uses != 0 {
		return uses
	}
	return hal.TextureUsesFromUsage(usage, tex != nil && isDepthStencilFormat(tex.format))
}

// textureBarrierState is the access, stage and layout a fine-grained usage
// of tex means in a barrier. The layout has to be the one every other path
// puts the image in (deriveImageLayout). tex may be nil, for a swapchain
// image.
//
// Reference: wgpu-hal vulkan/conv.rs map_texture_usage_to_barrier
func textureBarrierState(tex *Texture, uses hal.TextureUses) (vk.AccessFlags, vk.PipelineStageFlags, vk.ImageLayout) {
	var access vk.AccessFlags
	var stage vk.PipelineStageFlags

	if uses&hal.TextureUsesCopySrc != 0 {
		access |= vk.AccessFlags(vk.AccessTransferReadBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageTransferBit)
	}
	if uses&hal.TextureUsesCopyDst != 0 {
		access |= vk.AccessFlags(vk.AccessTransferWriteBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageTransferBit)
	}
	if uses&(hal.TextureUsesResource|hal.TextureUsesStorageRead) != 0 {
		access |= vk.AccessFlags(vk.AccessShaderReadBit)
		stage |= shaderStages
	}
	if uses&hal.TextureUsesStorageWrite != 0 {
		access |= vk.AccessFlags(vk.AccessShaderReadBit | vk.AccessShaderWriteBit)
		stage |= shaderStages
	}
	if uses&hal.TextureUsesColorTarget != 0 {
		access |= vk.AccessFlags(vk.AccessColorAttachmentWriteBit | vk.AccessColorAttachmentReadBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit)
	}
	if uses&hal.TextureUsesDepthStencilRead != 0 {
		access |= vk.AccessFlags(vk.AccessDepthStencilAttachmentReadBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageEarlyFragmentTestsBit | vk.PipelineStageLateFragmentTestsBit)
	}
	if uses&hal.TextureUsesDepthStencilWrite != 0 {
		access |= vk.AccessFlags(vk.AccessDepthStencilAttachmentWriteBit | vk.AccessDepthStencilAttachmentReadBit)
		stage |= vk.PipelineStageFlags(vk.PipelineStageEarlyFragmentTestsBit | vk.PipelineStageLateFragmentTestsBit)
	}

	// Uninitialized and Present have no access of their own.
	if stage == 0 {
		stage = vk.PipelineStageFlags(vk.PipelineStageTopOfPipeBit)
	}

	return access, stage, deriveImageLayout(tex, uses)
}

// deriveImageLayout is the layout an image of tex's format is in while it
// is in the fine-grained state uses. Every path that names a layout agrees
// on it: the barriers, the render pass attachment layouts and the sampled
// image descriptors.
//
//   - Uninitialized (and no state at all) is UNDEFINED, the layout a new
//     image starts in; naming GENERAL there would lie about the old layout.
//   - A depth/stencil format is only ever read in
//     DEPTH_STENCIL_READ_ONLY_OPTIMAL: sampled, attached read-only, or both
//     in the same pass. Written as an attachment it is in
//     DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
//   - A colour target that is also sampled (TextureBinding usage) ends a
//     pass in GENERAL, see offscreenFinalLayout, so its target state is
//     GENERAL too.
//   - Any other combination of read states is GENERAL, which every access
//     allows.
//
// Reference: wgpu-hal vulkan/conv.rs derive_image_layout
func deriveImageLayout(tex *Texture, uses hal.TextureUses) vk.ImageLayout {
	isColor := tex == nil || !isDepthStencilFormat(tex.format)
	switch uses {
	case 0, hal.TextureUsesUninitialized:
		return vk.ImageLayoutUndefined
	case hal.TextureUsesCopySrc:
		return vk.ImageLayoutTransferSrcOptimal
	case hal.TextureUsesCopyDst:
		return vk.ImageLayoutTransferDstOptimal
	case hal.TextureUsesColorTarget:
		if tex != nil && tex.usage&gputypes.TextureUsageTextureBinding != 0 {
			return vk.ImageLayoutGeneral
		}
		return vk.ImageLayoutColorAttachmentOptimal
	case hal.TextureUsesDepthStencilWrite:
		return vk.ImageLayoutDepthStencilAttachmentOptimal
	case hal.TextureUsesPresent:
		return vk.ImageLayoutPresentSrcKhr
	case hal.TextureUsesResource:
		if isColor {
			return vk.ImageLayoutShaderReadOnlyOptimal
		}
		return vk.ImageLayoutDepthStencilReadOnlyOptimal
	default:
		if isColor {
			return vk.ImageLayoutGeneral
		}
		return vk.ImageLayoutDepthStencilReadOnlyOptimal
	}
}

func mipLevelCountOrRemaining(count uint32) uint32 {
	if count == 0 {
		return vk.RemainingMipLevels
	}
	return count
}

func arrayLayerCountOrRemaining(count uint32) uint32 {
	if count == 0 {
		return vk.RemainingArrayLayers
	}
	return count
}

func loadOpToVk(op gputypes.LoadOp) vk.AttachmentLoadOp {
	switch op {
	case gputypes.LoadOpClear:
		return vk.AttachmentLoadOpClear
	case gputypes.LoadOpLoad:
		return vk.AttachmentLoadOpLoad
	default:
		return vk.AttachmentLoadOpDontCare
	}
}

func storeOpToVk(op gputypes.StoreOp) vk.AttachmentStoreOp {
	switch op {
	case gputypes.StoreOpStore:
		return vk.AttachmentStoreOpStore
	default:
		return vk.AttachmentStoreOpDontCare
	}
}

// --- Vulkan function wrappers ---

func vkBeginCommandBuffer(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, beginInfo *vk.CommandBufferBeginInfo) vk.Result {
	return cmds.BeginCommandBuffer(cmdBuffer, beginInfo)
}

func vkEndCommandBuffer(cmds *vk.Commands, cmdBuffer vk.CommandBuffer) vk.Result {
	return cmds.EndCommandBuffer(cmdBuffer)
}

func vkResetCommandPool(cmds *vk.Commands, device vk.Device, pool vk.CommandPool, flags vk.CommandPoolResetFlags) vk.Result {
	return cmds.ResetCommandPool(device, pool, flags)
}

//nolint:unparam // Vulkan API wrapper — signature mirrors vkCmdPipelineBarrier spec
func vkCmdPipelineBarrier(cmds *vk.Commands, cmdBuffer vk.CommandBuffer,
	srcStageMask, dstStageMask vk.PipelineStageFlags,
	dependencyFlags vk.DependencyFlags,
	memoryBarrierCount uint32, pMemoryBarriers *vk.MemoryBarrier,
	bufferMemoryBarrierCount uint32, pBufferMemoryBarriers *vk.BufferMemoryBarrier,
	imageMemoryBarrierCount uint32, pImageMemoryBarriers *vk.ImageMemoryBarrier) {
	cmds.CmdPipelineBarrier(cmdBuffer, srcStageMask, dstStageMask, dependencyFlags,
		memoryBarrierCount, pMemoryBarriers,
		bufferMemoryBarrierCount, pBufferMemoryBarriers,
		imageMemoryBarrierCount, pImageMemoryBarriers)
}

func vkCmdFillBuffer(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, buffer vk.Buffer, offset, size vk.DeviceSize, data uint32) {
	cmds.CmdFillBuffer(cmdBuffer, buffer, offset, size, data)
}

func vkCmdCopyBuffer(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, src, dst vk.Buffer, regionCount uint32, pRegions *vk.BufferCopy) {
	cmds.CmdCopyBuffer(cmdBuffer, src, dst, regionCount, pRegions)
}

func vkCmdCopyBufferToImage(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, src vk.Buffer, dst vk.Image, layout vk.ImageLayout, regionCount uint32, pRegions *vk.BufferImageCopy) {
	cmds.CmdCopyBufferToImage(cmdBuffer, src, dst, layout, regionCount, pRegions)
}

func vkCmdCopyImageToBuffer(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, src vk.Image, layout vk.ImageLayout, dst vk.Buffer, regionCount uint32, pRegions *vk.BufferImageCopy) {
	cmds.CmdCopyImageToBuffer(cmdBuffer, src, layout, dst, regionCount, pRegions)
}

func vkCmdCopyImage(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, src vk.Image, srcLayout vk.ImageLayout, dst vk.Image, dstLayout vk.ImageLayout, regionCount uint32, pRegions *vk.ImageCopy) {
	cmds.CmdCopyImage(cmdBuffer, src, srcLayout, dst, dstLayout, regionCount, pRegions)
}

//nolint:unused // Reserved for VK_KHR_dynamic_rendering support (disabled on Intel)
func vkCmdBeginRendering(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, renderingInfo *vk.RenderingInfo) {
	cmds.CmdBeginRendering(cmdBuffer, renderingInfo)
}

//nolint:unused // Reserved for VK_KHR_dynamic_rendering support (disabled on Intel)
func vkCmdEndRendering(cmds *vk.Commands, cmdBuffer vk.CommandBuffer) {
	cmds.CmdEndRendering(cmdBuffer)
}

func vkCmdBeginRenderPass(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, renderPassBegin *vk.RenderPassBeginInfo, contents vk.SubpassContents) {
	cmds.CmdBeginRenderPass(cmdBuffer, renderPassBegin, contents)
}

func vkCmdEndRenderPass(cmds *vk.Commands, cmdBuffer vk.CommandBuffer) {
	cmds.CmdEndRenderPass(cmdBuffer)
}

func vkCmdBindPipeline(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, bindPoint vk.PipelineBindPoint, pipeline vk.Pipeline) {
	cmds.CmdBindPipeline(cmdBuffer, bindPoint, pipeline)
}

func vkCmdBindDescriptorSets(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, bindPoint vk.PipelineBindPoint, layout vk.PipelineLayout, firstSet uint32, setCount uint32, pSets *vk.DescriptorSet, dynamicOffsetCount uint32, pDynamicOffsets *uint32) {
	cmds.CmdBindDescriptorSets(cmdBuffer, bindPoint, layout, firstSet, setCount, pSets, dynamicOffsetCount, pDynamicOffsets)
}

func vkCmdBindVertexBuffers(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, firstBinding, bindingCount uint32, pBuffers *vk.Buffer, pOffsets *vk.DeviceSize) {
	cmds.CmdBindVertexBuffers(cmdBuffer, firstBinding, bindingCount, pBuffers, pOffsets)
}

func vkCmdBindIndexBuffer(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, buffer vk.Buffer, offset vk.DeviceSize, indexType vk.IndexType) {
	cmds.CmdBindIndexBuffer(cmdBuffer, buffer, offset, indexType)
}

func vkCmdSetViewport(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, firstViewport, viewportCount uint32, pViewports *vk.Viewport) {
	cmds.CmdSetViewport(cmdBuffer, firstViewport, viewportCount, pViewports)
}

func vkCmdSetScissor(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, firstScissor, scissorCount uint32, pScissors *vk.Rect2D) {
	cmds.CmdSetScissor(cmdBuffer, firstScissor, scissorCount, pScissors)
}

func vkCmdSetBlendConstants(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, blendConstants *[4]float32) {
	cmds.CmdSetBlendConstants(cmdBuffer, *blendConstants)
}

func vkCmdSetStencilReference(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, faceMask vk.StencilFaceFlags, reference uint32) {
	cmds.CmdSetStencilReference(cmdBuffer, faceMask, reference)
}

func vkCmdDraw(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, vertexCount, instanceCount, firstVertex, firstInstance uint32) {
	cmds.CmdDraw(cmdBuffer, vertexCount, instanceCount, firstVertex, firstInstance)
}

func vkCmdDrawIndexed(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, indexCount, instanceCount, firstIndex uint32, vertexOffset int32, firstInstance uint32) {
	cmds.CmdDrawIndexed(cmdBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance)
}

func vkCmdDrawIndirect(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, buffer vk.Buffer, offset vk.DeviceSize, drawCount, stride uint32) {
	cmds.CmdDrawIndirect(cmdBuffer, buffer, offset, drawCount, stride)
}

func vkCmdDrawIndexedIndirect(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, buffer vk.Buffer, offset vk.DeviceSize, drawCount, stride uint32) {
	cmds.CmdDrawIndexedIndirect(cmdBuffer, buffer, offset, drawCount, stride)
}

func vkCmdDispatch(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, x, y, z uint32) {
	cmds.CmdDispatch(cmdBuffer, x, y, z)
}

func vkCmdDispatchIndirect(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, buffer vk.Buffer, offset vk.DeviceSize) {
	cmds.CmdDispatchIndirect(cmdBuffer, buffer, offset)
}

func vkCmdCopyQueryPoolResults(cmds *vk.Commands, cmdBuffer vk.CommandBuffer, queryPool vk.QueryPool, firstQuery, queryCount uint32, dstBuffer vk.Buffer, dstOffset, stride uint64, flags vk.QueryResultFlags) {
	cmds.CmdCopyQueryPoolResults(cmdBuffer, queryPool, firstQuery, queryCount, dstBuffer, vk.DeviceSize(dstOffset), vk.DeviceSize(stride), flags)
}
