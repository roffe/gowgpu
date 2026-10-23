//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

import (
	"fmt"
	"image"
	"runtime"
	"slices"
	"sync"
	"time"
	"unsafe"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// cmdBufferPool reuses slices of vk.CommandBuffer handles across Submit calls.
// Without pooling, every Submit allocates a new []vk.CommandBuffer on the heap
// (1-8 elements per frame at 60+ FPS = 60-480 allocations/second).
var cmdBufferPool = sync.Pool{
	New: func() any {
		s := make([]vk.CommandBuffer, 0, 8)
		return &s
	},
}

// relaySemaphores enforces GPU-side ordering between consecutive vkQueueSubmit calls.
//
// The wgpu_hal Queue trait (lib.rs:1059-1068) promises that if two calls to Submit
// are ordered, the first submission finishes on the GPU before the second begins.
// Vulkan only guarantees that submissions START in order, not that they finish in order.
// Relay semaphores close this gap by chaining a wait-then-signal dependency between
// every pair of consecutive submissions.
//
// We alternate between two binary semaphores instead of reusing one. This works around
// Mesa ANV driver bug #5508 (https://gitlab.freedesktop.org/mesa/mesa/-/issues/5508)
// where a single binary semaphore waited and signaled in consecutive submissions hangs.
// The bug is fixed in Mesa, but the workaround should be retained until at least Oct 2026.
//
// Reference: Rust wgpu-hal vulkan/mod.rs:526-598.
type relaySemaphores struct {
	// wait is the semaphore the next submission should wait on before beginning
	// execution on the GPU. Zero for the first submission (no dependency yet).
	wait vk.Semaphore

	// signal is the semaphore the next submission should signal when it finishes
	// execution on the GPU. Always valid (non-zero).
	signal vk.Semaphore
}

// newRelaySemaphores creates the initial relay state with one binary semaphore.
// The first submission will signal this semaphore without waiting on anything.
func newRelaySemaphores(cmds *vk.Commands, device vk.Device) (*relaySemaphores, error) {
	createInfo := vk.SemaphoreCreateInfo{
		SType: vk.StructureTypeSemaphoreCreateInfo,
	}
	var sem vk.Semaphore
	result := cmds.CreateSemaphore(device, &createInfo, nil, &sem)
	if result != vk.Success {
		return nil, fmt.Errorf("vulkan: vkCreateSemaphore (relay 1) failed: %d", result)
	}
	return &relaySemaphores{
		wait:   0, // first submission has no predecessor to wait on
		signal: sem,
	}, nil
}

// advance returns the (wait, signal) semaphores for the current submission and
// prepares the state for the next one.
//
// State machine:
//
//	Submit 1: returns (0, sem1) — no wait, signal sem1.    State becomes (sem1, sem2).
//	Submit 2: returns (sem1, sem2) — wait sem1, signal sem2. State becomes (sem2, sem1) [swap].
//	Submit 3: returns (sem2, sem1) — wait sem2, signal sem1. State becomes (sem1, sem2) [swap].
//	...alternating indefinitely.
//
// The second semaphore is created on demand (during the transition from first to second
// submission) to avoid allocating a semaphore that might never be needed.
func (r *relaySemaphores) advance(cmds *vk.Commands, device vk.Device) (wait, signal vk.Semaphore, err error) {
	// Capture current state for the caller.
	wait = r.wait
	signal = r.signal

	if r.wait == 0 {
		// First submission just happened. The second submission should wait on
		// what we just signaled, and signal a new semaphore.
		r.wait = r.signal
		createInfo := vk.SemaphoreCreateInfo{
			SType: vk.StructureTypeSemaphoreCreateInfo,
		}
		var sem2 vk.Semaphore
		result := cmds.CreateSemaphore(device, &createInfo, nil, &sem2)
		if result != vk.Success {
			return 0, 0, fmt.Errorf("vulkan: vkCreateSemaphore (relay 2) failed: %d", result)
		}
		r.signal = sem2
	} else {
		// Subsequent submissions: swap wait and signal so the next submission
		// waits on what this one signals, and signals the one this one waited on.
		r.wait, r.signal = r.signal, r.wait
	}

	return wait, signal, nil
}

// destroy releases both relay semaphores.
func (r *relaySemaphores) destroy(cmds *vk.Commands, device vk.Device) {
	if r.wait != 0 {
		cmds.DestroySemaphore(device, r.wait, nil)
		r.wait = 0
	}
	if r.signal != 0 {
		cmds.DestroySemaphore(device, r.signal, nil)
		r.signal = 0
	}
}

// Queue implements hal.Queue for Vulkan.
type Queue struct {
	handle      vk.Queue
	device      *Device
	familyIndex uint32
	// acquired are the swapchains with an image acquired that no Present
	// or DiscardTexture has given back. A Submit waits for the acquire
	// semaphore of each the first time, and signals a present semaphore of
	// each every time, which its present waits for.
	acquired []*Swapchain
	relay    *relaySemaphores
	mu       sync.Mutex // Protects Submit() and Present() from concurrent access

	// BUG-WGPU-VK-005: Offscreen submit semaphore suppression.
	// When swapchainSuppressed is true, Submit skips swapchain semaphore
	// binding; saved holds the acquired swapchains until it ends.
	// Same pattern as VK-004 (WriteTexture) but caller-controlled.
	swapchainSuppressed bool
	saved               []*Swapchain

	// submitScratch holds what Submit hands vkQueueSubmit through pointers
	// inside the VkSubmitInfo chain, kept so a submit allocates nothing once
	// they have grown. Guarded by mu.
	submitScratch struct {
		waits        []vk.Semaphore
		waitStages   []vk.PipelineStageFlags
		waitValues   []uint64
		signals      []vk.Semaphore
		signalValues []uint64
		consumed     []*Swapchain
		one          [1]*Swapchain
	}

	// Points of timeline semaphores of code sharing the device the next
	// Submit waits for and signals (AddExternalSync). Guarded by mu.
	externalWaits, externalSignals []TimelinePoint
}

// TimelinePoint is a value of a timeline semaphore.
type TimelinePoint struct {
	Semaphore vk.Semaphore
	Value     uint64
}

// maxExternalSync is how many points of others one submit can wait for, and
// signal.
const maxExternalSync = 8

// ExternalSyncFull reports whether the next Submit takes no more points of
// AddExternalSync.
func (q *Queue) ExternalSyncFull() bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.externalWaits) == maxExternalSync
}

// AddExternalSync has the next Submit wait for semaphore, a timeline
// semaphore of code sharing the device such as a video decoder's, to reach
// wait before its transfers, and signal signal once it completes.
func (q *Queue) AddExternalSync(semaphore, wait, signal uint64) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if !q.device.timelineFence.isTimeline {
		return fmt.Errorf("vulkan: external semaphores need timeline semaphores")
	}
	if len(q.externalWaits) == maxExternalSync {
		return fmt.Errorf("vulkan: a submit waits for at most %d external semaphores", maxExternalSync)
	}
	q.externalWaits = append(q.externalWaits, TimelinePoint{vk.Semaphore(semaphore), wait})
	q.externalSignals = append(q.externalSignals, TimelinePoint{vk.Semaphore(semaphore), signal})
	return nil
}

// acquireWaitStages are the stages the first submit of a frame holds until
// the acquire semaphore signals: the colour writes of a render pass on the
// swapchain image, and the transfer of a copy or queue write into it, which
// can be the first thing a frame does to the image.
const acquireWaitStages = vk.PipelineStageFlags(vk.PipelineStageColorAttachmentOutputBit | vk.PipelineStageTransferBit)

func validateSwapchainSubmission(swapchain *Swapchain, device *Device) error {
	if swapchain == nil {
		return nil
	}
	if swapchain.device != device {
		return fmt.Errorf("vulkan: active swapchain belongs to a different device")
	}
	if swapchain.destroyed {
		return fmt.Errorf("vulkan: active swapchain has been destroyed")
	}
	if swapchain.broken {
		if swapchain.failureErr != nil {
			return fmt.Errorf("vulkan: active swapchain is broken: %w", swapchain.failureErr)
		}
		return fmt.Errorf("vulkan: active swapchain is broken")
	}
	if !swapchain.imageAcquired {
		return fmt.Errorf("vulkan: no swapchain image acquired for submission")
	}
	if swapchain.currentAcquireIdx < 0 || swapchain.currentAcquireIdx >= len(swapchain.acquireSemaphores) || swapchain.currentAcquireIdx >= len(swapchain.acquireFenceValues) {
		return fmt.Errorf("vulkan: swapchain acquire semaphore state is inconsistent")
	}
	if swapchain.currentImage >= uint32(len(swapchain.presentPools)) {
		return fmt.Errorf("vulkan: swapchain present image index is out of range")
	}
	return nil
}

// holds reports whether sc has an image acquired that the queue's
// submissions synchronize with. q.mu is held.
func (q *Queue) holds(sc *Swapchain) bool {
	return slices.Contains(q.acquired, sc)
}

// hold has the queue's submissions synchronize with the image sc acquired.
// q.mu is held.
func (q *Queue) hold(sc *Swapchain) {
	if !q.holds(sc) {
		q.acquired = append(q.acquired, sc)
	}
}

// release forgets sc, whose image was presented or given back. q.mu is held.
func (q *Queue) release(sc *Swapchain) {
	isSC := func(s *Swapchain) bool { return s == sc }
	q.acquired = slices.DeleteFunc(q.acquired, isSC)
	q.saved = slices.DeleteFunc(q.saved, isSC)
}

// Submit submits command buffers to the GPU.
// Returns a monotonically increasing submission index for tracking completion.
// The HAL internally manages fence/timeline semaphore synchronization.
//
// VK-SYNC-001: Every submission chains through relay semaphores to guarantee
// GPU-side execution ordering between consecutive vkQueueSubmit calls.
func (q *Queue) Submit(commandBuffers []hal.CommandBuffer) (uint64, error) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(commandBuffers) == 0 {
		return 0, nil
	}
	for _, sc := range q.acquired {
		if err := validateSwapchainSubmission(sc, q.device); err != nil {
			return 0, err
		}
	}

	// Convert command buffers to Vulkan handles.
	// Use sync.Pool to avoid per-frame heap allocation (VK-PERF-001).
	pooledSlice := cmdBufferPool.Get().(*[]vk.CommandBuffer)
	vkCmdBuffers := (*pooledSlice)[:0]
	for _, cb := range commandBuffers {
		vkCB, ok := cb.(*CommandBuffer)
		if !ok {
			*pooledSlice = vkCmdBuffers
			cmdBufferPool.Put(pooledSlice)
			return 0, fmt.Errorf("vulkan: command buffer is not a Vulkan command buffer")
		}
		vkCmdBuffers = append(vkCmdBuffers, vkCB.handle)
	}
	defer func() {
		*pooledSlice = vkCmdBuffers[:0]
		cmdBufferPool.Put(pooledSlice)
	}()
	return q.submit(vkCmdBuffers, q.acquired)
}

// submit submits cmdBuffers synchronized with the acquired images of
// swapchains, and with the points of AddExternalSync, which it consumes.
// q.mu is held.
//
// ADR-058: an image's acquire semaphore is waited for once, by the first
// submission after the acquire, and every submission signals a present
// semaphore of the image, all of which its present waits for: a frame of
// several submits presents after the last of them.
func (q *Queue) submit(cmdBuffers []vk.CommandBuffer, swapchains []*Swapchain) (uint64, error) {
	defer func() { q.externalWaits, q.externalSignals = q.externalWaits[:0], q.externalSignals[:0] }()
	s := &q.submitScratch
	s.waits, s.waitStages, s.waitValues = s.waits[:0], s.waitStages[:0], s.waitValues[:0]
	s.signals, s.signalValues = s.signals[:0], s.signalValues[:0]
	wait := func(sem vk.Semaphore, stage vk.PipelineStageFlags, value uint64) {
		s.waits, s.waitStages, s.waitValues = append(s.waits, sem), append(s.waitStages, stage), append(s.waitValues, value)
	}
	signal := func(sem vk.Semaphore, value uint64) {
		s.signals, s.signalValues = append(s.signals, sem), append(s.signalValues, value)
	}
	broken := func(err error) error {
		for _, sc := range swapchains {
			sc.markBroken(err)
		}
		return err
	}

	s.consumed = s.consumed[:0]
	for _, sc := range swapchains {
		if !sc.acquireUsed {
			if sc.currentAcquireSem == 0 {
				return 0, broken(fmt.Errorf("vulkan: active swapchain acquire semaphore is invalid"))
			}
			wait(sc.currentAcquireSem, acquireWaitStages, 0)
			s.consumed = append(s.consumed, sc)
		}
		presentSem, err := sc.allocPresentSemaphore()
		if err != nil {
			return 0, broken(err)
		}
		signal(presentSem, 0)
	}

	// VK-SYNC-001: Add relay semaphores for GPU-side submission ordering.
	// This ensures that barriers from one submission are visible to the next
	// (required by the wgpu_hal Queue trait, not guaranteed by Vulkan spec).
	if q.relay != nil {
		relayWait, relaySignal, err := q.relay.advance(q.device.cmds, q.device.handle)
		if err != nil {
			return 0, broken(fmt.Errorf("vulkan: relay semaphore advance: %w", err))
		}
		if relayWait != 0 {
			wait(relayWait, vk.PipelineStageFlags(vk.PipelineStageTopOfPipeBit), 0)
		}
		signal(relaySignal, 0)
	}
	for i := range q.externalWaits {
		wait(q.externalWaits[i].Semaphore, vk.PipelineStageFlags(vk.PipelineStageTransferBit), q.externalWaits[i].Value)
		signal(q.externalSignals[i].Semaphore, q.externalSignals[i].Value)
	}

	tf := q.device.timelineFence
	signalValue := tf.nextSignalValue()
	fence := vk.Fence(0)
	var pins runtime.Pinner
	defer pins.Unpin()
	var stack struct {
		cmdBuffers               [8]vk.CommandBuffer
		waits, signals           [16]vk.Semaphore // maxExternalSync points, the relay, the timeline, swapchains
		waitStages               [16]vk.PipelineStageFlags
		waitValues, signalValues [16]uint64
	}
	submitInfo := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		CommandBufferCount: uint32(len(cmdBuffers)),
		PCommandBuffers:    onStackOrPinned(&pins, stack.cmdBuffers[:], cmdBuffers),
	}
	var timelineInfo vk.TimelineSemaphoreSubmitInfo
	if tf.isTimeline {
		signal(tf.timelineSemaphore, signalValue)
		timelineInfo = vk.TimelineSemaphoreSubmitInfo{
			SType:                     vk.StructureTypeTimelineSemaphoreSubmitInfo,
			WaitSemaphoreValueCount:   uint32(len(s.waitValues)),
			PWaitSemaphoreValues:      onStackOrPinned(&pins, stack.waitValues[:], s.waitValues),
			SignalSemaphoreValueCount: uint32(len(s.signalValues)),
			PSignalSemaphoreValues:    onStackOrPinned(&pins, stack.signalValues[:], s.signalValues),
		}
		submitInfo.PNext = (*uintptr)(unsafe.Pointer(&timelineInfo))
	} else {
		// Binary path (VK-IMPL-003): a fence of the pool tracks the submission.
		var err error
		if fence, err = tf.signal(q.device.cmds, q.device.handle, signalValue); err != nil {
			return 0, broken(fmt.Errorf("vulkan: Submit fencePool signal: %w", err))
		}
	}
	if len(s.waits) > 0 {
		submitInfo.WaitSemaphoreCount = uint32(len(s.waits))
		submitInfo.PWaitSemaphores = onStackOrPinned(&pins, stack.waits[:], s.waits)
		submitInfo.PWaitDstStageMask = onStackOrPinned(&pins, stack.waitStages[:], s.waitStages)
	}
	if len(s.signals) > 0 {
		submitInfo.SignalSemaphoreCount = uint32(len(s.signals))
		submitInfo.PSignalSemaphores = onStackOrPinned(&pins, stack.signals[:], s.signals)
	}

	if result := vkQueueSubmit(q, 1, &submitInfo, fence); result != vk.Success {
		return 0, broken(fmt.Errorf("vulkan: vkQueueSubmit failed: %d", result))
	}
	// VK-IMPL-004: the next acquire with the semaphore waits for this
	// submission first.
	for _, sc := range s.consumed {
		sc.acquireUsed = true
		sc.acquireFenceValues[sc.currentAcquireIdx] = signalValue
	}
	return signalValue, nil
}

// onStackOrPinned points a Vulkan call at the elements of src: at their
// copy in stack, an array on the caller's stack, when they fit, as they
// almost always do, and else at src, pinned in pins. Memory on the stack
// needs no pinning, and a Pinner allocates once it holds more than five
// objects.
func onStackOrPinned[T any](pins *runtime.Pinner, stack, src []T) *T {
	if len(src) == 0 {
		return nil
	}
	if len(src) <= len(stack) {
		copy(stack, src)
		return &stack[0]
	}
	return vk.Pin(pins, &src[0])
}

// consumeAcquire waits for sc's acquire semaphore in a submission of
// nothing, for a frame discarded without a Submit: vkAcquireNextImageKHR
// may only get the semaphore back unsignaled, and the fence value recorded
// here is what the next acquire with it waits for. q.mu is held.
func (q *Queue) consumeAcquire(sc *Swapchain) error {
	if sc.acquireUsed || sc.currentAcquireSem == 0 || sc.currentAcquireIdx < 0 || sc.currentAcquireIdx >= len(sc.acquireFenceValues) {
		return nil
	}
	tf := q.device.timelineFence
	signalValue := tf.nextSignalValue()
	waitSem := sc.currentAcquireSem
	waitStage := vk.PipelineStageFlags(vk.PipelineStageAllCommandsBit)
	waitValues := [1]uint64{0}
	signalSems := [1]vk.Semaphore{tf.timelineSemaphore}
	signalValues := [1]uint64{signalValue}
	submitInfo := vk.SubmitInfo{
		SType:              vk.StructureTypeSubmitInfo,
		WaitSemaphoreCount: 1,
		PWaitSemaphores:    &waitSem,
		PWaitDstStageMask:  &waitStage,
	}
	fence := vk.Fence(0)
	var timelineInfo vk.TimelineSemaphoreSubmitInfo
	if tf.isTimeline {
		submitInfo.SignalSemaphoreCount = 1
		submitInfo.PSignalSemaphores = &signalSems[0]
		timelineInfo = vk.TimelineSemaphoreSubmitInfo{
			SType:                     vk.StructureTypeTimelineSemaphoreSubmitInfo,
			WaitSemaphoreValueCount:   1,
			PWaitSemaphoreValues:      &waitValues[0],
			SignalSemaphoreValueCount: 1,
			PSignalSemaphoreValues:    &signalValues[0],
		}
		submitInfo.PNext = (*uintptr)(unsafe.Pointer(&timelineInfo))
	} else {
		var err error
		if fence, err = tf.signal(q.device.cmds, q.device.handle, signalValue); err != nil {
			return fmt.Errorf("vulkan: discard fence: %w", err)
		}
	}
	if result := vkQueueSubmit(q, 1, &submitInfo, fence); result != vk.Success {
		return fmt.Errorf("vulkan: vkQueueSubmit (discard) failed: %d", result)
	}
	sc.acquireFenceValues[sc.currentAcquireIdx] = signalValue
	sc.acquireUsed = true
	return nil
}

// PollCompleted returns the highest submission index known to be completed by the GPU.
// Non-blocking.
func (q *Queue) PollCompleted() uint64 {
	q.mu.Lock()
	defer q.mu.Unlock()
	return q.device.timelineFence.poll(q.device.cmds, q.device.handle)
}

// SubmitForPresent submits command buffers synchronized with the image
// swapchain acquired, the way Submit is with every acquired image.
//
// VK-SYNC-001: Every submission chains through relay semaphores to guarantee
// GPU-side execution ordering between consecutive vkQueueSubmit calls.
func (q *Queue) SubmitForPresent(commandBuffers []hal.CommandBuffer, swapchain *Swapchain) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(commandBuffers) == 0 {
		return nil
	}
	if swapchain == nil {
		return fmt.Errorf("vulkan: swapchain is nil")
	}
	if err := validateSwapchainSubmission(swapchain, q.device); err != nil {
		return err
	}

	// Convert command buffers to Vulkan handles.
	// Use sync.Pool to avoid per-frame heap allocation (VK-PERF-001).
	pooledSlice := cmdBufferPool.Get().(*[]vk.CommandBuffer)
	vkCmdBuffers := (*pooledSlice)[:0]
	for _, cb := range commandBuffers {
		vkCB, ok := cb.(*CommandBuffer)
		if !ok {
			*pooledSlice = vkCmdBuffers
			cmdBufferPool.Put(pooledSlice)
			return fmt.Errorf("vulkan: command buffer is not a Vulkan command buffer")
		}
		vkCmdBuffers = append(vkCmdBuffers, vkCB.handle)
	}
	defer func() {
		*pooledSlice = vkCmdBuffers[:0]
		cmdBufferPool.Put(pooledSlice)
	}()

	q.submitScratch.one[0] = swapchain
	_, err := q.submit(vkCmdBuffers, q.submitScratch.one[:])
	q.submitScratch.one[0] = nil
	if err != nil {
		return err
	}
	q.hold(swapchain)
	return nil
}

// WriteBuffer writes data to a host-visible buffer immediately, without
// waiting for the GPU: the caller owns the ordering. The staging belt only
// writes into chunks the GPU is done with, and it is the one caller on the
// frame path; until 2026-10-18 this waited for the latest submission on every
// call, which blocked the CPU on the previous frame's GPU work twice a frame.
func (q *Queue) WriteBuffer(buffer hal.Buffer, offset uint64, data []byte) error {
	vkBuffer, ok := buffer.(*Buffer)
	if !ok || vkBuffer.memory == nil {
		return fmt.Errorf("vulkan: WriteBuffer: invalid buffer")
	}

	// Buffer must be mapped (host-visible) for direct CPU writes.
	// Device-local buffers require staging at the public API level.
	if vkBuffer.memory.MappedPtr == 0 {
		return fmt.Errorf("vulkan: WriteBuffer: buffer is not mapped")
	}

	// Bounds check: verify the write fits within the mapped region (BUG-VK-001).
	// Without this, a partial/failed vkAllocateMemory that returned a too-small
	// mapping would cause SIGSEGV in copyToMappedMemory.
	if vkBuffer.memory.MappedSize > 0 && offset+uint64(len(data)) > vkBuffer.memory.MappedSize {
		return fmt.Errorf("vulkan: WriteBuffer: write of %d bytes at offset %d exceeds mapped size %d (BUG-VK-001)",
			len(data), offset, vkBuffer.memory.MappedSize)
	}

	// Direct copy using Vulkan mapped memory from vkMapMemory.
	// Use copyToMappedMemory to avoid go vet false positive about unsafe.Pointer.
	copyToMappedMemory(vkBuffer.memory.MappedPtr, offset, data)

	// Flush mapped memory to ensure GPU sees CPU writes.
	// Only needed for non-coherent memory. (BUG-VK-009 Fix 2/5)
	if !vkBuffer.memory.IsCoherent {
		alignedOffset, alignedSize := q.device.alignedMappedRange(vkBuffer.memory.Offset, vkBuffer.memory.Size)
		memRange := vk.MappedMemoryRange{
			SType:  vk.StructureTypeMappedMemoryRange,
			Memory: vkBuffer.memory.Memory,
			Offset: alignedOffset,
			Size:   alignedSize,
		}
		result := q.device.cmds.FlushMappedMemoryRanges(q.device.handle, 1, &memRange)
		if result != vk.Success {
			return fmt.Errorf("vulkan: WriteBuffer: FlushMappedMemoryRanges failed: %d", result)
		}
	}
	return nil
}

// WriteTexture writes data to a texture immediately and leaves the texture
// in the Resource state, which a later WriteTexture takes it from, keeping
// what it does not overwrite. The first takes the texture as uninitialized.
// Returns an error if any step fails (VK-003: no more silent error swallowing).
func (q *Queue) WriteTexture(dst *hal.ImageCopyTexture, data []byte, layout *hal.ImageDataLayout, size *hal.Extent3D) error {
	if dst == nil || dst.Texture == nil || len(data) == 0 || size == nil {
		return fmt.Errorf("vulkan: WriteTexture: invalid arguments")
	}

	vkTexture, ok := dst.Texture.(*Texture)
	if !ok || vkTexture == nil {
		return fmt.Errorf("vulkan: WriteTexture: invalid texture type")
	}

	// Create staging buffer
	stagingDesc := &hal.BufferDescriptor{
		Label: "staging-buffer-for-texture",
		Size:  uint64(len(data)),
		Usage: gputypes.BufferUsageCopySrc | gputypes.BufferUsageMapWrite,
	}

	stagingBuffer, err := q.device.CreateBuffer(stagingDesc)
	if err != nil {
		return fmt.Errorf("vulkan: WriteTexture: CreateBuffer failed: %w", err)
	}
	defer q.device.DestroyBuffer(stagingBuffer)

	if err := q.WriteBuffer(stagingBuffer, 0, data); err != nil {
		return fmt.Errorf("vulkan: WriteTexture: %w", err)
	}

	// Create one-shot command buffer
	cmdEncoder, err := q.device.CreateCommandEncoder(&hal.CommandEncoderDescriptor{
		Label: "texture-upload-encoder",
	})
	if err != nil {
		return fmt.Errorf("vulkan: WriteTexture: CreateCommandEncoder failed: %w", err)
	}

	encoder, ok := cmdEncoder.(*CommandEncoder)
	if !ok {
		return fmt.Errorf("vulkan: WriteTexture: unexpected encoder type")
	}

	// Begin recording
	if err := encoder.BeginEncoding("texture-upload"); err != nil {
		return fmt.Errorf("vulkan: WriteTexture: BeginEncoding failed: %w", err)
	}

	old, oldUsage := hal.TextureUsesUninitialized, gputypes.TextureUsage(0)
	if vkTexture.written {
		old, oldUsage = hal.TextureUsesResource, gputypes.TextureUsageTextureBinding
	}
	encoder.TransitionTextures([]hal.TextureBarrier{
		{
			Texture: dst.Texture,
			Usage: hal.TextureUsageTransition{
				OldUsage: oldUsage,
				NewUsage: gputypes.TextureUsageCopyDst,
				Old:      old,
				New:      hal.TextureUsesCopyDst,
			},
		},
	})

	// Copy from staging buffer to texture
	bytesPerRow := layout.BytesPerRow
	if bytesPerRow == 0 {
		// Calculate based on format and width
		bytesPerRow = size.Width * 4 // Assume 4 bytes per pixel for RGBA
	}

	rowsPerImage := layout.RowsPerImage
	if rowsPerImage == 0 {
		rowsPerImage = size.Height
	}

	regions := []hal.BufferTextureCopy{
		{
			BufferLayout: hal.ImageDataLayout{
				Offset:       layout.Offset,
				BytesPerRow:  bytesPerRow,
				RowsPerImage: rowsPerImage,
			},
			TextureBase: hal.ImageCopyTexture{
				Texture:  dst.Texture,
				MipLevel: dst.MipLevel,
				Origin: hal.Origin3D{
					X: dst.Origin.X,
					Y: dst.Origin.Y,
					Z: dst.Origin.Z,
				},
				Aspect: dst.Aspect,
			},
			Size: hal.Extent3D{
				Width:              size.Width,
				Height:             size.Height,
				DepthOrArrayLayers: size.DepthOrArrayLayers,
			},
		},
	}

	encoder.CopyBufferToTexture(stagingBuffer, dst.Texture, regions)

	// Transition texture to shader read layout
	encoder.TransitionTextures([]hal.TextureBarrier{
		{
			Texture: dst.Texture,
			Usage: hal.TextureUsageTransition{
				OldUsage: gputypes.TextureUsageCopyDst,
				NewUsage: gputypes.TextureUsageTextureBinding,
				Old:      hal.TextureUsesCopyDst,
				New:      hal.TextureUsesResource,
			},
		},
	})

	// End recording and submit
	cmdBuffer, err := encoder.EndEncoding()
	if err != nil {
		return fmt.Errorf("vulkan: WriteTexture: EndEncoding failed: %w", err)
	}

	// VK-004: Staging uploads must NOT consume swapchain semaphores.
	// When WriteTexture is called between BeginFrame/EndFrame (e.g., in onDraw),
	// the activeSwapchain acquire semaphore must be preserved for the render pass
	// Submit, not consumed by this staging upload. Temporarily clear activeSwapchain
	// so the internal Submit runs without render-pass synchronization.
	q.mu.Lock()
	saved := q.acquired
	q.acquired = nil
	q.mu.Unlock()
	defer func() {
		q.mu.Lock()
		q.acquired = saved
		q.mu.Unlock()
	}()

	// Submit and wait for completion — WriteTexture must block because
	// the staging buffer data must be fully uploaded before return.
	subIdx, err := q.Submit([]hal.CommandBuffer{cmdBuffer})
	if err != nil {
		return fmt.Errorf("vulkan: WriteTexture: Submit failed: %w", err)
	}
	vkTexture.written = true

	// Wait for the submission to complete (60 second timeout).
	timeoutNs := uint64(60 * time.Second)
	if waitErr := q.device.timelineFence.waitForValue(q.device.cmds, q.device.handle, subIdx, timeoutNs); waitErr != nil {
		hal.Logger().Warn("vulkan: WriteTexture: wait failed", "err", waitErr)
	}

	// Free command buffer back to pool after GPU finishes
	q.device.FreeCommandBuffer(cmdBuffer)

	hal.Logger().Debug("vulkan: WriteTexture completed",
		"width", size.Width,
		"height", size.Height,
		"dataSize", len(data),
	)

	return nil
}

// waitForGPU waits for the latest GPU submission to complete.
// Both paths use the unified deviceFence: timeline semaphore (VK-IMPL-001)
// or binary fence pool (VK-IMPL-003).
func (q *Queue) waitForGPU() {
	timeoutNs := uint64(60 * time.Second)
	_ = q.device.timelineFence.waitForLatest(q.device.cmds, q.device.handle, timeoutNs)
}

// Present presents a surface texture to the screen.
//
// damageRects is an optional list of rectangles (physical pixels, top-left
// origin) indicating which surface regions changed this frame. When non-empty
// and VK_KHR_incremental_present is supported, the rects are chained into
// VkPresentInfoKHR.PNext as a compositor hint. When empty or the extension
// is unavailable, the present path is identical to a full-surface present.
func (q *Queue) Present(surface hal.Surface, _ hal.SurfaceTexture, damageRects []image.Rectangle) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	vkSurface, ok := surface.(*Surface)
	if !ok {
		return fmt.Errorf("vulkan: surface is not a Vulkan surface")
	}

	if vkSurface.swapchain == nil {
		return fmt.Errorf("vulkan: surface not configured")
	}

	swapchain := vkSurface.swapchain
	err := swapchain.present(q, damageRects)
	q.release(swapchain)
	return err
}

// GetTimestampPeriod returns the timestamp period in nanoseconds per tick.
// Queried from VkPhysicalDeviceLimits.TimestampPeriod at device init.
// Intel typically 1.0, AMD/NVIDIA vary.
func (q *Queue) GetTimestampPeriod() float32 {
	return q.device.timestampPeriod
}

// SupportsCommandBufferCopies returns true for Vulkan.
// Vulkan uses command buffers for copy operations — PendingWrites batches them.
func (q *Queue) SupportsCommandBufferCopies() bool {
	return true
}

// SetSwapchainSuppressed temporarily disables swapchain semaphore binding
// for subsequent Submit calls. Used for offscreen renders that must not
// consume acquire/present semaphores intended for the compositor submit.
//
// BUG-WGPU-VK-005: When ui uses RepaintBoundary with GPU texture caching,
// there are two Submit calls per frame (offscreen + compositor). Without
// suppression, the first (offscreen) submit hijacks swapchain semaphores,
// leaving the compositor submit without synchronization -> race -> flickering.
//
// When suppressed is true the acquired swapchains are put aside, when false
// they are back, with any acquired in between.
//
// Same save/restore pattern as VK-004 in WriteTexture.
func (q *Queue) SetSwapchainSuppressed(suppressed bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if suppressed {
		if q.swapchainSuppressed {
			return
		}
		q.saved, q.acquired = q.acquired, q.saved[:0]
		q.swapchainSuppressed = true
	} else if q.swapchainSuppressed {
		for _, sc := range q.acquired {
			if !slices.Contains(q.saved, sc) {
				q.saved = append(q.saved, sc)
			}
		}
		q.acquired, q.saved = q.saved, q.acquired[:0]
		q.swapchainSuppressed = false
	}
}

// Vulkan function wrapper

//nolint:unparam // Vulkan API wrapper — signature mirrors vkQueueSubmit spec
func vkQueueSubmit(q *Queue, submitCount uint32, submits *vk.SubmitInfo, fence vk.Fence) vk.Result {
	return q.device.cmds.QueueSubmit(q.handle, submitCount, submits, fence)
}
