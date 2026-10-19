//go:build !(js && wasm)

package hal

import "github.com/roffe/gowgpu/gputypes"

// CommandEncoder records GPU commands.
//
// Lifecycle: After EndEncoding, the encoder is in "closed" state. After
// ResetAll (with completed command buffers), the encoder is ready for a
// new BeginEncoding cycle. This enables encoder pooling at the wgpu core
// level, matching Rust wgpu-core's CommandAllocator pattern.
type CommandEncoder interface {
	// BeginEncoding begins command recording with an optional label.
	BeginEncoding(label string) error

	// EndEncoding finishes command recording and returns a command buffer.
	// The encoder enters "closed" state but retains its internal resources
	// (e.g., DX12 allocator, Vulkan command pool). Call ResetAll after GPU
	// completion to prepare for the next BeginEncoding cycle.
	EndEncoding() (CommandBuffer, error)

	// DiscardEncoding discards the encoder without creating a command buffer.
	// Use this to cancel encoding that encountered errors.
	DiscardEncoding()

	// ResetAll resets the encoder and associated command buffers for reuse.
	// Must be called after the GPU has completed all commands from previous
	// EndEncoding calls. After ResetAll, the encoder is ready for BeginEncoding.
	// Not all backends need this (GLES/Software/Metal/Noop are no-ops).
	ResetAll(commandBuffers []CommandBuffer)

	// Destroy releases all GPU resources owned by this encoder.
	// Must be called when the encoder is no longer needed (e.g., during
	// device shutdown or encoder pool cleanup). The encoder must not be
	// in recording state when Destroy is called.
	Destroy()

	// TransitionBuffers transitions buffer states for synchronization.
	// This is required on some backends (Vulkan, DX12) but no-op on others (Metal).
	TransitionBuffers(barriers []BufferBarrier)

	// TransitionTextures transitions texture states for synchronization.
	// This is required on some backends (Vulkan, DX12) but no-op on others (Metal).
	TransitionTextures(barriers []TextureBarrier)

	// ClearBuffer clears a buffer region to zero. The core has checked the
	// range: offset and size are multiples of 4, size is not 0, and the
	// buffer is in the copy destination state.
	ClearBuffer(buffer Buffer, offset, size uint64)

	// CopyBufferToBuffer copies data between buffers.
	CopyBufferToBuffer(src, dst Buffer, regions []BufferCopy)

	// CopyBufferToTexture copies data from a buffer to a texture.
	CopyBufferToTexture(src Buffer, dst Texture, regions []BufferTextureCopy)

	// CopyTextureToBuffer copies data from a texture to a buffer.
	CopyTextureToBuffer(src Texture, dst Buffer, regions []BufferTextureCopy)

	// CopyTextureToTexture copies data between textures.
	CopyTextureToTexture(src, dst Texture, regions []TextureCopy)

	// ResolveQuerySet copies query results from a query set into a buffer.
	// firstQuery is the index of the first query to resolve.
	// queryCount is the number of queries to resolve.
	// destination is the buffer to write results to.
	// destinationOffset is the byte offset into the destination buffer.
	// Each timestamp result is a uint64 (8 bytes).
	ResolveQuerySet(querySet QuerySet, firstQuery, queryCount uint32, destination Buffer, destinationOffset uint64)

	// BeginRenderPass begins a render pass.
	// Returns a render pass encoder for recording draw commands.
	BeginRenderPass(desc *RenderPassDescriptor) RenderPassEncoder

	// BeginComputePass begins a compute pass.
	// Returns a compute pass encoder for recording dispatch commands.
	BeginComputePass(desc *ComputePassDescriptor) ComputePassEncoder

	// BuildAccelerationStructures builds one or more acceleration structures.
	// Batched to match Vulkan vkCmdBuildAccelerationStructuresKHR.
	// No-op on backends without RT support.
	BuildAccelerationStructures(descriptors []BuildAccelerationStructureDescriptor)

	// PlaceAccelerationStructureBarrier inserts an AS memory barrier.
	PlaceAccelerationStructureBarrier(barrier AccelerationStructureBarrier)

	// CopyAccelerationStructure copies or compacts an AS.
	CopyAccelerationStructure(src, dst AccelerationStructure, copyMode gputypes.AccelerationStructureCopyMode)

	// ReadAccelerationStructureCompactSize reads the post-compact size of an AS
	// into the given buffer. Used for the compaction state machine.
	ReadAccelerationStructureCompactSize(as AccelerationStructure, buffer Buffer, offset uint64)
}

// RenderPassEncoder records render commands within a render pass.
// Render passes define rendering targets and operations.
type RenderPassEncoder interface {
	// End finishes the render pass.
	// After this call, the encoder cannot be used again.
	End()

	// SetPipeline sets the active render pipeline.
	SetPipeline(pipeline RenderPipeline)

	// SetBindGroup sets a bind group for the given index.
	// offsets are dynamic offsets for dynamic uniform/storage buffers.
	SetBindGroup(index uint32, group BindGroup, offsets []uint32)

	// SetVertexBuffer sets a vertex buffer for the given slot.
	SetVertexBuffer(slot uint32, buffer Buffer, offset uint64)

	// SetIndexBuffer sets the index buffer.
	SetIndexBuffer(buffer Buffer, format gputypes.IndexFormat, offset uint64)

	// SetViewport sets the viewport transformation.
	SetViewport(vp gputypes.Viewport)

	// SetScissorRect sets the scissor rectangle for clipping.
	SetScissorRect(rect gputypes.ScissorRect)

	// SetBlendConstant sets the blend constant color.
	SetBlendConstant(color *gputypes.Color)

	// SetStencilReference sets the stencil reference value.
	SetStencilReference(reference uint32)

	// Draw draws primitives.
	Draw(args gputypes.DrawArgs)

	// DrawIndexed draws indexed primitives.
	DrawIndexed(args gputypes.DrawIndexedArgs)

	// DrawIndirect draws primitives with GPU-generated parameters.
	// buffer contains drawCount consecutive 16-byte DrawIndirectArgs records.
	DrawIndirect(buffer Buffer, offset uint64, drawCount uint32)

	// DrawIndexedIndirect draws indexed primitives with GPU-generated parameters.
	// buffer contains drawCount consecutive 20-byte DrawIndexedIndirectArgs records.
	DrawIndexedIndirect(buffer Buffer, offset uint64, drawCount uint32)

	// DrawIndirectCount draws primitives using a GPU count buffer (Vulkan 1.2+).
	// countBuffer holds a single uint32 draw count at countOffset.
	DrawIndirectCount(buffer Buffer, offset uint64, countBuffer Buffer, countOffset uint64, maxDrawCount uint32)

	// DrawIndexedIndirectCount draws indexed primitives using a GPU count buffer.
	DrawIndexedIndirectCount(buffer Buffer, offset uint64, countBuffer Buffer, countOffset uint64, maxDrawCount uint32)

	// ExecuteBundle executes a pre-recorded render bundle.
	// Bundles are an optimization for repeated draw calls.
	ExecuteBundle(bundle RenderBundle)
}

// ComputePassEncoder records compute commands within a compute pass.
type ComputePassEncoder interface {
	// End finishes the compute pass.
	// After this call, the encoder cannot be used again.
	End()

	// SetPipeline sets the active compute pipeline.
	SetPipeline(pipeline ComputePipeline)

	// SetBindGroup sets a bind group for the given index.
	// offsets are dynamic offsets for dynamic uniform/storage buffers.
	SetBindGroup(index uint32, group BindGroup, offsets []uint32)

	// Dispatch dispatches compute work.
	// x, y, z are the number of workgroups to dispatch in each dimension.
	Dispatch(x, y, z uint32)

	// DispatchIndirect dispatches compute work with GPU-generated parameters.
	// buffer contains DispatchIndirectArgs at the given offset.
	DispatchIndirect(buffer Buffer, offset uint64)
}

// RenderBundle is a pre-recorded set of render commands.
// Bundles can be executed multiple times for better performance.
type RenderBundle interface {
	Resource
}

// RenderBundleEncoder records commands into a render bundle.
// The recorded commands can be replayed multiple times via ExecuteBundle.
type RenderBundleEncoder interface {
	// SetPipeline sets the active render pipeline.
	SetPipeline(pipeline RenderPipeline)

	// SetBindGroup sets a bind group for the given index.
	SetBindGroup(index uint32, group BindGroup, offsets []uint32)

	// SetVertexBuffer sets a vertex buffer for the given slot.
	SetVertexBuffer(slot uint32, buffer Buffer, offset uint64)

	// SetIndexBuffer sets the index buffer.
	SetIndexBuffer(buffer Buffer, format gputypes.IndexFormat, offset uint64)

	// Draw draws primitives.
	Draw(args gputypes.DrawArgs)

	// DrawIndexed draws indexed primitives.
	DrawIndexed(args gputypes.DrawIndexedArgs)

	// Finish finalizes the bundle and returns it.
	// The encoder cannot be used after this call.
	Finish() RenderBundle
}

// BufferBarrier defines a buffer state transition.
type BufferBarrier struct {
	Buffer Buffer
	Usage  BufferUsageTransition
}

// TextureBarrier defines a texture state transition.
type TextureBarrier struct {
	Texture Texture
	Range   TextureRange
	Usage   TextureUsageTransition
}

// BufferUsageTransition defines a buffer usage state transition.
type BufferUsageTransition struct {
	OldUsage gputypes.BufferUsage
	NewUsage gputypes.BufferUsage
}

// TextureUsageTransition defines a texture usage state transition.
//
// OldUsage and NewUsage are the coarse WebGPU usages. Old and New are the
// fine-grained states behind them when the caller knows them (the core
// tracker always does): gputypes folds a read-only depth attachment and a
// written one into RenderAttachment, while Vulkan keeps them in different
// layouts. Zero means not known, and the backend derives the state from
// the coarse usage and the texture format (TextureUsesFromUsage).
type TextureUsageTransition struct {
	OldUsage gputypes.TextureUsage
	NewUsage gputypes.TextureUsage
	Old      TextureUses
	New      TextureUses
}

// TextureUses is the fine-grained usage state of a texture, the states the
// core tracker tells apart. core/track.TextureUses has the same bits, so a
// tracked state converts by value.
//
// Reference: wgpu-types TextureUses (wgpu-types/src/texture.rs).
type TextureUses uint32

// Texture usage states.
const (
	TextureUsesUninitialized     TextureUses = 1 << 0 // Unknown/junk contents
	TextureUsesPresent           TextureUses = 1 << 1 // Ready for surface presentation
	TextureUsesCopySrc           TextureUses = 1 << 2 // Source of a hardware copy
	TextureUsesCopyDst           TextureUses = 1 << 3 // Destination of a hardware copy
	TextureUsesResource          TextureUses = 1 << 4 // Read-only sampled or fetched
	TextureUsesColorTarget       TextureUses = 1 << 5 // Render pass color target
	TextureUsesDepthStencilRead  TextureUses = 1 << 6 // Read-only depth/stencil attachment
	TextureUsesDepthStencilWrite TextureUses = 1 << 7 // Read-write depth/stencil attachment
	TextureUsesStorageRead       TextureUses = 1 << 8 // Storage texture read-only
	TextureUsesStorageWrite      TextureUses = 1 << 9 // Storage texture write-only
)

// TextureUsesFromUsage is the fine-grained state a coarse usage stands for:
// RenderAttachment is the depth/stencil write state on a depth/stencil
// format and the color target otherwise, StorageBinding the exclusive
// storage write state so later uses are always fenced, and 0 stays 0 (the
// uninitialized state).
//
// Reference: wgpu-core conv.rs map_texture_usage
func TextureUsesFromUsage(usage gputypes.TextureUsage, depthStencil bool) TextureUses {
	var uses TextureUses
	if usage&gputypes.TextureUsageCopySrc != 0 {
		uses |= TextureUsesCopySrc
	}
	if usage&gputypes.TextureUsageCopyDst != 0 {
		uses |= TextureUsesCopyDst
	}
	if usage&gputypes.TextureUsageTextureBinding != 0 {
		uses |= TextureUsesResource
	}
	if usage&gputypes.TextureUsageStorageBinding != 0 {
		uses |= TextureUsesStorageWrite
	}
	if usage&gputypes.TextureUsageRenderAttachment != 0 {
		if depthStencil {
			uses |= TextureUsesDepthStencilWrite
		} else {
			uses |= TextureUsesColorTarget
		}
	}
	return uses
}

// TextureRange specifies a range of texture subresources.
type TextureRange struct {
	// Aspect specifies which aspect of the texture (color, depth, stencil).
	Aspect gputypes.TextureAspect

	// BaseMipLevel is the first mip level in the range.
	BaseMipLevel uint32

	// MipLevelCount is the number of mip levels (0 means all remaining levels).
	MipLevelCount uint32

	// BaseArrayLayer is the first array layer in the range.
	BaseArrayLayer uint32

	// ArrayLayerCount is the number of array layers (0 means all remaining layers).
	ArrayLayerCount uint32
}

// BufferCopy defines a buffer-to-buffer copy region.
type BufferCopy struct {
	SrcOffset uint64
	DstOffset uint64
	Size      uint64
}

// BufferTextureCopy defines a buffer-texture copy region.
type BufferTextureCopy struct {
	BufferLayout ImageDataLayout
	TextureBase  ImageCopyTexture
	Size         Extent3D
}

// TextureCopy defines a texture-to-texture copy region.
type TextureCopy struct {
	SrcBase ImageCopyTexture
	DstBase ImageCopyTexture
	Size    Extent3D
}

// ImageDataLayout describes the layout of image data in a buffer.
type ImageDataLayout struct {
	// Offset is the offset in bytes from the start of the buffer.
	Offset uint64

	// BytesPerRow is the stride in bytes between rows of the image.
	// Must be a multiple of 256 for texture copies.
	// Can be 0 for single-row images.
	BytesPerRow uint32

	// RowsPerImage is the number of rows per image slice.
	// Only needed for 3D textures.
	// Can be 0 to use the image height.
	RowsPerImage uint32
}

// ImageCopyTexture specifies a texture location for copying.
type ImageCopyTexture struct {
	// Texture is the texture to copy to/from.
	Texture Texture

	// MipLevel is the mip level to copy.
	MipLevel uint32

	// Origin is the starting point of the copy.
	Origin Origin3D

	// Aspect specifies which aspect to copy (color, depth, stencil).
	Aspect gputypes.TextureAspect
}

// Origin3D is a 3D origin point.
type Origin3D struct {
	X uint32
	Y uint32
	Z uint32
}

// Extent3D is a 3D extent.
type Extent3D struct {
	Width              uint32
	Height             uint32
	DepthOrArrayLayers uint32
}
