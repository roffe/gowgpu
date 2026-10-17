package gputypes

// DrawArgs describes parameters for a non-indexed draw call.
//
// Field layout matches VkDrawIndirectCommand and D3D12_DRAW_ARGUMENTS exactly,
// enabling zero-copy use as indirect draw argument buffers.
// Using a struct instead of 4 positional uint32 prevents silent
// argument swap bugs (Go has no named arguments).
type DrawArgs struct {
	// VertexCount is the number of vertices to draw.
	VertexCount uint32
	// InstanceCount is the number of instances to draw.
	InstanceCount uint32
	// FirstVertex is the index of the first vertex to draw.
	FirstVertex uint32
	// FirstInstance is the instance ID of the first instance to draw.
	// Must be 0 unless FeatureIndirectFirstInstance is enabled.
	FirstInstance uint32
}

// DrawIndexedArgs describes parameters for an indexed draw call.
//
// Field layout matches VkDrawIndexedIndirectCommand and D3D12_DRAW_INDEXED_ARGUMENTS
// exactly, enabling zero-copy use as indirect draw argument buffers.
type DrawIndexedArgs struct {
	// IndexCount is the number of indices to draw.
	IndexCount uint32
	// InstanceCount is the number of instances to draw.
	InstanceCount uint32
	// FirstIndex is the offset into the index buffer.
	FirstIndex uint32
	// BaseVertex is the value added to the vertex index before
	// indexing into the vertex buffer. Signed to allow negative offsets.
	BaseVertex int32
	// FirstInstance is the instance ID of the first instance to draw.
	// Must be 0 unless FeatureIndirectFirstInstance is enabled.
	FirstInstance uint32
}
