package gputypes

// AccelerationStructureUpdateMode determines how an acceleration structure is updated.
type AccelerationStructureUpdateMode uint8

const (
	// AccelerationStructureUpdateModeBuild always performs a full build.
	AccelerationStructureUpdateModeBuild AccelerationStructureUpdateMode = iota
	// AccelerationStructureUpdateModePreferUpdate performs an incremental update
	// if the hardware supports it (e.g. for skinned meshes), otherwise falls back to full build.
	AccelerationStructureUpdateModePreferUpdate
)

// AccelerationStructureCopyMode determines the type of copy operation.
type AccelerationStructureCopyMode uint8

const (
	// AccelerationStructureCopyModeClone creates a direct duplicate.
	AccelerationStructureCopyModeClone AccelerationStructureCopyMode = iota
	// AccelerationStructureCopyModeCompact duplicates and compacts (20-40% memory reduction).
	AccelerationStructureCopyModeCompact
)

// AccelerationStructureType describes what kind of primitives an acceleration structure contains.
type AccelerationStructureType uint8

const (
	// AccelerationStructureTypeTriangles contains triangle geometry.
	AccelerationStructureTypeTriangles AccelerationStructureType = iota
	// AccelerationStructureTypeAABBs contains axis-aligned bounding boxes.
	AccelerationStructureTypeAABBs
	// AccelerationStructureTypeInstances contains TLAS instance references to BLASes.
	AccelerationStructureTypeInstances
)

// AccelerationStructureFlags are optional flags for acceleration structure creation.
type AccelerationStructureFlags uint8

const (
	// ASFlagAllowUpdate enables incremental updates after initial build.
	ASFlagAllowUpdate AccelerationStructureFlags = 0x01
	// ASFlagAllowCompaction enables compaction via copy operation.
	ASFlagAllowCompaction AccelerationStructureFlags = 0x02
	// ASFlagPreferFastTrace optimizes for trace performance (static geometry).
	ASFlagPreferFastTrace AccelerationStructureFlags = 0x04
	// ASFlagPreferFastBuild optimizes for build performance (dynamic geometry).
	ASFlagPreferFastBuild AccelerationStructureFlags = 0x08
	// ASFlagLowMemory optimizes for low memory usage.
	ASFlagLowMemory AccelerationStructureFlags = 0x10
	// ASFlagUseTransform enables the transform buffer in BLAS triangle builds.
	ASFlagUseTransform AccelerationStructureFlags = 0x20
	// ASFlagAllowRayHitVertexReturn enables triangle vertex retrieval on ray hit.
	ASFlagAllowRayHitVertexReturn AccelerationStructureFlags = 0x40
)

// Contains returns true if all flags in other are set.
func (f AccelerationStructureFlags) Contains(other AccelerationStructureFlags) bool {
	return f&other == other
}

// AccelerationStructureGeometryFlags are per-geometry flags.
type AccelerationStructureGeometryFlags uint8

const (
	// ASGeometryFlagOpaque marks geometry as opaque (skips any-hit shader).
	ASGeometryFlagOpaque AccelerationStructureGeometryFlags = 0x01
	// ASGeometryFlagNoDuplicateAnyHitInvocation limits any-hit to one invocation per primitive.
	ASGeometryFlagNoDuplicateAnyHitInvocation AccelerationStructureGeometryFlags = 0x02
)

// Contains returns true if all flags in other are set.
func (f AccelerationStructureGeometryFlags) Contains(other AccelerationStructureGeometryFlags) bool {
	return f&other == other
}

const (
	// AABBGeometryMinStride is the minimum stride for AABB geometry data (two vec3<f32> = 24 bytes).
	AABBGeometryMinStride uint64 = 24
	// TransformBufferAlignment is the required alignment for transform buffers in BLAS builds.
	TransformBufferAlignment uint64 = 16
	// InstanceBufferAlignment is the required alignment for instance buffers in TLAS builds.
	InstanceBufferAlignment uint64 = 16
)

// BlasTriangleGeometrySizeDescriptor describes the size parameters for a BLAS triangle geometry.
type BlasTriangleGeometrySizeDescriptor struct {
	// VertexFormat is the vertex position format (must be Float32x3 unless
	// FeatureExtendedASVertexFormats is enabled).
	VertexFormat VertexFormat
	// VertexCount is the number of vertices.
	VertexCount uint32
	// IndexFormat is the index format (nil for non-indexed geometry).
	IndexFormat *IndexFormat
	// IndexCount is the number of indices (nil for non-indexed geometry).
	IndexCount *uint32
	// Flags are per-geometry flags.
	Flags AccelerationStructureGeometryFlags
}

// BlasAABBGeometrySizeDescriptor describes the size parameters for a BLAS AABB geometry.
type BlasAABBGeometrySizeDescriptor struct {
	// PrimitiveCount is the number of AABB primitives.
	PrimitiveCount uint32
	// Flags are per-geometry flags.
	Flags AccelerationStructureGeometryFlags
}

// BlasGeometrySizeDescriptors is a discriminated union of BLAS geometry size descriptors.
// Exactly one of Triangles or AABBs must be non-nil.
type BlasGeometrySizeDescriptors struct {
	// Triangles contains triangle geometry descriptors (mutually exclusive with AABBs).
	Triangles []BlasTriangleGeometrySizeDescriptor
	// AABBs contains AABB geometry descriptors (mutually exclusive with Triangles).
	AABBs []BlasAABBGeometrySizeDescriptor
}

// CreateBlasDescriptor describes a bottom-level acceleration structure to create.
type CreateBlasDescriptor struct {
	// Label is an optional debug label.
	Label string
	// Flags are acceleration structure creation flags.
	Flags AccelerationStructureFlags
	// UpdateMode determines how the BLAS is updated after initial build.
	UpdateMode AccelerationStructureUpdateMode
}

// CreateTlasDescriptor describes a top-level acceleration structure to create.
type CreateTlasDescriptor struct {
	// Label is an optional debug label.
	Label string
	// MaxInstances is the maximum number of instances this TLAS can hold.
	MaxInstances uint32
	// Flags are acceleration structure creation flags.
	Flags AccelerationStructureFlags
	// UpdateMode determines how the TLAS is updated after initial build.
	UpdateMode AccelerationStructureUpdateMode
}

// AccelerationStructureBindingLayout describes an acceleration structure binding
// in a bind group layout.
type AccelerationStructureBindingLayout struct {
	// VertexReturn enables triangle vertex data retrieval on ray hit.
	// Requires FeatureRayHitVertexReturn.
	VertexReturn bool
}
