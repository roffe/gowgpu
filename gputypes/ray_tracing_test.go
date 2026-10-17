package gputypes

import "testing"

func TestRTFeatureBits(t *testing.T) {
	t.Parallel()

	features := []struct {
		name string
		feat Feature
		bit  int
	}{
		{"RayQuery", FeatureRayQuery, 20},
		{"RayHitVertexReturn", FeatureRayHitVertexReturn, 21},
		{"ExtendedASVertexFormats", FeatureExtendedASVertexFormats, 22},
		{"ASBindingArray", FeatureASBindingArray, 23},
		{"RayTracingPipelines", FeatureRayTracingPipelines, 24},
	}

	for _, tt := range features {
		t.Run(tt.name, func(t *testing.T) {
			expected := Feature(1 << tt.bit)
			if tt.feat != expected {
				t.Errorf("%s = %d (bit %d), want %d (bit %d)",
					tt.name, tt.feat, bitPosition(uint64(tt.feat)), expected, tt.bit)
			}
			if tt.feat.String() != tt.name {
				t.Errorf("String() = %q, want %q", tt.feat.String(), tt.name)
			}
		})
	}
}

func TestRTFeatureContains(t *testing.T) {
	t.Parallel()

	var f Features
	f.Insert(FeatureRayQuery)
	f.Insert(FeatureRayHitVertexReturn)

	if !f.Contains(FeatureRayQuery) {
		t.Error("should contain RayQuery")
	}
	if !f.Contains(FeatureRayHitVertexReturn) {
		t.Error("should contain RayHitVertexReturn")
	}
	if f.Contains(FeatureRayTracingPipelines) {
		t.Error("should not contain RayTracingPipelines")
	}
	if f.Count() != 2 {
		t.Errorf("Count() = %d, want 2", f.Count())
	}
}

func TestRTFeatureNoOverlapWithExisting(t *testing.T) {
	t.Parallel()

	existing := []Feature{
		FeatureDepthClipControl, FeatureDepth32FloatStencil8,
		FeatureTextureCompressionBC, FeatureTextureCompressionETC2,
		FeatureTextureCompressionASTC, FeatureIndirectFirstInstance,
		FeatureShaderF16, FeatureRG11B10UfloatRenderable,
		FeatureBGRA8UnormStorage, FeatureFloat32Filterable,
		FeatureTimestampQuery, FeaturePipelineStatisticsQuery,
		FeatureMultiDrawIndirect, FeatureMultiDrawIndirectCount,
		FeaturePushConstants, FeatureTextureAdapterSpecificFormatFeatures,
		FeatureShaderFloat64, FeatureVertexAttribute64bit,
		FeatureSubgroupOperations, FeatureSubgroupBarrier,
	}
	rt := []Feature{
		FeatureRayQuery, FeatureRayHitVertexReturn,
		FeatureExtendedASVertexFormats, FeatureASBindingArray,
		FeatureRayTracingPipelines,
	}

	for _, e := range existing {
		for _, r := range rt {
			if e&r != 0 {
				t.Errorf("overlap: %s (bit %d) & %s (bit %d)",
					e.String(), bitPosition(uint64(e)),
					r.String(), bitPosition(uint64(r)))
			}
		}
	}
}

func TestRTBufferUsageBits(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		usage BufferUsage
		hex   uint64
	}{
		{"AccelerationStructureScratch", BufferUsageAccelerationStructureScratch, 0x0800},
		{"BlasInput", BufferUsageBlasInput, 0x1000},
		{"TlasInput", BufferUsageTlasInput, 0x2000},
		{"AccelerationStructureQuery", BufferUsageAccelerationStructureQuery, 0x4000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if uint64(tt.usage) != tt.hex {
				t.Errorf("%s = 0x%04X, want 0x%04X", tt.name, uint64(tt.usage), tt.hex)
			}
		})
	}
}

func TestRTBufferUsageNoOverlap(t *testing.T) {
	t.Parallel()

	existing := []BufferUsage{
		BufferUsageMapRead, BufferUsageMapWrite,
		BufferUsageCopySrc, BufferUsageCopyDst,
		BufferUsageIndex, BufferUsageVertex,
		BufferUsageUniform, BufferUsageStorage,
		BufferUsageIndirect, BufferUsageQueryResolve,
	}
	rt := []BufferUsage{
		BufferUsageAccelerationStructureScratch,
		BufferUsageBlasInput, BufferUsageTlasInput,
		BufferUsageAccelerationStructureQuery,
	}

	for _, e := range existing {
		for _, r := range rt {
			if e&r != 0 {
				t.Errorf("overlap: 0x%X & 0x%X", uint64(e), uint64(r))
			}
		}
	}
}

func TestRTBufferUsageContainsUnknownBits(t *testing.T) {
	t.Parallel()

	valid := BufferUsageVertex | BufferUsageBlasInput
	if valid.ContainsUnknownBits() {
		t.Error("valid RT usage should not contain unknown bits")
	}

	invalid := BufferUsage(0x80000000)
	if !invalid.ContainsUnknownBits() {
		t.Error("0x80000000 should be unknown")
	}
}

func TestASFlagsContains(t *testing.T) {
	t.Parallel()

	flags := ASFlagAllowUpdate | ASFlagAllowCompaction | ASFlagPreferFastTrace
	if !flags.Contains(ASFlagAllowUpdate) {
		t.Error("should contain AllowUpdate")
	}
	if !flags.Contains(ASFlagAllowCompaction) {
		t.Error("should contain AllowCompaction")
	}
	if flags.Contains(ASFlagLowMemory) {
		t.Error("should not contain LowMemory")
	}
}

func TestASGeometryFlagsContains(t *testing.T) {
	t.Parallel()

	flags := ASGeometryFlagOpaque | ASGeometryFlagNoDuplicateAnyHitInvocation
	if !flags.Contains(ASGeometryFlagOpaque) {
		t.Error("should contain Opaque")
	}
	if ASGeometryFlagOpaque.Contains(ASGeometryFlagNoDuplicateAnyHitInvocation) {
		t.Error("Opaque alone should not contain NoDuplicateAnyHit")
	}
}

func TestRTConstants(t *testing.T) {
	t.Parallel()

	if AABBGeometryMinStride != 24 {
		t.Errorf("AABBGeometryMinStride = %d, want 24", AABBGeometryMinStride)
	}
	if TransformBufferAlignment != 16 {
		t.Errorf("TransformBufferAlignment = %d, want 16", TransformBufferAlignment)
	}
	if InstanceBufferAlignment != 16 {
		t.Errorf("InstanceBufferAlignment = %d, want 16", InstanceBufferAlignment)
	}
}

func bitPosition(v uint64) int {
	pos := 0
	for v > 1 {
		v >>= 1
		pos++
	}
	return pos
}
