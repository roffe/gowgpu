package gputypes

import (
	"testing"
	"unsafe"
)

func TestDrawArgsFieldLayout(t *testing.T) {
	t.Parallel()

	// VkDrawIndirectCommand layout: vertexCount, instanceCount, firstVertex, firstInstance
	// All uint32, tightly packed = 16 bytes total.
	var args DrawArgs

	if got := unsafe.Sizeof(args); got != 16 {
		t.Errorf("sizeof(DrawArgs) = %d, want 16 (4 x uint32)", got)
	}
	if got := unsafe.Offsetof(args.VertexCount); got != 0 {
		t.Errorf("offset(VertexCount) = %d, want 0", got)
	}
	if got := unsafe.Offsetof(args.InstanceCount); got != 4 {
		t.Errorf("offset(InstanceCount) = %d, want 4", got)
	}
	if got := unsafe.Offsetof(args.FirstVertex); got != 8 {
		t.Errorf("offset(FirstVertex) = %d, want 8", got)
	}
	if got := unsafe.Offsetof(args.FirstInstance); got != 12 {
		t.Errorf("offset(FirstInstance) = %d, want 12", got)
	}
}

func TestDrawIndexedArgsFieldLayout(t *testing.T) {
	t.Parallel()

	// VkDrawIndexedIndirectCommand layout: indexCount, instanceCount, firstIndex, baseVertex (i32), firstInstance
	// 5 x 4 bytes = 20 bytes total.
	var args DrawIndexedArgs

	if got := unsafe.Sizeof(args); got != 20 {
		t.Errorf("sizeof(DrawIndexedArgs) = %d, want 20 (5 x 4 bytes)", got)
	}
	if got := unsafe.Offsetof(args.IndexCount); got != 0 {
		t.Errorf("offset(IndexCount) = %d, want 0", got)
	}
	if got := unsafe.Offsetof(args.InstanceCount); got != 4 {
		t.Errorf("offset(InstanceCount) = %d, want 4", got)
	}
	if got := unsafe.Offsetof(args.FirstIndex); got != 8 {
		t.Errorf("offset(FirstIndex) = %d, want 8", got)
	}
	if got := unsafe.Offsetof(args.BaseVertex); got != 12 {
		t.Errorf("offset(BaseVertex) = %d, want 12", got)
	}
	if got := unsafe.Offsetof(args.FirstInstance); got != 16 {
		t.Errorf("offset(FirstInstance) = %d, want 16", got)
	}
}

func TestDrawArgsZeroValue(t *testing.T) {
	t.Parallel()

	var args DrawArgs

	if args.VertexCount != 0 {
		t.Errorf("zero DrawArgs.VertexCount = %d, want 0", args.VertexCount)
	}
	if args.InstanceCount != 0 {
		t.Errorf("zero DrawArgs.InstanceCount = %d, want 0", args.InstanceCount)
	}
	if args.FirstVertex != 0 {
		t.Errorf("zero DrawArgs.FirstVertex = %d, want 0", args.FirstVertex)
	}
	if args.FirstInstance != 0 {
		t.Errorf("zero DrawArgs.FirstInstance = %d, want 0", args.FirstInstance)
	}
}

func TestDrawArgsVertexCountOnly(t *testing.T) {
	t.Parallel()

	// Setting only VertexCount leaves InstanceCount at 0.
	// Users must set InstanceCount explicitly (no implicit default of 1).
	args := DrawArgs{VertexCount: 36}

	if args.VertexCount != 36 {
		t.Errorf("VertexCount = %d, want 36", args.VertexCount)
	}
	if args.InstanceCount != 0 {
		t.Errorf("InstanceCount = %d, want 0 (must be set explicitly)", args.InstanceCount)
	}
}

func TestDrawIndexedArgsBaseVertexSigned(t *testing.T) {
	t.Parallel()

	args := DrawIndexedArgs{
		IndexCount:    100,
		InstanceCount: 1,
		BaseVertex:    -10,
	}

	if args.BaseVertex != -10 {
		t.Errorf("BaseVertex = %d, want -10", args.BaseVertex)
	}
}
