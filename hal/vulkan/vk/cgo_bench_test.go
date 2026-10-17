package vk_test

import (
	"testing"

	"github.com/roffe/gowgpu/hal/vulkan/vk"
	"github.com/roffe/gowgpu/hal/vulkan/vk/vknoop"
)

func cmds() *vk.Commands {
	return vk.NoopCommands(vknoop.Draw(), vknoop.Bind(), vknoop.Create(), vknoop.Scissor(), vknoop.Pipeline())
}

func BenchmarkCmdDraw(b *testing.B) {
	c := cmds()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		c.CmdDraw(1, 4, uint32(i), 0, 0)
	}
}

func BenchmarkCmdBindPipeline(b *testing.B) {
	c := cmds()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		c.CmdBindPipeline(1, vk.PipelineBindPointGraphics, vk.Pipeline(i))
	}
}

// As gogpu's command.go calls it: one set, one dynamic offset, on the stack.
func BenchmarkCmdBindDescriptorSets(b *testing.B) {
	c := cmds()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sets := [1]vk.DescriptorSet{vk.DescriptorSet(i + 1)}
		offs := [1]uint32{256}
		c.CmdBindDescriptorSets(1, vk.PipelineBindPointGraphics, 2, 0, 1, &sets[0], 1, &offs[0])
	}
}

func BenchmarkCmdSetScissor(b *testing.B) {
	c := cmds()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r := vk.Rect2D{Extent: vk.Extent2D{Width: uint32(i), Height: 1}}
		c.CmdSetScissor(1, 0, 1, &r)
	}
}

// A create info that points at more Go memory, as gogpu's do.
func BenchmarkCreateBuffer(b *testing.B) {
	c := cmds()
	families := []uint32{0}
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		info := vk.BufferCreateInfo{SType: vk.StructureTypeBufferCreateInfo, Size: vk.DeviceSize(i), QueueFamilyIndexCount: 1, PQueueFamilyIndices: &families[0]}
		var buf vk.Buffer
		if c.CreateBuffer(1, &info, nil, &buf) != vk.Success || buf != 2 {
			b.Fatal("bad result")
		}
	}
}

var heapInfo *vk.BufferCreateInfo

// A heap create info pointing at heap memory: cgocheck must not object.
func TestCreateBufferHeapInfo(t *testing.T) {
	c := cmds()
	families := new([4]uint32)
	heapInfo = &vk.BufferCreateInfo{SType: vk.StructureTypeBufferCreateInfo, Size: 64, QueueFamilyIndexCount: 1, PQueueFamilyIndices: &families[0]}
	var buf vk.Buffer
	if r := c.CreateBuffer(1, heapInfo, nil, &buf); r != vk.Success || buf != 2 {
		t.Fatalf("CreateBuffer = %d, %d", r, buf)
	}
}
