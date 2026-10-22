//go:build !(js && wasm)

package vulkan

import (
	"sync"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// liveSet holds the handles of one kind of child object of the device that
// are alive: added on create, removed on destroy, and swept by Device.Destroy.
type liveSet[H comparable] struct {
	mu sync.Mutex
	m  map[H]struct{}
}

func (s *liveSet[H]) add(h H) {
	s.mu.Lock()
	if s.m == nil {
		s.m = make(map[H]struct{})
	}
	s.m[h] = struct{}{}
	s.mu.Unlock()
}

func (s *liveSet[H]) remove(h H) {
	s.mu.Lock()
	delete(s.m, h)
	s.mu.Unlock()
}

// sweep destroys every handle of s with destroy and empties it, returning
// how many there were.
func (s *liveSet[H]) sweep(destroy func(H)) int {
	s.mu.Lock()
	m := s.m
	s.m = nil
	s.mu.Unlock()
	for h := range m {
		destroy(h)
	}
	return len(m)
}

// liveObjects are the device's child objects still alive, by kind.
type liveObjects struct {
	buffers                liveSet[vk.Buffer]
	images                 liveSet[vk.Image]
	imageViews             liveSet[vk.ImageView]
	samplers               liveSet[vk.Sampler]
	setLayouts             liveSet[vk.DescriptorSetLayout]
	pipelineLayouts        liveSet[vk.PipelineLayout]
	pipelines              liveSet[vk.Pipeline]
	shaderModules          liveSet[vk.ShaderModule]
	fences                 liveSet[vk.Fence]
	queryPools             liveSet[vk.QueryPool]
	commandPools           liveSet[vk.CommandPool]
	accelerationStructures liveSet[vk.AccelerationStructureKHR]
}

// sweepLive destroys every child object still alive, with the device idle
// and its framebuffers gone: views before images, acceleration structures
// before their buffers, pools last, and the memory of buffers and images is
// freed with the allocator after.
func (d *Device) sweepLive() {
	cmds, dev := d.cmds, d.handle
	n := d.live.imageViews.sweep(func(h vk.ImageView) { cmds.DestroyImageView(dev, h, nil) })
	n += d.live.pipelines.sweep(func(h vk.Pipeline) { cmds.DestroyPipeline(dev, h, nil) })
	n += d.live.pipelineLayouts.sweep(func(h vk.PipelineLayout) { cmds.DestroyPipelineLayout(dev, h, nil) })
	n += d.live.setLayouts.sweep(func(h vk.DescriptorSetLayout) { cmds.DestroyDescriptorSetLayout(dev, h, nil) })
	n += d.live.shaderModules.sweep(func(h vk.ShaderModule) { cmds.DestroyShaderModule(dev, h, nil) })
	n += d.live.samplers.sweep(func(h vk.Sampler) { cmds.DestroySampler(dev, h, nil) })
	n += d.live.accelerationStructures.sweep(func(h vk.AccelerationStructureKHR) { cmds.DestroyAccelerationStructureKHR(dev, h, nil) })
	n += d.live.queryPools.sweep(func(h vk.QueryPool) { cmds.DestroyQueryPool(dev, h, nil) })
	n += d.live.fences.sweep(func(h vk.Fence) { cmds.DestroyFence(dev, h, nil) })
	n += d.live.images.sweep(func(h vk.Image) { cmds.DestroyImage(dev, h, nil) })
	n += d.live.buffers.sweep(func(h vk.Buffer) { cmds.DestroyBuffer(dev, h, nil) })
	n += d.live.commandPools.sweep(func(h vk.CommandPool) { cmds.DestroyCommandPool(dev, h, nil) })
	if n > 0 {
		hal.Logger().Debug("vulkan: destroyed the child objects left on the device", "objects", n)
	}
}
