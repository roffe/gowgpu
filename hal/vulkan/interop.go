//go:build !(js && wasm)

package vulkan

import (
	"runtime"
	"strings"

	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

// Interop describes the device to code sharing it with gowgpu: a video
// decoder such as FFmpeg, which takes it in its AVVulkanDeviceContext, or
// another API filling memory the device reads, as OpenGL does through
// VK_KHR_external_memory_fd or _win32 where the device has them enabled.
// It has the handles, what was enabled, and the queue families a decoder
// may use.
type Interop struct {
	GetInstanceProcAddr uintptr // vkGetInstanceProcAddr
	Instance            vk.Instance
	PhysicalDevice      vk.PhysicalDevice
	Device              vk.Device
	InstanceExtensions  []string
	DeviceExtensions    []string

	// The features enabled, the 1.1 to 1.3 ones without their PNext.
	Features   vk.PhysicalDeviceFeatures
	Features11 vk.PhysicalDeviceVulkan11Features
	Features12 vk.PhysicalDeviceVulkan12Features
	Features13 vk.PhysicalDeviceVulkan13Features

	// QueueFamilies are, in the order to prefer them, a compute and a video
	// decode family whose queues gowgpu made for the decoder and does not use,
	// then gowgpu's own family, flagged graphics only: the decoder must not
	// use its queues, but shares the images it makes with it. Empty when the
	// GPU cannot decode video with Vulkan Video, or the device was made
	// without what it takes.
	QueueFamilies []InteropQueueFamily
}

// InteropQueueFamily is a queue family, how many of its queues the decoder
// may use, from the first, and what they can do.
type InteropQueueFamily struct {
	Index, Count uint32
	Flags        vk.QueueFlags
}

// Interop returns the device's description for code sharing it, or false
// for a device not made by an adapter.
func (d *Device) Interop() (Interop, bool) {
	if d.interop == nil {
		return Interop{}, false
	}
	return *d.interop, true
}

// externalMemoryExtension is the extension that exports device memory to
// other APIs on this platform.
func externalMemoryExtension() string {
	if runtime.GOOS == "windows" {
		return "VK_KHR_external_memory_win32"
	}
	return "VK_KHR_external_memory_fd"
}

// videoSetup is what a device needs for a video decoder to share it: the
// extensions to enable, with their NUL, and the compute and video decode
// queue families it gets a queue of each of.
type videoSetup struct {
	extensions      []string
	compute, decode uint32
}

// videoExtensions are the device extensions FFmpeg decodes video with, the
// first two required, the codecs at least one, the rest when there. Not the
// video maintenance ones: FFmpeg takes them as their features enabled too.
var videoExtensions = []string{
	"VK_KHR_video_queue",
	"VK_KHR_video_decode_queue",
	"VK_KHR_video_decode_h264",
	"VK_KHR_video_decode_h265",
	"VK_KHR_video_decode_av1",
	"VK_KHR_video_decode_vp9",
}

// videoDecodeSetup returns what the device needs for a video decoder to
// share it, or nil when the GPU has no Vulkan Video decoding, misses a
// feature the decoder relies on, or has no compute family besides graphics
// to give it a queue of its own.
func videoDecodeSetup(apiVersion uint32, available map[string]bool, families []vk.QueueFamilyProperties, graphics uint32,
	f11 *vk.PhysicalDeviceVulkan11Features, f12 *vk.PhysicalDeviceVulkan12Features, f13 *vk.PhysicalDeviceVulkan13Features) *videoSetup {
	if apiVersion < vkMakeVersion(1, 3, 0) || !available[videoExtensions[0]] || !available[videoExtensions[1]] ||
		f11.SamplerYcbcrConversion == 0 || f12.TimelineSemaphore == 0 || f13.Synchronization2 == 0 {
		return nil
	}
	s := &videoSetup{compute: ^uint32(0), decode: ^uint32(0)}
	codecs := 0
	for i, ext := range videoExtensions {
		if available[ext] {
			s.extensions = append(s.extensions, ext+"\x00")
			if i >= 2 && strings.HasPrefix(ext, "VK_KHR_video_decode_") {
				codecs++
			}
		}
	}
	for i, f := range families {
		flags := f.QueueFlags
		switch {
		case f.QueueCount == 0 || uint32(i) == graphics:
		case s.decode == ^uint32(0) && flags&vk.QueueFlags(vk.QueueVideoDecodeBitKhr) != 0:
			s.decode = uint32(i)
		case s.compute == ^uint32(0) && flags&vk.QueueFlags(vk.QueueComputeBit) != 0 && flags&vk.QueueFlags(vk.QueueGraphicsBit) == 0:
			s.compute = uint32(i)
		}
	}
	if codecs == 0 || s.decode == ^uint32(0) || s.compute == ^uint32(0) {
		return nil
	}
	return s
}
