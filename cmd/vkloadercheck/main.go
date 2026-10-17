// vkloadercheck exercises the cgo loader path against the system Vulkan
// loader without creating an instance: dlopen, vkGetInstanceProcAddr and one
// global command, vkEnumerateInstanceVersion.
package main

import (
	"fmt"
	"os"

	"github.com/roffe/wgpu/hal/vulkan/vk"
)

func main() {
	if err := vk.Init(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	c := vk.NewCommands()
	if err := c.LoadGlobal(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	var v uint32
	r := c.EnumerateInstanceVersion(&v)
	var n uint32
	r2 := c.EnumerateInstanceExtensionProperties(nil, &n, nil)
	fmt.Printf("vkEnumerateInstanceVersion=%d -> %d.%d.%d; instance extensions=%d (%d)\n", r, v>>22, (v>>12)&0x3ff, v&0xfff, n, r2)
}
