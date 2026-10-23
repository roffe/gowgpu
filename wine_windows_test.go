//go:build windows

package wgpu

import "golang.org/x/sys/windows"

// underWine reports a run under Wine, whose vkd3d leaves out some of D3D12.
func underWine() bool {
	return windows.NewLazySystemDLL("ntdll.dll").NewProc("wine_get_version").Find() == nil
}

// upstreamVkd3d reports a run under Wine with Wine's own d3d12.dll, which is
// built on upstream vkd3d and leaves out parts of D3D12 that vkd3d-proton and
// Windows have. Wine marks the DLLs it builds itself with "Wine builtin DLL"
// after their DOS header, and vkd3d-proton's d3d12.dll is a native one.
func upstreamVkd3d() bool {
	if !underWine() {
		return false
	}
	d3d12 := windows.NewLazySystemDLL("d3d12.dll")
	if d3d12.Load() != nil {
		return false
	}
	var mark [16]byte
	if windows.ReadProcessMemory(windows.CurrentProcess(), d3d12.Handle()+0x40, &mark[0], uintptr(len(mark)), nil) != nil {
		return false
	}
	return string(mark[:]) == "Wine builtin DLL"
}
