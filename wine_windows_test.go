//go:build windows

package wgpu

import "golang.org/x/sys/windows"

// underWine reports a run under Wine, whose vkd3d leaves out some of D3D12.
func underWine() bool {
	return windows.NewLazySystemDLL("ntdll.dll").NewProc("wine_get_version").Find() == nil
}
