//go:build !windows

package wgpu

func underWine() bool { return false }
