//go:build windows

package vk

import (
	"syscall"
	"unsafe"
)

// Windows needs no cgo to load: LoadLibrary and GetProcAddress are syscalls.

func openLibrary(name string) (uintptr, error) {
	h, err := syscall.LoadLibrary(name)
	return uintptr(h), err
}

func librarySymbol(lib uintptr, name string) (unsafe.Pointer, error) {
	p, err := syscall.GetProcAddress(syscall.Handle(lib), name)
	if err != nil {
		return nil, err
	}
	return cPointer(p), nil
}

func closeLibrary(lib uintptr) error {
	return syscall.FreeLibrary(syscall.Handle(lib))
}
