// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package d3d12

// On 386 a 64-bit argument takes two stack slots, low word first, and a
// 64-bit result comes back in EDX:EAX. Calls that pass one have a second
// SyscallN form for 386 with the value split by lo and hi.

func lo(v uint64) uintptr { return uintptr(v) }
func hi(v uint64) uintptr { return uintptr(v >> 32) }

// join64 rebuilds a 64-bit result from a syscall's r1 and r2.
func join64(r1, r2 uintptr) uint64 {
	if ptrSize == 4 {
		return uint64(r1) | uint64(r2)<<32
	}
	return uint64(r1)
}
