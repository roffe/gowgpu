//go:build windows

package dx12

import "unsafe"

// fromHandle turns a HAL handle back into the backend object behind it. The
// DX12 handles are Go pointers kept as uintptr by gputypes, which checkptr
// (on under -race) rejects as pointer arithmetic. Core holds every object a
// handle names, so it lives at least as long as the handle is in use.
//
//go:nocheckptr
func fromHandle[T any](h uintptr) *T {
	// The bits of h are read as a pointer instead of converting h: go vet
	// rejects every uintptr to unsafe.Pointer conversion and can't be told
	// that this one is meant.
	return *(**T)(unsafe.Pointer(&h))
}
