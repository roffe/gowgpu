//go:build !windows && !(js && wasm)

package vk

/*
#cgo linux LDFLAGS: -ldl
#include <stdint.h>

// <dlfcn.h>, declared by hand so no sysroot headers are needed to compile.
extern void *dlopen(const char *, int);
extern void *dlsym(void *, const char *);
extern int dlclose(void *);
extern char *dlerror(void);
#if defined(__APPLE__)
#define GVK_RTLD_NOW_LOCAL (0x2 | 0x4)
#else
#define GVK_RTLD_NOW_LOCAL 0x2 // RTLD_LOCAL is 0 on glibc, musl and 64-bit bionic
#endif

static uintptr_t gvk_dlopen(const char *name) { return (uintptr_t)dlopen(name, GVK_RTLD_NOW_LOCAL); }
static uintptr_t gvk_dlsym(uintptr_t lib, const char *name) { return (uintptr_t)dlsym((void *)lib, name); }
static int gvk_dlclose(uintptr_t lib) { return dlclose((void *)lib); }
static uintptr_t gvk_dlerror(void) { return (uintptr_t)dlerror(); }
*/
import "C"

import (
	"errors"
	"unsafe"
)

func cString(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}

func dlerror() error {
	p := uintptr(C.gvk_dlerror())
	if p == 0 {
		return errors.New("unknown dl error")
	}
	b := unsafe.Slice((*byte)(cPointer(p)), 4096)
	for i, c := range b {
		if c == 0 {
			return errors.New(string(b[:i]))
		}
	}
	return errors.New(string(b))
}

// The names cross as pointers. As integers cgo did nothing for them, and a
// short name's buffer sits on the stack, which can be copied elsewhere before
// C reads it.

func openLibrary(name string) (uintptr, error) {
	n := cString(name)
	lib := uintptr(C.gvk_dlopen((*C.char)(unsafe.Pointer(&n[0]))))
	if lib == 0 {
		return 0, dlerror()
	}
	return lib, nil
}

func librarySymbol(lib uintptr, name string) (unsafe.Pointer, error) {
	n := cString(name)
	p := uintptr(C.gvk_dlsym(C.uintptr_t(lib), (*C.char)(unsafe.Pointer(&n[0]))))
	if p == 0 {
		return nil, dlerror()
	}
	return cPointer(p), nil
}

func closeLibrary(lib uintptr) error {
	if C.gvk_dlclose(C.uintptr_t(lib)) != 0 {
		return dlerror()
	}
	return nil
}
