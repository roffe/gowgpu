//go:build !windows && !(js && wasm)

// Package dl opens shared libraries and looks up their symbols at run time
// through dlopen and dlsym, so the backends that call EGL, GL, X11 or Wayland
// need none of those libraries, nor their headers, at build time.
package dl

/*
#cgo linux LDFLAGS: -ldl
#include <stdint.h>

// <dlfcn.h>, declared by hand so no sysroot headers are needed to compile.
extern void *dlopen(const char *, int);
extern void *dlsym(void *, const char *);
extern char *dlerror(void);
#if defined(__APPLE__)
#define DL_RTLD_NOW_LOCAL (0x2 | 0x4)
#elif defined(__ANDROID__) && !defined(__LP64__)
#define DL_RTLD_NOW_LOCAL 0 // 32-bit bionic: RTLD_NOW and RTLD_LOCAL are 0, and 0x2 is RTLD_GLOBAL
#else
#define DL_RTLD_NOW_LOCAL 0x2 // RTLD_LOCAL is 0 on glibc, musl and 64-bit bionic
#endif

static void *dl_open(const uint8_t *name) { return dlopen((const char *)name, DL_RTLD_NOW_LOCAL); }
static void *dl_sym(void *lib, const uint8_t *name) { return dlsym(lib, (const char *)name); }
static char *dl_error(void) { return dlerror(); }
#cgo noescape dl_open
#cgo noescape dl_sym
#cgo nocallback dl_open
#cgo nocallback dl_sym
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

// Lib is an open shared library.
type Lib struct{ handle unsafe.Pointer }

// Open loads the first of names that loads, as dlopen(RTLD_NOW|RTLD_LOCAL).
func Open(names ...string) (Lib, error) {
	var errs []error
	for _, name := range names {
		n := cString(name)
		if h := C.dl_open(&n[0]); h != nil {
			return Lib{h}, nil
		}
		errs = append(errs, fmt.Errorf("%s: %w", name, lastError()))
	}
	return Lib{}, errors.Join(errs...)
}

// Sym returns the address of the named symbol.
func (l Lib) Sym(name string) (unsafe.Pointer, error) {
	n := cString(name)
	p := C.dl_sym(l.handle, &n[0])
	if p == nil {
		return nil, fmt.Errorf("%s: %w", name, lastError())
	}
	return p, nil
}

// SymOrNil returns the address of the named symbol, or nil when the library
// has none: for optional entry points.
func (l Lib) SymOrNil(name string) unsafe.Pointer {
	p, _ := l.Sym(name)
	return p
}

func lastError() error {
	if e := C.dl_error(); e != nil {
		return errors.New(C.GoString(e))
	}
	return errors.New("unknown dl error")
}

func cString(s string) []C.uint8_t {
	b := make([]C.uint8_t, len(s)+1)
	for i := 0; i < len(s); i++ {
		b[i] = C.uint8_t(s[i])
	}
	return b
}
