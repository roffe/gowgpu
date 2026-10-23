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

// dlerror's message is the calling thread's, and a later dl call on that
// thread may replace it, so each call copies it into msg, n bytes with the
// NUL, before it returns.
static void dl_error(uint8_t *msg, int n) {
	const char *e = dlerror();
	int i = 0;
	for (; e && e[i] && i < n - 1; i++) msg[i] = (uint8_t)e[i];
	msg[i] = 0;
}
static void *dl_open(const uint8_t *name, uint8_t *msg, int n) {
	void *h = dlopen((const char *)name, DL_RTLD_NOW_LOCAL);
	if (!h) dl_error(msg, n);
	return h;
}
static void *dl_sym(void *lib, const uint8_t *name, uint8_t *msg, int n) {
	dlerror();
	void *p = dlsym(lib, (const char *)name);
	if (!p) dl_error(msg, n);
	return p;
}
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
	var msg errMsg
	for _, name := range names {
		n := cString(name)
		if h := C.dl_open(&n[0], &msg[0], C.int(len(msg))); h != nil {
			return Lib{h}, nil
		}
		errs = append(errs, fmt.Errorf("%s: %w", name, msg.err()))
	}
	return Lib{}, errors.Join(errs...)
}

// Sym returns the address of the named symbol.
func (l Lib) Sym(name string) (unsafe.Pointer, error) {
	var msg errMsg
	n := cString(name)
	p := C.dl_sym(l.handle, &n[0], &msg[0], C.int(len(msg)))
	if p == nil {
		return nil, fmt.Errorf("%s: %w", name, msg.err())
	}
	return p, nil
}

// SymOrNil returns the address of the named symbol, or nil when the library
// has none: for optional entry points.
func (l Lib) SymOrNil(name string) unsafe.Pointer {
	p, _ := l.Sym(name)
	return p
}

// errMsg is the dlerror message of a failed call.
type errMsg [512]C.uint8_t

func (m *errMsg) err() error {
	if m[0] == 0 {
		return errors.New("unknown dl error")
	}
	return errors.New(C.GoString((*C.char)(unsafe.Pointer(&m[0]))))
}

func cString(s string) []C.uint8_t {
	b := make([]C.uint8_t, len(s)+1)
	for i := 0; i < len(s); i++ {
		b[i] = C.uint8_t(s[i])
	}
	return b
}
