// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build darwin && !(js && wasm)

package metal

/*
#cgo CFLAGS: -fblocks
#cgo noescape gm1p
#cgo nocallback gm1p
#cgo noescape gm3p
#cgo nocallback gm3p
#cgo noescape gm2e
#cgo nocallback gm2e
#cgo noescape gm3e
#cgo nocallback gm3e
#cgo noescape gm_v_6upppbpp
#cgo nocallback gm_v_6upppbpp
#cgo noescape gm_class
#cgo nocallback gm_class
#cgo noescape gm_sel
#cgo nocallback gm_sel
#cgo LDFLAGS: -lobjc -framework Foundation -framework Metal -framework QuartzCore
#include <stdint.h>

// <objc/message.h>, <objc/runtime.h>, <dlfcn.h>, <stdlib.h> and Block.h,
// declared by hand: compiling needs no SDK headers (linking does, as always).
extern void objc_msgSend(void);
#if defined(__x86_64__)
extern void objc_msgSend_stret(void);
#endif
extern void *objc_getClass(const char *);
extern void *sel_registerName(const char *);
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);
extern void *MTLCreateSystemDefaultDevice(void);
extern void *dlsym(void *, const char *);

typedef uintptr_t U;
typedef struct { double a, b; } G2d;                // CGSize
typedef struct { double a, b, c, d; } G4d;          // MTLClearColor
typedef struct { double a, b, c, d, e, f; } G6d;    // MTLViewport
typedef struct { U a, b; } G2u;                     // NSRange
typedef struct { U a, b, c; } G3u;                  // MTLOrigin, MTLSize, MTLAccelerationStructureSizes
typedef struct { U a, b, c, d; } G4u;               // MTLScissorRect
typedef struct { U a, b, c, d, e, f; } G6u;         // MTLRegion

// Integer sends by arity: ids, NSUIntegers, pointers and BOOLs all travel in
// integer registers (or 8-byte stack slots) on arm64 and x86_64.
static U gm0(U o, U s) { return ((U (*)(U, U))objc_msgSend)(o, s); }
static U gm1(U o, U s, U a) { return ((U (*)(U, U, U))objc_msgSend)(o, s, a); }
static U gm2(U o, U s, U a, U b) { return ((U (*)(U, U, U, U))objc_msgSend)(o, s, a, b); }
static U gm3(U o, U s, U a, U b, U c) { return ((U (*)(U, U, U, U, U))objc_msgSend)(o, s, a, b, c); }
static U gm4(U o, U s, U a, U b, U c, U d) { return ((U (*)(U, U, U, U, U, U))objc_msgSend)(o, s, a, b, c, d); }
static U gm5(U o, U s, U a, U b, U c, U d, U e) { return ((U (*)(U, U, U, U, U, U, U))objc_msgSend)(o, s, a, b, c, d, e); }
static U gm6(U o, U s, U a, U b, U c, U d, U e, U f) { return ((U (*)(U, U, U, U, U, U, U, U))objc_msgSend)(o, s, a, b, c, d, e, f); }
static U gm7(U o, U s, U a, U b, U c, U d, U e, U f, U g) { return ((U (*)(U, U, U, U, U, U, U, U, U))objc_msgSend)(o, s, a, b, c, d, e, f, g); }
static U gm8(U o, U s, U a, U b, U c, U d, U e, U f, U g, U h) { return ((U (*)(U, U, U, U, U, U, U, U, U, U))objc_msgSend)(o, s, a, b, c, d, e, f, g, h); }
static U gm9(U o, U s, U a, U b, U c, U d, U e, U f, U g, U h, U i) { return ((U (*)(U, U, U, U, U, U, U, U, U, U, U))objc_msgSend)(o, s, a, b, c, d, e, f, g, h, i); }
static U gm10(U o, U s, U a, U b, U c, U d, U e, U f, U g, U h, U i, U j) { return ((U (*)(U, U, U, U, U, U, U, U, U, U, U, U))objc_msgSend)(o, s, a, b, c, d, e, f, g, h, i, j); }

// The sends that take floats or structs by value: the compiler applies the
// HFA, stack and indirect rules of each ABI.
static void gm_v_f(U o, U s, float a) { ((void (*)(U, U, float))objc_msgSend)(o, s, a); }
static void gm_v_d(U o, U s, double a) { ((void (*)(U, U, double))objc_msgSend)(o, s, a); }
static void gm_v_fff(U o, U s, float a, float b, float c) { ((void (*)(U, U, float, float, float))objc_msgSend)(o, s, a, b, c); }
static void gm_v_ffff(U o, U s, float a, float b, float c, float d) { ((void (*)(U, U, float, float, float, float))objc_msgSend)(o, s, a, b, c, d); }
static void gm_v_2d(U o, U s, G2d a) { ((void (*)(U, U, G2d))objc_msgSend)(o, s, a); }
static void gm_v_4d(U o, U s, G4d a) { ((void (*)(U, U, G4d))objc_msgSend)(o, s, a); }
static void gm_v_6d(U o, U s, G6d a) { ((void (*)(U, U, G6d))objc_msgSend)(o, s, a); }
static void gm_v_2u(U o, U s, G2u a) { ((void (*)(U, U, G2u))objc_msgSend)(o, s, a); }
static void gm_v_4u(U o, U s, G4u a) { ((void (*)(U, U, G4u))objc_msgSend)(o, s, a); }
static void gm_v_p2u(U o, U s, U a, G2u b) { ((void (*)(U, U, U, G2u))objc_msgSend)(o, s, a, b); }
static void gm_v_3u3u(U o, U s, G3u a, G3u b) { ((void (*)(U, U, G3u, G3u))objc_msgSend)(o, s, a, b); }
static void gm_v_pp3u(U o, U s, U a, U b, G3u c) { ((void (*)(U, U, U, U, G3u))objc_msgSend)(o, s, a, b, c); }
static void gm_v_pppp3uppp3u(U o, U s, U a, U b, U c, U d, G3u e, U f, U g, U h, G3u i) {
	((void (*)(U, U, U, U, U, U, G3u, U, U, U, G3u))objc_msgSend)(o, s, a, b, c, d, e, f, g, h, i);
}
static void gm_v_ppp3u3uppp3u(U o, U s, U a, U b, U c, G3u d, G3u e, U f, U g, U h, G3u i) {
	((void (*)(U, U, U, U, U, G3u, G3u, U, U, U, G3u))objc_msgSend)(o, s, a, b, c, d, e, f, g, h, i);
}
static void gm_v_ppp3u3upppp(U o, U s, U a, U b, U c, G3u d, G3u e, U f, U g, U h, U i) {
	((void (*)(U, U, U, U, U, G3u, G3u, U, U, U, U))objc_msgSend)(o, s, a, b, c, d, e, f, g, h, i);
}
static void gm_v_6uppppp(U o, U s, G6u a, U b, U c, U d, U e, U f) {
	((void (*)(U, U, G6u, U, U, U, U, U))objc_msgSend)(o, s, a, b, c, d, e, f);
}
static U gm_pp2u2u(U o, U s, U a, U b, G2u c, G2u d) { return ((U (*)(U, U, U, U, G2u, G2u))objc_msgSend)(o, s, a, b, c, d); }

// A struct return over 16 bytes goes through objc_msgSend_stret on x86_64 only.
static G3u gm_3u_p(U o, U s, U a) {
#if defined(__x86_64__)
	return ((G3u (*)(U, U, U))objc_msgSend_stret)(o, s, a);
#else
	return ((G3u (*)(U, U, U))objc_msgSend)(o, s, a);
#endif
}

// Sends that hand Metal Go memory take it as a pointer to bytes, which cgo
// passes without a cgocheck walk and keeps adjustable by stack copying.
typedef const uint8_t *P;
static U gm1p(U o, U s, P a) { return ((U (*)(U, U, P))objc_msgSend)(o, s, a); }
static U gm3p(U o, U s, P bytes, U b, U c) { return ((U (*)(U, U, P, U, U))objc_msgSend)(o, s, bytes, b, c); }
static U gm2e(U o, U s, U a, uint8_t *err) { return ((U (*)(U, U, U, uint8_t *))objc_msgSend)(o, s, a, err); }
static U gm3e(U o, U s, U a, U b, uint8_t *err) { return ((U (*)(U, U, U, U, uint8_t *))objc_msgSend)(o, s, a, b, err); }
static void gm_v_6upppbpp(U o, U s, G6u a, U b, U c, P bytes, U e, U f) {
	((void (*)(U, U, G6u, U, U, P, U, U))objc_msgSend)(o, s, a, b, c, bytes, e, f);
}
static U gm_class(P name) { return (U)objc_getClass((const char *)name); }
static U gm_sel(P name) { return (U)sel_registerName((const char *)name); }
static U gm_pool_push(void) { return (U)objc_autoreleasePoolPush(); }
static void gm_pool_pop(U p) { objc_autoreleasePoolPop((void *)p); }
static U gm_default_device(void) { return (U)MTLCreateSystemDefaultDevice(); }

// MTLCopyAllDevices only exists on macOS: look it up instead of linking it.
static U gm_copy_all_devices(void) {
	// (void *)-2 is RTLD_DEFAULT.
	U (*fn)(void) = (U (*)(void))dlsym((void *)-2, "MTLCopyAllDevices");
	return fn ? fn() : 0;
}

// Completion blocks: real heap blocks (clang -fblocks, plain C) that call
// gogpuMetalBlock(kind, id). Metal copies a block it keeps, so the creator
// releases its own reference right after handing it over.
extern void *_Block_copy(const void *);
extern void _Block_release(const void *);
extern void gogpuMetalBlock(uint32_t kind, uint64_t id);
// void (^)(id<MTLCommandBuffer>)
static U gm_new_block1(uint32_t kind, uint64_t id) {
	return (U)_Block_copy(^(U cmdbuf) { gogpuMetalBlock(kind, id); });
}
// void (^)(id<MTLSharedEvent>, uint64_t)
static U gm_new_block2(uint32_t kind, uint64_t id) {
	return (U)_Block_copy(^(U event, uint64_t value) { gogpuMetalBlock(kind, id); });
}
static void gm_release_block(U b) { _Block_release((const void *)b); }
*/
import "C"

import (
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/roffe/gowgpu/hal"
)

// selectorCache caches registered selectors for performance.
var selectorCache sync.Map

// cPointer turns an address C returned (never Go memory) into a pointer.
func cPointer(p uintptr) unsafe.Pointer {
	return *(*unsafe.Pointer)(unsafe.Pointer(&p))
}

func cString(s string) []byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return b
}

// initObjCRuntime has nothing left to load: libobjc and the frameworks are
// linked. It stays for Init's sake.
func initObjCRuntime() error { return nil }

// classCache keeps the classes looked up so far: a class does not move, and
// a lookup made a C string every call on paths that run per pass.
var classCache sync.Map // string -> Class

// GetClass returns the Class for a given name.
func GetClass(name string) Class {
	if cached, ok := classCache.Load(name); ok {
		return cached.(Class)
	}
	b := cString(name)
	cls := Class(C.gm_class((*C.uint8_t)(&b[0])))
	if cls != 0 {
		classCache.Store(name, cls)
	}
	return cls
}

// RegisterSelector registers and returns a selector for the given name.
func RegisterSelector(name string) SEL {
	if cached, ok := selectorCache.Load(name); ok {
		return cached.(SEL)
	}
	b := cString(name)
	sel := SEL(C.gm_sel((*C.uint8_t)(&b[0])))
	selectorCache.Store(name, sel)
	return sel
}

// Sel is a convenience alias for RegisterSelector.
func Sel(name string) SEL {
	return RegisterSelector(name)
}

// structType names the by-value C struct an objcArg carries, or for a
// result, what msgSend returns.
type structType byte

const (
	retVoid structType = 1
	retInt  structType = 2
	retBool structType = 3

	// Each struct's shape letter (see shape) is '0'+its value: '4' to ':'.
	cgSizeType         structType = 4  // CGSize: 2 doubles
	mtlClearColorType  structType = 5  // 4 doubles
	mtlViewportType    structType = 6  // 6 doubles
	nsRangeType        structType = 7  // 2 NSUInteger
	mtlOriginType      structType = 8  // 3 NSUInteger
	mtlScissorRectType structType = 9  // 4 NSUInteger
	mtlRegionType      structType = 10 // 6 NSUInteger
)

// mtlSizeType has MTLOrigin's layout.
const mtlSizeType = mtlOriginType

var structSize = [...]uintptr{cgSizeType: 16, mtlClearColorType: 32, mtlViewportType: 48, nsRangeType: 16, mtlOriginType: 24, mtlScissorRectType: 32, mtlRegionType: 48}

// Argument kinds, as shape letters.
const (
	kindInt    = 'p' // id, pointer, NSUInteger, uint64
	kindBool   = 'B'
	kindFloat  = 'f'
	kindDouble = 'd'
	kindBytes  = 'b' // a pointer into Go memory, see argBytes
)

// objcArg is one message argument, held inline so building it allocates
// nothing.
type objcArg struct {
	kind byte
	w    [6]uint64
	ptr  unsafe.Pointer // Go memory, for kindBytes
}

func argPointer(val uintptr) objcArg { return objcArg{kind: kindInt, w: [6]uint64{uint64(val)}} }

func argUint64(val uint64) objcArg { return objcArg{kind: kindInt, w: [6]uint64{val}} }

// argBytes passes Go memory: it stays a pointer all the way to C, so a
// stack copy in between cannot leave C with a stale address.
func argBytes(p unsafe.Pointer) objcArg { return objcArg{kind: kindBytes, ptr: p} }

func argBool(val bool) objcArg {
	a := objcArg{kind: kindBool}
	if val {
		a.w[0] = 1
	}
	return a
}

func argFloat32(val float32) objcArg {
	a := objcArg{kind: kindFloat}
	*(*float32)(unsafe.Pointer(&a.w[0])) = val
	return a
}

func argFloat64(val float64) objcArg {
	a := objcArg{kind: kindDouble}
	*(*float64)(unsafe.Pointer(&a.w[0])) = val
	return a
}

func argStruct[T any](val T, td structType) objcArg {
	if unsafe.Sizeof(val) != structSize[td] {
		panic("metal: struct argument does not match its C type")
	}
	a := objcArg{kind: '0' + byte(td)}
	*(*T)(unsafe.Pointer(&a.w)) = val
	return a
}

func (a *objcArg) word() C.U { return C.U(a.w[0]) }

func (a *objcArg) f32() C.float  { return C.float(*(*float32)(unsafe.Pointer(&a.w[0]))) }
func (a *objcArg) f64() C.double { return C.double(*(*float64)(unsafe.Pointer(&a.w[0]))) }
func (a *objcArg) s2d() C.G2d    { return *(*C.G2d)(unsafe.Pointer(&a.w)) }
func (a *objcArg) s4d() C.G4d    { return *(*C.G4d)(unsafe.Pointer(&a.w)) }
func (a *objcArg) s6d() C.G6d    { return *(*C.G6d)(unsafe.Pointer(&a.w)) }
func (a *objcArg) s2u() C.G2u    { return *(*C.G2u)(unsafe.Pointer(&a.w)) }
func (a *objcArg) s3u() C.G3u    { return *(*C.G3u)(unsafe.Pointer(&a.w)) }
func (a *objcArg) s4u() C.G4u    { return *(*C.G4u)(unsafe.Pointer(&a.w)) }
func (a *objcArg) s6u() C.G6u    { return *(*C.G6u)(unsafe.Pointer(&a.w)) }

// msgInt sends a message whose arguments all travel as integers.
func msgInt(obj ID, sel SEL, a []uintptr) uintptr {
	o, s := C.U(obj), C.U(sel)
	u := func(i int) C.U { return C.U(a[i]) }
	switch len(a) {
	case 0:
		return uintptr(C.gm0(o, s))
	case 1:
		return uintptr(C.gm1(o, s, u(0)))
	case 2:
		return uintptr(C.gm2(o, s, u(0), u(1)))
	case 3:
		return uintptr(C.gm3(o, s, u(0), u(1), u(2)))
	case 4:
		return uintptr(C.gm4(o, s, u(0), u(1), u(2), u(3)))
	case 5:
		return uintptr(C.gm5(o, s, u(0), u(1), u(2), u(3), u(4)))
	case 6:
		return uintptr(C.gm6(o, s, u(0), u(1), u(2), u(3), u(4), u(5)))
	case 7:
		return uintptr(C.gm7(o, s, u(0), u(1), u(2), u(3), u(4), u(5), u(6)))
	case 8:
		return uintptr(C.gm8(o, s, u(0), u(1), u(2), u(3), u(4), u(5), u(6), u(7)))
	case 9:
		return uintptr(C.gm9(o, s, u(0), u(1), u(2), u(3), u(4), u(5), u(6), u(7), u(8)))
	case 10:
		return uintptr(C.gm10(o, s, u(0), u(1), u(2), u(3), u(4), u(5), u(6), u(7), u(8), u(9)))
	}
	panic("metal: objc_msgSend with more than 10 arguments")
}

// shape spells out an argument list: one letter per argument, digits for
// structs ('0'+structType).
func shape(args []objcArg) (key [12]byte, n int, integer bool) {
	integer = true
	for _, a := range args {
		key[n] = a.kind
		n++
		if a.kind != kindInt && a.kind != kindBool {
			integer = false
		}
	}
	return key, n, integer
}

// msgSend sends sel to obj with typed arguments, storing an integer or
// struct result at retPtr.
func msgSend(obj ID, sel SEL, retType structType, retPtr unsafe.Pointer, args ...objcArg) error {
	if obj == 0 || sel == 0 {
		return nil
	}
	key, n, integer := shape(args)
	o, s := C.U(obj), C.U(sel)
	if integer && retType <= retBool {
		var w [10]uintptr
		for i := range args {
			w[i] = uintptr(args[i].w[0])
		}
		r := msgInt(obj, sel, w[:len(args)])
		switch retType {
		case retInt:
			*(*uintptr)(retPtr) = r
		case retBool:
			*(*uint8)(retPtr) = uint8(r)
		}
		return nil
	}
	a := args
	switch k := string(key[:n]); k {
	case "f":
		C.gm_v_f(o, s, a[0].f32())
	case "d":
		C.gm_v_d(o, s, a[0].f64())
	case "fff":
		C.gm_v_fff(o, s, a[0].f32(), a[1].f32(), a[2].f32())
	case "ffff":
		C.gm_v_ffff(o, s, a[0].f32(), a[1].f32(), a[2].f32(), a[3].f32())
	case "4": // CGSize
		C.gm_v_2d(o, s, a[0].s2d())
	case "5": // MTLClearColor
		C.gm_v_4d(o, s, a[0].s4d())
	case "6": // MTLViewport
		C.gm_v_6d(o, s, a[0].s6d())
	case "7": // NSRange
		C.gm_v_2u(o, s, a[0].s2u())
	case "9": // MTLScissorRect
		C.gm_v_4u(o, s, a[0].s4u())
	case "p7":
		C.gm_v_p2u(o, s, a[0].word(), a[1].s2u())
	case "88":
		C.gm_v_3u3u(o, s, a[0].s3u(), a[1].s3u())
	case "pp8":
		C.gm_v_pp3u(o, s, a[0].word(), a[1].word(), a[2].s3u())
	case "pppp8ppp8":
		C.gm_v_pppp3uppp3u(o, s, a[0].word(), a[1].word(), a[2].word(), a[3].word(), a[4].s3u(), a[5].word(), a[6].word(), a[7].word(), a[8].s3u())
	case "ppp88ppp8":
		C.gm_v_ppp3u3uppp3u(o, s, a[0].word(), a[1].word(), a[2].word(), a[3].s3u(), a[4].s3u(), a[5].word(), a[6].word(), a[7].word(), a[8].s3u())
	case "ppp88pppp":
		C.gm_v_ppp3u3upppp(o, s, a[0].word(), a[1].word(), a[2].word(), a[3].s3u(), a[4].s3u(), a[5].word(), a[6].word(), a[7].word(), a[8].word())
	case ":ppppp": // MTLRegion, then integers
		C.gm_v_6uppppp(o, s, a[0].s6u(), a[1].word(), a[2].word(), a[3].word(), a[4].word(), a[5].word())
	case ":ppbpp": // replaceRegion:mipmapLevel:slice:withBytes:bytesPerRow:bytesPerImage:
		C.gm_v_6upppbpp(o, s, a[0].s6u(), a[1].word(), a[2].word(), (*C.uint8_t)(a[3].ptr), a[4].word(), a[5].word())
	case "pp77":
		r := uintptr(C.gm_pp2u2u(o, s, a[0].word(), a[1].word(), a[2].s2u(), a[3].s2u()))
		if retPtr != nil {
			*(*uintptr)(retPtr) = r
		}
	case "p":
		if retType != mtlOriginType {
			panic("metal: unexpected struct result")
		}
		*(*C.G3u)(retPtr) = C.gm_3u_p(o, s, a[0].word())
	default:
		panic("metal: no objc_msgSend trampoline for argument shape " + k)
	}
	return nil
}

func msgSendVoid(obj ID, sel SEL, args ...objcArg) {
	_ = msgSend(obj, sel, retVoid, nil, args...)
}

func msgSendSetBytes(obj ID, sel SEL, bytes unsafe.Pointer, length, index NSUInteger) error {
	_ = msgSendBytes(obj, sel, bytes, uintptr(length), uintptr(index))
	return nil
}

// msgSendBytes sends sel with Go memory and two integers:
// setBytes:length:atIndex:, newBufferWithBytes:length:options:.
func msgSendBytes(obj ID, sel SEL, bytes unsafe.Pointer, a, b uintptr) ID {
	if obj == 0 || sel == 0 {
		return 0
	}
	return ID(C.gm3p(C.U(obj), C.U(sel), (*C.uint8_t)(bytes), C.U(a), C.U(b)))
}

// msgSendErr sends sel with integer arguments and a trailing NSError **,
// which may point at the caller's stack.
func msgSendErr(obj ID, sel SEL, errOut *ID, args ...uintptr) ID {
	if obj == 0 || sel == 0 {
		return 0
	}
	e := (*C.uint8_t)(unsafe.Pointer(errOut))
	switch len(args) {
	case 1:
		return ID(C.gm2e(C.U(obj), C.U(sel), C.U(args[0]), e))
	case 2:
		return ID(C.gm3e(C.U(obj), C.U(sel), C.U(args[0]), C.U(args[1]), e))
	}
	panic("metal: msgSendErr arity")
}

func msgSendID(obj ID, sel SEL, args ...objcArg) ID {
	var result ID
	_ = msgSend(obj, sel, retInt, unsafe.Pointer(&result), args...)
	return result
}

func msgSendUint(obj ID, sel SEL, args ...objcArg) uint {
	var result uint
	_ = msgSend(obj, sel, retInt, unsafe.Pointer(&result), args...)
	return result
}

func msgSendBool(obj ID, sel SEL, args ...objcArg) bool {
	var result uint8
	_ = msgSend(obj, sel, retBool, unsafe.Pointer(&result), args...)
	return result != 0
}

// MsgSend calls an Objective-C method whose arguments are all integers,
// ids or pointers and returns the result as ID.
//
// For void-returning ObjC methods (setters like setTextureType:, setLabel:,
// commit, waitUntilCompleted), the return value is meaningless — objc_msgSend
// always returns a register-sized value regardless of the method's actual return
// type. Callers discard with `_ = MsgSend(...)`.
func MsgSend(obj ID, sel SEL, args ...uintptr) ID {
	if obj == 0 || sel == 0 {
		return 0
	}
	return ID(msgInt(obj, sel, args))
}

// MsgSendUint calls a method and returns a uint result.
func MsgSendUint(obj ID, sel SEL, args ...uintptr) uint {
	return uint(MsgSend(obj, sel, args...))
}

// MsgSendBool calls a method and returns a bool result.
func MsgSendBool(obj ID, sel SEL, args ...uintptr) bool {
	return uint8(MsgSend(obj, sel, args...)) != 0
}

// Retain increments the reference count of an object.
func Retain(obj ID) ID {
	if obj == 0 {
		return 0
	}
	return MsgSend(obj, Sel("retain"))
}

// Release decrements the reference count of an object.
func Release(obj ID) {
	if obj == 0 {
		return
	}
	_ = MsgSend(obj, Sel("release"))
}

// AutoreleasePool manages an Objective-C autorelease pool. The caller must
// call Drain once from the same goroutine that created the pool. Pool values
// must not be copied, shared, or drained concurrently.
type AutoreleasePool struct {
	pool   ID
	locked bool
	unlock func()
	drain  func(ID)
}

type autoreleasePoolCallbacks struct {
	lock   func()
	unlock func()
	create func() ID
	drain  func(ID)
}

func newAutoreleasePoolWithCallbacks(callbacks autoreleasePoolCallbacks) (pool *AutoreleasePool) {
	locked := callbacks.lock != nil
	if callbacks.lock != nil {
		callbacks.lock()
	}

	pool = &AutoreleasePool{
		locked: locked,
		unlock: callbacks.unlock,
		drain:  callbacks.drain,
	}
	created := false
	defer func() {
		if created {
			return
		}
		// A create callback can panic after the OS-thread lock is acquired. Make
		// the partially constructed pool terminal before releasing that lock so
		// a future recovery cannot accidentally unlock it twice.
		pool.pool = 0
		if pool.locked {
			pool.locked = false
		}
		if locked && callbacks.unlock != nil {
			callbacks.unlock()
		}
	}()

	pool.pool = callbacks.create()
	created = true
	return pool
}

// NewAutoreleasePool creates a new autorelease pool.
func NewAutoreleasePool() *AutoreleasePool {
	return newAutoreleasePoolWithCallbacks(autoreleasePoolCallbacks{
		lock:   runtime.LockOSThread,
		unlock: runtime.UnlockOSThread,
		create: func() ID { return ID(C.gm_pool_push()) },
		drain:  func(pool ID) { C.gm_pool_pop(C.U(pool)) },
	})
}

// Drain drains the autorelease pool.
func (p *AutoreleasePool) Drain() {
	if p == nil || !p.locked {
		return
	}
	// Mark terminal before invoking Objective-C. Drain can panic, but a caller
	// recovering that panic must not be able to drain or unlock this pool again.
	pool := p.pool
	p.pool = 0
	p.locked = false
	if p.unlock != nil {
		defer p.unlock()
	}
	if pool != 0 && p.drain != nil {
		p.drain(pool)
	}
}

// NSString creates an NSString from a Go string.
// Returns a +1 retained object that the caller must Release().
// Uses alloc/initWithUTF8String: instead of stringWithUTF8String:
// to return a retained object (not autoreleased) for explicit ownership.
func NSString(s string) ID {
	nsStringClass := ID(GetClass("NSString"))
	if s == "" {
		// Use alloc/init for empty string to get +1 retained object
		obj := MsgSend(nsStringClass, Sel("alloc"))
		return MsgSend(obj, Sel("init"))
	}
	cstr := cString(s)
	obj := MsgSend(nsStringClass, Sel("alloc"))
	return ID(C.gm1p(C.U(obj), C.U(Sel("initWithUTF8String:")), (*C.uint8_t)(&cstr[0])))
}

// GoString converts an NSString to a Go string.
func GoString(nsstr ID) string {
	if nsstr == 0 {
		return ""
	}
	cstr := MsgSend(nsstr, Sel("UTF8String"))
	if cstr == 0 {
		return ""
	}
	return goStringFromCStr(uintptr(cstr))
}

func goStringFromCStr(cstr uintptr) string {
	if cstr == 0 {
		return ""
	}
	length := 0
	ptr := (*byte)(cPointer(cstr))
	for i := 0; i < 4096; i++ {
		b := unsafe.Slice(ptr, i+1)
		if b[i] == 0 {
			length = i
			break
		}
	}
	if length == 0 {
		return ""
	}
	result := unsafe.Slice(ptr, length)
	return string(result)
}

// --------------------------------------------------------------------------
// Completion blocks
// --------------------------------------------------------------------------
//
// newBlock makes a heap block whose invoke calls gogpuMetalBlock(kind, id).
// Each kind keeps its Go-side state in a registry keyed by id, as before.

const (
	blockSharedEvent uint32 = iota + 1
	blockStaging
	blockFrame
	blockGPU
)

// blockRegistryEntry holds Go-side state for an active shared event block.
type blockRegistryEntry struct {
	done chan struct{} // Signaled when the block is invoked
}

var (
	blockRegistry            sync.Map // map[uint64]*blockRegistryEntry
	completedHandlerRegistry sync.Map // map[uint64]ID
	frameCompletionRegistry  sync.Map // map[uint64]chan<- struct{}
	gpuCompletionRegistry    sync.Map // map[uint64]*gpuCompletionEntry
	blockIDCounter           uint64
)

type gpuCompletionEntry struct {
	target *atomic.Uint64
	value  uint64
}

func initBlockSupport() {}

func nextBlockID() uint64 {
	return atomic.AddUint64(&blockIDCounter, 1)
}

func newBlock(kind uint32, id uint64) uintptr {
	if kind == blockSharedEvent {
		return uintptr(C.gm_new_block2(C.uint32_t(kind), C.uint64_t(id)))
	}
	return uintptr(C.gm_new_block1(C.uint32_t(kind), C.uint64_t(id)))
}

// releaseBlockRef drops the creator's reference to a block, once the method
// that keeps it (addCompletedHandler:, notifyListener:atValue:block:) has
// copied it.
func releaseBlockRef(block uintptr) {
	if block != 0 {
		C.gm_release_block(C.U(block))
	}
}

//export gogpuMetalBlock
func gogpuMetalBlock(kind C.uint32_t, cid C.uint64_t) {
	id := uint64(cid)
	switch uint32(kind) {
	case blockSharedEvent:
		hal.Logger().Debug("metal: shared event notification fired", "blockID", id)
		if entry, ok := blockRegistry.Load(id); ok {
			select {
			case entry.(*blockRegistryEntry).done <- struct{}{}:
			default:
			}
		}
	case blockStaging:
		hal.Logger().Debug("metal: completion handler fired", "blockID", id)
		if val, ok := completedHandlerRegistry.LoadAndDelete(id); ok {
			if staging := val.(ID); staging != 0 {
				Release(staging)
			}
		}
	case blockFrame:
		if hal.DebugEnabled() {
			hal.Logger().Debug("metal: frame completion fired", "blockID", id)
		}
		if val, ok := frameCompletionRegistry.LoadAndDelete(id); ok {
			if ch := val.(chan<- struct{}); ch != nil {
				ch <- struct{}{}
			}
		}
	case blockGPU:
		if hal.DebugEnabled() {
			hal.Logger().Debug("metal: GPU completion tracking fired", "blockID", id)
		}
		if val, ok := gpuCompletionRegistry.LoadAndDelete(id); ok {
			entry := val.(*gpuCompletionEntry)
			for {
				current := entry.target.Load()
				if entry.value <= current || entry.target.CompareAndSwap(current, entry.value) {
					break
				}
			}
		}
	}
}

// newSharedEventNotificationBlock creates the block for
// notifyListener:atValue:block:. The caller releases its block reference
// once the listener has it, and calls releaseBlock(id) when it stops waiting.
func newSharedEventNotificationBlock() (uintptr, uint64, chan struct{}) {
	id := nextBlockID()
	done := make(chan struct{}, 1)
	blockRegistry.Store(id, &blockRegistryEntry{done: done})
	b := newBlock(blockSharedEvent, id)
	if b == 0 {
		blockRegistry.Delete(id)
		return 0, 0, nil
	}
	return b, id, done
}

func releaseBlock(id uint64) {
	blockRegistry.Delete(id)
}

// newCompletedHandlerBlock creates an addCompletedHandler: block that
// releases stagingBuffer once the GPU is done with it.
func newCompletedHandlerBlock(stagingBuffer ID) uintptr {
	id := nextBlockID()
	completedHandlerRegistry.Store(id, stagingBuffer)
	b := newBlock(blockStaging, id)
	if b == 0 {
		completedHandlerRegistry.Delete(id)
	}
	return b
}

// newFrameCompletionBlock creates an addCompletedHandler: block that
// signals frameSemaphore.
func newFrameCompletionBlock(frameSemaphore chan struct{}) uintptr {
	if frameSemaphore == nil {
		return 0
	}
	id := nextBlockID()
	var sendOnly chan<- struct{} = frameSemaphore
	frameCompletionRegistry.Store(id, sendOnly)
	b := newBlock(blockFrame, id)
	if b == 0 {
		frameCompletionRegistry.Delete(id)
	}
	return b
}

// newGPUCompletionBlock creates an addCompletedHandler: block that raises
// target to submissionIndex.
func newGPUCompletionBlock(target *atomic.Uint64, submissionIndex uint64) uintptr {
	if target == nil {
		return 0
	}
	id := nextBlockID()
	gpuCompletionRegistry.Store(id, &gpuCompletionEntry{target: target, value: submissionIndex})
	b := newBlock(blockGPU, id)
	if b == 0 {
		gpuCompletionRegistry.Delete(id)
	}
	return b
}

func cDefaultDevice() ID  { return ID(C.gm_default_device()) }
func cCopyAllDevices() ID { return ID(C.gm_copy_all_devices()) }

// blockSupport is whether completion blocks can be made: always, now that
// they are built in C.
const blockSupport = true
