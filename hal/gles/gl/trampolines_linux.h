// One trampoline per C signature of the GL functions context_linux.go calls,
// through the pointers eglGetProcAddress returned. No GL headers: GLenum and
// friends are uint32_t, pointers const uint8_t *, or const void * for one to
// more pointers, which cgo checks, GLintptr, GLsizeiptr and GLsync
// uintptr_t, GLuint64 uint64_t, GLboolean results uint8_t.
#ifndef GLES_GL_TRAMPOLINES_LINUX_H
#define GLES_GL_TRAMPOLINES_LINUX_H
#include <stdint.h>

static inline uint32_t gl_i_(uintptr_t fn) { return ((uint32_t (*)(void))fn)(); }
static inline uint32_t gl_i_i(uintptr_t fn, uint32_t a0) { return ((uint32_t (*)(uint32_t))fn)(a0); }
static inline uint32_t gl_i_ip(uintptr_t fn, uint32_t a0, const uint8_t * a1) { return ((uint32_t (*)(uint32_t, const void *))fn)(a0, a1); }
static inline uint32_t gl_i_uiq(uintptr_t fn, uintptr_t a0, uint32_t a1, uint64_t a2) { return ((uint32_t (*)(uintptr_t, uint32_t, uint64_t))fn)(a0, a1, a2); }
static inline uintptr_t gl_u_ii(uintptr_t fn, uint32_t a0, uint32_t a1) { return ((uintptr_t (*)(uint32_t, uint32_t))fn)(a0, a1); }
static inline uintptr_t gl_u_iuui(uintptr_t fn, uint32_t a0, uintptr_t a1, uintptr_t a2, uint32_t a3) { return ((uintptr_t (*)(uint32_t, uintptr_t, uintptr_t, uint32_t))fn)(a0, a1, a2, a3); }
static inline void gl_v_(uintptr_t fn) { ((void (*)(void))fn)(); }
static inline void gl_v_d(uintptr_t fn, double a0) { ((void (*)(double))fn)(a0); }
static inline void gl_v_dd(uintptr_t fn, double a0, double a1) { ((void (*)(double, double))fn)(a0, a1); }
static inline void gl_v_f(uintptr_t fn, float a0) { ((void (*)(float))fn)(a0); }
static inline void gl_v_ff(uintptr_t fn, float a0, float a1) { ((void (*)(float, float))fn)(a0, a1); }
static inline void gl_v_fff(uintptr_t fn, float a0, float a1, float a2) { ((void (*)(float, float, float))fn)(a0, a1, a2); }
static inline void gl_v_ffff(uintptr_t fn, float a0, float a1, float a2, float a3) { ((void (*)(float, float, float, float))fn)(a0, a1, a2, a3); }
static inline void gl_v_iff(uintptr_t fn, uint32_t a0, float a1, float a2) { ((void (*)(uint32_t, float, float))fn)(a0, a1, a2); }
static inline void gl_v_i(uintptr_t fn, uint32_t a0) { ((void (*)(uint32_t))fn)(a0); }
static inline void gl_v_ii(uintptr_t fn, uint32_t a0, uint32_t a1) { ((void (*)(uint32_t, uint32_t))fn)(a0, a1); }
static inline void gl_v_iif(uintptr_t fn, uint32_t a0, uint32_t a1, float a2) { ((void (*)(uint32_t, uint32_t, float))fn)(a0, a1, a2); }
static inline void gl_v_iii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2) { ((void (*)(uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2); }
static inline void gl_v_iiii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3); }
static inline void gl_v_iiiii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4); }
static inline void gl_v_iiiiii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4, a5); }
static inline void gl_v_iiiiiiii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7); }
static inline void gl_v_iiiiiiiiii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, uint32_t a8, uint32_t a9) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }
static inline void gl_v_iiiiiiiip(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, const uint8_t * a8) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const void *))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8); }
static inline void gl_v_iiiiiiiiu(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, uintptr_t a8) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8); }
static inline void gl_v_iiiiiip(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, const uint8_t * a6) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const void *))fn)(a0, a1, a2, a3, a4, a5, a6); }
static inline void gl_v_iiiiiu(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uintptr_t a5) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5); }
static inline void gl_v_iiiiu(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uintptr_t a4) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uintptr_t))fn)(a0, a1, a2, a3, a4); }
static inline void gl_v_iiiu(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uintptr_t a3) { ((void (*)(uint32_t, uint32_t, uint32_t, uintptr_t))fn)(a0, a1, a2, a3); }
static inline void gl_v_iiiui(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uintptr_t a3, uint32_t a4) { ((void (*)(uint32_t, uint32_t, uint32_t, uintptr_t, uint32_t))fn)(a0, a1, a2, a3, a4); }
static inline void gl_v_iiiuiii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uintptr_t a3, uint32_t a4, uint32_t a5, uint32_t a6) { ((void (*)(uint32_t, uint32_t, uint32_t, uintptr_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4, a5, a6); }
static inline void gl_v_iiiuii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uintptr_t a3, uint32_t a4, uint32_t a5) { ((void (*)(uint32_t, uint32_t, uint32_t, uintptr_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4, a5); }
static inline void gl_v_iip(uintptr_t fn, uint32_t a0, uint32_t a1, const uint8_t * a2) { ((void (*)(uint32_t, uint32_t, const void *))fn)(a0, a1, a2); }
static inline void gl_v_iipp(uintptr_t fn, uint32_t a0, uint32_t a1, const uint8_t * a2, const uint8_t * a3) { ((void (*)(uint32_t, uint32_t, const void *, const void *))fn)(a0, a1, a2, a3); }
static inline void gl_v_iigp(uintptr_t fn, uint32_t a0, uint32_t a1, const void * a2, const uint8_t * a3) { ((void (*)(uint32_t, uint32_t, const void *, const void *))fn)(a0, a1, a2, a3); }
static inline void gl_v_iiuuu(uintptr_t fn, uint32_t a0, uint32_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4) { ((void (*)(uint32_t, uint32_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3, a4); }
static inline void gl_v_ip(uintptr_t fn, uint32_t a0, const uint8_t * a1) { ((void (*)(uint32_t, const void *))fn)(a0, a1); }
static inline void gl_v_iuui(uintptr_t fn, uint32_t a0, uintptr_t a1, uintptr_t a2, uint32_t a3) { ((void (*)(uint32_t, uintptr_t, uintptr_t, uint32_t))fn)(a0, a1, a2, a3); }
static inline void gl_v_iuup(uintptr_t fn, uint32_t a0, uintptr_t a1, uintptr_t a2, const uint8_t * a3) { ((void (*)(uint32_t, uintptr_t, uintptr_t, const void *))fn)(a0, a1, a2, a3); }
static inline void gl_v_u(uintptr_t fn, uintptr_t a0) { ((void (*)(uintptr_t))fn)(a0); }
static inline void gl_v_uiipp(uintptr_t fn, uintptr_t a0, uint32_t a1, uint32_t a2, const uint8_t * a3, const uint8_t * a4) { ((void (*)(uintptr_t, uint32_t, uint32_t, const void *, const void *))fn)(a0, a1, a2, a3, a4); }
static inline void * gl_P_i(uintptr_t fn, uint32_t a0) { return ((void * (*)(uint32_t))fn)(a0); }
static inline void * gl_P_ii(uintptr_t fn, uint32_t a0, uint32_t a1) { return ((void * (*)(uint32_t, uint32_t))fn)(a0, a1); }
static inline uint8_t gl_b_i(uintptr_t fn, uint32_t a0) { return ((uint8_t (*)(uint32_t))fn)(a0); }
static inline void gl_v_iiiuu(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uintptr_t a3, uintptr_t a4) { ((void (*)(uint32_t, uint32_t, uint32_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3, a4); }

static inline void gl_v_iiiiiiiii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, uint32_t a8) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8); }
static inline void gl_v_iiiiiiiiiu(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, uint32_t a8, uintptr_t a9) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }
static inline void gl_v_iiiiiiiiiip(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, uint32_t a8, uint32_t a9, const uint8_t * a10) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, const void *))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }
static inline void gl_v_iiiiiiiiiiu(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, uint32_t a8, uint32_t a9, uintptr_t a10) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uintptr_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10); }
static inline void gl_v_iu(uintptr_t fn, uint32_t a0, uintptr_t a1) { ((void (*)(uint32_t, uintptr_t))fn)(a0, a1); }
static inline void gl_v_iiu(uintptr_t fn, uint32_t a0, uint32_t a1, uintptr_t a2) { ((void (*)(uint32_t, uint32_t, uintptr_t))fn)(a0, a1, a2); }
static inline void gl_v_iiiiiii(uintptr_t fn, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4, uint32_t a5, uint32_t a6) { ((void (*)(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4, a5, a6); }
#endif
