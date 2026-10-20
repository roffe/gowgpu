// One trampoline per C signature of the EGL, X11 and wayland-egl functions
// this package calls, through the pointers dlsym and eglGetProcAddress
// returned. No headers: EGL handles, native windows and displays are
// uintptr_t, EGLint, EGLenum and EGLBoolean uint32_t.
#ifndef GLES_EGL_TRAMPOLINES_LINUX_H
#define GLES_EGL_TRAMPOLINES_LINUX_H
#include <stdint.h>

static inline uint32_t egl_i_(uintptr_t fn) { return ((uint32_t (*)(void))fn)(); }
static inline uint32_t egl_i_i(uintptr_t fn, uint32_t a0) { return ((uint32_t (*)(uint32_t))fn)(a0); }
static inline uint32_t egl_i_u(uintptr_t fn, uintptr_t a0) { return ((uint32_t (*)(uintptr_t))fn)(a0); }
static inline uint32_t egl_i_ui(uintptr_t fn, uintptr_t a0, uint32_t a1) { return ((uint32_t (*)(uintptr_t, uint32_t))fn)(a0, a1); }
static inline uint32_t egl_i_upp(uintptr_t fn, uintptr_t a0, const uint8_t * a1, const uint8_t * a2) { return ((uint32_t (*)(uintptr_t, const void *, const void *))fn)(a0, a1, a2); }
static inline uint32_t egl_i_uppip(uintptr_t fn, uintptr_t a0, const uint8_t * a1, const uint8_t * a2, uint32_t a3, const uint8_t * a4) { return ((uint32_t (*)(uintptr_t, const void *, const void *, uint32_t, const void *))fn)(a0, a1, a2, a3, a4); }
static inline uint32_t egl_i_uu(uintptr_t fn, uintptr_t a0, uintptr_t a1) { return ((uint32_t (*)(uintptr_t, uintptr_t))fn)(a0, a1); }
static inline uint32_t egl_i_uuip(uintptr_t fn, uintptr_t a0, uintptr_t a1, uint32_t a2, const uint8_t * a3) { return ((uint32_t (*)(uintptr_t, uintptr_t, uint32_t, const void *))fn)(a0, a1, a2, a3); }
static inline uint32_t egl_i_uupi(uintptr_t fn, uintptr_t a0, uintptr_t a1, const uint8_t * a2, uint32_t a3) { return ((uint32_t (*)(uintptr_t, uintptr_t, const void *, uint32_t))fn)(a0, a1, a2, a3); }
static inline uint32_t egl_i_uuuu(uintptr_t fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3) { return ((uint32_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3); }
static inline uintptr_t egl_u_(uintptr_t fn) { return ((uintptr_t (*)(void))fn)(); }
static inline uintptr_t egl_u_i(uintptr_t fn, uint32_t a0) { return ((uintptr_t (*)(uint32_t))fn)(a0); }
static inline uintptr_t egl_u_iup(uintptr_t fn, uint32_t a0, uintptr_t a1, const uint8_t * a2) { return ((uintptr_t (*)(uint32_t, uintptr_t, const void *))fn)(a0, a1, a2); }
static inline uintptr_t egl_u_u(uintptr_t fn, uintptr_t a0) { return ((uintptr_t (*)(uintptr_t))fn)(a0); }
static inline uintptr_t egl_u_uii(uintptr_t fn, uintptr_t a0, uint32_t a1, uint32_t a2) { return ((uintptr_t (*)(uintptr_t, uint32_t, uint32_t))fn)(a0, a1, a2); }
static inline uintptr_t egl_u_uup(uintptr_t fn, uintptr_t a0, uintptr_t a1, const uint8_t * a2) { return ((uintptr_t (*)(uintptr_t, uintptr_t, const void *))fn)(a0, a1, a2); }
static inline uintptr_t egl_u_uuup(uintptr_t fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, const uint8_t * a3) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, const void *))fn)(a0, a1, a2, a3); }
static inline void * egl_P_p(uintptr_t fn, const uint8_t * a0) { return ((void * (*)(const void *))fn)(a0); }
static inline void * egl_P_ui(uintptr_t fn, uintptr_t a0, uint32_t a1) { return ((void * (*)(uintptr_t, uint32_t))fn)(a0, a1); }
static inline void egl_v_u(uintptr_t fn, uintptr_t a0) { ((void (*)(uintptr_t))fn)(a0); }
static inline void egl_v_uiiii(uintptr_t fn, uintptr_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t a4) { ((void (*)(uintptr_t, uint32_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4); }

#endif
