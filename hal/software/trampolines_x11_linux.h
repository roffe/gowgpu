// One trampoline per C signature of the libX11 functions blit_linux.go
// calls, through the pointers dlsym returned. No X11 headers: Display*, GC
// and Drawable are uintptr_t, int and unsigned int uint32_t.
#ifndef SOFTWARE_TRAMPOLINES_X11_LINUX_H
#define SOFTWARE_TRAMPOLINES_X11_LINUX_H
#include <stdint.h>

static inline uint32_t sw_i_p(uintptr_t fn, const uint8_t * a0) { return ((uint32_t (*)(const void *))fn)(a0); }
static inline uintptr_t sw_u_uuuu(uintptr_t fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, uintptr_t a3) { return ((uintptr_t (*)(uintptr_t, uintptr_t, uintptr_t, uintptr_t))fn)(a0, a1, a2, a3); }
static inline void sw_v_u(uintptr_t fn, uintptr_t a0) { ((void (*)(uintptr_t))fn)(a0); }
static inline void sw_v_uu(uintptr_t fn, uintptr_t a0, uintptr_t a1) { ((void (*)(uintptr_t, uintptr_t))fn)(a0, a1); }
static inline void sw_v_uuupiiiiii(uintptr_t fn, uintptr_t a0, uintptr_t a1, uintptr_t a2, const uint8_t * a3, uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7, uint32_t a8, uint32_t a9) { ((void (*)(uintptr_t, uintptr_t, uintptr_t, const void *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t))fn)(a0, a1, a2, a3, a4, a5, a6, a7, a8, a9); }

#endif
