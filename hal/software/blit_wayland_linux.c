//go:build !(js && wasm)

// libwayland-client calls for the software backend's Wayland SHM blit. The
// entry points come from dlsym (blit_wayland.go), so neither libwayland nor
// its headers are needed to build. wl_proxy_marshal* are variadic: every
// message shape the blit sends has its own function here, so the compiler
// applies the variadic calling convention, which differs from a plain call
// on some ABIs.

#include <stddef.h>
#include <stdint.h>
#include "_cgo_export.h"

typedef void *(*swl_marshal_constructor_fn)(void *proxy, uint32_t opcode, const void *iface, ...);
typedef void *(*swl_marshal_constructor_versioned_fn)(void *proxy, uint32_t opcode, const void *iface, uint32_t version, ...);
typedef void (*swl_marshal_fn)(void *proxy, uint32_t opcode, ...);

uintptr_t swl_call_u_u(void *fn, uintptr_t a) { return (uintptr_t)((void *(*)(void *))fn)((void *)a); }
void swl_call_v_u(void *fn, uintptr_t a) { ((void (*)(void *))fn)((void *)a); }
void swl_call_v_uu(void *fn, uintptr_t a, uintptr_t b) { ((void (*)(void *, void *))fn)((void *)a, (void *)b); }
int swl_call_i_u(void *fn, uintptr_t a) { return ((int (*)(void *))fn)((void *)a); }
// wl_display_roundtrip_queue dispatches events, which runs the listeners
// below and through them Go: never mark this one nocallback.
int swl_call_i_uu(void *fn, uintptr_t a, uintptr_t b) { return ((int (*)(void *, void *))fn)((void *)a, (void *)b); }

// wl_display.get_registry: opcode 1, signature "n".
uintptr_t swl_get_registry(void *fn, uintptr_t display, const void *registry_iface) {
	return (uintptr_t)((swl_marshal_constructor_fn)fn)((void *)display, 1, registry_iface, NULL);
}

// wl_registry.bind: opcode 0, signature "usun".
uintptr_t swl_registry_bind(void *fn, uintptr_t registry, uint32_t name, const void *iface, const char *iface_name, uint32_t version) {
	return (uintptr_t)((swl_marshal_constructor_versioned_fn)fn)((void *)registry, 0, iface, version, name, iface_name, version, NULL);
}

// wl_shm.create_pool: opcode 0, signature "nhi".
uintptr_t swl_create_pool(void *fn, uintptr_t shm, const void *pool_iface, int32_t fd, int32_t size) {
	return (uintptr_t)((swl_marshal_constructor_fn)fn)((void *)shm, 0, pool_iface, NULL, fd, size);
}

// wl_shm_pool.create_buffer: opcode 0, signature "niiiiu".
uintptr_t swl_create_buffer(void *fn, uintptr_t pool, const void *buffer_iface, int32_t offset, int32_t width, int32_t height, int32_t stride, uint32_t format) {
	return (uintptr_t)((swl_marshal_constructor_fn)fn)((void *)pool, 0, buffer_iface, NULL, offset, width, height, stride, format);
}

// A request without arguments: wl_surface.commit, wl_shm_pool.destroy,
// wl_buffer.destroy.
void swl_marshal0(void *fn, uintptr_t proxy, uint32_t opcode) {
	((swl_marshal_fn)fn)((void *)proxy, opcode);
}

// wl_surface.attach: opcode 1, signature "?oii".
void swl_surface_attach(void *fn, uintptr_t surface, uintptr_t buffer, int32_t x, int32_t y) {
	((swl_marshal_fn)fn)((void *)surface, 1, (void *)buffer, x, y);
}

// wl_surface.damage_buffer: opcode 9, signature "iiii".
void swl_surface_damage_buffer(void *fn, uintptr_t surface, int32_t x, int32_t y, int32_t w, int32_t h) {
	((swl_marshal_fn)fn)((void *)surface, 9, x, y, w, h);
}

static void swl_registry_global(void *data, void *registry, uint32_t name, const char *iface, uint32_t version) {
	(void)data;
	(void)registry;
	goSwRegistryGlobal(name, (char *)iface, version);
}

static void swl_registry_global_remove(void *data, void *registry, uint32_t name) {
	(void)data;
	(void)registry;
	(void)name;
}

static void swl_buffer_release(void *data, void *buffer) {
	(void)data;
	goSwBufferRelease((uintptr_t)buffer);
}

// Listener tables in the layout wl_proxy_add_listener expects: one function
// pointer per event, in event order.
static void (*const swl_registry_listener[2])(void) = {
	(void (*)(void))swl_registry_global,
	(void (*)(void))swl_registry_global_remove,
};
static void (*const swl_buffer_listener[1])(void) = {
	(void (*)(void))swl_buffer_release,
};

int swl_add_registry_listener(void *fn, uintptr_t registry) {
	return ((int (*)(void *, const void *, void *))fn)((void *)registry, swl_registry_listener, NULL);
}

int swl_add_buffer_listener(void *fn, uintptr_t buffer) {
	return ((int (*)(void *, const void *, void *))fn)((void *)buffer, swl_buffer_listener, NULL);
}
