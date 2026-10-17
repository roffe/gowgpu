//go:build !(js && wasm)

// Presents the software backend's framebuffer in a macOS/iOS layer: into a
// CAMetalLayer's next drawable when the window has one, otherwise as a
// CALayer's contents. libobjc, CoreGraphics and Metal are opened with dlopen
// and their functions declared here by hand, so neither the frameworks nor
// the SDK headers are needed to build.

#include <stddef.h>
#include <stdint.h>

// <dlfcn.h>, <stdlib.h> and <string.h>, declared by hand like the rest.
extern void *dlopen(const char *, int);
extern void *dlsym(void *, const char *);
extern void *malloc(size_t);
extern void free(void *);
extern void *memcpy(void *, const void *, size_t);
#define SWB_RTLD_NOW_LOCAL (0x2 | 0x4)

typedef void *id;
typedef void *SEL;
typedef struct { double width, height; } swb_cgsize;            // CGSize
typedef struct { double x, y, width, height; } swb_cgrect;     // CGRect
typedef struct { uint64_t x, y, z, width, height, depth; } swb_region; // MTLRegion

static id (*swb_objc_lookUpClass)(const char *);
static id (*swb_object_getClass)(id);
static SEL (*swb_sel_registerName)(const char *);
static void *swb_objc_msgSend;
static void *(*swb_autoreleasePoolPush)(void);
static void (*swb_autoreleasePoolPop)(void *);

static void *(*swb_CGColorSpaceCreateDeviceRGB)(void);
static void *(*swb_CGDataProviderCreateWithData)(void *, const void *, size_t, void (*)(void *, const void *, size_t));
static void *(*swb_CGImageCreate)(size_t, size_t, size_t, size_t, size_t, void *, uint32_t, void *, const double *, _Bool, int32_t);
static void (*swb_CGImageRelease)(void *);
static void (*swb_CGDataProviderRelease)(void *);

static id swb_metal_layer_class, swb_device, swb_queue;
static void *swb_colorspace;

static SEL sel_device, sel_setDevice, sel_colorspace, sel_setColorspace, sel_setDrawableSize,
	sel_nextDrawable, sel_texture, sel_replaceRegion, sel_commandBuffer, sel_presentDrawable,
	sel_commit, sel_setNeedsDisplay, sel_setNeedsDisplayInRect, sel_setContents, sel_newCommandQueue;

#define SWB_SYM(lib, var, name) (*(void **)&(var) = dlsym((lib), (name)))

// swb_init resolves everything once. 0 when the CALayer path is usable; the
// Metal path additionally needs swb_queue.
int swb_init(void) {
	void *objc = dlopen("/usr/lib/libobjc.A.dylib", SWB_RTLD_NOW_LOCAL);
	void *cg = dlopen("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics", SWB_RTLD_NOW_LOCAL);
	if (!objc || !cg) {
		return -1;
	}
	if (!SWB_SYM(objc, swb_objc_lookUpClass, "objc_lookUpClass") ||
		!SWB_SYM(objc, swb_object_getClass, "object_getClass") ||
		!SWB_SYM(objc, swb_sel_registerName, "sel_registerName") ||
		!(swb_objc_msgSend = dlsym(objc, "objc_msgSend")) ||
		!SWB_SYM(objc, swb_autoreleasePoolPush, "objc_autoreleasePoolPush") ||
		!SWB_SYM(objc, swb_autoreleasePoolPop, "objc_autoreleasePoolPop") ||
		!SWB_SYM(cg, swb_CGColorSpaceCreateDeviceRGB, "CGColorSpaceCreateDeviceRGB") ||
		!SWB_SYM(cg, swb_CGDataProviderCreateWithData, "CGDataProviderCreateWithData") ||
		!SWB_SYM(cg, swb_CGImageCreate, "CGImageCreate") ||
		!SWB_SYM(cg, swb_CGImageRelease, "CGImageRelease") ||
		!SWB_SYM(cg, swb_CGDataProviderRelease, "CGDataProviderRelease")) {
		return -1;
	}
	// Lives as long as the process: layers keep a reference to it.
	swb_colorspace = swb_CGColorSpaceCreateDeviceRGB();

	sel_device = swb_sel_registerName("device");
	sel_setDevice = swb_sel_registerName("setDevice:");
	sel_colorspace = swb_sel_registerName("colorspace");
	sel_setColorspace = swb_sel_registerName("setColorspace:");
	sel_setDrawableSize = swb_sel_registerName("setDrawableSize:");
	sel_nextDrawable = swb_sel_registerName("nextDrawable");
	sel_texture = swb_sel_registerName("texture");
	sel_replaceRegion = swb_sel_registerName("replaceRegion:mipmapLevel:withBytes:bytesPerRow:");
	sel_commandBuffer = swb_sel_registerName("commandBuffer");
	sel_presentDrawable = swb_sel_registerName("presentDrawable:");
	sel_commit = swb_sel_registerName("commit");
	sel_setNeedsDisplay = swb_sel_registerName("setNeedsDisplay");
	sel_setNeedsDisplayInRect = swb_sel_registerName("setNeedsDisplayInRect:");
	sel_setContents = swb_sel_registerName("setContents:");
	sel_newCommandQueue = swb_sel_registerName("newCommandQueue");

	swb_metal_layer_class = swb_objc_lookUpClass("CAMetalLayer");
	void *metal = dlopen("/System/Library/Frameworks/Metal.framework/Metal", SWB_RTLD_NOW_LOCAL);
	id (*create_device)(void) = metal ? (id (*)(void))dlsym(metal, "MTLCreateSystemDefaultDevice") : NULL;
	if (create_device && (swb_device = create_device())) {
		swb_queue = ((id (*)(id, SEL))swb_objc_msgSend)(swb_device, sel_newCommandQueue);
	}
	return 0;
}

static void swb_send(id obj, SEL sel) { ((void (*)(id, SEL))swb_objc_msgSend)(obj, sel); }
static id swb_get(id obj, SEL sel) { return ((id (*)(id, SEL))swb_objc_msgSend)(obj, sel); }
static void swb_set(id obj, SEL sel, id v) { ((void (*)(id, SEL, id))swb_objc_msgSend)(obj, sel, v); }

// swb_blit_metal draws width x height BGRA pixels into the CAMetalLayer's
// next drawable. replaceRegion copies them, so data is not kept.
static void swb_blit_metal(id layer, const uint8_t *data, int32_t width, int32_t height) {
	if (!swb_get(layer, sel_device)) {
		swb_set(layer, sel_setDevice, swb_device);
	}
	if (!swb_get(layer, sel_colorspace)) {
		swb_set(layer, sel_setColorspace, swb_colorspace);
	}
	swb_cgsize size = {(double)width, (double)height};
	((void (*)(id, SEL, swb_cgsize))swb_objc_msgSend)(layer, sel_setDrawableSize, size);

	id drawable = swb_get(layer, sel_nextDrawable);
	id texture = drawable ? swb_get(drawable, sel_texture) : NULL;
	if (!texture) {
		return;
	}
	swb_region region = {0, 0, 0, (uint64_t)width, (uint64_t)height, 1};
	((void (*)(id, SEL, swb_region, uint64_t, const void *, uint64_t))swb_objc_msgSend)(
		texture, sel_replaceRegion, region, 0, data, 4 * (uint64_t)width);

	id buffer = swb_get(swb_queue, sel_commandBuffer);
	if (!buffer) {
		return;
	}
	swb_set(buffer, sel_presentDrawable, drawable);
	swb_send(buffer, sel_commit);
	swb_send(layer, sel_setNeedsDisplay);
}

static void swb_free_pixels(void *info, const void *data, size_t size) {
	(void)data;
	(void)size;
	free(info);
}

// swb_blit_layer makes a CGImage of a copy of the pixels the layer's
// contents: Core Animation keeps the image past this call, so it must not
// read the caller's buffer.
static void swb_blit_layer(id layer, const uint8_t *data, int32_t width, int32_t height, const int32_t *rects, int32_t nrects) {
	size_t len = 4 * (size_t)width * (size_t)height;
	void *pixels = malloc(len);
	if (!pixels) {
		return;
	}
	memcpy(pixels, data, len);
	void *provider = swb_CGDataProviderCreateWithData(pixels, pixels, len, swb_free_pixels);
	if (!provider) {
		free(pixels);
		return;
	}
	// kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst: BGRA in memory.
	void *image = swb_CGImageCreate((size_t)width, (size_t)height, 8, 32, 4 * (size_t)width,
		swb_colorspace, 0x2000 | 2, provider, NULL, 0, 0);
	swb_CGDataProviderRelease(provider);
	if (!image) {
		return;
	}
	swb_set(layer, sel_setContents, image);
	swb_CGImageRelease(image);

	if (nrects == 0) {
		swb_send(layer, sel_setNeedsDisplay);
		return;
	}
	for (int32_t i = 0; i < nrects; i++) {
		const int32_t *r = rects + 4 * i;
		swb_cgrect rect = {r[0], r[1], r[2], r[3]};
		((void (*)(id, SEL, swb_cgrect))swb_objc_msgSend)(layer, sel_setNeedsDisplayInRect, rect);
	}
}

// swb_blit presents to layer (a CALayer or CAMetalLayer). rects holds nrects
// x, y, width, height damage rectangles; with none the whole layer is redrawn.
void swb_blit(uintptr_t layer, const uint8_t *data, int32_t width, int32_t height, const int32_t *rects, int32_t nrects) {
	void *pool = swb_autoreleasePoolPush();
	id l = (id)layer;
	if (swb_queue && swb_metal_layer_class && swb_object_getClass(l) == swb_metal_layer_class) {
		swb_blit_metal(l, data, width, height); // a CAMetalLayer has no settable contents
	} else {
		swb_blit_layer(l, data, width, height, rects, nrects);
	}
	swb_autoreleasePoolPop(pool);
}
