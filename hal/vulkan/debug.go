//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

/*
#include <stdint.h>
#if defined(__arm__)
#define VKAPI_PTR __attribute__((pcs("aapcs-vfp")))
#elif defined(_WIN32) && !defined(_WIN64)
#define VKAPI_PTR __stdcall
#else
#define VKAPI_PTR
#endif

// The validation layer calls the messenger from inside the Vulkan call that
// tripped it. Calling back into Go from there would break the nocallback
// promise every vk trampoline makes, so the callback stays in C: it copies
// the message into a ring that Go drains.

// The leading fields of VkDebugUtilsMessengerCallbackDataEXT.
typedef struct {
	uint32_t sType;
	const void *pNext;
	uint32_t flags;
	const char *pMessageIdName;
	int32_t messageIdNumber;
	const char *pMessage;
} gvk_debug_data;

enum { GVK_DEBUG_SLOTS = 256, GVK_DEBUG_ID = 128, GVK_DEBUG_MSG = 1024 };

typedef struct {
	uint32_t severity, types;
	char id[GVK_DEBUG_ID];
	char msg[GVK_DEBUG_MSG];
} gvk_debug_entry;

static gvk_debug_entry gvk_debug_ring[GVK_DEBUG_SLOTS];
static uint32_t gvk_debug_head, gvk_debug_tail, gvk_debug_dropped;
static int gvk_debug_lock;

static void gvk_debug_acquire(void) { while (__atomic_exchange_n(&gvk_debug_lock, 1, __ATOMIC_ACQUIRE)) {} }
static void gvk_debug_release(void) { __atomic_store_n(&gvk_debug_lock, 0, __ATOMIC_RELEASE); }

static void gvk_debug_copy(char *dst, const char *src, int n) {
	int i = 0;
	if (src) {
		for (; i < n - 1 && src[i]; i++) {
			dst[i] = src[i];
		}
	}
	dst[i] = 0;
}

// PFN_vkDebugUtilsMessengerCallbackEXT
static uint32_t VKAPI_PTR gvk_debug_callback(uint32_t severity, uint32_t types, const void *data, void *user) {
	const gvk_debug_data *d = (const gvk_debug_data *)data;
	gvk_debug_acquire();
	if (gvk_debug_head - gvk_debug_tail < GVK_DEBUG_SLOTS) {
		gvk_debug_entry *e = &gvk_debug_ring[gvk_debug_head % GVK_DEBUG_SLOTS];
		e->severity = severity;
		e->types = types;
		gvk_debug_copy(e->id, d ? d->pMessageIdName : 0, GVK_DEBUG_ID);
		gvk_debug_copy(e->msg, d ? d->pMessage : 0, GVK_DEBUG_MSG);
		gvk_debug_head++;
	} else {
		gvk_debug_dropped++;
	}
	gvk_debug_release();
	return 0; // VK_FALSE: never abort the call
}

static uintptr_t gvk_debug_callback_ptr(void) { return (uintptr_t)&gvk_debug_callback; }

// gvk_debug_pop moves the oldest message to out, returning 0 when there is none.
static int gvk_debug_pop(gvk_debug_entry *out) {
	int ok = 0;
	gvk_debug_acquire();
	if (gvk_debug_tail != gvk_debug_head) {
		*out = gvk_debug_ring[gvk_debug_tail % GVK_DEBUG_SLOTS];
		gvk_debug_tail++;
		ok = 1;
	}
	gvk_debug_release();
	return ok;
}

// gvk_debug_take_dropped returns how many messages found the ring full
// since the last call.
static uint32_t gvk_debug_take_dropped(void) {
	gvk_debug_acquire();
	uint32_t n = gvk_debug_dropped;
	gvk_debug_dropped = 0;
	gvk_debug_release();
	return n;
}
*/
import "C"

import (
	"context"
	"log/slog"
	"runtime"
	"sync"
	"time"

	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/vulkan/vk"
)

var drainOnce sync.Once

// drainDebugMessages logs what the C messenger collected, every 50ms, for as
// long as the process runs; only debug instances start it.
func drainDebugMessages() {
	drainOnce.Do(func() {
		go func() {
			var e C.gvk_debug_entry
			for range time.Tick(50 * time.Millisecond) {
				for C.gvk_debug_pop(&e) != 0 {
					logDebugMessage(uint32(e.severity), uint32(e.types),
						C.GoString(&e.id[0]), C.GoString(&e.msg[0]))
				}
				if n := C.gvk_debug_take_dropped(); n != 0 {
					hal.Logger().Error("vulkan: Validation layer messages lost, the debug messenger's ring was full", "count", uint32(n))
				}
			}
		}()
	})
}

// debugCallbackPtr holds the callback function pointer to prevent GC collection.
// Once created, the callback lives for the process lifetime (Vulkan requirement).
var debugCallbackPtr uintptr

// logDebugMessage logs one validation message the C messenger collected.
func logDebugMessage(severity, types uint32, msgID, msg string) {
	if msg == "" {
		msg = "(no message)"
	}

	// Map Vulkan severity to slog level.
	severityBits := vk.DebugUtilsMessageSeverityFlagBitsEXT(severity)
	var level slog.Level
	switch {
	case severityBits&vk.DebugUtilsMessageSeverityErrorBitExt != 0:
		level = slog.LevelError
	case severityBits&vk.DebugUtilsMessageSeverityWarningBitExt != 0:
		level = slog.LevelWarn
	case severityBits&vk.DebugUtilsMessageSeverityInfoBitExt != 0:
		level = slog.LevelInfo
	default:
		level = slog.LevelDebug
	}

	// Determine message type.
	typeBits := vk.DebugUtilsMessageTypeFlagBitsEXT(types)
	var typeStr string
	switch {
	case typeBits&vk.DebugUtilsMessageTypeValidationBitExt != 0:
		typeStr = "Validation"
	case typeBits&vk.DebugUtilsMessageTypePerformanceBitExt != 0:
		typeStr = "Performance"
	default:
		typeStr = "General"
	}

	attrs := []slog.Attr{
		slog.String("type", typeStr),
	}
	if msgID != "" {
		attrs = append(attrs, slog.String("id", msgID))
	}
	hal.Logger().LogAttrs(context.Background(), level, "vulkan: "+msg, attrs...)
}

// createDebugMessenger creates a Vulkan debug utils messenger for the instance.
// Its callback is C (gvk_debug_callback); drainDebugMessages logs what it collects.
// Returns the messenger handle, or 0 if creation fails (non-fatal).
func createDebugMessenger(instance *Instance) vk.DebugUtilsMessengerEXT {
	// Create callback pointer once (safe for concurrent use after initialization).
	if debugCallbackPtr == 0 {
		debugCallbackPtr = uintptr(C.gvk_debug_callback_ptr())
	}
	drainDebugMessages()

	severityFlags := vk.DebugUtilsMessageSeverityFlagsEXT(
		vk.DebugUtilsMessageSeverityWarningBitExt |
			vk.DebugUtilsMessageSeverityErrorBitExt,
	)

	typeFlags := vk.DebugUtilsMessageTypeFlagsEXT(
		vk.DebugUtilsMessageTypeGeneralBitExt |
			vk.DebugUtilsMessageTypeValidationBitExt |
			vk.DebugUtilsMessageTypePerformanceBitExt,
	)

	createInfo := vk.DebugUtilsMessengerCreateInfoEXT{
		SType:           vk.StructureTypeDebugUtilsMessengerCreateInfoExt,
		MessageSeverity: severityFlags,
		MessageType:     typeFlags,
		PfnUserCallback: debugCallbackPtr,
	}

	var messenger vk.DebugUtilsMessengerEXT
	result := instance.cmds.CreateDebugUtilsMessengerEXT(
		instance.handle, &createInfo, nil, &messenger,
	)
	if result != vk.Success {
		hal.Logger().Warn("vulkan: failed to create debug messenger", "result", result)
		return 0
	}

	// Keep the callback function alive for the lifetime of the messenger.
	runtime.KeepAlive(debugCallbackPtr)

	return messenger
}

// destroyDebugMessenger destroys the debug utils messenger.
func destroyDebugMessenger(instance *Instance, messenger vk.DebugUtilsMessengerEXT) {
	if messenger != 0 {
		instance.cmds.DestroyDebugUtilsMessengerEXT(instance.handle, messenger, nil)
	}
}

// setObjectName names a Vulkan object for validation messages and GPU
// debuggers, when the instance enables VK_EXT_debug_utils.
func (d *Device) setObjectName(objectType vk.ObjectType, handle uint64, name string) {
	if !d.cmds.HasDebugUtils() || handle == 0 {
		return
	}

	cname := append([]byte(name), 0)
	nameInfo := vk.DebugUtilsObjectNameInfoEXT{
		SType:        vk.StructureTypeDebugUtilsObjectNameInfoExt,
		ObjectType:   objectType,
		ObjectHandle: handle,
		PObjectName:  &cname[0],
	}
	_ = d.cmds.SetDebugUtilsObjectNameEXT(d.handle, &nameInfo)
	runtime.KeepAlive(cname)
}
