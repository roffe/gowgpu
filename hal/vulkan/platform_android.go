//go:build android && (arm64 || amd64)

// Copyright 2026 The GoGPU Authors
// SPDX-License-Identifier: MIT

package vulkan

/*
#include <stdint.h>
extern int atoi(const char *);

// <sys/system_properties.h>, declared by hand: available at every API level,
// unlike android_get_device_api_level (API 29+).
extern int __system_property_get(const char *, char *);

static int gvk_android_sdk(void) {
	char value[92] = {0}; // PROP_VALUE_MAX
	if (__system_property_get("ro.build.version.sdk", value) <= 0) {
		return -1;
	}
	return atoi(value);
}
*/
import "C"

import (
	"fmt"

	"github.com/roffe/wgpu/hal"
)

type platformInstanceState struct {
	swapchain swapchainPlatformPolicy
}

func newPlatformInstanceState(desc *hal.InstanceDescriptor) (platformInstanceState, error) {
	if desc != nil {
		if err := validateAndroidInstanceFlags(desc.Flags); err != nil {
			return platformInstanceState{}, err
		}
	}

	sdk, err := loadAndroidSDKVersion()
	if err != nil {
		return platformInstanceState{}, fmt.Errorf("vulkan: query Android SDK version: %w", err)
	}
	if err := validateAndroidSDKVersion(sdk); err != nil {
		return platformInstanceState{}, err
	}

	return platformInstanceState{swapchain: androidSwapchainPlatformPolicy(sdk)}, nil
}

func loadAndroidSDKVersion() (uint32, error) {
	sdk := int32(C.gvk_android_sdk())
	if sdk < 0 {
		return 0, fmt.Errorf("invalid SDK version %d", sdk)
	}
	return uint32(sdk), nil
}
