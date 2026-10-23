// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build !(js && wasm)

// Package vk holds the Vulkan bindings: types, constants and the function
// pointers of a Commands table, generated from the Khronos vk.xml.
//
// Every call goes through one of the C trampolines in trampolines.h, one per
// C signature, with the function pointer as the first argument. Pointers
// cross as typed pointers under #cgo noescape and nocallback, so nothing
// escapes and nothing allocates. cgocheck walks every argument that points
// at more pointers, each element of an array, and those may only lead to Go
// memory that is pinned (see Pin), the caller's stack or C memory. Building
// without cgo stops with a message saying so.
//
// # Generation
//
// cmd/vk-gen writes the bindings from vk.xml, then cmd/vk-cgo moves the
// wrappers onto the trampolines:
//
//	(cd cmd/vk-gen && go run . -spec vk.xml -out ../../hal/vulkan/vk/) && go run ./cmd/vk-cgo hal/vulkan/vk
//
// # Usage
//
//	if err := vk.Init(); err != nil {
//	    log.Fatal(err)
//	}
//
//	var cmds vk.Commands
//	if err := cmds.LoadGlobal(); err != nil {
//	    log.Fatal(err)
//	}
//	// create the instance, then
//	err := cmds.LoadInstance(instance)
package vk
