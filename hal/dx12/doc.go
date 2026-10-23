//go:build !(js && wasm)

// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

// Package dx12 is the DirectX 12 backend of the HAL, for 64-bit Windows 10
// and later, amd64 and arm64. It calls D3D12 and DXGI through syscall, so it
// builds with CGO_ENABLED=0.
//
// An instance needs DXGI 1.4, which every Windows 10 has, and an adapter
// needs Direct3D 12 at feature level 11_0 with resource binding tier 2.
// Windows 10 1803 and later add IDXGIFactory6: the adapters then come by GPU
// preference and a surface presents Fifo, Mailbox, and Immediate where the
// display can tear. Without it discrete GPUs come first and a surface
// presents Fifo only. A surface needs DXGI 1.5, Windows 10 1607 and later.
//
// Shaders are naga's HLSL compiled by FXC, d3dcompiler_47.dll, which comes
// with Windows, or naga's DXIL with GOGPU_DX12_DXIL=1.
//
// InstanceFlagsDebug turns on the D3D12 debug layer, DRED on Windows 10 1903
// and later, and with InstanceFlagsValidation also GPU-based validation, all
// of which slow everything down a lot.
//
// The COM bindings are in the d3d12, dxgi and d3dcompile packages.
package dx12
