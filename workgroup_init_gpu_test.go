//go:build !(js && wasm)

package wgpu

import "testing"

// dirtyWorkgroupShader fills workgroup memory with data[0] and copies it
// out, so the next dispatch can find it there.
const dirtyWorkgroupShader = `
var<workgroup> shared_data: array<u32, 64>;
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(local_invocation_index) i: u32) {
	let marker = data[0];
	workgroupBarrier();
	shared_data[i] = marker;
	workgroupBarrier();
	data[i] = shared_data[63u - i];
}
`

// readWorkgroupShader copies workgroup memory out without writing it.
const readWorkgroupShader = `
var<workgroup> shared_data: array<u32, 64>;
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@compute @workgroup_size(64) fn cs_main(@builtin(local_invocation_index) i: u32) {
	data[i] = shared_data[i];
}
`

// TestWorkgroupMemoryStartsZeroed reads workgroup memory that a dispatch
// before filled with a marker, which WebGPU requires to read zero.
func TestWorkgroupMemoryStartsZeroed(t *testing.T) {
	device, h := newValidatedDevice(t)
	in := make([]uint32, 64)
	in[0] = 0xdead
	if got := runU32Compute(t, device, dirtyWorkgroupShader, nil, in)[5]; got != 0xdead {
		t.Fatalf("filling workgroup memory read back %#x, want 0xdead", got)
	}
	for i, got := range runU32Compute(t, device, readWorkgroupShader, nil, make([]uint32, 64)) {
		if got != 0 {
			t.Fatalf("workgroup memory [%d] = %#x, want 0", i, got)
		}
	}
	assertNoValidation(t, h)
}
