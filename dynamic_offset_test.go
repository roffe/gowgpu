//go:build !(js && wasm)

package wgpu_test

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu"
	"github.com/roffe/gowgpu/gputypes"
)

// SetBindGroup takes one dynamic offset per dynamic binding of the group,
// aligned to the device's limit for the binding type and keeping the binding
// inside its buffer; anything else is an encoder error, not a driver call.
func TestSetBindGroupChecksDynamicOffsets(t *testing.T) {
	_, _, device := newDevice(t)
	defer device.Release()
	requireHAL(t, device)

	buf, err := device.CreateBuffer(&wgpu.BufferDescriptor{Size: 512, Usage: wgpu.BufferUsageUniform | wgpu.BufferUsageStorage})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	layout, err := device.CreateBindGroupLayout(&wgpu.BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 1, Visibility: gputypes.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{
			Type: gputypes.BufferBindingTypeStorage, HasDynamicOffset: true, MinBindingSize: 16,
		}},
		{Binding: 0, Visibility: gputypes.ShaderStageCompute, Buffer: &gputypes.BufferBindingLayout{
			Type: gputypes.BufferBindingTypeUniform, HasDynamicOffset: true, MinBindingSize: 16,
		}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer layout.Release()
	group, err := device.CreateBindGroup(&wgpu.BindGroupDescriptor{Layout: layout, Entries: []wgpu.BindGroupEntry{
		{Binding: 1, Buffer: buf, Size: 32},
		{Binding: 0, Buffer: buf, Size: 16},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroup: %v", err)
	}
	defer group.Release()

	limits := device.Limits()
	uniform, storage := limits.MinUniformBufferOffsetAlignment, limits.MinStorageBufferOffsetAlignment
	// The last aligned offset that keeps a binding of size in the buffer,
	// and the one after it.
	last := func(size, alignment uint32) uint32 { return (512 - size) / alignment * alignment }
	past := func(size, alignment uint32) uint32 { return last(size, alignment) + alignment }
	for _, tc := range []struct {
		name    string
		offsets []uint32
		wantErr string
	}{
		{"none", nil, "dynamic offsets"},
		{"one short", []uint32{0}, "dynamic offsets"},
		{"one too many", []uint32{0, 0, 0}, "dynamic offsets"},
		{"in range, by binding number", []uint32{last(16, uniform), last(32, storage)}, ""},
		{"uniform unaligned", []uint32{uniform / 2, 0}, "not aligned"},
		{"storage unaligned", []uint32{0, storage / 2}, "not aligned"},
		{"uniform past the end", []uint32{past(16, uniform), 0}, "runs past"},
		{"storage past the end", []uint32{0, past(32, storage)}, "runs past"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			encoder, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := encoder.BeginComputePass(nil)
			if err != nil {
				t.Fatalf("BeginComputePass: %v", err)
			}
			pass.SetBindGroup(0, group, tc.offsets)
			_ = pass.End()
			cb, err := encoder.Finish()
			if cb != nil {
				cb.Release()
			}
			switch {
			case tc.wantErr == "" && err != nil:
				t.Errorf("SetBindGroup(%v): %v", tc.offsets, err)
			case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
				t.Errorf("SetBindGroup(%v): %v, want an error about %q", tc.offsets, err, tc.wantErr)
			}
		})
	}
}
