//go:build !(js && wasm)

package wgpu

import (
	"errors"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// An indirect offset so large that offset+12 wraps is past the buffer, not
// inside it.
func TestDispatchIndirectOffsetThatWraps(t *testing.T) {
	device, h := newValidatedDevice(t)
	shader, err := device.CreateShaderModule(&ShaderModuleDescriptor{WGSL: "@compute @workgroup_size(1) fn main() {}"})
	if err != nil {
		t.Fatalf("CreateShaderModule: %v", err)
	}
	defer shader.Release()
	pipeline, err := device.CreateComputePipeline(&ComputePipelineDescriptor{Module: shader, EntryPoint: "main"})
	if err != nil {
		t.Skipf("CreateComputePipeline: %v", err)
	}
	defer pipeline.Release()
	buf, err := device.CreateBuffer(&BufferDescriptor{Size: 16, Usage: gputypes.BufferUsageIndirect})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	enc, err := device.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	pass, err := enc.BeginComputePass(nil)
	if err != nil {
		t.Fatalf("BeginComputePass: %v", err)
	}
	pass.SetPipeline(pipeline)
	pass.DispatchIndirect(buf, ^uint64(0)-3)
	_ = pass.End()
	if _, err := enc.Finish(); !errors.Is(err, ErrDispatchIndirectBufferOverrun) {
		t.Errorf("Finish = %v, want ErrDispatchIndirectBufferOverrun", err)
	}
	assertNoValidation(t, h)
}
