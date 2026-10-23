//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/noop"
)

// An indexed draw with a base vertex needs an adapter with
// DownlevelFlagsBaseVertex: without it the draw fails the encoder.
func TestDrawIndexedBaseVertexNeedsTheDownlevelFlag(t *testing.T) {
	for _, tc := range []struct {
		name       string
		flags      gputypes.DownlevelFlags
		baseVertex int32
		wantErr    bool
	}{
		{"base vertex with the flag", gputypes.DownlevelFlagsBaseVertex, 2, false},
		{"no base vertex without the flag", 0, 0, false},
		{"base vertex without the flag", 0, 2, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			adapter := &core.Adapter{DownlevelCapabilities: gputypes.DownlevelCapabilities{Flags: tc.flags}}
			command := &captureCommandEncoder{CommandEncoder: &noop.CommandEncoder{},
				renderPass: &captureRenderPass{RenderPassEncoder: &noop.RenderPassEncoder{}}}
			h := &captureDevice{Device: &noop.Device{}, commandEncoder: command}
			device := &Device{core: core.NewDevice(h, adapter, 0, DefaultLimits(), "base-vertex")}
			index, err := device.CreateBuffer(&BufferDescriptor{Size: 64, Usage: BufferUsageIndex})
			if err != nil {
				t.Fatalf("CreateBuffer: %v", err)
			}
			enc, err := device.CreateCommandEncoder(nil)
			if err != nil {
				t.Fatalf("CreateCommandEncoder: %v", err)
			}
			pass, err := enc.BeginRenderPass(&RenderPassDescriptor{ColorAttachments: captureAttachments(t, device)})
			if err != nil {
				t.Fatalf("BeginRenderPass: %v", err)
			}
			pipeline := &RenderPipeline{}
			pipeline.SetTestRequiredVertexBuffers(0)
			pass.SetPipeline(pipeline)
			pass.SetIndexBuffer(index, gputypes.IndexFormatUint16, 0)
			pass.DrawIndexed(gputypes.DrawIndexedArgs{IndexCount: 3, InstanceCount: 1, BaseVertex: tc.baseVertex})
			_ = pass.End()
			_, err = enc.Finish()
			switch {
			case tc.wantErr && (err == nil || !strings.Contains(err.Error(), "DownlevelFlagsBaseVertex")):
				t.Errorf("Finish: %v, want an error about DownlevelFlagsBaseVertex", err)
			case !tc.wantErr && err != nil:
				t.Errorf("Finish: %v", err)
			}
		})
	}
}
