//go:build js && wasm

package browser

import (
	"strings"
	"syscall/js"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

func TestLimitsCarryInterStageVariablesAndBindGroupsPlusVertexBuffers(t *testing.T) {
	limits := ExtractLimits(js.Global().Get("JSON").Call("parse", `{"maxInterStageShaderVariables": 28, "maxBindGroupsPlusVertexBuffers": 30}`))
	if limits.MaxInterStageShaderVariables != 28 || limits.MaxBindGroupsPlusVertexBuffers != 30 {
		t.Fatalf("ExtractLimits = %d inter-stage variables, %d bind groups plus vertex buffers, want 28 and 30",
			limits.MaxInterStageShaderVariables, limits.MaxBindGroupsPlusVertexBuffers)
	}
	obj := limitsToJSObject(gputypes.Limits{MaxInterStageShaderVariables: 28, MaxBindGroupsPlusVertexBuffers: 30})
	if obj.Get("maxInterStageShaderVariables").Int() != 28 || obj.Get("maxBindGroupsPlusVertexBuffers").Int() != 30 {
		t.Fatalf("limitsToJSObject = %v, want both limits", js.Global().Get("JSON").Call("stringify", obj))
	}
}

func TestRequiredFeatures(t *testing.T) {
	features, err := featuresToJSArray(gputypes.Features(gputypes.FeatureSubgroupOperations | gputypes.FeatureShaderF16))
	if err != nil {
		t.Fatalf("featuresToJSArray: %v", err)
	}
	if got := js.Global().Get("JSON").Call("stringify", features).String(); got != `["shader-f16","subgroups"]` {
		t.Fatalf("features %s, want shader-f16 and subgroups", got)
	}

	_, err = featuresToJSArray(gputypes.Features(gputypes.FeatureShaderF16 | gputypes.FeaturePushConstants))
	if err == nil || !strings.Contains(err.Error(), "PushConstants") || strings.Contains(err.Error(), "ShaderF16") {
		t.Fatalf("featuresToJSArray with a native-only feature = %v, want an error naming PushConstants only", err)
	}

	supported := js.Global().Get("Set").New(js.ValueOf([]any{"subgroups", "texture-formats-tier1"}))
	if got := ExtractFeatures(supported); !got.Contains(gputypes.FeatureSubgroupOperations) || !got.Contains(gputypes.FeatureTextureFormatsTier1) {
		t.Fatalf("ExtractFeatures = %#x, want subgroups and texture-formats-tier1", uint64(got))
	}
}
