//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

func TestFeaturesLeaveOutShaderF16(t *testing.T) {
	a := &Adapter{capabilities: AdapterCapabilities{FeatureLevel: d3d12.D3D_FEATURE_LEVEL_12_1, ShaderModel: d3d12.D3D_SHADER_MODEL_6_7}}
	if a.Features().Contains(gputypes.FeatureShaderF16) {
		t.Fatal("ShaderF16 advertised, but shaders are compiled for shader model 5.1 or 6.0, which have no 16-bit floats")
	}
}
