//go:build windows && !(js && wasm)

package dx12

import (
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
	"github.com/roffe/gowgpu/hal/dx12/d3d12"
)

func TestBuildDepthStencilDescStencilDefaults(t *testing.T) {
	equal := hal.StencilFaceState{Compare: gputypes.CompareFunctionEqual}
	for _, tc := range []struct {
		name    string
		face    hal.StencilFaceState
		masks   uint32
		enabled bool
	}{
		{"zero faces and masks", hal.StencilFaceState{}, 0, false},
		{"zero faces with masks", hal.StencilFaceState{}, 0xFF, false},
		{"always and keep with masks", hal.StencilFaceState{Compare: gputypes.CompareFunctionAlways, FailOp: hal.StencilOperationKeep, DepthFailOp: hal.StencilOperationKeep, PassOp: hal.StencilOperationKeep}, 0xFF, false},
		{"a test without masks", equal, 0, false},
		{"a test with masks", equal, 0xFF, true},
		{"a write with masks", hal.StencilFaceState{PassOp: hal.StencilOperationReplace}, 0xFF, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			desc := buildDepthStencilDesc(&hal.DepthStencilState{
				Format:           gputypes.TextureFormatDepth24PlusStencil8,
				DepthCompare:     gputypes.CompareFunctionLess,
				StencilFront:     tc.face,
				StencilBack:      tc.face,
				StencilReadMask:  tc.masks,
				StencilWriteMask: tc.masks,
			})
			if got := desc.StencilEnable != 0; got != tc.enabled {
				t.Fatalf("StencilEnable = %v, want %v", got, tc.enabled)
			}
			if !tc.enabled {
				return
			}
			want := d3d12.D3D12_COMPARISON_FUNC_ALWAYS
			if tc.face.Compare != gputypes.CompareFunctionUndefined {
				want = compareFunctionToD3D12(tc.face.Compare)
			}
			if desc.FrontFace.StencilFunc != want || desc.BackFace.StencilFunc != want {
				t.Fatalf("stencil funcs = %d/%d, want %d", desc.FrontFace.StencilFunc, desc.BackFace.StencilFunc, want)
			}
			if desc.FrontFace.StencilFailOp != d3d12.D3D12_STENCIL_OP_KEEP {
				t.Fatalf("undefined fail op = %d, want KEEP", desc.FrontFace.StencilFailOp)
			}
		})
	}
}

func TestBuildDepthStencilDescDepthDefaults(t *testing.T) {
	for _, tc := range []struct {
		name    string
		compare gputypes.CompareFunction
		write   bool
		enabled bool
		fn      d3d12.D3D12_COMPARISON_FUNC
	}{
		{"unset compare without writes", gputypes.CompareFunctionUndefined, false, false, d3d12.D3D12_COMPARISON_FUNC_ALWAYS},
		{"unset compare with writes", gputypes.CompareFunctionUndefined, true, true, d3d12.D3D12_COMPARISON_FUNC_ALWAYS},
		{"always without writes", gputypes.CompareFunctionAlways, false, false, d3d12.D3D12_COMPARISON_FUNC_ALWAYS},
		{"less", gputypes.CompareFunctionLess, false, true, d3d12.D3D12_COMPARISON_FUNC_LESS},
		{"never", gputypes.CompareFunctionNever, false, true, d3d12.D3D12_COMPARISON_FUNC_NEVER},
	} {
		t.Run(tc.name, func(t *testing.T) {
			desc := buildDepthStencilDesc(&hal.DepthStencilState{
				Format:            gputypes.TextureFormatDepth24PlusStencil8,
				DepthCompare:      tc.compare,
				DepthWriteEnabled: tc.write,
			})
			if got := desc.DepthEnable != 0; got != tc.enabled || desc.DepthFunc != tc.fn {
				t.Fatalf("DepthEnable, DepthFunc = %v, %d, want %v, %d", got, desc.DepthFunc, tc.enabled, tc.fn)
			}
		})
	}
}
