package gputypes

import (
	"strings"
	"testing"
)

func TestDownlevelFlagsBitPositions(t *testing.T) {
	// Each flag must have exactly the correct bit position matching Rust wgpu-types.
	tests := []struct {
		name string
		flag DownlevelFlags
		bit  uint
	}{
		{"ComputeShaders", DownlevelFlagsComputeShaders, 0},
		{"FragmentWritableStorage", DownlevelFlagsFragmentWritableStorage, 1},
		{"IndirectExecution", DownlevelFlagsIndirectExecution, 2},
		{"BaseVertex", DownlevelFlagsBaseVertex, 3},
		{"ReadOnlyDepthStencil", DownlevelFlagsReadOnlyDepthStencil, 4},
		{"NonPowerOfTwoMipmappedTextures", DownlevelFlagsNonPowerOfTwoMipmappedTextures, 5},
		{"CubeArrayTextures", DownlevelFlagsCubeArrayTextures, 6},
		{"ComparisonSamplers", DownlevelFlagsComparisonSamplers, 7},
		{"IndependentBlend", DownlevelFlagsIndependentBlend, 8},
		{"VertexStorage", DownlevelFlagsVertexStorage, 9},
		{"AnisotropicFiltering", DownlevelFlagsAnisotropicFiltering, 10},
		{"FragmentStorage", DownlevelFlagsFragmentStorage, 11},
		{"MultisampledShading", DownlevelFlagsMultisampledShading, 12},
		{"DepthTextureAndBufferCopies", DownlevelFlagsDepthTextureAndBufferCopies, 13},
		{"WebGPUTextureFormatSupport", DownlevelFlagsWebGPUTextureFormatSupport, 14},
		{"BufferBindingsNot16ByteAligned", DownlevelFlagsBufferBindingsNot16ByteAligned, 15},
		{"UnrestrictedIndexBuffer", DownlevelFlagsUnrestrictedIndexBuffer, 16},
		{"FullDrawIndexUint32", DownlevelFlagsFullDrawIndexUint32, 17},
		{"DepthBiasClamp", DownlevelFlagsDepthBiasClamp, 18},
		{"ViewFormats", DownlevelFlagsViewFormats, 19},
		{"UnrestrictedExternalTextureCopies", DownlevelFlagsUnrestrictedExternalTextureCopies, 20},
		{"SurfaceViewFormats", DownlevelFlagsSurfaceViewFormats, 21},
		{"NonblockingQueryResolve", DownlevelFlagsNonblockingQueryResolve, 22},
		{"ShaderF16InF32", DownlevelFlagsShaderF16InF32, 23},
		{"MSL21", DownlevelFlagsMSL21, 24},
		{"TextureCompression", DownlevelFlagsTextureCompression, 25},
		{"LinearInterpolation", DownlevelFlagsLinearInterpolation, 26},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expected := DownlevelFlags(1 << tt.bit)
			if tt.flag != expected {
				t.Errorf("%s = %#x, want %#x (1 << %d)", tt.name, uint32(tt.flag), uint32(expected), tt.bit)
			}
		})
	}
}

func TestDownlevelFlagsContains(t *testing.T) {
	tests := []struct {
		name  string
		flags DownlevelFlags
		query DownlevelFlags
		want  bool
	}{
		{
			name:  "single flag present",
			flags: DownlevelFlagsComputeShaders,
			query: DownlevelFlagsComputeShaders,
			want:  true,
		},
		{
			name:  "single flag absent",
			flags: DownlevelFlagsComputeShaders,
			query: DownlevelFlagsFragmentWritableStorage,
			want:  false,
		},
		{
			name:  "multiple flags all present",
			flags: DownlevelFlagsComputeShaders | DownlevelFlagsIndirectExecution | DownlevelFlagsBaseVertex,
			query: DownlevelFlagsComputeShaders | DownlevelFlagsIndirectExecution,
			want:  true,
		},
		{
			name:  "multiple flags partial",
			flags: DownlevelFlagsComputeShaders | DownlevelFlagsIndirectExecution,
			query: DownlevelFlagsComputeShaders | DownlevelFlagsBaseVertex,
			want:  false,
		},
		{
			name:  "zero query always true",
			flags: DownlevelFlagsComputeShaders,
			query: 0,
			want:  true,
		},
		{
			name:  "zero flags with non-zero query",
			flags: 0,
			query: DownlevelFlagsComputeShaders,
			want:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.flags.Contains(tt.query); got != tt.want {
				t.Errorf("DownlevelFlags(%#x).Contains(%#x) = %v, want %v",
					uint32(tt.flags), uint32(tt.query), got, tt.want)
			}
		})
	}
}

func TestDownlevelFlagsAll(t *testing.T) {
	all := DownlevelFlagsAll()

	// Must have exactly 27 bits set.
	count := 0
	for v := uint32(all); v != 0; v &= v - 1 {
		count++
	}
	if count != 27 {
		t.Errorf("DownlevelFlagsAll() has %d bits set, want 27", count)
	}

	// Must equal (1 << 27) - 1 = 0x7FFFFFF.
	if uint32(all) != 0x7FFFFFF {
		t.Errorf("DownlevelFlagsAll() = %#x, want %#x", uint32(all), uint32(0x7FFFFFF))
	}

	// Every defined flag must be present in All.
	definedFlags := []DownlevelFlags{
		DownlevelFlagsComputeShaders,
		DownlevelFlagsFragmentWritableStorage,
		DownlevelFlagsIndirectExecution,
		DownlevelFlagsBaseVertex,
		DownlevelFlagsReadOnlyDepthStencil,
		DownlevelFlagsNonPowerOfTwoMipmappedTextures,
		DownlevelFlagsCubeArrayTextures,
		DownlevelFlagsComparisonSamplers,
		DownlevelFlagsIndependentBlend,
		DownlevelFlagsVertexStorage,
		DownlevelFlagsAnisotropicFiltering,
		DownlevelFlagsFragmentStorage,
		DownlevelFlagsMultisampledShading,
		DownlevelFlagsDepthTextureAndBufferCopies,
		DownlevelFlagsWebGPUTextureFormatSupport,
		DownlevelFlagsBufferBindingsNot16ByteAligned,
		DownlevelFlagsUnrestrictedIndexBuffer,
		DownlevelFlagsFullDrawIndexUint32,
		DownlevelFlagsDepthBiasClamp,
		DownlevelFlagsViewFormats,
		DownlevelFlagsUnrestrictedExternalTextureCopies,
		DownlevelFlagsSurfaceViewFormats,
		DownlevelFlagsNonblockingQueryResolve,
		DownlevelFlagsShaderF16InF32,
		DownlevelFlagsMSL21,
		DownlevelFlagsTextureCompression,
		DownlevelFlagsLinearInterpolation,
	}
	for _, f := range definedFlags {
		if !all.Contains(f) {
			t.Errorf("DownlevelFlagsAll() does not contain flag %#x", uint32(f))
		}
	}
}

func TestDownlevelFlagsCompliant(t *testing.T) {
	compliant := DownlevelFlagsCompliant()
	all := DownlevelFlagsAll()

	// Compliant = All minus AnisotropicFiltering.
	expected := all &^ DownlevelFlagsAnisotropicFiltering
	if compliant != expected {
		t.Errorf("DownlevelFlagsCompliant() = %#x, want %#x", uint32(compliant), uint32(expected))
	}

	// Compliant must NOT contain AnisotropicFiltering.
	if compliant.Contains(DownlevelFlagsAnisotropicFiltering) {
		t.Error("DownlevelFlagsCompliant() should not contain AnisotropicFiltering")
	}

	// Compliant must contain ComputeShaders (representative check).
	if !compliant.Contains(DownlevelFlagsComputeShaders) {
		t.Error("DownlevelFlagsCompliant() should contain ComputeShaders")
	}

	// Must have 26 bits set (27 minus 1).
	count := 0
	for v := uint32(compliant); v != 0; v &= v - 1 {
		count++
	}
	if count != 26 {
		t.Errorf("DownlevelFlagsCompliant() has %d bits set, want 26", count)
	}
}

func TestDownlevelFlagsString(t *testing.T) {
	tests := []struct {
		name  string
		flags DownlevelFlags
		want  string
	}{
		{
			name:  "none",
			flags: 0,
			want:  "(none)",
		},
		{
			name:  "single",
			flags: DownlevelFlagsComputeShaders,
			want:  "ComputeShaders",
		},
		{
			name:  "multiple",
			flags: DownlevelFlagsComputeShaders | DownlevelFlagsBaseVertex,
			want:  "ComputeShaders|BaseVertex",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.flags.String()
			if got != tt.want {
				t.Errorf("DownlevelFlags(%#x).String() = %q, want %q", uint32(tt.flags), got, tt.want)
			}
		})
	}
}

func TestDownlevelFlagsStringAllFlags(t *testing.T) {
	// All flags set should produce a string with all 27 names.
	all := DownlevelFlagsAll()
	s := all.String()
	parts := strings.Split(s, "|")
	if len(parts) != 27 {
		t.Errorf("DownlevelFlagsAll().String() has %d parts, want 27: %q", len(parts), s)
	}
}

func TestShaderModelString(t *testing.T) {
	tests := []struct {
		model ShaderModel
		want  string
	}{
		{ShaderModelSm2, "Sm2"},
		{ShaderModelSm4, "Sm4"},
		{ShaderModelSm5, "Sm5"},
		{ShaderModel(99), "Unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.model.String(); got != tt.want {
				t.Errorf("ShaderModel(%d).String() = %q, want %q", tt.model, got, tt.want)
			}
		})
	}
}

func TestShaderModelValues(t *testing.T) {
	// Shader model values must match D3D conventions.
	if ShaderModelSm2 != 2 {
		t.Errorf("ShaderModelSm2 = %d, want 2", ShaderModelSm2)
	}
	if ShaderModelSm4 != 4 {
		t.Errorf("ShaderModelSm4 = %d, want 4", ShaderModelSm4)
	}
	if ShaderModelSm5 != 5 {
		t.Errorf("ShaderModelSm5 = %d, want 5", ShaderModelSm5)
	}
}

func TestShaderModelOrdering(t *testing.T) {
	// ShaderModel must be orderable: Sm2 < Sm4 < Sm5.
	if !(ShaderModelSm2 < ShaderModelSm4) {
		t.Error("expected Sm2 < Sm4")
	}
	if !(ShaderModelSm4 < ShaderModelSm5) {
		t.Error("expected Sm4 < Sm5")
	}
}

func TestDefaultDownlevelCapabilities(t *testing.T) {
	dc := DefaultDownlevelCapabilities()

	if dc.Flags != DownlevelFlagsAll() {
		t.Errorf("DefaultDownlevelCapabilities().Flags = %#x, want %#x",
			uint32(dc.Flags), uint32(DownlevelFlagsAll()))
	}
	if dc.Limits != (DownlevelLimits{}) {
		t.Error("DefaultDownlevelCapabilities().Limits should be empty")
	}
	if dc.ShaderModel != ShaderModelSm5 {
		t.Errorf("DefaultDownlevelCapabilities().ShaderModel = %d, want %d",
			dc.ShaderModel, ShaderModelSm5)
	}
}

func TestIsWebGPUCompliant(t *testing.T) {
	tests := []struct {
		name string
		dc   DownlevelCapabilities
		want bool
	}{
		{
			name: "default is compliant",
			dc:   DefaultDownlevelCapabilities(),
			want: true,
		},
		{
			name: "compliant flags + Sm5",
			dc: DownlevelCapabilities{
				Flags:       DownlevelFlagsCompliant(),
				Limits:      DownlevelLimits{},
				ShaderModel: ShaderModelSm5,
			},
			want: true,
		},
		{
			name: "all flags + Sm5 is compliant",
			dc: DownlevelCapabilities{
				Flags:       DownlevelFlagsAll(),
				Limits:      DownlevelLimits{},
				ShaderModel: ShaderModelSm5,
			},
			want: true,
		},
		{
			name: "missing compute shaders",
			dc: DownlevelCapabilities{
				Flags:       DownlevelFlagsCompliant() &^ DownlevelFlagsComputeShaders,
				Limits:      DownlevelLimits{},
				ShaderModel: ShaderModelSm5,
			},
			want: false,
		},
		{
			name: "Sm4 is not compliant",
			dc: DownlevelCapabilities{
				Flags:       DownlevelFlagsCompliant(),
				Limits:      DownlevelLimits{},
				ShaderModel: ShaderModelSm4,
			},
			want: false,
		},
		{
			name: "Sm2 is not compliant",
			dc: DownlevelCapabilities{
				Flags:       DownlevelFlagsCompliant(),
				Limits:      DownlevelLimits{},
				ShaderModel: ShaderModelSm2,
			},
			want: false,
		},
		{
			name: "zero flags not compliant",
			dc: DownlevelCapabilities{
				Flags:       0,
				Limits:      DownlevelLimits{},
				ShaderModel: ShaderModelSm5,
			},
			want: false,
		},
		{
			name: "only aniso missing is still compliant",
			dc: DownlevelCapabilities{
				Flags:       DownlevelFlagsAll() &^ DownlevelFlagsAnisotropicFiltering,
				Limits:      DownlevelLimits{},
				ShaderModel: ShaderModelSm5,
			},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.dc.IsWebGPUCompliant(); got != tt.want {
				t.Errorf("IsWebGPUCompliant() = %v, want %v", got, tt.want)
			}
		})
	}
}
