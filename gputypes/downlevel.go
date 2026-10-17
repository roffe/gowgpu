package gputypes

import "strings"

// DownlevelFlags represents binary flags listing features that may or may not
// be present on downlevel adapters.
//
// The W3C WebGPU spec assumes all adapters meet a baseline (compute, indirect draw,
// etc.) and simply excludes non-conformant hardware from requestAdapter(). As a
// native Go library, we support backends below that baseline (GLES 3.0, CPU software
// rasterizer) and degrade gracefully instead of refusing to run. DownlevelFlags
// tracks exactly what each backend supports, enabling consumers to make informed
// decisions (e.g., GPU compute vs CPU fallback).
//
// This is a Rust wgpu extension — the term "downlevel" does not appear in the
// W3C WebGPU specification. Of 27 flags, 24 track capabilities required by the
// spec for core (conformant) adapters, 1 (AnisotropicFiltering) is not required
// by the spec, and 2 (MSL21, SurfaceViewFormats) are backend-specific.
//
// You can check whether a set of flags is compliant through the
// DownlevelCapabilities.IsWebGPUCompliant method.
//
// Bit positions match Rust wgpu-types (limits.rs:1102-1246).
type DownlevelFlags uint32

const (
	// DownlevelFlagsComputeShaders indicates the device supports compiling and using compute shaders.
	// WebGL2 and GLES 3.0 devices do not support compute.
	DownlevelFlagsComputeShaders DownlevelFlags = 1 << 0

	// DownlevelFlagsFragmentWritableStorage indicates support for binding storage buffers
	// and textures to fragment shaders.
	DownlevelFlagsFragmentWritableStorage DownlevelFlags = 1 << 1

	// DownlevelFlagsIndirectExecution indicates support for indirect drawing and dispatching.
	// DownlevelFlagsComputeShaders must be present for this flag.
	// WebGL2, GLES 3.0, and Metal on Apple1/Apple2 GPUs do not support indirect.
	DownlevelFlagsIndirectExecution DownlevelFlags = 1 << 2

	// DownlevelFlagsBaseVertex indicates support for non-zero base_vertex parameter
	// to direct indexed draw calls.
	// Indirect calls, if supported, always support non-zero base_vertex.
	DownlevelFlagsBaseVertex DownlevelFlags = 1 << 3

	// DownlevelFlagsReadOnlyDepthStencil indicates support for reading from a depth/stencil
	// texture while using it as a read-only depth/stencil attachment.
	// The WebGL2 and GLES backends do not support RODS.
	DownlevelFlagsReadOnlyDepthStencil DownlevelFlags = 1 << 4

	// DownlevelFlagsNonPowerOfTwoMipmappedTextures indicates support for textures with mipmaps
	// which have a non power of two size.
	DownlevelFlagsNonPowerOfTwoMipmappedTextures DownlevelFlags = 1 << 5

	// DownlevelFlagsCubeArrayTextures indicates support for textures that are cube arrays.
	DownlevelFlagsCubeArrayTextures DownlevelFlags = 1 << 6

	// DownlevelFlagsComparisonSamplers indicates support for comparison samplers.
	DownlevelFlagsComparisonSamplers DownlevelFlags = 1 << 7

	// DownlevelFlagsIndependentBlend indicates support for different blend operations
	// per color attachment.
	DownlevelFlagsIndependentBlend DownlevelFlags = 1 << 8

	// DownlevelFlagsVertexStorage indicates support for storage buffers in vertex shaders.
	DownlevelFlagsVertexStorage DownlevelFlags = 1 << 9

	// DownlevelFlagsAnisotropicFiltering indicates support for samplers with anisotropic filtering.
	// Note this isn't actually required by WebGPU; the implementation is allowed to completely
	// ignore aniso clamp. This flag is here for native backends so they can communicate
	// to the user if aniso is enabled.
	// All backends and all devices support anisotropic filtering.
	DownlevelFlagsAnisotropicFiltering DownlevelFlags = 1 << 10

	// DownlevelFlagsFragmentStorage indicates support for storage buffers in fragment shaders.
	DownlevelFlagsFragmentStorage DownlevelFlags = 1 << 11

	// DownlevelFlagsMultisampledShading indicates support for sample-rate shading.
	DownlevelFlagsMultisampledShading DownlevelFlags = 1 << 12

	// DownlevelFlagsDepthTextureAndBufferCopies indicates support for copies between
	// depth textures and buffers.
	// GLES/WebGL don't support this.
	DownlevelFlagsDepthTextureAndBufferCopies DownlevelFlags = 1 << 13

	// DownlevelFlagsWebGPUTextureFormatSupport indicates support for all the texture usages
	// described in WebGPU. If this isn't supported, call GetTextureFormatFeatures to determine
	// how you can use textures of a given format.
	DownlevelFlagsWebGPUTextureFormatSupport DownlevelFlags = 1 << 14

	// DownlevelFlagsBufferBindingsNot16ByteAligned indicates support for buffer bindings
	// with sizes that aren't a multiple of 16.
	// WebGL doesn't support this.
	DownlevelFlagsBufferBindingsNot16ByteAligned DownlevelFlags = 1 << 15

	// DownlevelFlagsUnrestrictedIndexBuffer indicates support for buffers to combine INDEX
	// usage with usages other than COPY_DST and COPY_SRC.
	// WebGL doesn't support this.
	DownlevelFlagsUnrestrictedIndexBuffer DownlevelFlags = 1 << 16

	// DownlevelFlagsFullDrawIndexUint32 indicates support for full 32-bit range indices
	// (2^32-1 as opposed to 2^24-1 without this flag).
	// Corresponds to Vulkan's VkPhysicalDeviceFeatures.fullDrawIndexUint32.
	DownlevelFlagsFullDrawIndexUint32 DownlevelFlags = 1 << 17

	// DownlevelFlagsDepthBiasClamp indicates support for depth bias clamping.
	// Corresponds to Vulkan's VkPhysicalDeviceFeatures.depthBiasClamp.
	DownlevelFlagsDepthBiasClamp DownlevelFlags = 1 << 18

	// DownlevelFlagsViewFormats indicates support for specifying which view format values
	// are allowed when create_view() is called on a texture.
	// The WebGL and GLES backends don't support this.
	DownlevelFlagsViewFormats DownlevelFlags = 1 << 19

	// DownlevelFlagsUnrestrictedExternalTextureCopies indicates support for unrestricted
	// external texture copy operations.
	// WebGL doesn't support this. WebGPU does.
	DownlevelFlagsUnrestrictedExternalTextureCopies DownlevelFlags = 1 << 20

	// DownlevelFlagsSurfaceViewFormats indicates support for specifying which view formats
	// are allowed when calling create_view on the texture returned by Surface.GetCurrentTexture.
	// The GLES/WebGL and Vulkan on Android don't support this.
	DownlevelFlagsSurfaceViewFormats DownlevelFlags = 1 << 21

	// DownlevelFlagsNonblockingQueryResolve indicates that calls to
	// CommandEncoder.ResolveQuerySet will be performed on the queue timeline.
	// If false, resolve will be performed on the device (CPU) timeline and will block
	// until the query has data.
	DownlevelFlagsNonblockingQueryResolve DownlevelFlags = 1 << 22

	// DownlevelFlagsShaderF16InF32 allows shaders to use quantizeToF16, pack2x16float,
	// and unpack2x16float, which operate on f16-precision values stored in f32s.
	// Not supported by Vulkan on Mesa when FeatureShaderF16 is absent.
	DownlevelFlagsShaderF16InF32 DownlevelFlags = 1 << 23

	// DownlevelFlagsMSL21 indicates support for features introduced in MSL 2.1.
	DownlevelFlagsMSL21 DownlevelFlags = 1 << 24

	// DownlevelFlagsTextureCompression indicates the adapter supports the WebGPU texture
	// compression requirement: BC || (ETC2 && ASTC).
	// See https://www.w3.org/TR/webgpu/#adapter-capability-guarantees.
	DownlevelFlagsTextureCompression DownlevelFlags = 1 << 25

	// DownlevelFlagsLinearInterpolation indicates support for @interpolate(linear)
	// (a.k.a. noperspective) on shader inter-stage variables.
	// GLSL ES has no noperspective qualifier, so the GLES backend only supports this
	// on desktop OpenGL, not on GLES/WebGL2.
	DownlevelFlagsLinearInterpolation DownlevelFlags = 1 << 26
)

// downlevelFlagCount is the total number of defined DownlevelFlags constants.
const downlevelFlagCount = 27

// Contains reports whether f contains all bits in flag.
func (f DownlevelFlags) Contains(flag DownlevelFlags) bool {
	return f&flag == flag
}

// String returns a human-readable representation of the flags.
func (f DownlevelFlags) String() string {
	if f == 0 {
		return "(none)"
	}

	type flagName struct {
		flag DownlevelFlags
		name string
	}

	flags := []flagName{
		{DownlevelFlagsComputeShaders, "ComputeShaders"},
		{DownlevelFlagsFragmentWritableStorage, "FragmentWritableStorage"},
		{DownlevelFlagsIndirectExecution, "IndirectExecution"},
		{DownlevelFlagsBaseVertex, "BaseVertex"},
		{DownlevelFlagsReadOnlyDepthStencil, "ReadOnlyDepthStencil"},
		{DownlevelFlagsNonPowerOfTwoMipmappedTextures, "NonPowerOfTwoMipmappedTextures"},
		{DownlevelFlagsCubeArrayTextures, "CubeArrayTextures"},
		{DownlevelFlagsComparisonSamplers, "ComparisonSamplers"},
		{DownlevelFlagsIndependentBlend, "IndependentBlend"},
		{DownlevelFlagsVertexStorage, "VertexStorage"},
		{DownlevelFlagsAnisotropicFiltering, "AnisotropicFiltering"},
		{DownlevelFlagsFragmentStorage, "FragmentStorage"},
		{DownlevelFlagsMultisampledShading, "MultisampledShading"},
		{DownlevelFlagsDepthTextureAndBufferCopies, "DepthTextureAndBufferCopies"},
		{DownlevelFlagsWebGPUTextureFormatSupport, "WebGPUTextureFormatSupport"},
		{DownlevelFlagsBufferBindingsNot16ByteAligned, "BufferBindingsNot16ByteAligned"},
		{DownlevelFlagsUnrestrictedIndexBuffer, "UnrestrictedIndexBuffer"},
		{DownlevelFlagsFullDrawIndexUint32, "FullDrawIndexUint32"},
		{DownlevelFlagsDepthBiasClamp, "DepthBiasClamp"},
		{DownlevelFlagsViewFormats, "ViewFormats"},
		{DownlevelFlagsUnrestrictedExternalTextureCopies, "UnrestrictedExternalTextureCopies"},
		{DownlevelFlagsSurfaceViewFormats, "SurfaceViewFormats"},
		{DownlevelFlagsNonblockingQueryResolve, "NonblockingQueryResolve"},
		{DownlevelFlagsShaderF16InF32, "ShaderF16InF32"},
		{DownlevelFlagsMSL21, "MSL21"},
		{DownlevelFlagsTextureCompression, "TextureCompression"},
		{DownlevelFlagsLinearInterpolation, "LinearInterpolation"},
	}

	var parts []string
	for _, fn := range flags {
		if f.Contains(fn.flag) {
			parts = append(parts, fn.name)
		}
	}

	if len(parts) == 0 {
		return "(none)"
	}

	return strings.Join(parts, "|")
}

// DownlevelFlagsAll returns a DownlevelFlags value with all 27 bits set.
func DownlevelFlagsAll() DownlevelFlags {
	return (1 << downlevelFlagCount) - 1
}

// DownlevelFlagsCompliant returns all flags that indicate WebGPU compliance.
// This is all flags except AnisotropicFiltering, which is not required by WebGPU.
// Matches Rust wgpu-types DownlevelFlags::compliant() (limits.rs:1249-1257).
func DownlevelFlagsCompliant() DownlevelFlags {
	return DownlevelFlagsAll() &^ DownlevelFlagsAnisotropicFiltering
}

// ShaderModel represents the collections of shader features a device supports
// if it supports less than WebGPU normally allows.
// Defined in terms of D3D's shader models.
type ShaderModel uint32

const (
	// ShaderModelSm2 represents extremely limited shaders, including a total instruction limit.
	ShaderModelSm2 ShaderModel = 2
	// ShaderModelSm4 represents shaders missing minor features and storage images.
	ShaderModelSm4 ShaderModel = 4
	// ShaderModelSm5 represents WebGPU-level shader support (shader model 5).
	ShaderModelSm5 ShaderModel = 5
)

// String returns the shader model name.
func (sm ShaderModel) String() string {
	switch sm {
	case ShaderModelSm2:
		return "Sm2"
	case ShaderModelSm4:
		return "Sm4"
	case ShaderModelSm5:
		return "Sm5"
	default:
		return "Unknown"
	}
}

// DownlevelLimits represents additional limits on a downlevel adapter.
// Currently empty, reserved for future use.
// Matches Rust wgpu-types DownlevelLimits (limits.rs:1044).
type DownlevelLimits struct{}

// DownlevelCapabilities lists various ways the underlying platform does not
// conform to the WebGPU standard.
// Matches Rust wgpu-types DownlevelCapabilities (limits.rs:1056-1063).
type DownlevelCapabilities struct {
	// Flags is the combined boolean flags.
	Flags DownlevelFlags
	// Limits is the additional limits for downlevel adapters.
	Limits DownlevelLimits
	// ShaderModel indicates which collections of shader features are supported.
	ShaderModel ShaderModel
}

// DefaultDownlevelCapabilities returns the default DownlevelCapabilities,
// representing a fully WebGPU-compliant adapter.
// Matches Rust Default for DownlevelCapabilities (limits.rs:1065-1072).
func DefaultDownlevelCapabilities() DownlevelCapabilities {
	return DownlevelCapabilities{
		Flags:       DownlevelFlagsAll(),
		Limits:      DownlevelLimits{},
		ShaderModel: ShaderModelSm5,
	}
}

// IsWebGPUCompliant returns true if the underlying platform offers complete
// support of the baseline WebGPU standard.
//
// If this returns false, some parts of the API will result in validation errors
// where they would not normally. These parts can be determined by the values
// in this structure.
// Matches Rust DownlevelCapabilities::is_webgpu_compliant() (limits.rs:1075-1085).
func (dc DownlevelCapabilities) IsWebGPUCompliant() bool {
	return dc.Flags.Contains(DownlevelFlagsCompliant()) &&
		dc.Limits == DownlevelLimits{} &&
		dc.ShaderModel >= ShaderModelSm5
}
