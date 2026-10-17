// Package gputypes provides the WebGPU type definitions gowgpu's API is
// written in: enums, structs and constants, with no dependencies.
//
// It is github.com/gogpu/gputypes v0.8.0 (MIT, see LICENSE), carried in
// gowgpu so the module depends on nothing of the gogpu ecosystem but naga.
//
// # Type Categories
//
// The package provides types in several categories:
//
// Texture types: TextureFormat, TextureUsage, TextureDimension, TextureDescriptor, etc.
//
// Buffer types: BufferUsage, BufferDescriptor, IndexFormat, etc.
//
// Sampler types: AddressMode, FilterMode, CompareFunction, SamplerDescriptor, etc.
//
// Render types: BlendState, BlendFactor, BlendOperation, PrimitiveTopology, etc.
//
// Shader types: ShaderStage, ShaderModuleDescriptor, etc.
//
// Vertex types: VertexFormat, VertexStepMode, VertexAttribute, etc.
//
// Binding types: BindGroupLayoutEntry, BufferBindingLayout, TextureBindingLayout, etc.
//
// Adapter types: DeviceType, AdapterInfo, PowerPreference, Backend, etc.
//
// Limits and Features: Limits struct, Features flags, etc.
//
// Geometry types: Extent3D, Origin3D, Color, etc.
//
// Ray Tracing types (experimental): AccelerationStructureFlags, BlasTriangleGeometrySizeDescriptor,
// CreateBlasDescriptor, CreateTlasDescriptor, AccelerationStructureBindingLayout, etc.
//
// # Usage
//
// Import the package and use types directly:
//
//	import "github.com/roffe/gowgpu/gputypes"
//
//	format := gputypes.TextureFormatRGBA8Unorm
//	usage := gputypes.TextureUsageCopySrc | gputypes.TextureUsageRenderAttachment
//
//	desc := gputypes.TextureDescriptor{
//	    Size:   gputypes.Extent3D{Width: 800, Height: 600, DepthOrArrayLayers: 1},
//	    Format: format,
//	    Usage:  usage,
//	}
//
// # WebGPU Specification
//
// Types follow the W3C WebGPU specification:
// https://www.w3.org/TR/webgpu/
//
// Naming conventions follow wgpu-types (Rust) where applicable:
// https://docs.rs/wgpu-types
package gputypes
