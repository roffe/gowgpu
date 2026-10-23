//go:build js && wasm

package browser

import "syscall/js"

// RenderPipeline wraps a browser GPURenderPipeline.
type RenderPipeline struct {
	// ref_ is the GPURenderPipeline JavaScript object.
	ref_ js.Value
}

// NewRenderPipeline constructs a RenderPipeline from a GPURenderPipeline js.Value.
func NewRenderPipeline(ref js.Value) *RenderPipeline {
	return &RenderPipeline{ref_: ref}
}

// Ref returns the underlying GPURenderPipeline js.Value.
func (p *RenderPipeline) Ref() js.Value { return p.ref_ }

// GetBindGroupLayout returns the bind group layout at the given index, or
// the exception GPURenderPipeline.getBindGroupLayout(index) throws as an error.
func (p *RenderPipeline) GetBindGroupLayout(index uint32) (*BindGroupLayout, error) {
	var jsLayout js.Value
	if err := catchJS(func() { jsLayout = p.ref_.Call("getBindGroupLayout", index) }); err != nil {
		return nil, err
	}
	return NewBindGroupLayout(jsLayout), nil
}

// ComputePipeline wraps a browser GPUComputePipeline.
type ComputePipeline struct {
	// ref_ is the GPUComputePipeline JavaScript object.
	ref_ js.Value
}

// NewComputePipeline constructs a ComputePipeline from a GPUComputePipeline js.Value.
func NewComputePipeline(ref js.Value) *ComputePipeline {
	return &ComputePipeline{ref_: ref}
}

// Ref returns the underlying GPUComputePipeline js.Value.
func (p *ComputePipeline) Ref() js.Value { return p.ref_ }

// GetBindGroupLayout returns the bind group layout at the given index, or
// the exception GPUComputePipeline.getBindGroupLayout(index) throws as an error.
func (p *ComputePipeline) GetBindGroupLayout(index uint32) (*BindGroupLayout, error) {
	var jsLayout js.Value
	if err := catchJS(func() { jsLayout = p.ref_.Call("getBindGroupLayout", index) }); err != nil {
		return nil, err
	}
	return NewBindGroupLayout(jsLayout), nil
}
