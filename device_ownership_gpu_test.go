//go:build !(js && wasm)

package wgpu

import (
	"strings"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
)

// An object made on one device can't be used with another: a view of
// another device's texture, a bind group with its buffer, a pass with its
// pipeline, a write to its buffer and a submit of its command buffer.
func TestObjectsStayWithTheirDevice(t *testing.T) {
	a, h := newValidatedDevice(t)
	b, _ := newValidatedDevice(t)

	tex, err := a.CreateTexture(&TextureDescriptor{
		Size: Extent3D{Width: 4, Height: 4, DepthOrArrayLayers: 1}, MipLevelCount: 1, SampleCount: 1,
		Dimension: TextureDimension2D, Format: TextureFormatRGBA8Unorm, Usage: TextureUsageTextureBinding,
	})
	if err != nil {
		t.Fatalf("CreateTexture: %v", err)
	}
	defer tex.Release()
	buf, err := a.CreateBuffer(&BufferDescriptor{Size: 256, Usage: BufferUsageUniform | BufferUsageCopyDst})
	if err != nil {
		t.Fatalf("CreateBuffer: %v", err)
	}
	defer buf.Release()
	bgl, err := b.CreateBindGroupLayout(&BindGroupLayoutDescriptor{Entries: []gputypes.BindGroupLayoutEntry{
		{Binding: 0, Visibility: gputypes.ShaderStageFragment, Buffer: &gputypes.BufferBindingLayout{Type: gputypes.BufferBindingTypeUniform}},
	}})
	if err != nil {
		t.Fatalf("CreateBindGroupLayout: %v", err)
	}
	defer bgl.Release()
	foreign := func(what string, err error) {
		t.Helper()
		if err == nil || !strings.Contains(err.Error(), "another device") {
			t.Errorf("%s: %v, want an error about another device", what, err)
		}
	}

	_, err = b.CreateTextureView(tex, nil)
	foreign("CreateTextureView", err)
	_, err = b.CreateBindGroup(&BindGroupDescriptor{Layout: bgl, Entries: []BindGroupEntry{{Binding: 0, Buffer: buf, Size: 256}}})
	foreign("CreateBindGroup", err)
	foreign("WriteBuffer", b.Queue().WriteBuffer(buf, 0, make([]byte, 4)))

	enc, err := a.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	cb, err := enc.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	_, err = b.Queue().Submit(cb)
	foreign("Submit", err)

	enc, err = b.CreateCommandEncoder(nil)
	if err != nil {
		t.Fatalf("CreateCommandEncoder: %v", err)
	}
	enc.ClearBuffer(buf, 0, 0)
	_, err = enc.Finish()
	foreign("ClearBuffer", err)
	assertNoValidation(t, h)
}
