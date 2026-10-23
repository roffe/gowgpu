//go:build !(js && wasm)

package core

import (
	"testing"

	"github.com/roffe/gowgpu/core/track"
	"github.com/roffe/gowgpu/gputypes"
)

// A submitted command buffer's trackers and pass scope go back to the
// device, emptied, and the next encoder records into them.
func TestCommandBufferScopesReused(t *testing.T) {
	device := NewDevice(&mockHALDevice{}, &Adapter{}, gputypes.Features(0), gputypes.DefaultLimits(), "TestDevice")

	enc, err := device.CreateCommandEncoder("first")
	if err != nil {
		t.Fatal(err)
	}
	scopes := enc.Mutable()
	scopes.TextureScope().SetSingle(5, track.TextureUsesColorTarget)
	scopes.BufferScope().SetSingle(3, track.BufferUsesVertex)
	if err := scopes.pass.addTexture(7, track.TextureSelector{}, track.TextureSelector{}, nil, track.TextureUsesResource); err != nil {
		t.Fatal(err)
	}
	cb, err := enc.Finish()
	if err != nil {
		t.Fatal(err)
	}
	if enc.Mutable() != nil {
		t.Fatal("a finished encoder still holds the scopes its command buffer owns")
	}
	if !cb.TextureScope().IsUsed(5) || !cb.BufferScope().IsUsed(3) {
		t.Fatal("the command buffer lost the usage its encoder recorded")
	}

	cb.ReleaseScopes()
	if cb.TextureScope() != nil || cb.BufferScope() != nil {
		t.Fatal("ReleaseScopes left the scopes on the command buffer")
	}
	cb.ReleaseScopes() // a second call is a no-op

	next, err := device.CreateCommandEncoder("second")
	if err != nil {
		t.Fatal(err)
	}
	if next.Mutable() != scopes {
		t.Fatal("the next encoder did not get the released scopes")
	}
	if !scopes.TextureScope().IsEmpty() || scopes.TextureScope().IsUsed(5) || scopes.BufferScope().IsUsed(3) {
		t.Fatal("released trackers came back with usage in them")
	}
	if !scopes.pass.textures.IsEmpty() || len(scopes.pass.texHAL) != 0 {
		t.Fatal("released pass scope came back with usage in it")
	}
}
