// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

package gl

// How much of the context's state a Shadow follows: texture and sampler
// units, indexed buffer bindings per target, and vertex attribute locations.
// A binding past these limits is sent every time.
const (
	MaxShadowUnits   = 32
	MaxShadowIndexed = 32
	MaxShadowAttribs = 16
)

// Shadow is what the replay of one submission knows the context's state to
// be, so a command whose state is in place already sends no GL call: fyne
// switches pipelines before nearly every draw, and each switch sent the
// program, the cull, depth, stencil, blend and mask state and every vertex
// attribute again, ~20 calls where one UseProgram would do (2026-10-20,
// 2179 commands a frame in the widgets bench).
//
// A submission starts from the zero Shadow, nothing known, and ends with it:
// what runs between submissions (uploads, texture creation, the present
// blit) is not followed. Within a submission every command that changes
// state the Shadow follows either records the change or forgets the entry.
type Shadow struct {
	Program    uint32
	ProgramSet bool

	// Pipeline is the pipeline state command applied in full last, nil when
	// a clear or a stencil reference changed part of that state since.
	Pipeline any

	VAO    uint32
	VAOSet bool

	ArrayBuffer    uint32
	ArrayBufferSet bool

	ActiveUnit    uint32
	ActiveUnitSet bool
	Textures      [MaxShadowUnits]ShadowObject // the texture bound last per unit
	Samplers      [MaxShadowUnits]ShadowObject

	Uniforms [MaxShadowIndexed]ShadowRange // GL_UNIFORM_BUFFER indexed bindings
	Storages [MaxShadowIndexed]ShadowRange // GL_SHADER_STORAGE_BUFFER indexed bindings

	// Attribs is the bound VAO's attribute state; a VAO switch forgets it.
	Attribs [MaxShadowAttribs]ShadowAttrib

	// Scissor is set when the scissor test is enabled with this rectangle.
	Scissor ShadowScissor

	// Restart is the primitive restart in place, RestartSet once known:
	// 0 for none, the restart index otherwise (1 for the fixed one).
	Restart    uint32
	RestartSet bool
}

// ShadowObject is an object name bound somewhere, Set once known, and for a
// texture the target it is bound to.
type ShadowObject struct {
	ID     uint32
	Target uint32
	Set    bool
}

// ShadowRange is an indexed buffer binding, Size 0 for the whole buffer.
type ShadowRange struct {
	Buffer       uint32
	Offset, Size int
	Set          bool
}

// ShadowAttrib is one enabled vertex attribute's pointer and divisor.
type ShadowAttrib struct {
	Buffer     uint32
	Size       int32
	Type       uint32
	Normalized bool
	Stride     int32
	Offset     uintptr
	Divisor    uint32
	Set        bool
}

// ShadowScissor is the scissor rectangle while the test is enabled.
type ShadowScissor struct {
	X, Y, W, H int32
	Set        bool
}

// ForgetTextures drops what is known about the active unit and the texture
// bound to every unit, for code that binds textures outside the bind groups.
func (s *Shadow) ForgetTextures() {
	s.ActiveUnitSet = false
	s.Textures = [MaxShadowUnits]ShadowObject{}
}
