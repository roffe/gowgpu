// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build linux && !(js && wasm)

// Error handling follows ADR-049 three-tier strategy:
//
//	Tier 1 (creation/submit): check the API result code
//	Tier 2 (void/hot path): infallible by GPU API contract — Vulkan §6.6, WebGPU §21.2
//	Tier 3 (platform syscalls): use errno for diagnostics (Wayland, X11, Win32)
//
// Enterprise reference: Rust wgpu-hal returns () for all draw/destroy/barrier commands.
//
// All GL commands in this file are Tier 2: GL errors are reported via glGetError(),
// not through the call itself.
package gl

/*
#include "trampolines_linux.h"
#cgo noescape gl_P_i
#cgo nocallback gl_P_i
#cgo noescape gl_P_ii
#cgo nocallback gl_P_ii
#cgo noescape gl_b_i
#cgo nocallback gl_b_i
#cgo noescape gl_i_
#cgo nocallback gl_i_
#cgo noescape gl_i_i
#cgo nocallback gl_i_i
#cgo noescape gl_i_ip
#cgo nocallback gl_i_ip
#cgo noescape gl_i_uiq
#cgo nocallback gl_i_uiq
#cgo noescape gl_u_ii
#cgo nocallback gl_u_ii
#cgo noescape gl_v_
#cgo nocallback gl_v_
#cgo noescape gl_v_d
#cgo nocallback gl_v_d
#cgo noescape gl_v_dd
#cgo nocallback gl_v_dd
#cgo noescape gl_v_f
#cgo nocallback gl_v_f
#cgo noescape gl_v_ff
#cgo nocallback gl_v_ff
#cgo noescape gl_v_ffff
#cgo nocallback gl_v_ffff
#cgo noescape gl_v_i
#cgo nocallback gl_v_i
#cgo noescape gl_v_ii
#cgo nocallback gl_v_ii
#cgo noescape gl_v_iif
#cgo nocallback gl_v_iif
#cgo noescape gl_v_iii
#cgo nocallback gl_v_iii
#cgo noescape gl_v_iiii
#cgo nocallback gl_v_iiii
#cgo noescape gl_v_iiiii
#cgo nocallback gl_v_iiiii
#cgo noescape gl_v_iiiiii
#cgo nocallback gl_v_iiiiii
#cgo noescape gl_v_iiiiiiii
#cgo nocallback gl_v_iiiiiiii
#cgo noescape gl_v_iiiiiiiiii
#cgo nocallback gl_v_iiiiiiiiii
#cgo noescape gl_v_iiiiiiiip
#cgo nocallback gl_v_iiiiiiiip
#cgo noescape gl_v_iiiiiiiiu
#cgo nocallback gl_v_iiiiiiiiu
#cgo noescape gl_v_iiiiiip
#cgo nocallback gl_v_iiiiiip
#cgo noescape gl_v_iiiiiu
#cgo nocallback gl_v_iiiiiu
#cgo noescape gl_v_iiiu
#cgo nocallback gl_v_iiiu
#cgo noescape gl_v_iiiui
#cgo nocallback gl_v_iiiui
#cgo noescape gl_v_iiiuii
#cgo nocallback gl_v_iiiuii
#cgo noescape gl_v_iiiuu
#cgo nocallback gl_v_iiiuu
#cgo noescape gl_v_iip
#cgo nocallback gl_v_iip
#cgo noescape gl_v_iipp
#cgo nocallback gl_v_iipp
#cgo noescape gl_v_iiuuu
#cgo nocallback gl_v_iiuuu
#cgo noescape gl_v_ip
#cgo nocallback gl_v_ip
#cgo noescape gl_v_iuui
#cgo nocallback gl_v_iuui
#cgo noescape gl_v_iuup
#cgo nocallback gl_v_iuup
#cgo noescape gl_v_u
#cgo nocallback gl_v_u
#cgo noescape gl_v_uiipp
#cgo nocallback gl_v_uiipp
*/
import "C"

import "unsafe"

// Context holds OpenGL function pointers loaded at runtime, called through the
// C trampolines in trampolines_linux.h.
// Functions are loaded via eglGetProcAddress for all OpenGL functions.
type Context struct {
	isGLES bool

	// Core GL 1.1
	glGetError     unsafe.Pointer
	glGetString    unsafe.Pointer
	glGetIntegerv  unsafe.Pointer
	glEnable       unsafe.Pointer
	glDisable      unsafe.Pointer
	glClear        unsafe.Pointer
	glClearColor   unsafe.Pointer
	glClearDepth   unsafe.Pointer
	glViewport     unsafe.Pointer
	glScissor      unsafe.Pointer
	glDrawArrays   unsafe.Pointer
	glDrawElements unsafe.Pointer
	glFlush        unsafe.Pointer
	glFinish       unsafe.Pointer

	// Shaders (GL 2.0+)
	glCreateShader       unsafe.Pointer
	glDeleteShader       unsafe.Pointer
	glShaderSource       unsafe.Pointer
	glCompileShader      unsafe.Pointer
	glGetShaderiv        unsafe.Pointer
	glGetShaderInfoLog   unsafe.Pointer
	glCreateProgram      unsafe.Pointer
	glDeleteProgram      unsafe.Pointer
	glAttachShader       unsafe.Pointer
	glDetachShader       unsafe.Pointer
	glLinkProgram        unsafe.Pointer
	glUseProgram         unsafe.Pointer
	glGetProgramiv       unsafe.Pointer
	glGetProgramInfoLog  unsafe.Pointer
	glGetUniformLocation unsafe.Pointer
	glGetAttribLocation  unsafe.Pointer

	// Uniforms (GL 2.0+)
	glUniform1i        unsafe.Pointer
	glUniform1f        unsafe.Pointer
	glUniform2f        unsafe.Pointer
	glUniform3f        unsafe.Pointer
	glUniform4f        unsafe.Pointer
	glUniform1iv       unsafe.Pointer
	glUniform1fv       unsafe.Pointer
	glUniform2fv       unsafe.Pointer
	glUniform3fv       unsafe.Pointer
	glUniform4fv       unsafe.Pointer
	glUniformMatrix4fv unsafe.Pointer

	// Buffers (GL 1.5+)
	glGenBuffers    unsafe.Pointer
	glDeleteBuffers unsafe.Pointer
	glBindBuffer    unsafe.Pointer
	glBufferData    unsafe.Pointer
	glBufferSubData unsafe.Pointer
	glMapBuffer     unsafe.Pointer
	glUnmapBuffer   unsafe.Pointer

	// VAO (GL 3.0+)
	glGenVertexArrays    unsafe.Pointer
	glDeleteVertexArrays unsafe.Pointer
	glBindVertexArray    unsafe.Pointer

	// Vertex attributes (GL 2.0+)
	glEnableVertexAttribArray  unsafe.Pointer
	glDisableVertexAttribArray unsafe.Pointer
	glVertexAttribPointer      unsafe.Pointer

	// Textures (GL 1.1+)
	glGenTextures    unsafe.Pointer
	glDeleteTextures unsafe.Pointer
	glBindTexture    unsafe.Pointer
	glActiveTexture  unsafe.Pointer
	glTexImage2D     unsafe.Pointer
	glTexSubImage2D  unsafe.Pointer
	glTexParameteri  unsafe.Pointer
	glGenerateMipmap unsafe.Pointer

	// Framebuffers (GL 3.0+)
	glGenFramebuffers        unsafe.Pointer
	glDeleteFramebuffers     unsafe.Pointer
	glBindFramebuffer        unsafe.Pointer
	glFramebufferTexture2D   unsafe.Pointer
	glCheckFramebufferStatus unsafe.Pointer
	glDrawBuffers            unsafe.Pointer
	glClearBufferfv          unsafe.Pointer // GL 3.0+ — per-buffer float clear

	// Pixel read/store (GL 1.0+)
	glReadPixels  unsafe.Pointer
	glPixelStorei unsafe.Pointer

	// Renderbuffers (GL 3.0+)
	glGenRenderbuffers        unsafe.Pointer
	glDeleteRenderbuffers     unsafe.Pointer
	glBindRenderbuffer        unsafe.Pointer
	glRenderbufferStorage     unsafe.Pointer
	glFramebufferRenderbuffer unsafe.Pointer

	// Blending (GL 1.4+)
	glBlendFunc             unsafe.Pointer
	glBlendFuncSeparate     unsafe.Pointer
	glBlendEquation         unsafe.Pointer
	glBlendEquationSeparate unsafe.Pointer
	glBlendColor            unsafe.Pointer

	// Depth/Stencil
	glDepthFunc           unsafe.Pointer
	glDepthMask           unsafe.Pointer
	glDepthRange          unsafe.Pointer
	glStencilFunc         unsafe.Pointer
	glStencilOp           unsafe.Pointer
	glStencilMask         unsafe.Pointer
	glStencilFuncSeparate unsafe.Pointer
	glStencilOpSeparate   unsafe.Pointer
	glStencilMaskSeparate unsafe.Pointer
	glColorMask           unsafe.Pointer

	// Face culling
	glCullFace  unsafe.Pointer
	glFrontFace unsafe.Pointer

	// Sync (GL 3.2+)
	glFenceSync      unsafe.Pointer
	glDeleteSync     unsafe.Pointer
	glClientWaitSync unsafe.Pointer
	glWaitSync       unsafe.Pointer

	// UBO (GL 3.1+)
	glBindBufferBase       unsafe.Pointer
	glBindBufferRange      unsafe.Pointer
	glGetUniformBlockIndex unsafe.Pointer
	glUniformBlockBinding  unsafe.Pointer

	// Instancing (GL 3.1+)
	glDrawArraysInstanced   unsafe.Pointer
	glDrawElementsInstanced unsafe.Pointer
	glVertexAttribDivisor   unsafe.Pointer

	// Base instance (GL 4.2 / ARB_base_instance, ES EXT_base_instance; optional)
	glDrawArraysInstancedBaseInstance   unsafe.Pointer
	glDrawElementsInstancedBaseInstance unsafe.Pointer

	// Compute shaders (GL 4.3+ / ES 3.1+)
	glDispatchCompute         unsafe.Pointer
	glDispatchComputeIndirect unsafe.Pointer
	glMemoryBarrier           unsafe.Pointer

	// MSAA (GL 3.2+ / ES 3.1+)
	glTexImage2DMultisample unsafe.Pointer
	glBlitFramebuffer       unsafe.Pointer

	// Sampler objects (GL 3.3+ / ES 3.0+)
	glGenSamplers       unsafe.Pointer
	glDeleteSamplers    unsafe.Pointer
	glBindSampler       unsafe.Pointer
	glSamplerParameteri unsafe.Pointer
	glSamplerParameterf unsafe.Pointer

	// Query objects (GL 3.3+ / GL_EXT_disjoint_timer_query)
	glGenQueries          unsafe.Pointer
	glDeleteQueries       unsafe.Pointer
	glQueryCounter        unsafe.Pointer
	glGetQueryObjectui64v unsafe.Pointer
	glCopyBufferSubData   unsafe.Pointer
	glCopyTexSubImage2D   unsafe.Pointer
	glGetSynciv           unsafe.Pointer

	// Indexed string query (GL 3.0+ / ES 3.0+)
	glGetStringi unsafe.Pointer

	// Shadow is what the command replay knows the context's state to be.
	Shadow Shadow
}

// ProcAddressFunc is a function that returns the address of an OpenGL function.
type ProcAddressFunc func(name string) unsafe.Pointer

// Load loads all OpenGL function pointers using the provided loader.
// isGLES indicates the context is OpenGL ES — some functions have different
// names (glClearDepthf vs glClearDepth, glMapBufferRange vs glMapBuffer).
// Mesa d3d12 gallium returns dispatch stubs for desktop GL names on GLES
// contexts that generate GL_INVALID_ENUM at call time.
func (c *Context) Load(getProcAddr ProcAddressFunc, isGLES ...bool) error {
	gles := len(isGLES) > 0 && isGLES[0]
	c.isGLES = gles

	// Core GL 1.1
	c.glGetError = getProcAddr("glGetError")
	c.glGetString = getProcAddr("glGetString")
	c.glGetIntegerv = getProcAddr("glGetIntegerv")
	c.glEnable = getProcAddr("glEnable")
	c.glDisable = getProcAddr("glDisable")
	c.glClear = getProcAddr("glClear")
	c.glClearColor = getProcAddr("glClearColor")
	if gles {
		c.glClearDepth = getProcAddr("glClearDepthf")
	} else {
		c.glClearDepth = getProcAddr("glClearDepth")
	}
	c.glViewport = getProcAddr("glViewport")
	c.glScissor = getProcAddr("glScissor")
	c.glDrawArrays = getProcAddr("glDrawArrays")
	c.glDrawElements = getProcAddr("glDrawElements")
	c.glFlush = getProcAddr("glFlush")
	c.glFinish = getProcAddr("glFinish")

	// Shaders
	c.glCreateShader = getProcAddr("glCreateShader")
	c.glDeleteShader = getProcAddr("glDeleteShader")
	c.glShaderSource = getProcAddr("glShaderSource")
	c.glCompileShader = getProcAddr("glCompileShader")
	c.glGetShaderiv = getProcAddr("glGetShaderiv")
	c.glGetShaderInfoLog = getProcAddr("glGetShaderInfoLog")
	c.glCreateProgram = getProcAddr("glCreateProgram")
	c.glDeleteProgram = getProcAddr("glDeleteProgram")
	c.glAttachShader = getProcAddr("glAttachShader")
	c.glDetachShader = getProcAddr("glDetachShader")
	c.glLinkProgram = getProcAddr("glLinkProgram")
	c.glUseProgram = getProcAddr("glUseProgram")
	c.glGetProgramiv = getProcAddr("glGetProgramiv")
	c.glGetProgramInfoLog = getProcAddr("glGetProgramInfoLog")
	c.glGetUniformLocation = getProcAddr("glGetUniformLocation")
	c.glGetAttribLocation = getProcAddr("glGetAttribLocation")

	// Uniforms
	c.glUniform1i = getProcAddr("glUniform1i")
	c.glUniform1f = getProcAddr("glUniform1f")
	c.glUniform2f = getProcAddr("glUniform2f")
	c.glUniform3f = getProcAddr("glUniform3f")
	c.glUniform4f = getProcAddr("glUniform4f")
	c.glUniform1iv = getProcAddr("glUniform1iv")
	c.glUniform1fv = getProcAddr("glUniform1fv")
	c.glUniform2fv = getProcAddr("glUniform2fv")
	c.glUniform3fv = getProcAddr("glUniform3fv")
	c.glUniform4fv = getProcAddr("glUniform4fv")
	c.glUniformMatrix4fv = getProcAddr("glUniformMatrix4fv")

	// Buffers
	c.glGenBuffers = getProcAddr("glGenBuffers")
	c.glDeleteBuffers = getProcAddr("glDeleteBuffers")
	c.glBindBuffer = getProcAddr("glBindBuffer")
	c.glBufferData = getProcAddr("glBufferData")
	c.glBufferSubData = getProcAddr("glBufferSubData")
	c.glMapBuffer = getProcAddr("glMapBuffer")
	if gles && c.glMapBuffer == nil {
		c.glMapBuffer = getProcAddr("glMapBufferOES")
	}
	c.glUnmapBuffer = getProcAddr("glUnmapBuffer")

	// VAO — GLES may need OES suffix on some Mesa dispatch paths.
	c.glGenVertexArrays = getProcAddr("glGenVertexArrays")
	c.glDeleteVertexArrays = getProcAddr("glDeleteVertexArrays")
	c.glBindVertexArray = getProcAddr("glBindVertexArray")
	if gles {
		if p := getProcAddr("glGenVertexArraysOES"); p != nil {
			c.glGenVertexArrays = p
		}
		if p := getProcAddr("glDeleteVertexArraysOES"); p != nil {
			c.glDeleteVertexArrays = p
		}
		if p := getProcAddr("glBindVertexArrayOES"); p != nil {
			c.glBindVertexArray = p
		}
	}

	// Vertex attributes
	c.glEnableVertexAttribArray = getProcAddr("glEnableVertexAttribArray")
	c.glDisableVertexAttribArray = getProcAddr("glDisableVertexAttribArray")
	c.glVertexAttribPointer = getProcAddr("glVertexAttribPointer")

	// Textures
	c.glGenTextures = getProcAddr("glGenTextures")
	c.glDeleteTextures = getProcAddr("glDeleteTextures")
	c.glBindTexture = getProcAddr("glBindTexture")
	c.glActiveTexture = getProcAddr("glActiveTexture")
	c.glTexImage2D = getProcAddr("glTexImage2D")
	c.glTexSubImage2D = getProcAddr("glTexSubImage2D")
	c.glTexParameteri = getProcAddr("glTexParameteri")
	c.glGenerateMipmap = getProcAddr("glGenerateMipmap")

	// Framebuffers
	c.glGenFramebuffers = getProcAddr("glGenFramebuffers")
	c.glDeleteFramebuffers = getProcAddr("glDeleteFramebuffers")
	c.glBindFramebuffer = getProcAddr("glBindFramebuffer")
	c.glFramebufferTexture2D = getProcAddr("glFramebufferTexture2D")
	c.glCheckFramebufferStatus = getProcAddr("glCheckFramebufferStatus")
	c.glDrawBuffers = getProcAddr("glDrawBuffers")
	c.glClearBufferfv = getProcAddr("glClearBufferfv")

	// Pixel read/store
	c.glReadPixels = getProcAddr("glReadPixels")
	c.glPixelStorei = getProcAddr("glPixelStorei")

	// Renderbuffers
	c.glGenRenderbuffers = getProcAddr("glGenRenderbuffers")
	c.glDeleteRenderbuffers = getProcAddr("glDeleteRenderbuffers")
	c.glBindRenderbuffer = getProcAddr("glBindRenderbuffer")
	c.glRenderbufferStorage = getProcAddr("glRenderbufferStorage")
	c.glFramebufferRenderbuffer = getProcAddr("glFramebufferRenderbuffer")

	// Blending
	c.glBlendFunc = getProcAddr("glBlendFunc")
	c.glBlendFuncSeparate = getProcAddr("glBlendFuncSeparate")
	c.glBlendEquation = getProcAddr("glBlendEquation")
	c.glBlendEquationSeparate = getProcAddr("glBlendEquationSeparate")
	c.glBlendColor = getProcAddr("glBlendColor")

	// Depth/Stencil
	c.glDepthFunc = getProcAddr("glDepthFunc")
	c.glDepthMask = getProcAddr("glDepthMask")
	if gles {
		c.glDepthRange = getProcAddr("glDepthRangef")
	} else {
		c.glDepthRange = getProcAddr("glDepthRange")
	}
	c.glStencilFunc = getProcAddr("glStencilFunc")
	c.glStencilOp = getProcAddr("glStencilOp")
	c.glStencilMask = getProcAddr("glStencilMask")
	c.glStencilFuncSeparate = getProcAddr("glStencilFuncSeparate")
	c.glStencilOpSeparate = getProcAddr("glStencilOpSeparate")
	c.glStencilMaskSeparate = getProcAddr("glStencilMaskSeparate")
	c.glColorMask = getProcAddr("glColorMask")

	// Face culling
	c.glCullFace = getProcAddr("glCullFace")
	c.glFrontFace = getProcAddr("glFrontFace")

	// Sync
	c.glFenceSync = getProcAddr("glFenceSync")
	c.glDeleteSync = getProcAddr("glDeleteSync")
	c.glClientWaitSync = getProcAddr("glClientWaitSync")
	c.glWaitSync = getProcAddr("glWaitSync")

	// UBO
	c.glBindBufferBase = getProcAddr("glBindBufferBase")
	c.glBindBufferRange = getProcAddr("glBindBufferRange")
	c.glGetUniformBlockIndex = getProcAddr("glGetUniformBlockIndex")
	c.glUniformBlockBinding = getProcAddr("glUniformBlockBinding")

	// Instancing
	c.glDrawArraysInstanced = getProcAddr("glDrawArraysInstanced")
	c.glDrawElementsInstanced = getProcAddr("glDrawElementsInstanced")
	c.glVertexAttribDivisor = getProcAddr("glVertexAttribDivisor")
	if gles {
		c.glDrawArraysInstancedBaseInstance = getProcAddr("glDrawArraysInstancedBaseInstanceEXT")
		c.glDrawElementsInstancedBaseInstance = getProcAddr("glDrawElementsInstancedBaseInstanceEXT")
	} else {
		c.glDrawArraysInstancedBaseInstance = getProcAddr("glDrawArraysInstancedBaseInstance")
		c.glDrawElementsInstancedBaseInstance = getProcAddr("glDrawElementsInstancedBaseInstance")
	}

	// Compute shaders (optional - may be nil on older GL versions)
	c.glDispatchCompute = getProcAddr("glDispatchCompute")
	c.glDispatchComputeIndirect = getProcAddr("glDispatchComputeIndirect")
	c.glMemoryBarrier = getProcAddr("glMemoryBarrier")

	// MSAA (optional - may be nil on older GL versions)
	c.glTexImage2DMultisample = getProcAddr("glTexImage2DMultisample")
	c.glBlitFramebuffer = getProcAddr("glBlitFramebuffer")

	// Sampler objects (optional - GL 3.3+ / ES 3.0+)
	c.glGenSamplers = getProcAddr("glGenSamplers")
	c.glDeleteSamplers = getProcAddr("glDeleteSamplers")
	c.glBindSampler = getProcAddr("glBindSampler")
	c.glSamplerParameteri = getProcAddr("glSamplerParameteri")
	c.glSamplerParameterf = getProcAddr("glSamplerParameterf")

	// Query objects (optional - GL 3.3+ / GL_EXT_disjoint_timer_query)
	c.glGenQueries = getProcAddr("glGenQueries")
	c.glDeleteQueries = getProcAddr("glDeleteQueries")
	c.glQueryCounter = getProcAddr("glQueryCounter")
	c.glGetQueryObjectui64v = getProcAddr("glGetQueryObjectui64v")
	c.glCopyBufferSubData = getProcAddr("glCopyBufferSubData")
	c.glCopyTexSubImage2D = getProcAddr("glCopyTexSubImage2D")
	c.glGetSynciv = getProcAddr("glGetSynciv")

	// Indexed string query (GL 3.0+ / ES 3.0+)
	c.glGetStringi = getProcAddr("glGetStringi")

	return nil
}

// --- GL Function Wrappers ---

func (c *Context) GetError() uint32 {
	var result uint32
	result = uint32(C.gl_i_(C.uintptr_t(uintptr(c.glGetError))))
	return result
}

func (c *Context) GetString(name uint32) string {
	return goString(C.gl_P_i(C.uintptr_t(uintptr(c.glGetString)), C.uint32_t(name)))
}

func (c *Context) GetIntegerv(pname uint32, data *int32) {
	C.gl_v_ip(C.uintptr_t(uintptr(c.glGetIntegerv)), C.uint32_t(pname), (*C.uint8_t)(unsafe.Pointer(data)))
}

// GetStringi returns an indexed string from an OpenGL string array (GL 3.0+).
// Used for querying GL_EXTENSIONS one at a time (the modern way).
// Returns "" if glGetStringi is not available or the index is out of range.
func (c *Context) GetStringi(name, index uint32) string {
	if c.glGetStringi == nil {
		return ""
	}
	return goString(C.gl_P_ii(C.uintptr_t(uintptr(c.glGetStringi)), C.uint32_t(name), C.uint32_t(index)))
}

func (c *Context) Enable(capability uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glEnable)), C.uint32_t(capability))
}

func (c *Context) Disable(capability uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glDisable)), C.uint32_t(capability))
}

func (c *Context) Clear(mask uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glClear)), C.uint32_t(mask))
}

func (c *Context) ClearColor(r, g, b, a float32) {
	C.gl_v_ffff(C.uintptr_t(uintptr(c.glClearColor)), C.float(r), C.float(g), C.float(b), C.float(a))
}

// ClearDepth sets the depth value used by glClear(GL_DEPTH_BUFFER_BIT):
// glClearDepthf on GLES, glClearDepth with its double on desktop GL.
func (c *Context) ClearDepth(depth float64) {
	if c.isGLES {
		C.gl_v_f(C.uintptr_t(uintptr(c.glClearDepth)), C.float(float32(depth)))
		return
	}
	C.gl_v_d(C.uintptr_t(uintptr(c.glClearDepth)), C.double(depth))
}

// DepthRange sets the mapping of NDC depth to window depth: glDepthRangef
// on GLES, glDepthRange with its doubles on desktop GL.
func (c *Context) DepthRange(near, far float64) {
	if c.isGLES {
		C.gl_v_ff(C.uintptr_t(uintptr(c.glDepthRange)), C.float(float32(near)), C.float(float32(far)))
		return
	}
	C.gl_v_dd(C.uintptr_t(uintptr(c.glDepthRange)), C.double(near), C.double(far))
}

func (c *Context) Viewport(x, y, width, height int32) {
	// Convert int32 to uint32 for API compatibility
	ux, uy, uw, uh := uint32(x), uint32(y), uint32(width), uint32(height)
	C.gl_v_iiii(C.uintptr_t(uintptr(c.glViewport)), C.uint32_t(ux), C.uint32_t(uy), C.uint32_t(uw), C.uint32_t(uh))
}

func (c *Context) Scissor(x, y, width, height int32) {
	// Convert int32 to uint32 for API compatibility
	ux, uy, uw, uh := uint32(x), uint32(y), uint32(width), uint32(height)
	C.gl_v_iiii(C.uintptr_t(uintptr(c.glScissor)), C.uint32_t(ux), C.uint32_t(uy), C.uint32_t(uw), C.uint32_t(uh))
}

func (c *Context) DrawArrays(mode uint32, first, count int32) {
	ufirst, ucount := uint32(first), uint32(count)
	C.gl_v_iii(C.uintptr_t(uintptr(c.glDrawArrays)), C.uint32_t(mode), C.uint32_t(ufirst), C.uint32_t(ucount))
}

func (c *Context) DrawElements(mode uint32, count int32, typ uint32, indices uintptr) {
	ucount := uint32(count)
	C.gl_v_iiiu(C.uintptr_t(uintptr(c.glDrawElements)), C.uint32_t(mode), C.uint32_t(ucount), C.uint32_t(typ), C.uintptr_t(indices))
}

func (c *Context) Flush() {
	C.gl_v_(C.uintptr_t(uintptr(c.glFlush)))
}

func (c *Context) Finish() {
	C.gl_v_(C.uintptr_t(uintptr(c.glFinish)))
}

// --- Shaders ---

func (c *Context) CreateShader(shaderType uint32) uint32 {
	var result uint32
	result = uint32(C.gl_i_i(C.uintptr_t(uintptr(c.glCreateShader)), C.uint32_t(shaderType)))
	return result
}

func (c *Context) DeleteShader(shader uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glDeleteShader)), C.uint32_t(shader))
}

func (c *Context) ShaderSource(shader uint32, source string) {
	csource, free := cString(source)
	defer free()
	count := int32(1)
	length := int32(len(source))
	stringsPtr := &csource // const GLchar **: one string
	lengthsPtr := &length  // const GLint *: its length
	C.gl_v_iipp(C.uintptr_t(uintptr(c.glShaderSource)), C.uint32_t(shader), C.uint32_t(count), (*C.uint8_t)(unsafe.Pointer(stringsPtr)), (*C.uint8_t)(unsafe.Pointer(lengthsPtr)))
}

func (c *Context) CompileShader(shader uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glCompileShader)), C.uint32_t(shader))
}

func (c *Context) GetShaderiv(shader uint32, pname uint32, params *int32) {
	C.gl_v_iip(C.uintptr_t(uintptr(c.glGetShaderiv)), C.uint32_t(shader), C.uint32_t(pname), (*C.uint8_t)(unsafe.Pointer(params)))
}

func (c *Context) GetShaderInfoLog(shader uint32) string {
	var length int32
	c.GetShaderiv(shader, INFO_LOG_LENGTH, &length)
	if length == 0 {
		return ""
	}
	buf := make([]byte, length)
	maxLen := uint32(length)
	lenPtr := &length // GL writes the actual log length here
	bufPtr := &buf[0] // and the log text here
	C.gl_v_iipp(C.uintptr_t(uintptr(c.glGetShaderInfoLog)), C.uint32_t(shader), C.uint32_t(maxLen), (*C.uint8_t)(unsafe.Pointer(lenPtr)), (*C.uint8_t)(unsafe.Pointer(bufPtr)))
	return string(buf[:length])
}

func (c *Context) CreateProgram() uint32 {
	var result uint32
	result = uint32(C.gl_i_(C.uintptr_t(uintptr(c.glCreateProgram))))
	return result
}

func (c *Context) DeleteProgram(program uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glDeleteProgram)), C.uint32_t(program))
}

func (c *Context) AttachShader(program, shader uint32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glAttachShader)), C.uint32_t(program), C.uint32_t(shader))
}

func (c *Context) LinkProgram(program uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glLinkProgram)), C.uint32_t(program))
}

func (c *Context) UseProgram(program uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glUseProgram)), C.uint32_t(program))
}

func (c *Context) GetProgramiv(program uint32, pname uint32, params *int32) {
	C.gl_v_iip(C.uintptr_t(uintptr(c.glGetProgramiv)), C.uint32_t(program), C.uint32_t(pname), (*C.uint8_t)(unsafe.Pointer(params)))
}

func (c *Context) GetProgramInfoLog(program uint32) string {
	var length int32
	c.GetProgramiv(program, INFO_LOG_LENGTH, &length)
	if length == 0 {
		return ""
	}
	buf := make([]byte, length)
	maxLen := uint32(length)
	lenPtr := &length // GL writes the actual log length here
	bufPtr := &buf[0] // and the log text here
	C.gl_v_iipp(C.uintptr_t(uintptr(c.glGetProgramInfoLog)), C.uint32_t(program), C.uint32_t(maxLen), (*C.uint8_t)(unsafe.Pointer(lenPtr)), (*C.uint8_t)(unsafe.Pointer(bufPtr)))
	return string(buf[:length])
}

func (c *Context) GetUniformLocation(program uint32, name string) int32 {
	cname, free := cString(name)
	defer free()
	var result int32
	result = int32(C.gl_i_ip(C.uintptr_t(uintptr(c.glGetUniformLocation)), C.uint32_t(program), (*C.uint8_t)(unsafe.Pointer(cname))))
	return result
}

func (c *Context) GetAttribLocation(program uint32, name string) int32 {
	cname, free := cString(name)
	defer free()
	var result int32
	result = int32(C.gl_i_ip(C.uintptr_t(uintptr(c.glGetAttribLocation)), C.uint32_t(program), (*C.uint8_t)(unsafe.Pointer(cname))))
	return result
}

// --- Buffers ---

func (c *Context) GenBuffers(n int32) uint32 {
	var buffer uint32
	pBuffer := &buffer // GL writes the new ID here
	C.gl_v_ip(C.uintptr_t(uintptr(c.glGenBuffers)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pBuffer)))
	return buffer
}

func (c *Context) DeleteBuffers(buffers ...uint32) {
	n := int32(len(buffers))
	pBuffers := &buffers[0]
	C.gl_v_ip(C.uintptr_t(uintptr(c.glDeleteBuffers)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pBuffers)))
}

func (c *Context) BindBuffer(target, buffer uint32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glBindBuffer)), C.uint32_t(target), C.uint32_t(buffer))
}

func (c *Context) BufferData(target uint32, size int, data uintptr, usage uint32) {
	sizePtr := uintptr(size)
	C.gl_v_iuui(C.uintptr_t(uintptr(c.glBufferData)), C.uint32_t(target), C.uintptr_t(sizePtr), C.uintptr_t(data), C.uint32_t(usage))
}

func (c *Context) BufferSubData(target uint32, offset, size int, data unsafe.Pointer) {
	offsetPtr := uintptr(offset)
	sizePtr := uintptr(size)
	dataPtr := data
	C.gl_v_iuup(C.uintptr_t(uintptr(c.glBufferSubData)), C.uint32_t(target), C.uintptr_t(offsetPtr), C.uintptr_t(sizePtr), (*C.uint8_t)(dataPtr))
}

// MapBuffer maps a buffer object's data store into the client's address space.
// target: GL_ARRAY_BUFFER, GL_COPY_READ_BUFFER, etc.
// access: GL_READ_ONLY, GL_WRITE_ONLY, GL_READ_WRITE.
// Returns the mapped pointer, or 0 if glMapBuffer is not available or the call fails.
func (c *Context) MapBuffer(target, access uint32) uintptr {
	if c.glMapBuffer == nil {
		return 0
	}
	var ptr uintptr
	ptr = uintptr(C.gl_u_ii(C.uintptr_t(uintptr(c.glMapBuffer)), C.uint32_t(target), C.uint32_t(access)))
	return ptr
}

// UnmapBuffer releases the mapping of a buffer object's data store.
// Returns true if the buffer was successfully unmapped, false if the buffer contents
// became corrupt during the mapping (GL_FALSE from driver) or glUnmapBuffer is unavailable.
func (c *Context) UnmapBuffer(target uint32) bool {
	if c.glUnmapBuffer == nil {
		return false
	}
	return C.gl_b_i(C.uintptr_t(uintptr(c.glUnmapBuffer)), C.uint32_t(target)) != 0 // a GLboolean
}

// --- VAO ---

func (c *Context) GenVertexArrays(n int32) uint32 {
	var vao uint32
	pVao := &vao
	C.gl_v_ip(C.uintptr_t(uintptr(c.glGenVertexArrays)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pVao)))
	return vao
}

func (c *Context) DeleteVertexArrays(arrays ...uint32) {
	n := int32(len(arrays))
	pArrays := &arrays[0]
	C.gl_v_ip(C.uintptr_t(uintptr(c.glDeleteVertexArrays)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pArrays)))
}

func (c *Context) BindVertexArray(array uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glBindVertexArray)), C.uint32_t(array))
}

// --- Vertex Attributes ---

func (c *Context) EnableVertexAttribArray(index uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glEnableVertexAttribArray)), C.uint32_t(index))
}

func (c *Context) DisableVertexAttribArray(index uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glDisableVertexAttribArray)), C.uint32_t(index))
}

func (c *Context) VertexAttribPointer(index uint32, size int32, typ uint32, normalized bool, stride int32, offset uintptr) {
	var norm uint8
	if normalized {
		norm = 1
	}
	C.gl_v_iiiiiu(C.uintptr_t(uintptr(c.glVertexAttribPointer)), C.uint32_t(index), C.uint32_t(size), C.uint32_t(typ), C.uint32_t(norm), C.uint32_t(stride), C.uintptr_t(offset))
}

// VertexAttribDivisor sets the instance divisor for a vertex attribute.
// divisor=0 means per-vertex, divisor=1 means per-instance.
// Matches Rust wgpu-hal/src/gles/queue.rs vertex_attrib_divisor call.
func (c *Context) VertexAttribDivisor(index, divisor uint32) {
	if c.glVertexAttribDivisor == nil {
		return
	}
	C.gl_v_ii(C.uintptr_t(uintptr(c.glVertexAttribDivisor)), C.uint32_t(index), C.uint32_t(divisor))
}

// --- Textures ---

func (c *Context) GenTextures(n int32) uint32 {
	var tex uint32
	pTex := &tex
	C.gl_v_ip(C.uintptr_t(uintptr(c.glGenTextures)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pTex)))
	return tex
}

func (c *Context) DeleteTextures(textures ...uint32) {
	n := int32(len(textures))
	pTextures := &textures[0]
	C.gl_v_ip(C.uintptr_t(uintptr(c.glDeleteTextures)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pTextures)))
}

func (c *Context) BindTexture(target, texture uint32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glBindTexture)), C.uint32_t(target), C.uint32_t(texture))
}

func (c *Context) ActiveTexture(texture uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glActiveTexture)), C.uint32_t(texture))
}

func (c *Context) TexParameteri(target, pname uint32, param int32) {
	C.gl_v_iii(C.uintptr_t(uintptr(c.glTexParameteri)), C.uint32_t(target), C.uint32_t(pname), C.uint32_t(param))
}

func (c *Context) TexImage2D(target uint32, level int32, internalformat int32, width, height int32, border int32, format, typ uint32, pixels uintptr) {
	C.gl_v_iiiiiiiiu(C.uintptr_t(uintptr(c.glTexImage2D)), C.uint32_t(target), C.uint32_t(level), C.uint32_t(internalformat), C.uint32_t(width), C.uint32_t(height), C.uint32_t(border), C.uint32_t(format), C.uint32_t(typ), C.uintptr_t(pixels))
}

func (c *Context) TexSubImage2D(target uint32, level int32, xoffset, yoffset, width, height int32, format, typ uint32, pixels unsafe.Pointer) {
	pixelsPtr := pixels
	C.gl_v_iiiiiiiip(C.uintptr_t(uintptr(c.glTexSubImage2D)), C.uint32_t(target), C.uint32_t(level), C.uint32_t(xoffset), C.uint32_t(yoffset), C.uint32_t(width), C.uint32_t(height), C.uint32_t(format), C.uint32_t(typ), (*C.uint8_t)(pixelsPtr))
}

// TexSubImage2DOffset is glTexSubImage2D from the bound
// GL_PIXEL_UNPACK_BUFFER: offset is a byte offset into that buffer, not a
// pointer, so it crosses as an integer.
func (c *Context) TexSubImage2DOffset(target uint32, level int32, xoffset, yoffset, width, height int32, format, typ uint32, offset uintptr) {
	C.gl_v_iiiiiiiiu(C.uintptr_t(uintptr(c.glTexSubImage2D)), C.uint32_t(target), C.uint32_t(level), C.uint32_t(xoffset), C.uint32_t(yoffset), C.uint32_t(width), C.uint32_t(height), C.uint32_t(format), C.uint32_t(typ), C.uintptr_t(offset))
}

func (c *Context) GenerateMipmap(target uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glGenerateMipmap)), C.uint32_t(target))
}

// TexImage2DMultisample creates a multisample 2D texture image.
// Requires OpenGL 3.2+ or OpenGL ES 3.1+.
// No-op if not supported.
func (c *Context) TexImage2DMultisample(target uint32, samples int32, internalformat uint32, width, height int32, fixedsamplelocations bool) {
	if c.glTexImage2DMultisample == nil {
		return
	}
	var fixed uint8
	if fixedsamplelocations {
		fixed = 1
	}
	C.gl_v_iiiiii(C.uintptr_t(uintptr(c.glTexImage2DMultisample)), C.uint32_t(target), C.uint32_t(samples), C.uint32_t(internalformat), C.uint32_t(width), C.uint32_t(height), C.uint32_t(fixed))
}

// BlitFramebuffer copies a block of pixels between framebuffers.
// Requires OpenGL 3.0+ or OpenGL ES 3.0+.
// No-op if not supported.
func (c *Context) BlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1 int32, mask, filter uint32) {
	if c.glBlitFramebuffer == nil {
		return
	}
	C.gl_v_iiiiiiiiii(C.uintptr_t(uintptr(c.glBlitFramebuffer)), C.uint32_t(srcX0), C.uint32_t(srcY0), C.uint32_t(srcX1), C.uint32_t(srcY1), C.uint32_t(dstX0), C.uint32_t(dstY0), C.uint32_t(dstX1), C.uint32_t(dstY1), C.uint32_t(mask), C.uint32_t(filter))
}

// --- Sampler Objects ---

// GenSamplers generates sampler object names.
// Requires OpenGL 3.3+ or OpenGL ES 3.0+.
// Returns 0 if sampler objects are not supported.
func (c *Context) GenSamplers(n int32) uint32 {
	if c.glGenSamplers == nil {
		return 0
	}
	var sampler uint32
	pSampler := &sampler
	C.gl_v_ip(C.uintptr_t(uintptr(c.glGenSamplers)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pSampler)))
	return sampler
}

// DeleteSamplers deletes sampler objects.
// No-op if sampler objects are not supported.
func (c *Context) DeleteSamplers(samplers ...uint32) {
	if c.glDeleteSamplers == nil {
		return
	}
	n := int32(len(samplers))
	pSamplers := &samplers[0]
	C.gl_v_ip(C.uintptr_t(uintptr(c.glDeleteSamplers)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pSamplers)))
}

// BindSampler binds a sampler object to a texture unit.
// No-op if sampler objects are not supported.
func (c *Context) BindSampler(unit, sampler uint32) {
	if c.glBindSampler == nil {
		return
	}
	C.gl_v_ii(C.uintptr_t(uintptr(c.glBindSampler)), C.uint32_t(unit), C.uint32_t(sampler))
}

// SamplerParameteri sets an integer parameter on a sampler object.
// No-op if sampler objects are not supported.
func (c *Context) SamplerParameteri(sampler, pname uint32, param int32) {
	if c.glSamplerParameteri == nil {
		return
	}
	C.gl_v_iii(C.uintptr_t(uintptr(c.glSamplerParameteri)), C.uint32_t(sampler), C.uint32_t(pname), C.uint32_t(param))
}

// SamplerParameterf sets a float parameter on a sampler object.
// No-op if sampler objects are not supported.
func (c *Context) SamplerParameterf(sampler, pname uint32, param float32) {
	if c.glSamplerParameterf == nil {
		return
	}
	C.gl_v_iif(C.uintptr_t(uintptr(c.glSamplerParameterf)), C.uint32_t(sampler), C.uint32_t(pname), C.float(param))
}

// SupportsSamplerObjects returns true if GL sampler objects are supported.
func (c *Context) SupportsSamplerObjects() bool {
	return c.glGenSamplers != nil
}

// --- Framebuffers ---

func (c *Context) GenFramebuffers(n int32) uint32 {
	var fbo uint32
	pFbo := &fbo
	C.gl_v_ip(C.uintptr_t(uintptr(c.glGenFramebuffers)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pFbo)))
	return fbo
}

func (c *Context) DeleteFramebuffers(framebuffers ...uint32) {
	n := int32(len(framebuffers))
	pFramebuffers := &framebuffers[0]
	C.gl_v_ip(C.uintptr_t(uintptr(c.glDeleteFramebuffers)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pFramebuffers)))
}

func (c *Context) BindFramebuffer(target, framebuffer uint32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glBindFramebuffer)), C.uint32_t(target), C.uint32_t(framebuffer))
}

func (c *Context) FramebufferTexture2D(target, attachment, textarget, texture uint32, level int32) {
	C.gl_v_iiiii(C.uintptr_t(uintptr(c.glFramebufferTexture2D)), C.uint32_t(target), C.uint32_t(attachment), C.uint32_t(textarget), C.uint32_t(texture), C.uint32_t(level))
}

func (c *Context) CheckFramebufferStatus(target uint32) uint32 {
	var result uint32
	result = uint32(C.gl_i_i(C.uintptr_t(uintptr(c.glCheckFramebufferStatus)), C.uint32_t(target)))
	return result
}

// DrawBuffers specifies a list of color buffers to be drawn into.
// Matches Rust wgpu-hal GLES SetDrawColorBuffers (queue.rs:1202-1207).
func (c *Context) DrawBuffers(bufs []uint32) {
	if len(bufs) == 0 {
		return
	}
	n := int32(len(bufs))
	pBufs := &bufs[0]
	C.gl_v_ip(C.uintptr_t(uintptr(c.glDrawBuffers)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pBufs)))
}

// ClearBufferfv clears a specific draw buffer with float values.
// buffer must be GL_COLOR, drawBuffer is the index (0..MAX_DRAW_BUFFERS-1),
// value points to 4 float32 values (RGBA).
// Matches Rust wgpu-hal GLES ClearColorF (queue.rs:1222).
func (c *Context) ClearBufferfv(buffer uint32, drawBuffer int32, value *[4]float32) {
	db := uint32(drawBuffer)
	pValue := unsafe.Pointer(value)
	C.gl_v_iip(C.uintptr_t(uintptr(c.glClearBufferfv)), C.uint32_t(buffer), C.uint32_t(db), (*C.uint8_t)(pValue))
}

// --- Renderbuffers ---

// GenRenderbuffers generates a single renderbuffer object and returns its name.
func (c *Context) GenRenderbuffers(n int32) uint32 {
	var rbo uint32
	pRbo := &rbo
	C.gl_v_ip(C.uintptr_t(uintptr(c.glGenRenderbuffers)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pRbo)))
	return rbo
}

// DeleteRenderbuffers deletes the named renderbuffer objects.
func (c *Context) DeleteRenderbuffers(renderbuffers ...uint32) {
	if len(renderbuffers) == 0 {
		return
	}
	n := int32(len(renderbuffers))
	pRenderbuffers := &renderbuffers[0]
	C.gl_v_ip(C.uintptr_t(uintptr(c.glDeleteRenderbuffers)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pRenderbuffers)))
}

// BindRenderbuffer binds a renderbuffer object to the given target
// (typically GL_RENDERBUFFER).
func (c *Context) BindRenderbuffer(target, renderbuffer uint32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glBindRenderbuffer)), C.uint32_t(target), C.uint32_t(renderbuffer))
}

// RenderbufferStorage allocates storage for the currently bound renderbuffer.
// internalFormat is a sized format like GL_RGBA8 or GL_DEPTH24_STENCIL8.
func (c *Context) RenderbufferStorage(target, internalFormat uint32, width, height int32) {
	uw, uh := uint32(width), uint32(height)
	C.gl_v_iiii(C.uintptr_t(uintptr(c.glRenderbufferStorage)), C.uint32_t(target), C.uint32_t(internalFormat), C.uint32_t(uw), C.uint32_t(uh))
}

// FramebufferRenderbuffer attaches a renderbuffer to the currently bound
// framebuffer at the specified attachment point.
func (c *Context) FramebufferRenderbuffer(target, attachment, renderbufferTarget, renderbuffer uint32) {
	C.gl_v_iiii(C.uintptr_t(uintptr(c.glFramebufferRenderbuffer)), C.uint32_t(target), C.uint32_t(attachment), C.uint32_t(renderbufferTarget), C.uint32_t(renderbuffer))
}

// --- Pixel Read/Store ---

// ReadPixels reads a block of pixels from the framebuffer.
func (c *Context) ReadPixels(x, y, width, height int32, format, dataType uint32, pixels unsafe.Pointer) {
	pixelsPtr := pixels
	C.gl_v_iiiiiip(C.uintptr_t(uintptr(c.glReadPixels)), C.uint32_t(x), C.uint32_t(y), C.uint32_t(width), C.uint32_t(height), C.uint32_t(format), C.uint32_t(dataType), (*C.uint8_t)(pixelsPtr))
}

// PixelStorei sets pixel storage modes that affect ReadPixels and TexImage operations.
func (c *Context) PixelStorei(pname uint32, param int32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glPixelStorei)), C.uint32_t(pname), C.uint32_t(param))
}

// --- Blending ---

func (c *Context) BlendFunc(sfactor, dfactor uint32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glBlendFunc)), C.uint32_t(sfactor), C.uint32_t(dfactor))
}

func (c *Context) BlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha uint32) {
	C.gl_v_iiii(C.uintptr_t(uintptr(c.glBlendFuncSeparate)), C.uint32_t(srcRGB), C.uint32_t(dstRGB), C.uint32_t(srcAlpha), C.uint32_t(dstAlpha))
}

func (c *Context) BlendEquation(mode uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glBlendEquation)), C.uint32_t(mode))
}

// BlendEquationSeparate sets separate blend equations for RGB and alpha.
func (c *Context) BlendEquationSeparate(modeRGB, modeAlpha uint32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glBlendEquationSeparate)), C.uint32_t(modeRGB), C.uint32_t(modeAlpha))
}

// BlendColor sets the constant blend color.
func (c *Context) BlendColor(r, g, b, a float32) {
	C.gl_v_ffff(C.uintptr_t(uintptr(c.glBlendColor)), C.float(r), C.float(g), C.float(b), C.float(a))
}

// --- UBO ---

// BindBufferBase binds a buffer to an indexed binding point.
func (c *Context) BindBufferBase(target, index, buffer uint32) {
	C.gl_v_iii(C.uintptr_t(uintptr(c.glBindBufferBase)), C.uint32_t(target), C.uint32_t(index), C.uint32_t(buffer))
}

// BindBufferRange binds a range of a buffer to an indexed binding point.
func (c *Context) BindBufferRange(target, index, buffer uint32, offset, size int) {
	// GLintptr offset and GLsizeiptr size are pointer-sized.
	C.gl_v_iiiuu(C.uintptr_t(uintptr(c.glBindBufferRange)), C.uint32_t(target), C.uint32_t(index), C.uint32_t(buffer), C.uintptr_t(offset), C.uintptr_t(size))
}

// GetUniformBlockIndex returns the index of a named uniform block.
func (c *Context) GetUniformBlockIndex(program uint32, name string) uint32 {
	cname, free := cString(name)
	defer free()
	var result int32
	result = int32(C.gl_i_ip(C.uintptr_t(uintptr(c.glGetUniformBlockIndex)), C.uint32_t(program), (*C.uint8_t)(unsafe.Pointer(cname))))
	return uint32(result)
}

// UniformBlockBinding assigns a uniform block to a binding point.
func (c *Context) UniformBlockBinding(program, blockIndex, blockBinding uint32) {
	C.gl_v_iii(C.uintptr_t(uintptr(c.glUniformBlockBinding)), C.uint32_t(program), C.uint32_t(blockIndex), C.uint32_t(blockBinding))
}

// --- Uniforms ---

// Uniform1i sets an integer uniform value.
func (c *Context) Uniform1i(location, value int32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glUniform1i)), C.uint32_t(location), C.uint32_t(value))
}

// --- Depth/Stencil ---

func (c *Context) DepthFunc(fn uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glDepthFunc)), C.uint32_t(fn))
}

func (c *Context) DepthMask(flag bool) {
	var f uint32
	if flag {
		f = 1
	}
	C.gl_v_i(C.uintptr_t(uintptr(c.glDepthMask)), C.uint32_t(f))
}

// StencilFuncSeparate sets stencil test function per face.
func (c *Context) StencilFuncSeparate(face, fn uint32, ref int32, mask uint32) {
	C.gl_v_iiii(C.uintptr_t(uintptr(c.glStencilFuncSeparate)), C.uint32_t(face), C.uint32_t(fn), C.uint32_t(ref), C.uint32_t(mask))
}

// StencilOpSeparate sets stencil operations per face.
func (c *Context) StencilOpSeparate(face, sfail, dpfail, dppass uint32) {
	C.gl_v_iiii(C.uintptr_t(uintptr(c.glStencilOpSeparate)), C.uint32_t(face), C.uint32_t(sfail), C.uint32_t(dpfail), C.uint32_t(dppass))
}

// StencilMaskSeparate sets stencil write mask per face.
func (c *Context) StencilMaskSeparate(face, mask uint32) {
	C.gl_v_ii(C.uintptr_t(uintptr(c.glStencilMaskSeparate)), C.uint32_t(face), C.uint32_t(mask))
}

// ColorMask enables or disables writing of color components.
func (c *Context) ColorMask(r, g, b, a bool) {
	var rv, gv, bv, av uint32
	if r {
		rv = 1
	}
	if g {
		gv = 1
	}
	if b {
		bv = 1
	}
	if a {
		av = 1
	}
	C.gl_v_iiii(C.uintptr_t(uintptr(c.glColorMask)), C.uint32_t(rv), C.uint32_t(gv), C.uint32_t(bv), C.uint32_t(av))
}

// --- Face Culling ---

func (c *Context) CullFace(mode uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glCullFace)), C.uint32_t(mode))
}

func (c *Context) FrontFace(mode uint32) {
	C.gl_v_i(C.uintptr_t(uintptr(c.glFrontFace)), C.uint32_t(mode))
}

// --- Instancing ---

func (c *Context) DrawArraysInstanced(mode uint32, first, count, instanceCount int32) {
	C.gl_v_iiii(C.uintptr_t(uintptr(c.glDrawArraysInstanced)), C.uint32_t(mode), C.uint32_t(first), C.uint32_t(count), C.uint32_t(instanceCount))
}

func (c *Context) DrawElementsInstanced(mode uint32, count int32, typ uint32, indices uintptr, instanceCount int32) {
	C.gl_v_iiiui(C.uintptr_t(uintptr(c.glDrawElementsInstanced)), C.uint32_t(mode), C.uint32_t(count), C.uint32_t(typ), C.uintptr_t(indices), C.uint32_t(instanceCount))
}

// DrawArraysInstancedBaseInstance is DrawArraysInstanced with the instance
// the per-instance attributes are fetched from first.
func (c *Context) DrawArraysInstancedBaseInstance(mode uint32, first, count, instanceCount int32, baseInstance uint32) {
	C.gl_v_iiiii(C.uintptr_t(uintptr(c.glDrawArraysInstancedBaseInstance)), C.uint32_t(mode), C.uint32_t(first), C.uint32_t(count), C.uint32_t(instanceCount), C.uint32_t(baseInstance))
}

// DrawElementsInstancedBaseInstance is DrawElementsInstanced with the
// instance the per-instance attributes are fetched from first.
func (c *Context) DrawElementsInstancedBaseInstance(mode uint32, count int32, typ uint32, indices uintptr, instanceCount int32, baseInstance uint32) {
	C.gl_v_iiiuii(C.uintptr_t(uintptr(c.glDrawElementsInstancedBaseInstance)), C.uint32_t(mode), C.uint32_t(count), C.uint32_t(typ), C.uintptr_t(indices), C.uint32_t(instanceCount), C.uint32_t(baseInstance))
}

// SupportsBaseInstance reports whether draws can start at an instance other
// than the first.
func (c *Context) SupportsBaseInstance() bool {
	return c.glDrawArraysInstancedBaseInstance != nil && c.glDrawElementsInstancedBaseInstance != nil
}

// --- Compute Shaders ---

// DispatchCompute launches compute shader workgroups.
// Requires OpenGL 4.3+ or OpenGL ES 3.1+.
// No-op if compute shaders are not supported.
func (c *Context) DispatchCompute(numGroupsX, numGroupsY, numGroupsZ uint32) {
	if c.glDispatchCompute == nil {
		return
	}
	C.gl_v_iii(C.uintptr_t(uintptr(c.glDispatchCompute)), C.uint32_t(numGroupsX), C.uint32_t(numGroupsY), C.uint32_t(numGroupsZ))
}

// DispatchComputeIndirect launches compute workgroups with parameters from a buffer.
// The indirect parameter is an offset into the currently bound GL_DISPATCH_INDIRECT_BUFFER.
// Requires OpenGL 4.3+ or OpenGL ES 3.1+.
// No-op if compute shaders are not supported.
func (c *Context) DispatchComputeIndirect(indirect uintptr) {
	if c.glDispatchComputeIndirect == nil {
		return
	}
	C.gl_v_u(C.uintptr_t(uintptr(c.glDispatchComputeIndirect)), C.uintptr_t(indirect))
}

// MemoryBarrier inserts a memory barrier for specified access types.
// barriers is a bitwise OR of GL_*_BARRIER_BIT constants.
// Requires OpenGL 4.2+ or OpenGL ES 3.1+.
// No-op if memory barriers are not supported.
func (c *Context) MemoryBarrier(barriers uint32) {
	if c.glMemoryBarrier == nil {
		return
	}
	C.gl_v_i(C.uintptr_t(uintptr(c.glMemoryBarrier)), C.uint32_t(barriers))
}

// SupportsCompute returns true if compute shaders are supported.
func (c *Context) SupportsCompute() bool {
	return c.glDispatchCompute != nil
}

// --- Query Objects ---

// GenQueries generates query object names.
// Returns 0 if query objects are not supported.
func (c *Context) GenQueries(n int32) uint32 {
	if c.glGenQueries == nil {
		return 0
	}
	var query uint32
	pQuery := &query
	C.gl_v_ip(C.uintptr_t(uintptr(c.glGenQueries)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(pQuery)))
	return query
}

// DeleteQueries deletes query objects.
func (c *Context) DeleteQueries(n int32, queries *uint32) {
	if c.glDeleteQueries == nil {
		return
	}
	C.gl_v_ip(C.uintptr_t(uintptr(c.glDeleteQueries)), C.uint32_t(n), (*C.uint8_t)(unsafe.Pointer(queries)))
}

// QueryCounter records a timestamp into a query object.
// Requires GL_ARB_timer_query (GL 3.3+) or GL_EXT_disjoint_timer_query (ES 3.0+).
func (c *Context) QueryCounter(query, target uint32) {
	if c.glQueryCounter == nil {
		return
	}
	C.gl_v_ii(C.uintptr_t(uintptr(c.glQueryCounter)), C.uint32_t(query), C.uint32_t(target))
}

// GetQueryObjectui64v retrieves a 64-bit query result.
func (c *Context) GetQueryObjectui64v(query, pname uint32, result *uint64) {
	if c.glGetQueryObjectui64v == nil {
		return
	}
	C.gl_v_iip(C.uintptr_t(uintptr(c.glGetQueryObjectui64v)), C.uint32_t(query), C.uint32_t(pname), (*C.uint8_t)(unsafe.Pointer(result)))
}

// SupportsTimestampQueries returns true if timestamp queries are supported.
func (c *Context) SupportsTimestampQueries() bool {
	return c.glQueryCounter != nil && c.glGenQueries != nil
}

// --- Sync Object Wrappers ---

// FenceSync creates a sync object and inserts it into the GL command stream.
// Returns the sync object handle, or 0 on failure.
func (c *Context) FenceSync(condition, flags uint32) uintptr {
	if c.glFenceSync == nil {
		return 0
	}
	var result uintptr
	result = uintptr(C.gl_u_ii(C.uintptr_t(uintptr(c.glFenceSync)), C.uint32_t(condition), C.uint32_t(flags)))
	return result
}

// DeleteSync deletes a sync object.
func (c *Context) DeleteSync(sync uintptr) {
	if c.glDeleteSync == nil || sync == 0 {
		return
	}
	C.gl_v_u(C.uintptr_t(uintptr(c.glDeleteSync)), C.uintptr_t(sync))
}

// ClientWaitSync blocks until the sync object is signaled or timeout expires.
// Returns ALREADY_SIGNALED, CONDITION_SATISFIED, TIMEOUT_EXPIRED, or WAIT_FAILED.
func (c *Context) ClientWaitSync(sync uintptr, flags uint32, timeout uint64) uint32 {
	if c.glClientWaitSync == nil || sync == 0 {
		return WAIT_FAILED
	}
	var result uint32
	result = uint32(C.gl_i_uiq(C.uintptr_t(uintptr(c.glClientWaitSync)), C.uintptr_t(sync), C.uint32_t(flags), C.uint64_t(timeout)))
	return result
}

// GetSyncStatus returns the signaled status of a sync object.
func (c *Context) GetSyncStatus(sync uintptr) uint32 {
	if c.glGetSynciv == nil || sync == 0 {
		return SIGNALED // assume signaled if not supported
	}
	pname := uint32(SYNC_STATUS)
	count := int32(1)
	var length int32
	var value int32
	pLength := &length
	pValue := &value
	C.gl_v_uiipp(C.uintptr_t(uintptr(c.glGetSynciv)), C.uintptr_t(sync), C.uint32_t(pname), C.uint32_t(count), (*C.uint8_t)(unsafe.Pointer(pLength)), (*C.uint8_t)(unsafe.Pointer(pValue)))
	return uint32(value)
}

// SupportsFenceSync returns true if GL fence sync objects are available.
func (c *Context) SupportsFenceSync() bool {
	return c.glFenceSync != nil
}

// --- Copy Operations ---

// CopyBufferSubData copies part of one buffer to another.
// Requires GL 3.1+ / ES 3.0+.
func (c *Context) CopyBufferSubData(readTarget, writeTarget uint32, readOffset, writeOffset, size int) {
	if c.glCopyBufferSubData == nil {
		return
	}
	ro := uintptr(readOffset)
	wo := uintptr(writeOffset)
	sz := uintptr(size)
	C.gl_v_iiuuu(C.uintptr_t(uintptr(c.glCopyBufferSubData)), C.uint32_t(readTarget), C.uint32_t(writeTarget), C.uintptr_t(ro), C.uintptr_t(wo), C.uintptr_t(sz))
}

// CopyTexSubImage2D copies pixels from the read framebuffer into a 2D texture.
// Reads from the currently bound GL_READ_FRAMEBUFFER.
func (c *Context) CopyTexSubImage2D(target uint32, level, xoffset, yoffset, x, y, width, height int32) {
	if c.glCopyTexSubImage2D == nil {
		return
	}
	C.gl_v_iiiiiiii(C.uintptr_t(uintptr(c.glCopyTexSubImage2D)), C.uint32_t(target), C.uint32_t(level), C.uint32_t(xoffset), C.uint32_t(yoffset), C.uint32_t(x), C.uint32_t(y), C.uint32_t(width), C.uint32_t(height))
}

// --- Helpers ---
// goString copies a NUL-terminated string GL returned, "" for NULL.
func goString(cstr unsafe.Pointer) string {
	if cstr == nil {
		return ""
	}
	return C.GoString((*C.char)(cstr))
}

// cString converts a Go string to a null-terminated C string.
// Returns the pointer and a function to free it.
func cString(s string) (*byte, func()) {
	buf := make([]byte, len(s)+1)
	copy(buf, s)
	return &buf[0], func() {} // No-op free since Go manages memory
}
