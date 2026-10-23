// Copyright 2025 The GoGPU Authors
// SPDX-License-Identifier: MIT

//go:build windows && !(js && wasm)

package gl

import (
	"math"
	"syscall"
	"unsafe"
)

// Context holds OpenGL function pointers loaded at runtime.
// Functions are loaded via wglGetProcAddress for GL 2.0+ or
// directly from opengl32.dll for GL 1.1 functions.
type Context struct {
	// Core GL 1.1 (from opengl32.dll)
	glGetError     uintptr
	glGetString    uintptr
	glGetIntegerv  uintptr
	glEnable       uintptr
	glDisable      uintptr
	glClear        uintptr
	glClearColor   uintptr
	glClearDepth   uintptr
	glViewport     uintptr
	glScissor      uintptr
	glDrawArrays   uintptr
	glDrawElements uintptr
	glFlush        uintptr
	glFinish       uintptr

	// Shaders (GL 2.0+)
	glCreateShader       uintptr
	glDeleteShader       uintptr
	glShaderSource       uintptr
	glCompileShader      uintptr
	glGetShaderiv        uintptr
	glGetShaderInfoLog   uintptr
	glCreateProgram      uintptr
	glDeleteProgram      uintptr
	glAttachShader       uintptr
	glDetachShader       uintptr
	glLinkProgram        uintptr
	glUseProgram         uintptr
	glGetProgramiv       uintptr
	glGetProgramInfoLog  uintptr
	glGetUniformLocation uintptr
	glGetAttribLocation  uintptr

	// Uniforms (GL 2.0+)
	glUniform1i        uintptr
	glUniform1f        uintptr
	glUniform2f        uintptr
	glUniform3f        uintptr
	glUniform4f        uintptr
	glUniform1iv       uintptr
	glUniform1fv       uintptr
	glUniform2fv       uintptr
	glUniform3fv       uintptr
	glUniform4fv       uintptr
	glUniformMatrix4fv uintptr

	// Buffers (GL 1.5+)
	glGenBuffers     uintptr
	glDeleteBuffers  uintptr
	glBindBuffer     uintptr
	glBufferData     uintptr
	glBufferSubData  uintptr
	glMapBuffer      uintptr
	glMapBufferRange uintptr
	glUnmapBuffer    uintptr

	// VAO (GL 3.0+)
	glGenVertexArrays    uintptr
	glDeleteVertexArrays uintptr
	glBindVertexArray    uintptr

	// Vertex attributes (GL 2.0+)
	glEnableVertexAttribArray  uintptr
	glDisableVertexAttribArray uintptr
	glVertexAttribPointer      uintptr

	// Textures (GL 1.1+)
	glGenTextures    uintptr
	glDeleteTextures uintptr
	glBindTexture    uintptr
	glActiveTexture  uintptr
	glTexImage2D     uintptr
	glTexSubImage2D  uintptr
	glTexParameteri  uintptr
	glGenerateMipmap uintptr

	// Immutable storage and compressed uploads (GL 4.2)
	glTexStorage2D            uintptr
	glCompressedTexSubImage2D uintptr

	// Framebuffers (GL 3.0+)
	glGenFramebuffers        uintptr
	glDeleteFramebuffers     uintptr
	glBindFramebuffer        uintptr
	glFramebufferTexture2D   uintptr
	glCheckFramebufferStatus uintptr
	glDrawBuffer             uintptr
	glDrawBuffers            uintptr
	glClearBufferfv          uintptr // GL 3.0+ — per-buffer float clear

	// Pixel read/store (GL 1.0+)
	glReadPixels  uintptr
	glPixelStorei uintptr

	// Renderbuffers (GL 3.0+)
	glGenRenderbuffers        uintptr
	glDeleteRenderbuffers     uintptr
	glBindRenderbuffer        uintptr
	glRenderbufferStorage     uintptr
	glFramebufferRenderbuffer uintptr

	// Blending (GL 1.4+)
	glBlendFunc             uintptr
	glBlendFuncSeparate     uintptr
	glBlendEquation         uintptr
	glBlendEquationSeparate uintptr
	glBlendColor            uintptr

	// Depth/Stencil
	glDepthFunc           uintptr
	glDepthMask           uintptr
	glDepthRange          uintptr
	glStencilFunc         uintptr
	glStencilOp           uintptr
	glStencilMask         uintptr
	glStencilFuncSeparate uintptr
	glStencilOpSeparate   uintptr
	glStencilMaskSeparate uintptr
	glColorMask           uintptr
	glPolygonOffset       uintptr
	glPolygonOffsetClamp  uintptr // GL 4.6 / ARB_polygon_offset_clamp, EXT_polygon_offset_clamp; optional

	// Face culling
	glCullFace  uintptr
	glFrontFace uintptr

	// Sync (GL 3.2+)
	glFenceSync      uintptr
	glDeleteSync     uintptr
	glClientWaitSync uintptr
	glWaitSync       uintptr

	// UBO (GL 3.1+)
	glBindBufferBase       uintptr
	glBindBufferRange      uintptr
	glGetUniformBlockIndex uintptr
	glUniformBlockBinding  uintptr

	// Instancing (GL 3.1+)
	glDrawArraysInstanced   uintptr
	glDrawElementsInstanced uintptr
	glVertexAttribDivisor   uintptr

	// Base instance (GL 4.2 / ARB_base_instance; optional)
	glDrawArraysInstancedBaseInstance   uintptr
	glDrawElementsInstancedBaseInstance uintptr

	// Compute shaders (GL 4.3+ / ES 3.1+)
	glDispatchCompute         uintptr
	glDispatchComputeIndirect uintptr
	glMemoryBarrier           uintptr

	// MSAA (GL 3.2+ / ES 3.1+)
	glTexImage2DMultisample uintptr
	glBlitFramebuffer       uintptr

	// Sampler objects (GL 3.3+ / ES 3.0+)
	glGenSamplers       uintptr
	glDeleteSamplers    uintptr
	glBindSampler       uintptr
	glSamplerParameteri uintptr
	glSamplerParameterf uintptr

	// Query objects (GL 3.3+ / GL_EXT_disjoint_timer_query)
	glGenQueries          uintptr
	glDeleteQueries       uintptr
	glQueryCounter        uintptr
	glGetQueryObjectui64v uintptr
	glCopyBufferSubData   uintptr
	glCopyTexSubImage2D   uintptr
	glGetSynciv           uintptr

	// Indexed string query (GL 3.0+ / ES 3.0+)
	glGetStringi uintptr

	// Shadow is what the command replay knows the context's state to be.
	Shadow Shadow
}

// ProcAddressFunc is a function that returns the address of an OpenGL function.
type ProcAddressFunc func(name string) uintptr

// Load loads all OpenGL function pointers using the provided loader.
func (c *Context) Load(getProcAddr ProcAddressFunc) error {
	// Core GL 1.1
	c.glGetError = getProcAddr("glGetError")
	c.glGetString = getProcAddr("glGetString")
	c.glGetIntegerv = getProcAddr("glGetIntegerv")
	c.glEnable = getProcAddr("glEnable")
	c.glDisable = getProcAddr("glDisable")
	c.glClear = getProcAddr("glClear")
	c.glClearColor = getProcAddr("glClearColor")
	c.glClearDepth = getProcAddr("glClearDepth")
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
	c.glMapBufferRange = getProcAddr("glMapBufferRange")
	c.glUnmapBuffer = getProcAddr("glUnmapBuffer")

	// VAO
	c.glGenVertexArrays = getProcAddr("glGenVertexArrays")
	c.glDeleteVertexArrays = getProcAddr("glDeleteVertexArrays")
	c.glBindVertexArray = getProcAddr("glBindVertexArray")

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
	c.glTexStorage2D = getProcAddr("glTexStorage2D")
	c.glCompressedTexSubImage2D = getProcAddr("glCompressedTexSubImage2D")

	// Framebuffers
	c.glGenFramebuffers = getProcAddr("glGenFramebuffers")
	c.glDeleteFramebuffers = getProcAddr("glDeleteFramebuffers")
	c.glBindFramebuffer = getProcAddr("glBindFramebuffer")
	c.glFramebufferTexture2D = getProcAddr("glFramebufferTexture2D")
	c.glCheckFramebufferStatus = getProcAddr("glCheckFramebufferStatus")
	c.glDrawBuffer = getProcAddr("glDrawBuffer")
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
	c.glDepthRange = getProcAddr("glDepthRange")
	c.glStencilFunc = getProcAddr("glStencilFunc")
	c.glStencilOp = getProcAddr("glStencilOp")
	c.glStencilMask = getProcAddr("glStencilMask")
	c.glStencilFuncSeparate = getProcAddr("glStencilFuncSeparate")
	c.glStencilOpSeparate = getProcAddr("glStencilOpSeparate")
	c.glStencilMaskSeparate = getProcAddr("glStencilMaskSeparate")
	c.glColorMask = getProcAddr("glColorMask")
	c.glPolygonOffset = getProcAddr("glPolygonOffset")
	c.glPolygonOffsetClamp = getProcAddr("glPolygonOffsetClamp")
	if c.glPolygonOffsetClamp == 0 {
		c.glPolygonOffsetClamp = getProcAddr("glPolygonOffsetClampEXT")
	}

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
	c.glDrawArraysInstancedBaseInstance = getProcAddr("glDrawArraysInstancedBaseInstance")
	c.glDrawElementsInstancedBaseInstance = getProcAddr("glDrawElementsInstancedBaseInstance")

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
	r, _, _ := syscall.SyscallN(c.glGetError)
	return uint32(r)
}

func (c *Context) GetString(name uint32) string {
	r, _, _ := syscall.SyscallN(c.glGetString, uintptr(name))
	if r == 0 {
		return ""
	}
	return goString(r)
}

func (c *Context) GetIntegerv(pname uint32, data *int32) {
	syscall.SyscallN(c.glGetIntegerv, uintptr(pname), uintptr(unsafe.Pointer(data)))
}

// GetStringi returns an indexed string from an OpenGL string array (GL 3.0+).
// Used for querying GL_EXTENSIONS one at a time (the modern way).
// Returns "" if glGetStringi is not available or the index is out of range.
func (c *Context) GetStringi(name, index uint32) string {
	if c.glGetStringi == 0 {
		return ""
	}
	r, _, _ := syscall.SyscallN(c.glGetStringi, uintptr(name), uintptr(index))
	if r == 0 {
		return ""
	}
	return goString(r)
}

func (c *Context) Enable(capability uint32) {
	syscall.SyscallN(c.glEnable, uintptr(capability))
}

func (c *Context) Disable(capability uint32) {
	syscall.SyscallN(c.glDisable, uintptr(capability))
}

func (c *Context) Clear(mask uint32) {
	syscall.SyscallN(c.glClear, uintptr(mask))
}

func (c *Context) ClearColor(r, g, b, a float32) {
	syscall.SyscallN(c.glClearColor,
		uintptr(*(*uint32)(unsafe.Pointer(&r))),
		uintptr(*(*uint32)(unsafe.Pointer(&g))),
		uintptr(*(*uint32)(unsafe.Pointer(&b))),
		uintptr(*(*uint32)(unsafe.Pointer(&a))))
}

// ClearDepth sets the depth value used by glClear(GL_DEPTH_BUFFER_BIT).
// Desktop GL uses glClearDepth (double); GLES uses glClearDepthf (float).
// Both variants are loaded via getProcAddr in init.
func (c *Context) ClearDepth(depth float64) {
	syscall.SyscallN(c.glClearDepth, doubleArg(depth)...)
}

// DepthRange sets the mapping of NDC depth to window depth.
// Desktop GL: glDepthRange(double, double). GLES: glDepthRangef(float, float).
func (c *Context) DepthRange(near, far float64) {
	args := append(doubleArg(near), doubleArg(far)...)
	syscall.SyscallN(c.glDepthRange, args...)
}

// doubleArg is a double argument as SyscallN passes it: one uintptr on
// 64-bit Windows, which puts it in a register, and both halves of it on 386,
// which puts it on the stack.
func doubleArg(v float64) []uintptr {
	bits := math.Float64bits(v)
	if unsafe.Sizeof(uintptr(0)) < 8 {
		return []uintptr{uintptr(bits), uintptr(bits >> 32)}
	}
	return []uintptr{uintptr(bits)}
}

func (c *Context) Viewport(x, y, width, height int32) {
	syscall.SyscallN(c.glViewport, uintptr(x), uintptr(y), uintptr(width), uintptr(height))
}

func (c *Context) Scissor(x, y, width, height int32) {
	syscall.SyscallN(c.glScissor, uintptr(x), uintptr(y), uintptr(width), uintptr(height))
}

func (c *Context) DrawArrays(mode uint32, first, count int32) {
	syscall.SyscallN(c.glDrawArrays, uintptr(mode), uintptr(first), uintptr(count))
}

func (c *Context) DrawElements(mode uint32, count int32, typ uint32, indices uintptr) {
	syscall.SyscallN(c.glDrawElements, uintptr(mode), uintptr(count), uintptr(typ), indices)
}

func (c *Context) Flush() {
	syscall.SyscallN(c.glFlush)
}

func (c *Context) Finish() {
	syscall.SyscallN(c.glFinish)
}

// --- Shaders ---

func (c *Context) CreateShader(shaderType uint32) uint32 {
	r, _, _ := syscall.SyscallN(c.glCreateShader, uintptr(shaderType))
	return uint32(r)
}

func (c *Context) DeleteShader(shader uint32) {
	syscall.SyscallN(c.glDeleteShader, uintptr(shader))
}

func (c *Context) ShaderSource(shader uint32, source string) {
	csource, free := cString(source)
	defer free()
	length := int32(len(source))
	syscall.SyscallN(c.glShaderSource, uintptr(shader), 1,
		uintptr(unsafe.Pointer(&csource)),
		uintptr(unsafe.Pointer(&length)))
}

func (c *Context) CompileShader(shader uint32) {
	syscall.SyscallN(c.glCompileShader, uintptr(shader))
}

func (c *Context) GetShaderiv(shader uint32, pname uint32, params *int32) {
	syscall.SyscallN(c.glGetShaderiv, uintptr(shader), uintptr(pname),
		uintptr(unsafe.Pointer(params)))
}

func (c *Context) GetShaderInfoLog(shader uint32) string {
	var length int32
	c.GetShaderiv(shader, INFO_LOG_LENGTH, &length)
	if length == 0 {
		return ""
	}
	buf := make([]byte, length)
	syscall.SyscallN(c.glGetShaderInfoLog, uintptr(shader), uintptr(length),
		uintptr(unsafe.Pointer(&length)), uintptr(unsafe.Pointer(&buf[0])))
	return string(buf[:length])
}

func (c *Context) CreateProgram() uint32 {
	r, _, _ := syscall.SyscallN(c.glCreateProgram)
	return uint32(r)
}

func (c *Context) DeleteProgram(program uint32) {
	syscall.SyscallN(c.glDeleteProgram, uintptr(program))
}

func (c *Context) AttachShader(program, shader uint32) {
	syscall.SyscallN(c.glAttachShader, uintptr(program), uintptr(shader))
}

func (c *Context) LinkProgram(program uint32) {
	syscall.SyscallN(c.glLinkProgram, uintptr(program))
}

func (c *Context) UseProgram(program uint32) {
	syscall.SyscallN(c.glUseProgram, uintptr(program))
}

func (c *Context) GetProgramiv(program uint32, pname uint32, params *int32) {
	syscall.SyscallN(c.glGetProgramiv, uintptr(program), uintptr(pname),
		uintptr(unsafe.Pointer(params)))
}

func (c *Context) GetProgramInfoLog(program uint32) string {
	var length int32
	c.GetProgramiv(program, INFO_LOG_LENGTH, &length)
	if length == 0 {
		return ""
	}
	buf := make([]byte, length)
	syscall.SyscallN(c.glGetProgramInfoLog, uintptr(program), uintptr(length),
		uintptr(unsafe.Pointer(&length)), uintptr(unsafe.Pointer(&buf[0])))
	return string(buf[:length])
}

func (c *Context) GetUniformLocation(program uint32, name string) int32 {
	cname, free := cString(name)
	defer free()
	r, _, _ := syscall.SyscallN(c.glGetUniformLocation, uintptr(program), uintptr(unsafe.Pointer(cname)))
	return int32(r)
}

func (c *Context) GetAttribLocation(program uint32, name string) int32 {
	cname, free := cString(name)
	defer free()
	r, _, _ := syscall.SyscallN(c.glGetAttribLocation, uintptr(program), uintptr(unsafe.Pointer(cname)))
	return int32(r)
}

// --- Buffers ---

func (c *Context) GenBuffers(n int32) uint32 {
	var buffer uint32
	syscall.SyscallN(c.glGenBuffers, uintptr(n), uintptr(unsafe.Pointer(&buffer)))
	return buffer
}

func (c *Context) DeleteBuffers(buffers ...uint32) {
	syscall.SyscallN(c.glDeleteBuffers, uintptr(len(buffers)),
		uintptr(unsafe.Pointer(&buffers[0])))
}

func (c *Context) BindBuffer(target, buffer uint32) {
	syscall.SyscallN(c.glBindBuffer, uintptr(target), uintptr(buffer))
}

func (c *Context) BufferData(target uint32, size int, data unsafe.Pointer, usage uint32) {
	syscall.SyscallN(c.glBufferData, uintptr(target), uintptr(size),
		uintptr(data), uintptr(usage))
}

func (c *Context) BufferSubData(target uint32, offset, size int, data unsafe.Pointer) {
	syscall.SyscallN(c.glBufferSubData, uintptr(target), uintptr(offset),
		uintptr(size), uintptr(data))
}

// MapBuffer maps a buffer object's data store into the client's address space.
// target: GL_ARRAY_BUFFER, GL_COPY_READ_BUFFER, etc.
// access: GL_READ_ONLY, GL_WRITE_ONLY, GL_READ_WRITE.
// Returns the mapped pointer, or 0 if glMapBuffer is not available or the call fails.
func (c *Context) MapBuffer(target, access uint32) uintptr {
	if c.glMapBuffer == 0 {
		return 0
	}
	r, _, _ := syscall.SyscallN(c.glMapBuffer, uintptr(target), uintptr(access))
	return r
}

// MapBufferRange maps length bytes of the buffer bound to target from offset
// for the MAP_*_BIT access given (GL 3.0 / ES 3.0). Returns the mapped
// pointer, or 0 when the call fails.
func (c *Context) MapBufferRange(target uint32, offset, length int, access uint32) uintptr {
	if c.glMapBufferRange == 0 {
		return 0
	}
	r, _, _ := syscall.SyscallN(c.glMapBufferRange, uintptr(target), uintptr(offset), uintptr(length), uintptr(access))
	return r
}

// UnmapBuffer releases the mapping of a buffer object's data store.
// Returns true if the buffer was successfully unmapped, false if the buffer contents
// became corrupt during the mapping (GL_FALSE from driver) or glUnmapBuffer is unavailable.
func (c *Context) UnmapBuffer(target uint32) bool {
	if c.glUnmapBuffer == 0 {
		return false
	}
	r, _, _ := syscall.SyscallN(c.glUnmapBuffer, uintptr(target))
	return r != 0
}

// --- VAO ---

func (c *Context) GenVertexArrays(n int32) uint32 {
	var vao uint32
	syscall.SyscallN(c.glGenVertexArrays, uintptr(n), uintptr(unsafe.Pointer(&vao)))
	return vao
}

func (c *Context) DeleteVertexArrays(arrays ...uint32) {
	syscall.SyscallN(c.glDeleteVertexArrays, uintptr(len(arrays)),
		uintptr(unsafe.Pointer(&arrays[0])))
}

func (c *Context) BindVertexArray(array uint32) {
	syscall.SyscallN(c.glBindVertexArray, uintptr(array))
}

// --- Vertex Attributes ---

func (c *Context) EnableVertexAttribArray(index uint32) {
	syscall.SyscallN(c.glEnableVertexAttribArray, uintptr(index))
}

func (c *Context) DisableVertexAttribArray(index uint32) {
	syscall.SyscallN(c.glDisableVertexAttribArray, uintptr(index))
}

func (c *Context) VertexAttribPointer(index uint32, size int32, typ uint32, normalized bool, stride int32, offset uintptr) {
	var norm uintptr
	if normalized {
		norm = TRUE
	}
	syscall.SyscallN(c.glVertexAttribPointer, uintptr(index), uintptr(size),
		uintptr(typ), norm, uintptr(stride), offset)
}

// VertexAttribDivisor sets the instance divisor for a vertex attribute.
// divisor=0 means per-vertex, divisor=1 means per-instance.
// Matches Rust wgpu-hal/src/gles/queue.rs vertex_attrib_divisor call.
func (c *Context) VertexAttribDivisor(index, divisor uint32) {
	syscall.SyscallN(c.glVertexAttribDivisor, uintptr(index), uintptr(divisor))
}

// --- Textures ---

func (c *Context) GenTextures(n int32) uint32 {
	var tex uint32
	syscall.SyscallN(c.glGenTextures, uintptr(n), uintptr(unsafe.Pointer(&tex)))
	return tex
}

func (c *Context) DeleteTextures(textures ...uint32) {
	syscall.SyscallN(c.glDeleteTextures, uintptr(len(textures)),
		uintptr(unsafe.Pointer(&textures[0])))
}

func (c *Context) BindTexture(target, texture uint32) {
	syscall.SyscallN(c.glBindTexture, uintptr(target), uintptr(texture))
}

func (c *Context) ActiveTexture(texture uint32) {
	syscall.SyscallN(c.glActiveTexture, uintptr(texture))
}

func (c *Context) TexParameteri(target, pname uint32, param int32) {
	syscall.SyscallN(c.glTexParameteri, uintptr(target), uintptr(pname), uintptr(param))
}

// TexImage2D specifies a 2D image of a texture level. pixels is 0 for no
// data, or an offset into the bound GL_PIXEL_UNPACK_BUFFER.
func (c *Context) TexImage2D(target uint32, level int32, internalformat int32, width, height int32, border int32, format, typ uint32, pixels uintptr) {
	syscall.SyscallN(c.glTexImage2D, uintptr(target), uintptr(level),
		uintptr(internalformat), uintptr(width), uintptr(height), uintptr(border),
		uintptr(format), uintptr(typ), pixels)
}

func (c *Context) TexSubImage2D(target uint32, level int32, xoffset, yoffset, width, height int32, format, typ uint32, pixels unsafe.Pointer) {
	syscall.SyscallN(c.glTexSubImage2D, uintptr(target), uintptr(level),
		uintptr(xoffset), uintptr(yoffset), uintptr(width), uintptr(height),
		uintptr(format), uintptr(typ), uintptr(pixels))
}

// TexSubImage2DOffset is glTexSubImage2D from the bound
// GL_PIXEL_UNPACK_BUFFER: offset is a byte offset into that buffer, not a
// pointer.
func (c *Context) TexSubImage2DOffset(target uint32, level int32, xoffset, yoffset, width, height int32, format, typ uint32, offset uintptr) {
	syscall.SyscallN(c.glTexSubImage2D, uintptr(target), uintptr(level),
		uintptr(xoffset), uintptr(yoffset), uintptr(width), uintptr(height),
		uintptr(format), uintptr(typ), offset)
}

func (c *Context) GenerateMipmap(target uint32) {
	syscall.SyscallN(c.glGenerateMipmap, uintptr(target))
}

// TexStorage2D allocates immutable storage for levels mip levels of the
// bound texture (GL 4.2, ARB_texture_storage).
func (c *Context) TexStorage2D(target uint32, levels int32, internalformat uint32, width, height int32) {
	syscall.SyscallN(c.glTexStorage2D, uintptr(target), uintptr(levels), uintptr(internalformat), uintptr(width), uintptr(height))
}

// CompressedTexSubImage2D replaces a block-aligned region of a compressed
// texture level with imageSize bytes of blocks from data.
func (c *Context) CompressedTexSubImage2D(target uint32, level, xoffset, yoffset, width, height int32, format uint32, imageSize int32, data unsafe.Pointer) {
	syscall.SyscallN(c.glCompressedTexSubImage2D, uintptr(target), uintptr(level), uintptr(xoffset), uintptr(yoffset),
		uintptr(width), uintptr(height), uintptr(format), uintptr(imageSize), uintptr(data))
}

// CompressedTexSubImage2DOffset is CompressedTexSubImage2D from the bound
// GL_PIXEL_UNPACK_BUFFER, offset bytes into it.
func (c *Context) CompressedTexSubImage2DOffset(target uint32, level, xoffset, yoffset, width, height int32, format uint32, imageSize int32, offset uintptr) {
	syscall.SyscallN(c.glCompressedTexSubImage2D, uintptr(target), uintptr(level), uintptr(xoffset), uintptr(yoffset),
		uintptr(width), uintptr(height), uintptr(format), uintptr(imageSize), offset)
}

// TexImage2DMultisample creates a multisample 2D texture image.
// Requires OpenGL 3.2+ or OpenGL ES 3.1+.
// No-op if not supported.
func (c *Context) TexImage2DMultisample(target uint32, samples int32, internalformat uint32, width, height int32, fixedsamplelocations bool) {
	if c.glTexImage2DMultisample == 0 {
		return
	}
	var fixed uintptr
	if fixedsamplelocations {
		fixed = TRUE
	}
	syscall.SyscallN(c.glTexImage2DMultisample, uintptr(target), uintptr(samples),
		uintptr(internalformat), uintptr(width), uintptr(height), fixed)
}

// BlitFramebuffer copies a block of pixels between framebuffers.
// Requires OpenGL 3.0+ or OpenGL ES 3.0+.
// No-op if not supported.
func (c *Context) BlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1 int32, mask, filter uint32) {
	if c.glBlitFramebuffer == 0 {
		return
	}
	syscall.SyscallN(c.glBlitFramebuffer,
		uintptr(srcX0), uintptr(srcY0), uintptr(srcX1), uintptr(srcY1),
		uintptr(dstX0), uintptr(dstY0), uintptr(dstX1), uintptr(dstY1),
		uintptr(mask), uintptr(filter))
}

// --- Sampler Objects ---

// GenSamplers generates sampler object names.
// Requires OpenGL 3.3+ or OpenGL ES 3.0+.
// Returns 0 if sampler objects are not supported.
func (c *Context) GenSamplers(n int32) uint32 {
	if c.glGenSamplers == 0 {
		return 0
	}
	var sampler uint32
	syscall.SyscallN(c.glGenSamplers, uintptr(n), uintptr(unsafe.Pointer(&sampler)))
	return sampler
}

// DeleteSamplers deletes sampler objects.
// No-op if sampler objects are not supported.
func (c *Context) DeleteSamplers(samplers ...uint32) {
	if c.glDeleteSamplers == 0 {
		return
	}
	syscall.SyscallN(c.glDeleteSamplers, uintptr(len(samplers)),
		uintptr(unsafe.Pointer(&samplers[0])))
}

// BindSampler binds a sampler object to a texture unit.
// No-op if sampler objects are not supported.
func (c *Context) BindSampler(unit, sampler uint32) {
	if c.glBindSampler == 0 {
		return
	}
	syscall.SyscallN(c.glBindSampler, uintptr(unit), uintptr(sampler))
}

// SamplerParameteri sets an integer parameter on a sampler object.
// No-op if sampler objects are not supported.
func (c *Context) SamplerParameteri(sampler, pname uint32, param int32) {
	if c.glSamplerParameteri == 0 {
		return
	}
	syscall.SyscallN(c.glSamplerParameteri, uintptr(sampler), uintptr(pname), uintptr(param))
}

// SamplerParameterf sets a float parameter on a sampler object.
// No-op if sampler objects are not supported.
func (c *Context) SamplerParameterf(sampler, pname uint32, param float32) {
	if c.glSamplerParameterf == 0 {
		return
	}
	syscall.SyscallN(c.glSamplerParameterf, uintptr(sampler), uintptr(pname),
		uintptr(*(*uint32)(unsafe.Pointer(&param))))
}

// SupportsSamplerObjects returns true if GL sampler objects are supported.
func (c *Context) SupportsSamplerObjects() bool {
	return c.glGenSamplers != 0
}

// --- Framebuffers ---

func (c *Context) GenFramebuffers(n int32) uint32 {
	var fbo uint32
	syscall.SyscallN(c.glGenFramebuffers, uintptr(n), uintptr(unsafe.Pointer(&fbo)))
	return fbo
}

func (c *Context) DeleteFramebuffers(framebuffers ...uint32) {
	syscall.SyscallN(c.glDeleteFramebuffers, uintptr(len(framebuffers)),
		uintptr(unsafe.Pointer(&framebuffers[0])))
}

func (c *Context) BindFramebuffer(target, framebuffer uint32) {
	syscall.SyscallN(c.glBindFramebuffer, uintptr(target), uintptr(framebuffer))
}

func (c *Context) FramebufferTexture2D(target, attachment, textarget, texture uint32, level int32) {
	syscall.SyscallN(c.glFramebufferTexture2D, uintptr(target), uintptr(attachment),
		uintptr(textarget), uintptr(texture), uintptr(level))
}

func (c *Context) CheckFramebufferStatus(target uint32) uint32 {
	r, _, _ := syscall.SyscallN(c.glCheckFramebufferStatus, uintptr(target))
	return uint32(r)
}

// IsGLES reports whether the context is OpenGL ES rather than desktop GL;
// WGL contexts are desktop GL.
func (c *Context) IsGLES() bool { return false }

// DrawBuffer selects the draw buffer of the default framebuffer on desktop
// GL, where glDrawBuffers takes GL_BACK only since 4.5.
func (c *Context) DrawBuffer(buf uint32) {
	syscall.SyscallN(c.glDrawBuffer, uintptr(buf))
}

// DrawBuffers specifies a list of color buffers to be drawn into.
// Matches Rust wgpu-hal GLES SetDrawColorBuffers (queue.rs:1202-1207).
// bufs must contain GL_COLOR_ATTACHMENT0..N or GL_NONE values.
func (c *Context) DrawBuffers(bufs []uint32) {
	if len(bufs) == 0 {
		return
	}
	syscall.SyscallN(c.glDrawBuffers, uintptr(len(bufs)),
		uintptr(unsafe.Pointer(&bufs[0])))
}

// ClearBufferfv clears a specific draw buffer with float values.
// buffer must be GL_COLOR, drawBuffer is the index (0..MAX_DRAW_BUFFERS-1),
// value points to 4 float32 values (RGBA).
// Matches Rust wgpu-hal GLES ClearColorF (queue.rs:1222).
func (c *Context) ClearBufferfv(buffer uint32, drawBuffer int32, value *[4]float32) {
	syscall.SyscallN(c.glClearBufferfv, uintptr(buffer), uintptr(drawBuffer),
		uintptr(unsafe.Pointer(value)))
}

// --- Renderbuffers ---

// GenRenderbuffers generates a single renderbuffer object and returns its name.
func (c *Context) GenRenderbuffers(n int32) uint32 {
	var rbo uint32
	syscall.SyscallN(c.glGenRenderbuffers, uintptr(n), uintptr(unsafe.Pointer(&rbo)))
	return rbo
}

// DeleteRenderbuffers deletes the named renderbuffer objects.
func (c *Context) DeleteRenderbuffers(renderbuffers ...uint32) {
	if len(renderbuffers) == 0 {
		return
	}
	syscall.SyscallN(c.glDeleteRenderbuffers, uintptr(len(renderbuffers)),
		uintptr(unsafe.Pointer(&renderbuffers[0])))
}

// BindRenderbuffer binds a renderbuffer object to the given target
// (typically GL_RENDERBUFFER).
func (c *Context) BindRenderbuffer(target, renderbuffer uint32) {
	syscall.SyscallN(c.glBindRenderbuffer, uintptr(target), uintptr(renderbuffer))
}

// RenderbufferStorage allocates storage for the currently bound renderbuffer.
// internalFormat is a sized format like GL_RGBA8 or GL_DEPTH24_STENCIL8.
func (c *Context) RenderbufferStorage(target, internalFormat uint32, width, height int32) {
	syscall.SyscallN(c.glRenderbufferStorage, uintptr(target), uintptr(internalFormat),
		uintptr(width), uintptr(height))
}

// FramebufferRenderbuffer attaches a renderbuffer to the currently bound
// framebuffer at the specified attachment point.
func (c *Context) FramebufferRenderbuffer(target, attachment, renderbufferTarget, renderbuffer uint32) {
	syscall.SyscallN(c.glFramebufferRenderbuffer, uintptr(target), uintptr(attachment),
		uintptr(renderbufferTarget), uintptr(renderbuffer))
}

// --- Pixel Read/Store ---

// ReadPixels reads a block of pixels from the framebuffer.
func (c *Context) ReadPixels(x, y, width, height int32, format, dataType uint32, pixels unsafe.Pointer) {
	syscall.SyscallN(c.glReadPixels, uintptr(x), uintptr(y), uintptr(width), uintptr(height),
		uintptr(format), uintptr(dataType), uintptr(pixels))
}

// PixelStorei sets pixel storage modes that affect ReadPixels and TexImage operations.
func (c *Context) PixelStorei(pname uint32, param int32) {
	syscall.SyscallN(c.glPixelStorei, uintptr(pname), uintptr(param))
}

// --- UBO ---

// BindBufferBase binds a buffer to an indexed binding point.
func (c *Context) BindBufferBase(target, index, buffer uint32) {
	syscall.SyscallN(c.glBindBufferBase, uintptr(target), uintptr(index), uintptr(buffer))
}

// BindBufferRange binds a range of a buffer to an indexed binding point.
func (c *Context) BindBufferRange(target, index, buffer uint32, offset, size int) {
	syscall.SyscallN(c.glBindBufferRange, uintptr(target), uintptr(index), uintptr(buffer),
		uintptr(offset), uintptr(size))
}

// GetUniformBlockIndex returns the index of a named uniform block.
func (c *Context) GetUniformBlockIndex(program uint32, name string) uint32 {
	cname, free := cString(name)
	defer free()
	r, _, _ := syscall.SyscallN(c.glGetUniformBlockIndex, uintptr(program), uintptr(unsafe.Pointer(cname)))
	return uint32(r)
}

// UniformBlockBinding assigns a uniform block to a binding point.
func (c *Context) UniformBlockBinding(program, blockIndex, blockBinding uint32) {
	syscall.SyscallN(c.glUniformBlockBinding, uintptr(program), uintptr(blockIndex), uintptr(blockBinding))
}

// --- Uniforms ---

// Uniform1i sets an integer uniform value.
func (c *Context) Uniform1i(location, value int32) {
	syscall.SyscallN(c.glUniform1i, uintptr(location), uintptr(value))
}

// --- Blending ---

func (c *Context) BlendFunc(sfactor, dfactor uint32) {
	syscall.SyscallN(c.glBlendFunc, uintptr(sfactor), uintptr(dfactor))
}

func (c *Context) BlendFuncSeparate(srcRGB, dstRGB, srcAlpha, dstAlpha uint32) {
	syscall.SyscallN(c.glBlendFuncSeparate, uintptr(srcRGB), uintptr(dstRGB),
		uintptr(srcAlpha), uintptr(dstAlpha))
}

func (c *Context) BlendEquation(mode uint32) {
	syscall.SyscallN(c.glBlendEquation, uintptr(mode))
}

// BlendEquationSeparate sets separate blend equations for RGB and alpha.
func (c *Context) BlendEquationSeparate(modeRGB, modeAlpha uint32) {
	syscall.SyscallN(c.glBlendEquationSeparate, uintptr(modeRGB), uintptr(modeAlpha))
}

// BlendColor sets the constant blend color.
func (c *Context) BlendColor(r, g, b, a float32) {
	syscall.SyscallN(c.glBlendColor,
		uintptr(*(*uint32)(unsafe.Pointer(&r))),
		uintptr(*(*uint32)(unsafe.Pointer(&g))),
		uintptr(*(*uint32)(unsafe.Pointer(&b))),
		uintptr(*(*uint32)(unsafe.Pointer(&a))))
}

// --- Depth/Stencil ---

func (c *Context) DepthFunc(fn uint32) {
	syscall.SyscallN(c.glDepthFunc, uintptr(fn))
}

func (c *Context) DepthMask(flag bool) {
	var f uintptr
	if flag {
		f = TRUE
	}
	syscall.SyscallN(c.glDepthMask, f)
}

// StencilFuncSeparate sets stencil test function per face.
func (c *Context) StencilFuncSeparate(face, fn uint32, ref int32, mask uint32) {
	syscall.SyscallN(c.glStencilFuncSeparate, uintptr(face), uintptr(fn), uintptr(ref), uintptr(mask))
}

// StencilOpSeparate sets stencil operations per face.
func (c *Context) StencilOpSeparate(face, sfail, dpfail, dppass uint32) {
	syscall.SyscallN(c.glStencilOpSeparate, uintptr(face), uintptr(sfail), uintptr(dpfail), uintptr(dppass))
}

// StencilMaskSeparate sets stencil write mask per face.
func (c *Context) StencilMaskSeparate(face, mask uint32) {
	syscall.SyscallN(c.glStencilMaskSeparate, uintptr(face), uintptr(mask))
}

// ColorMask enables or disables writing of color components.
func (c *Context) ColorMask(r, g, b, a bool) {
	var rv, gv, bv, av uintptr
	if r {
		rv = TRUE
	}
	if g {
		gv = TRUE
	}
	if b {
		bv = TRUE
	}
	if a {
		av = TRUE
	}
	syscall.SyscallN(c.glColorMask, rv, gv, bv, av)
}

// --- Face Culling ---

func (c *Context) CullFace(mode uint32) {
	syscall.SyscallN(c.glCullFace, uintptr(mode))
}

func (c *Context) FrontFace(mode uint32) {
	syscall.SyscallN(c.glFrontFace, uintptr(mode))
}

// PolygonOffset sets the depth bias of filled primitives: units are added
// as a constant and factor times the depth slope.
func (c *Context) PolygonOffset(factor, units float32) {
	syscall.SyscallN(c.glPolygonOffset,
		uintptr(*(*uint32)(unsafe.Pointer(&factor))),
		uintptr(*(*uint32)(unsafe.Pointer(&units))))
}

// PolygonOffsetClamp is PolygonOffset with the bias clamped to at most clamp.
func (c *Context) PolygonOffsetClamp(factor, units, clamp float32) {
	syscall.SyscallN(c.glPolygonOffsetClamp,
		uintptr(*(*uint32)(unsafe.Pointer(&factor))),
		uintptr(*(*uint32)(unsafe.Pointer(&units))),
		uintptr(*(*uint32)(unsafe.Pointer(&clamp))))
}

// SupportsPolygonOffsetClamp reports whether the depth bias can be clamped.
func (c *Context) SupportsPolygonOffsetClamp() bool {
	return c.glPolygonOffsetClamp != 0
}

// --- Instancing ---

func (c *Context) DrawArraysInstanced(mode uint32, first, count, instanceCount int32) {
	syscall.SyscallN(c.glDrawArraysInstanced, uintptr(mode), uintptr(first),
		uintptr(count), uintptr(instanceCount))
}

func (c *Context) DrawElementsInstanced(mode uint32, count int32, typ uint32, indices uintptr, instanceCount int32) {
	syscall.SyscallN(c.glDrawElementsInstanced, uintptr(mode), uintptr(count),
		uintptr(typ), indices, uintptr(instanceCount))
}

// DrawArraysInstancedBaseInstance is DrawArraysInstanced with the instance
// the per-instance attributes are fetched from first.
func (c *Context) DrawArraysInstancedBaseInstance(mode uint32, first, count, instanceCount int32, baseInstance uint32) {
	syscall.SyscallN(c.glDrawArraysInstancedBaseInstance, uintptr(mode), uintptr(first),
		uintptr(count), uintptr(instanceCount), uintptr(baseInstance))
}

// DrawElementsInstancedBaseInstance is DrawElementsInstanced with the
// instance the per-instance attributes are fetched from first.
func (c *Context) DrawElementsInstancedBaseInstance(mode uint32, count int32, typ uint32, indices uintptr, instanceCount int32, baseInstance uint32) {
	syscall.SyscallN(c.glDrawElementsInstancedBaseInstance, uintptr(mode), uintptr(count),
		uintptr(typ), indices, uintptr(instanceCount), uintptr(baseInstance))
}

// SupportsBaseInstance reports whether draws can start at an instance other
// than the first.
func (c *Context) SupportsBaseInstance() bool {
	return c.glDrawArraysInstancedBaseInstance != 0 && c.glDrawElementsInstancedBaseInstance != 0
}

// --- Compute Shaders ---

// DispatchCompute launches compute shader workgroups.
// Requires OpenGL 4.3+ or OpenGL ES 3.1+.
// No-op if compute shaders are not supported.
func (c *Context) DispatchCompute(numGroupsX, numGroupsY, numGroupsZ uint32) {
	if c.glDispatchCompute == 0 {
		return
	}
	syscall.SyscallN(c.glDispatchCompute, uintptr(numGroupsX), uintptr(numGroupsY), uintptr(numGroupsZ))
}

// DispatchComputeIndirect launches compute workgroups with parameters from a buffer.
// The indirect parameter is an offset into the currently bound GL_DISPATCH_INDIRECT_BUFFER.
// Requires OpenGL 4.3+ or OpenGL ES 3.1+.
// No-op if compute shaders are not supported.
func (c *Context) DispatchComputeIndirect(indirect uintptr) {
	if c.glDispatchComputeIndirect == 0 {
		return
	}
	syscall.SyscallN(c.glDispatchComputeIndirect, indirect)
}

// MemoryBarrier inserts a memory barrier for specified access types.
// barriers is a bitwise OR of GL_*_BARRIER_BIT constants.
// Requires OpenGL 4.2+ or OpenGL ES 3.1+.
// No-op if memory barriers are not supported.
func (c *Context) MemoryBarrier(barriers uint32) {
	if c.glMemoryBarrier == 0 {
		return
	}
	syscall.SyscallN(c.glMemoryBarrier, uintptr(barriers))
}

// SupportsCompute returns true if compute shaders are supported.
func (c *Context) SupportsCompute() bool {
	return c.glDispatchCompute != 0
}

// --- Query Objects ---

// GenQueries generates query object names.
// Returns 0 if query objects are not supported.
func (c *Context) GenQueries(n int32) uint32 {
	if c.glGenQueries == 0 {
		return 0
	}
	var query uint32
	syscall.SyscallN(c.glGenQueries, uintptr(n), uintptr(unsafe.Pointer(&query)))
	return query
}

// DeleteQueries deletes query objects.
func (c *Context) DeleteQueries(n int32, queries *uint32) {
	if c.glDeleteQueries == 0 {
		return
	}
	syscall.SyscallN(c.glDeleteQueries, uintptr(n), uintptr(unsafe.Pointer(queries)))
}

// QueryCounter records a timestamp into a query object.
// Requires GL_ARB_timer_query (GL 3.3+) or GL_EXT_disjoint_timer_query (ES 3.0+).
func (c *Context) QueryCounter(query, target uint32) {
	if c.glQueryCounter == 0 {
		return
	}
	syscall.SyscallN(c.glQueryCounter, uintptr(query), uintptr(target))
}

// GetQueryObjectui64v retrieves a 64-bit query result.
func (c *Context) GetQueryObjectui64v(query, pname uint32, result *uint64) {
	if c.glGetQueryObjectui64v == 0 {
		return
	}
	syscall.SyscallN(c.glGetQueryObjectui64v, uintptr(query), uintptr(pname), uintptr(unsafe.Pointer(result)))
}

// SupportsTimestampQueries returns true if timestamp queries are supported.
func (c *Context) SupportsTimestampQueries() bool {
	return c.glQueryCounter != 0 && c.glGenQueries != 0
}

// --- Sync Object Wrappers ---

// FenceSync creates a sync object and inserts it into the GL command stream.
// Returns the sync object handle, or 0 on failure.
func (c *Context) FenceSync(condition, flags uint32) uintptr {
	if c.glFenceSync == 0 {
		return 0
	}
	r, _, _ := syscall.SyscallN(c.glFenceSync, uintptr(condition), uintptr(flags))
	return r
}

// DeleteSync deletes a sync object.
func (c *Context) DeleteSync(sync uintptr) {
	if c.glDeleteSync == 0 || sync == 0 {
		return
	}
	syscall.SyscallN(c.glDeleteSync, sync)
}

// ClientWaitSync blocks until the sync object is signaled or timeout expires.
// Returns ALREADY_SIGNALED, CONDITION_SATISFIED, TIMEOUT_EXPIRED, or WAIT_FAILED.
func (c *Context) ClientWaitSync(sync uintptr, flags uint32, timeout uint64) uint32 {
	if c.glClientWaitSync == 0 || sync == 0 {
		return WAIT_FAILED
	}
	r, _, _ := syscall.SyscallN(c.glClientWaitSync, sync, uintptr(flags),
		uintptr(timeout), uintptr(timeout>>32))
	return uint32(r)
}

// GetSyncStatus returns the signaled status of a sync object.
func (c *Context) GetSyncStatus(sync uintptr) uint32 {
	if c.glGetSynciv == 0 || sync == 0 {
		return SIGNALED // assume signaled if not supported
	}
	var value int32
	var length int32
	pname := uint32(SYNC_STATUS)
	one := int32(1)
	syscall.SyscallN(c.glGetSynciv, sync, uintptr(pname), uintptr(one),
		uintptr(unsafe.Pointer(&length)), uintptr(unsafe.Pointer(&value)))
	return uint32(value)
}

// SupportsFenceSync returns true if GL fence sync objects are available.
func (c *Context) SupportsFenceSync() bool {
	return c.glFenceSync != 0
}

// --- Copy Operations ---

// CopyBufferSubData copies part of one buffer to another.
// Requires GL 3.1+ / ES 3.0+.
func (c *Context) CopyBufferSubData(readTarget, writeTarget uint32, readOffset, writeOffset, size int) {
	if c.glCopyBufferSubData == 0 {
		return
	}
	syscall.SyscallN(c.glCopyBufferSubData,
		uintptr(readTarget), uintptr(writeTarget),
		uintptr(readOffset), uintptr(writeOffset), uintptr(size))
}

// CopyTexSubImage2D copies pixels from the read framebuffer into a 2D texture.
// Reads from the currently bound GL_READ_FRAMEBUFFER.
func (c *Context) CopyTexSubImage2D(target uint32, level, xoffset, yoffset, x, y, width, height int32) {
	if c.glCopyTexSubImage2D == 0 {
		return
	}
	syscall.SyscallN(c.glCopyTexSubImage2D,
		uintptr(target), uintptr(level),
		uintptr(xoffset), uintptr(yoffset),
		uintptr(x), uintptr(y),
		uintptr(width), uintptr(height))
}

// --- Helpers ---

// ptrFromUintptr converts a uintptr (from FFI) to *byte without triggering go vet warning.
// This uses double pointer indirection pattern from ebitengine/purego.
// Reference: https://github.com/golang/go/issues/56487
func ptrFromUintptr(ptr uintptr) *byte {
	return *(**byte)(unsafe.Pointer(&ptr))
}

// goString converts a null-terminated C string pointer to Go string.
// The pointer must be valid and point to a null-terminated string.
// This is safe because the pointer comes from OpenGL and remains valid
// for the duration of this function call.
func goString(cstr uintptr) string {
	if cstr == 0 {
		return ""
	}
	// Find string length first (max 4096 to prevent infinite loops)
	// Use double pointer indirection to satisfy go vet (pattern from ebitengine/purego)
	length := 0
	for i := 0; i < 4096; i++ {
		b := unsafe.Slice(ptrFromUintptr(cstr), i+1)
		if b[i] == 0 {
			length = i
			break
		}
	}
	if length == 0 {
		return ""
	}
	// Create slice and copy to Go-managed memory
	result := unsafe.Slice(ptrFromUintptr(cstr), length)
	return string(result)
}

// cString converts a Go string to a null-terminated C string.
// Returns the pointer and a function to free it.
func cString(s string) (*byte, func()) {
	buf := make([]byte, len(s)+1)
	copy(buf, s)
	return &buf[0], func() {} // No-op free since Go manages memory
}
