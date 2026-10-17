package vk

import "unsafe"

// NoopCommands fills the commands the benchmarks call with do-nothing C.
func NoopCommands(draw, bind, create, scissor, pipeline unsafe.Pointer) *Commands {
	return &Commands{cmdDraw: draw, cmdBindDescriptorSets: bind, createBuffer: create, cmdSetScissor: scissor, cmdBindPipeline: pipeline}
}
