module github.com/roffe/gowgpu/examples/desktop

go 1.25.0

require (
	github.com/go-gl/glfw/v3.4/glfw v0.1.0-pre.2
	github.com/roffe/gowgpu v0.0.0
)

require (
	github.com/gogpu/naga v0.20.6-0.20261031111303-7e8984a54a3c // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/roffe/gowgpu => ../..
