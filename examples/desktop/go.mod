module github.com/roffe/gowgpu/examples/desktop

go 1.25.0

require (
	github.com/go-gl/glfw/v3.4/glfw v0.1.0-pre.2
	github.com/roffe/gowgpu v0.0.0
)

require (
	github.com/roffe/naga v0.0.0-20261023154236-1c7aa3a6b264 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/roffe/gowgpu => ../..
