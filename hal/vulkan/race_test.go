//go:build race && !(js && wasm)

package vulkan

// raceEnabled is set when the tests run under the race detector.
const raceEnabled = true
