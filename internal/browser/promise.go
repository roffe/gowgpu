//go:build js && wasm

package browser

import (
	"fmt"
	"syscall/js"
)

// AwaitPromise blocks the calling goroutine until a JS Promise resolves or rejects.
//
// On WASM, Go goroutines are cooperative (single-threaded). This function uses
// Promise.then with a channel to yield the goroutine until the JS event
// loop resolves the promise. The caller MUST be on a goroutine (not the main
// goroutine) or the program will deadlock.
//
// Pattern matches Rust wgpu's wasm_bindgen_futures::JsFuture::from(promise).
func AwaitPromise(promise js.Value) (js.Value, error) {
	ch := make(chan promiseResult, 1)
	OnSettled(promise,
		func(v js.Value) { ch <- promiseResult{value: v} },
		func(v js.Value) { ch <- promiseResult{value: v, rejected: true} })
	result := <-ch

	if result.rejected {
		msg := extractErrorMessage(result.value)
		return js.Undefined(), fmt.Errorf("promise rejected: %s", msg)
	}

	return result.value, nil
}

// OnSettled calls onResolve with the value promise resolves with, or
// onReject with the reason it rejects with. They run in a JavaScript
// callback, so they must not block.
//
// Matches Rust wgpu's register_then_closures.
func OnSettled(promise js.Value, onResolve, onReject func(js.Value)) {
	var resolve, reject js.Func
	settle := func(fn func(js.Value), args []js.Value) any {
		resolve.Release()
		reject.Release()
		v := js.Undefined()
		if len(args) > 0 {
			v = args[0]
		}
		fn(v)
		return nil
	}
	resolve = js.FuncOf(func(_ js.Value, args []js.Value) any { return settle(onResolve, args) })
	reject = js.FuncOf(func(_ js.Value, args []js.Value) any { return settle(onReject, args) })
	promise.Call("then", resolve, reject)
}

// promiseResult carries either a resolved value or a rejection.
type promiseResult struct {
	value    js.Value
	rejected bool
}

// extractErrorMessage extracts a human-readable message from a JS error value.
func extractErrorMessage(v js.Value) string {
	if v.IsUndefined() || v.IsNull() {
		return "unknown error"
	}

	// Try .message property first (standard Error objects).
	msg := v.Get("message")
	if !msg.IsUndefined() && !msg.IsNull() {
		return msg.String()
	}

	// Fall back to toString().
	return v.Call("toString").String()
}
