//go:build js && wasm

package browser

import (
	"syscall/js"
	"testing"
)

func jsObject(t *testing.T, json string) js.Value {
	t.Helper()
	return js.Global().Get("JSON").Call("parse", json)
}

func fakeDevice(lost js.Value) js.Value {
	dev := js.Global().Get("Object").New()
	dev.Set("queue", js.Global().Get("Object").New())
	dev.Set("lost", lost)
	return dev
}

func yieldToJS(t *testing.T) {
	t.Helper()
	if _, err := AwaitPromise(js.Global().Get("Promise").Call("resolve")); err != nil {
		t.Fatal(err)
	}
}

func TestDeviceRecordsTheLoss(t *testing.T) {
	var resolve js.Value
	executor := js.FuncOf(func(_ js.Value, args []js.Value) any {
		resolve = args[0]
		return nil
	})
	defer executor.Release()
	d := NewDevice(fakeDevice(js.Global().Get("Promise").New(executor)))

	yieldToJS(t)
	if _, lost := d.Lost(); lost {
		t.Fatal("device lost before its lost promise resolved")
	}
	resolve.Invoke(jsObject(t, `{"reason":"unknown","message":"GPU process crashed"}`))
	yieldToJS(t)
	if msg, lost := d.Lost(); !lost || msg != "GPU process crashed" {
		t.Fatalf("Lost() = %q, %v, want the browser's message and true", msg, lost)
	}
}

func TestRequestDeviceReportsADeviceThatArrivesLost(t *testing.T) {
	info := jsObject(t, `{"reason":"unknown","message":"adapter expired"}`)
	device := fakeDevice(js.Global().Get("Promise").Call("resolve", info))
	adapter := js.Global().Get("Object").New()
	adapter.Set("requestDevice", js.Global().Get("Function").New("d", "return Promise.resolve(d)").Call("bind", nil, device))

	d, err := NewAdapter(adapter).RequestDevice(js.Undefined())
	if err != nil {
		t.Fatalf("RequestDevice: %v", err)
	}
	if msg, lost := d.Lost(); !lost || msg != "adapter expired" {
		t.Fatalf("Lost() = %q, %v, want the device lost on arrival", msg, lost)
	}
}
