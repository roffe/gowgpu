//go:build js && wasm

package wgpu

import (
	"errors"
	"syscall/js"
	"testing"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/internal/browser"
)

func jsEval(expr string) js.Value {
	return js.Global().Get("Function").New("return " + expr).Invoke()
}

func yieldToJS(t *testing.T) {
	t.Helper()
	if _, err := browser.AwaitPromise(jsEval("Promise.resolve()")); err != nil {
		t.Fatal(err)
	}
}

// fakeDevice returns a GPUDevice stand-in whose lost promise is lost and
// whose error scope stack records the filters it is given in pushed.
func fakeDevice(lost js.Value) (dev, pushed js.Value) {
	dev = jsEval(`{
		queue: {submit() {}},
		pushed: [],
		pushErrorScope(filter) { this.pushed.push(filter) },
		popErrorScope() {
			this.pushed.pop()
			return Promise.resolve(this.pushed.length === 1 ? {message: "inner scope error"} : null)
		},
	}`)
	dev.Set("lost", lost)
	return dev, dev.Get("pushed")
}

func newFakeDevice(lost js.Value) (*Device, js.Value) {
	dev, pushed := fakeDevice(lost)
	bd := browser.NewDevice(dev)
	d := &Device{browser: bd, queue: &Queue{browser: bd.Queue()}, limits: gputypes.DefaultLimits()}
	d.queue.device = d
	return d, pushed
}

func TestBrowserErrorScopes(t *testing.T) {
	d, pushed := newFakeDevice(jsEval("new Promise(() => {})"))
	d.PushErrorScope(ErrorFilterOutOfMemory)
	d.PushErrorScope(ErrorFilterValidation)
	if got := js.Global().Get("JSON").Call("stringify", pushed).String(); got != `["out-of-memory","validation"]` {
		t.Fatalf("pushed filters %s, want out-of-memory then validation", got)
	}

	gpuErr := d.PopErrorScope()
	if gpuErr == nil || gpuErr.Type != ErrorFilterValidation || gpuErr.Message != "inner scope error" {
		t.Fatalf("inner PopErrorScope = %+v, want the validation scope's error", gpuErr)
	}
	if gpuErr := d.PopErrorScope(); gpuErr != nil {
		t.Fatalf("outer PopErrorScope = %+v, want nil", gpuErr)
	}

	defer func() {
		if recover() == nil {
			t.Fatal("PopErrorScope without a pushed scope did not panic")
		}
	}()
	d.PopErrorScope()
}

func TestBrowserLostDeviceRefusesWork(t *testing.T) {
	d, _ := newFakeDevice(jsEval(`Promise.resolve({reason: "unknown", message: "GPU process crashed"})`))
	yieldToJS(t)

	if _, err := d.CreateBuffer(&BufferDescriptor{Size: 16, Usage: BufferUsageCopyDst}); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("CreateBuffer on a lost device: %v, want ErrDeviceLost", err)
	}
	if _, err := d.CreateCommandEncoder(nil); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("CreateCommandEncoder on a lost device: %v, want ErrDeviceLost", err)
	}
	if _, err := d.Queue().Submit(); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("Submit on a lost device: %v, want ErrDeviceLost", err)
	}
	if err := d.Queue().WriteBuffer(&Buffer{browser: &browser.Buffer{}}, 0, nil); !errors.Is(err, ErrDeviceLost) {
		t.Errorf("WriteBuffer on a lost device: %v, want ErrDeviceLost", err)
	}
	buf := &Buffer{browser: &browser.Buffer{}, device: d, size: 16, usage: BufferUsageMapRead}
	if _, err := buf.MapAsync(MapModeRead, 0, 16); !errors.Is(err, ErrMapDeviceLost) {
		t.Errorf("MapAsync on a lost device: %v, want ErrMapDeviceLost", err)
	}
}
