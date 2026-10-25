// Command wasmchrome runs a js/wasm test binary in headless Chrome, where
// WebGPU is real, for go test's -exec:
//
//	GOOS=js GOARCH=wasm go test -exec "env GOOS= GOARCH= go run ./cmd/wasmchrome" -run BrowserGPU -v .
//
// It serves the binary and Go's wasm_exec.js on localhost, opens them in a
// headless Chrome with WebGPU turned on, prints what the test writes and
// exits with its exit code. CHROME names the browser, google-chrome-stable
// by default (Edge works too). WASMCHROME_FLAGS adds Chrome flags.
//
// Headless Chrome on Linux gives WebGPU to SwiftShader, its CPU renderer,
// which runs anywhere. WASMCHROME_WINDOW=1 opens a window on the display
// instead, where WebGPU runs on the GPU.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const page = `<!doctype html>
<script src="wasm_exec.js"></script>
<script>
// Posts go out one after another, so the output keeps its order and the
// exit comes last.
let queue = Promise.resolve();
const post = (path, body) => (queue = queue.then(() => fetch(path, {method: "POST", body})));
const decoder = new TextDecoder();
globalThis.fs.writeSync = (fd, buf) => { post("/out", decoder.decode(buf)); return buf.length; };
(async () => {
	const go = new Go();
	go.argv = ["test.wasm", ...await (await fetch("/args")).json()];
	go.exit = code => post("/exit", String(code));
	try {
		const {instance} = await WebAssembly.instantiateStreaming(fetch("/test.wasm"), go.importObject);
		await go.run(instance);
	} catch (e) {
		post("/out", "wasmchrome: " + e + "\n");
		post("/exit", "1");
	}
})();
</script>
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: wasmchrome test.wasm [test flags]")
		os.Exit(2)
	}
	code, err := run(os.Args[1], os.Args[2:])
	if err != nil {
		fmt.Fprintln(os.Stderr, "wasmchrome:", err)
		os.Exit(1)
	}
	os.Exit(code)
}

func run(wasm string, args []string) (int, error) {
	wasmExec, err := wasmExecJS()
	if err != nil {
		return 0, err
	}
	exit := make(chan int, 1)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		io.WriteString(w, page) //nolint:errcheck // the page sees a short read
	})
	mux.HandleFunc("GET /wasm_exec.js", func(w http.ResponseWriter, r *http.Request) { http.ServeFile(w, r, wasmExec) })
	mux.HandleFunc("GET /test.wasm", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/wasm")
		http.ServeFile(w, r, wasm)
	})
	mux.HandleFunc("GET /args", func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(args) //nolint:errcheck,errchkjson // the page sees no args
	})
	mux.HandleFunc("POST /out", func(_ http.ResponseWriter, r *http.Request) {
		io.Copy(os.Stdout, r.Body) //nolint:errcheck // nowhere to report it
	})
	mux.HandleFunc("POST /exit", func(_ http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		code, err := strconv.Atoi(strings.TrimSpace(string(b)))
		if err != nil {
			code = 1
		}
		select {
		case exit <- code:
		default:
		}
	})
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, err
	}
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second}
	go srv.Serve(ln) //nolint:errcheck // ends with Close
	defer srv.Close()

	profile, err := os.MkdirTemp("", "wasmchrome-")
	if err != nil {
		return 0, err
	}
	ctx, cancel := context.WithCancel(context.Background())
	chrome := exec.CommandContext(ctx, browser(), append(chromeFlags(profile), "http://"+ln.Addr().String()+"/")...) //nolint:gosec // the user's own browser
	// A session bus that goes nowhere: on one, Chrome starts the desktop's
	// portals and keyring, which outlive the run and hold its output open.
	chrome.Env = append(os.Environ(), "DBUS_SESSION_BUS_ADDRESS=unix:path=/dev/null")
	if !windowed() {
		chrome.Env = append(chrome.Env, "DISPLAY=", "WAYLAND_DISPLAY=")
	}
	chrome.Stderr = chromeLog(profile)
	if err := chrome.Start(); err != nil {
		cancel()
		os.RemoveAll(profile)
		return 0, err
	}
	var died error
	gone := make(chan struct{})
	go func() { died = chrome.Wait(); close(gone) }()
	defer func() {
		cancel()
		<-gone // Chrome writes to its profile until it is gone
		os.RemoveAll(profile)
	}()

	select {
	case code := <-exit:
		return code, nil
	case <-gone:
		log, _ := os.ReadFile(filepath.Join(profile, "chrome.log"))
		return 0, fmt.Errorf("%s exited before the test did: %v\n%s", browser(), died, log)
	case <-time.After(timeout()):
		return 0, fmt.Errorf("no exit from the test after %v", timeout())
	}
}

func browser() string {
	if b := os.Getenv("CHROME"); b != "" {
		return b
	}
	return "google-chrome-stable"
}

// chromeFlags turn WebGPU on in a headless Chrome of its own that asks no
// keyring for passwords and reports no crashes.
func chromeFlags(profile string) []string {
	flags := []string{
		"--enable-unsafe-webgpu",
		"--enable-features=Vulkan",
		"--user-data-dir=" + profile,
		"--no-first-run",
		"--no-default-browser-check",
		"--password-store=basic",
		"--disable-breakpad",
		"--disable-crash-reporter",
	}
	if !windowed() {
		flags = append(flags, "--headless=new")
	}
	return append(flags, strings.Fields(os.Getenv("WASMCHROME_FLAGS"))...)
}

func windowed() bool { return os.Getenv("WASMCHROME_WINDOW") == "1" }

// chromeLog is where Chrome's own messages go, so they do not mix with the
// test's output.
func chromeLog(profile string) io.Writer {
	f, err := os.Create(filepath.Join(profile, "chrome.log"))
	if err != nil {
		return io.Discard
	}
	return f
}

// wasmExecJS is the wasm_exec.js of the Go that built the binary.
func wasmExecJS() (string, error) {
	root := os.Getenv("GOROOT")
	if root == "" {
		out, err := exec.Command("go", "env", "GOROOT").Output()
		if err != nil {
			return "", fmt.Errorf("go env GOROOT: %w", err)
		}
		root = strings.TrimSpace(string(out))
	}
	return filepath.Join(root, "lib", "wasm", "wasm_exec.js"), nil
}

// timeout is how long the test may take, a bit over go test's own default.
func timeout() time.Duration {
	return 11 * time.Minute
}
