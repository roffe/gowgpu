//go:build !(js && wasm)

package vulkan

import (
	"os"
	"runtime"
	"testing"
)

// DISABLE_RTSS_LAYER is set on Windows only, and a value the user set stays.
func TestDisableRTSSLayer(t *testing.T) {
	t.Setenv("DISABLE_RTSS_LAYER", "")
	os.Unsetenv("DISABLE_RTSS_LAYER")
	disableRTSSLayer()
	got, set := os.LookupEnv("DISABLE_RTSS_LAYER")
	if windows := runtime.GOOS == "windows"; set != windows || (windows && got != "1") {
		t.Errorf("DISABLE_RTSS_LAYER = %q, set %v on %s", got, set, runtime.GOOS)
	}

	t.Setenv("DISABLE_RTSS_LAYER", "0")
	disableRTSSLayer()
	if got := os.Getenv("DISABLE_RTSS_LAYER"); got != "0" {
		t.Errorf("a value the user set was replaced: %q", got)
	}
}
