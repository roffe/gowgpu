//go:build !(js && wasm)

package core

import (
	"errors"
	"fmt"
	"math/bits"
	"reflect"
	"strings"

	"github.com/roffe/gowgpu/gputypes"
)

// ErrLimitExceeded is what ResolveRequiredLimits wraps for a limit the
// adapter can't give.
var ErrLimitExceeded = errors.New("limit better than the adapter's")

// ResolveRequiredLimits returns the limits of a device asked for with
// required on an adapter with the limits adapter: required's limits, and the
// adapter's for those it leaves zero. A maximum above the adapter's, or an
// alignment below it or not a power of two, is an error.
//
// Reference: wgpu-core instance.rs Adapter::create_device_and_queue
// (check_limits), WebGPU requestDevice
func ResolveRequiredLimits(adapter, required gputypes.Limits) (gputypes.Limits, error) {
	resolved := required
	have := reflect.ValueOf(adapter)
	out := reflect.ValueOf(&resolved).Elem()
	for i := range out.NumField() {
		name, field := out.Type().Field(i).Name, out.Field(i)
		limit := have.Field(i).Uint()
		asked := field.Uint()
		switch {
		case asked == 0:
			field.SetUint(limit)
		case strings.HasPrefix(name, "Min"):
			if asked < limit {
				return gputypes.Limits{}, fmt.Errorf("%w: %s %d, the adapter aligns to %d", ErrLimitExceeded, name, asked, limit)
			}
			if bits.OnesCount64(asked) != 1 {
				return gputypes.Limits{}, fmt.Errorf("%w: %s %d is not a power of two", ErrLimitExceeded, name, asked)
			}
		case asked > limit:
			return gputypes.Limits{}, fmt.Errorf("%w: %s %d, the adapter has %d", ErrLimitExceeded, name, asked, limit)
		}
	}
	return resolved, nil
}
