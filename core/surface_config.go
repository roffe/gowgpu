//go:build !(js && wasm)

package core

import (
	"errors"
	"fmt"
	"slices"

	"github.com/roffe/gowgpu/gputypes"
	"github.com/roffe/gowgpu/hal"
)

// ErrSurfaceConfiguration is what ValidateSurfaceConfiguration wraps for a
// configuration the surface does not support.
var ErrSurfaceConfiguration = errors.New("unsupported surface configuration")

// ValidateSurfaceConfiguration checks config against what the surface
// supports on the device's adapter: a format, present mode and alpha mode
// from caps (an undefined present mode and the automatic alpha mode leave
// the choice to the backend) and a size of at least 1x1 and at most
// MaxTextureDimension2D.
//
// Reference: wgpu-core device/surface_config.rs
// validate_surface_configuration
func ValidateSurfaceConfiguration(config *hal.SurfaceConfiguration, caps *hal.SurfaceCapabilities, limits gputypes.Limits) error {
	switch {
	case config.Width == 0 || config.Height == 0:
		return fmt.Errorf("%w: size %dx%d", ErrSurfaceConfiguration, config.Width, config.Height)
	case config.Width > limits.MaxTextureDimension2D || config.Height > limits.MaxTextureDimension2D:
		return fmt.Errorf("%w: size %dx%d is over MaxTextureDimension2D %d", ErrSurfaceConfiguration, config.Width, config.Height, limits.MaxTextureDimension2D)
	case !slices.Contains(caps.Formats, config.Format):
		return fmt.Errorf("%w: format %v, the surface has %v", ErrSurfaceConfiguration, config.Format, caps.Formats)
	case config.PresentMode != gputypes.PresentModeUndefined && !slices.Contains(caps.PresentModes, config.PresentMode):
		return fmt.Errorf("%w: present mode %v, the surface has %v", ErrSurfaceConfiguration, config.PresentMode, caps.PresentModes)
	case config.AlphaMode != gputypes.CompositeAlphaModeAuto && !slices.Contains(caps.AlphaModes, config.AlphaMode):
		return fmt.Errorf("%w: alpha mode %v, the surface has %v", ErrSurfaceConfiguration, config.AlphaMode, caps.AlphaModes)
	}
	return nil
}
