//go:build !(js && wasm)

package testutil_test

import (
	"testing"

	"github.com/roffe/gowgpu/core"
	"github.com/roffe/gowgpu/core/testutil"
	"github.com/roffe/gowgpu/gputypes"
)

func TestAssertNoError(t *testing.T) {
	t.Parallel()
	testutil.AssertNoError(t, nil)
}

func TestAssertFeatureError(t *testing.T) {
	t.Parallel()
	err := core.RequireFeature(gputypes.Features(0), gputypes.FeaturePushConstants, "CreatePipelineLayout")
	testutil.AssertFeatureError(t, err, gputypes.FeaturePushConstants, "CreatePipelineLayout")
}
