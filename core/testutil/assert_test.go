//go:build !(js && wasm)

package testutil_test

import (
	"testing"

	"github.com/gogpu/gputypes"
	"github.com/roffe/wgpu/core"
	"github.com/roffe/wgpu/core/testutil"
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
