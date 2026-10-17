package gputypes

import "testing"

func TestFeaturesContains(t *testing.T) {
	const unknownFeature Feature = 1 << 63

	tests := []struct {
		name     string
		features Features
		query    Feature
		want     bool
	}{
		{
			name:     "single",
			features: Features(FeatureDepthClipControl),
			query:    FeatureDepthClipControl,
			want:     true,
		},
		{
			name:     "all",
			features: Features(FeatureDepthClipControl | FeatureDepth32FloatStencil8),
			query:    FeatureDepthClipControl | FeatureDepth32FloatStencil8,
			want:     true,
		},
		{
			name:     "partial overlap",
			features: Features(FeatureDepthClipControl | FeatureDepth32FloatStencil8),
			query:    FeatureDepthClipControl | FeatureTextureCompressionBC,
			want:     false,
		},
		{
			name:     "disjoint",
			features: Features(FeatureDepthClipControl),
			query:    FeatureTextureCompressionBC,
			want:     false,
		},
		{
			name:     "zero query",
			features: Features(FeatureDepthClipControl),
			query:    0,
			want:     true,
		},
		{
			name:     "unknown query absent",
			features: Features(FeatureDepthClipControl),
			query:    unknownFeature,
			want:     false,
		},
		{
			name:     "unknown query present",
			features: Features(FeatureDepthClipControl) | Features(unknownFeature),
			query:    unknownFeature,
			want:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.features.Contains(tt.query); got != tt.want {
				t.Errorf("Features(%#x).Contains(%#x) = %v, want %v", uint64(tt.features), uint64(tt.query), got, tt.want)
			}
		})
	}
}
