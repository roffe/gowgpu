# Validation testing

How the validation in core is tested: feature gates, texture usage rules and the rest of the "Phase C" checks (the `VAL-C*` IDs in the code).

## Goals

1. Fast, deterministic unit tests. Most of the validation coverage needs no GPU.
2. Typed errors. Negative tests assert `*core.FeatureError`, `*core.CreateTextureError` and so on, not just `err != nil`.
3. A complete registry. A new feature gate goes into `AllFeatureRequirements` once and gets wired into the tests from there.
4. A thin smoke layer. `wgpu_test` checks the public API path on a real HAL, which is whatever GPU the machine has.

## Two tiers

| Tier | Package | Backend | Use for |
|------|---------|---------|---------|
| 1 | `core/` + `core/testutil/` | None, calls `Validate*` directly | Feature gates, descriptor validation, texture matrix |
| 2 | `wgpu` (`wgpu_test`) | The machine's GPU through `hal/allbackends` | Encode -> `Finish()` error paths, `CreateQuerySet`, multi-draw |

Rule: the bulk of the feature coverage lives in tier 1. Tier 2 gets one or two smoke tests per feature group, not 25 copies of the same case.

Tier 2 needs an adapter. On a machine without one the tests that need a device skip ("no adapters available"). To run them anyway, see [Running tests](#running-tests) for lavapipe.

## ValidationEnv (P9)

Location: [`core/testutil/validation_env.go`](../core/testutil/validation_env.go)

A small fixture builder, not a DI framework. One per `t.Run`, safe with `t.Parallel()`.

```go
testutil.NewValidationEnv(t)                              // no optional features
testutil.NewValidationEnv(t, testutil.WithFeature(f))     // one feature
testutil.NewValidationEnv(t, testutil.WithFeatures(fs))   // a feature set
testutil.NewValidationEnv(t, testutil.WithLimits(limits)) // custom limits
```

### Action helpers

Each method calls the matching core `Validate*` function and returns its error, for table-driven negative tests:

| Method | Validates | VAL-C |
|--------|-----------|-------|
| `CreatePipelineLayoutWithPushConstants()` | Push constant ranges | C4 |
| `CreateTextureWithFormat(format, usage)` | Format feature gates | C6..C11 |
| `CreateRenderPipelineUnclippedDepth()` | Unclipped depth | C5 |
| `CreateShaderModuleWithWGSL(source)` | WGSL f16 and subgroups | C12, C19, C20 |
| `CreateQuerySetTimestamp()` | Timestamp queries | C24 |
| `CreateBindGroupWithR32FloatTexture()` | Float32 filterable | C21 |

To add a scenario: one method on `ValidationEnv`, then one `featureGateCase` row.

## Assertion helpers (P2)

Location: [`core/testutil/assert.go`](../core/testutil/assert.go)

```go
testutil.AssertFeatureError(t, err, gputypes.FeaturePushConstants, "CreatePipelineLayout")
testutil.AssertNoError(t, err)
```

Use the typed assertions in negative tests, not a bare `if err == nil`.

## Registry-driven feature gates (P10)

Location: [`core/feature_gate_cases_test.go`](../core/feature_gate_cases_test.go)

```go
type featureGateCase struct {
	name     string
	feature  gputypes.Feature
	resource string
	act      func(env *testutil.ValidationEnv) error
}
```

`TestFeatureGates` runs two subtests per case:

- `.../negative`: `NewValidationEnv(t)` with no features, expects a `*core.FeatureError`
- `.../positive`: `NewValidationEnv(t, WithFeature(tc.feature))`, expects nil

`TestFeatureGateCasesCoverRegistryTargets` checks that every feature in `wiredFeatureGateTargets` has a case in `featureGateCases`.

### Feature registry

Location: [`core/feature_validate.go`](../core/feature_validate.go)

- `RequireFeature(features, feature, resource)` is the one gate helper (VAL-C0).
- `AllFeatureRequirements` lists all 25 features with their `Resource` entry point and a Rust wgpu-core reference.
- `TestFeatureGateRegistryComplete` (in `feature_validate_test.go`) checks for duplicates and empty metadata, and wants exactly 25 entries. Bump that number when you add one.

## Plain Validate* unit tests (P1)

Location: [`core/validate_test.go`](../core/validate_test.go), [`core/feature_format_test.go`](../core/feature_format_test.go) and friends.

For descriptor-only validation without a device:

```go
err := core.ValidateTextureDescriptor(desc, gputypes.DefaultLimits(), gputypes.Features(0))
if !core.IsFeatureError(err) { ... }
```

Build `desc` with `validTextureDesc()` or one of the other factories in `validate_test.go`.

## Texture usage matrix (P3 + P12)

Location: [`core/track/texture_compat_matrix_test.go`](../core/track/texture_compat_matrix_test.go)

Every pair of single-bit `TextureUses` goes through `IsCompatible` and gets compared with an independent `referenceCompatible` function (P12, the golden reference).

The creation-time checks are in [`core/texture_usage_validate.go`](../core/texture_usage_validate.go) (VAL-C14..C18), called from `ValidateTextureDescriptor`.

## Public API smoke tests (P6)

Location: [`wgpu_feature_gate_test.go`](../wgpu_feature_gate_test.go)

```go
newDeviceWithoutFeatures(t)                  // RequiredFeatures: 0
newDeviceWithFeatures(t, features)           // skips if the adapter lacks a feature
newEncoderWithRenderPassForDevice(t, device) // encoder plus render pass on that device
```

These go through `newAdapter`, which fails the test (it doesn't skip) when there's no adapter.

Typical flow for a draw-time gate:

1. Create a device without the feature
2. Record the draw, indirect or count-buffer call on a render pass
3. `encoder.Finish()` -> expect a `*core.FeatureError`

Gates that are only tested here, not in `featureGateCases`:

- `FeatureMultiDrawIndirect`: `MultiDrawIndirect` with `drawCount > 1`.
- `FeatureMultiDrawIndirectCount`: the count-buffer indirect calls (`MultiDrawIndirectCount`, `MultiDrawIndexedIndirectCount`).
- `FeatureIndirectFirstInstance`: the tests check that direct `Draw` and `DrawIndexed` with `firstInstance != 0` work without it, since WebGPU only gates indirect draws. Nothing checks the indirect case yet.

## Multi-draw indirect count (VAL-C2)

The count-buffer indirect draws are split over three layers, and each layer has focused unit tests that need no GPU:

| Layer | Code | Tests |
|-------|------|-------|
| HAL helper | [`hal/indirect_count.go`](../hal/indirect_count.go): `RecordIndirectCountMax` | [`hal/indirect_count_test.go`](../hal/indirect_count_test.go) |
| Core encoder | [`core/command_indirect_count.go`](../core/command_indirect_count.go): `indirectCountInactive`, `forwardIndirectCount` | [`core/indirect_count_test.go`](../core/indirect_count_test.go) |
| Public API | [`renderpass_indirect_count.go`](../renderpass_indirect_count.go): `executeIndirectCountDraw` (feature gate, buffer validation through `validateIndirectCountBuffers` in `renderpass_native.go`, HAL record) | [`renderpass_indirect_count_test.go`](../renderpass_indirect_count_test.go) |

### NewTestBuffer (validation-only buffers)

Location: [`export_test.go`](../export_test.go)

Makes a `*Buffer` with core state only, no HAL buffer. Use it in package-internal tests that check size, usage, offsets and labels without a device:

```go
pass := &RenderPassEncoder{}
indirect := NewTestBuffer(64, gputypes.BufferUsageIndirect, "indirect")
count := NewTestBuffer(4, gputypes.BufferUsageIndirect, "count")
err := pass.validateIndirectCountBuffers(indirect, 0, count, 0, 2, drawIndirectRecordSize)
```

### What each test file covers

- `hal/indirect_count_test.go`: `RecordIndirectCountMax` passes `maxDrawCount` on, a zero count is a no-op, and the count buffer isn't passed to the record callback.
- `core/indirect_count_test.go`: `indirectCountInactive` (ended pass, zero count). Core `DrawIndirectCount` and `DrawIndexedIndirectCount` forward to the HAL once and skip a nil buffer, an ended pass or a zero count.
- `renderpass_indirect_count_test.go`: table-driven `validateIndirectCountBuffers` (alignment, usage, span overrun), `validateIndexedIndirectCountPreconditions` (missing index buffer, strip format mismatch) and `executeIndirectCountDraw`.

The tier 2 smoke tests for the gate are `TestFeatureGate_MultiDrawIndirectCount_NoFeature` and `TestFeatureGate_MultiDrawIndirectCount_WithFeature` in `wgpu_feature_gate_test.go` (encode -> `Finish()` -> `*core.FeatureError`).

## Test pattern reference (P1..P14)

| ID | Pattern | When to use | Primary file |
|----|---------|-------------|--------------|
| P1 | Valid factory | Plain `Validate*` | `validate_test.go` |
| P2 | Typed error assert | Every negative test | `core/testutil/assert.go` |
| P3 | Table-driven + `t.Parallel()` | Combinatorial matrices | `texture_compat_matrix_test.go` |
| P4 | Mock HAL | Core tests that need a HAL device | `core/device_hal_test.go` (`mockHALDevice`) |
| P5 | Device tracker | Submit-time texture transitions | `core/device_tracker_test.go` |
| P6 | `newDeviceWithFeatures` | Public API smoke | `wgpu_feature_gate_test.go` |
| P7 | Mock context | Package-local (ray tracing) | `internal/raytracing/` (`mockDeviceContext`) |
| P8 | VAL-C* ID in code and test | Every new check | production and test comments |
| P9 | ValidationEnv | Create and draw paths that need features | `core/testutil/validation_env.go` |
| P10 | Registry-driven | Bulk feature gates | `core/feature_gate_cases_test.go` |
| P11 | Guard / completeness | Registry and cases stay in sync | `core/feature_validate_test.go` |
| P12 | Golden reference function | Matrix checks itself | `texture_compat_matrix_test.go` |
| P13 | `NewTestBuffer` | Public validation without a HAL | `export_test.go`, `renderpass_indirect_count_test.go` |
| P14 | Layered indirect-count tests | Draw helpers split over layers | `hal/`, `core/`, `renderpass_indirect_count_test.go` |

## Adding a new feature gate

1. Registry: add a row to `AllFeatureRequirements` in `core/feature_validate.go`, and bump the count in `TestFeatureGateRegistryComplete`.
2. Production: call `RequireFeature` at the right entry point, with a `VAL-C*` comment.
3. ValidationEnv: add a helper method if the gate can be tested without a full device encode.
4. P10 case: add a `featureGateCase`, and extend `wiredFeatureGateTargets` if it applies.
5. P1 unit test: if the gate lives in a plain `Validate*` function.
6. P6 smoke: if the gate is draw-time or public API only (`wgpu_feature_gate_test.go`).
7. CHANGELOG: a line under `[Unreleased]`.

## Running tests

```bash
# Everything outside the root package, no GPU needed
go test ./core/... ./hal/... ./internal/... ./gputypes/...

# Core validation only (fast)
go test ./core/... -count=1

# Feature gate registry
go test ./core/... -run TestFeatureGates -v

# Texture matrix
go test ./core/track/... -run IsCompatible -v

# Public API smoke tests, on this machine's GPU
go test . -run FeatureGate -v

# Indirect count: hal + core + public API
go test ./hal/... ./core/... . -run 'IndirectCount|ValidateIndirect|RecordIndirectCount' -count=1

# Lint
golangci-lint run --timeout=5m
```

The root package (`go test .`, and `go test ./...` which includes it) runs on whatever GPU the machine has, through `hal/allbackends`. On a machine without one, the tests that need a device skip.

State on Vulkan right now (NVIDIA, Linux):

- `TestFeatureGate_CreateQuerySet_WithFeature` skips, because the Vulkan adapter doesn't report `FeatureTimestampQuery`.

### Vulkan validation layer and lavapipe, without root

Neither needs to be installed system-wide. Download Arch's `vulkan-validation-layers` or `vulkan-swrast` package (`pacman -Sp` prints the URL without root), extract it into a scratch dir, and point `library_path` in the JSON manifest at the extracted `.so` (absolute path).

```bash
# Khronos validation layer. With no debug messenger registered it prints
# to stdout, so grep for "Validation Error".
VK_LAYER_PATH=<dir>/usr/share/vulkan/explicit_layer.d \
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
go test .

# lavapipe, Mesa's CPU Vulkan driver. Also the way to run the root
# package tests on a machine without a GPU.
VK_ICD_FILENAMES=<dir>/usr/share/vulkan/icd.d/lvp_icd.json go test .

# Act like a machine without Vulkan: point the loader at a manifest
# whose library_path doesn't exist.
VK_ICD_FILENAMES=/path/to/bogus_icd.json go test .
```

The validation layer currently reports image layout errors from the Vulkan layout tracking.

An app can also turn the layer on itself with `wgpu.InstanceDescriptor{Flags: gputypes.InstanceFlagsDebug}`. That enables `VK_LAYER_KHRONOS_validation` when the loader can find it, and silently skips it when it can't.

### Build tags

- `core/testutil` and most of the validation tests: `!(js && wasm)`, so none of it builds for the browser.
- The root package files also carry `!rust`, a leftover from gogpu's wgpu-native backend, which is gone.
- A few tests sit behind `-tags integration`: `core/hal_integration_test.go`, `depth_only_pass_repro_test.go`, `hal/gles/integration_test.go` (Linux), and on macOS `arraylength_repro_test.go`, `indirect_count_integration_test.go` and `metal_icb_benchmark_test.go`.

## Key source files

| Area | Files |
|------|-------|
| Infra | `core/feature_validate.go`, `core/testutil/*` |
| Format gates | `core/feature_format.go`, `core/validate.go` |
| Float32 filterable | `core/format_filterable.go` |
| Shader gates | `core/shader_features.go`, `core/spirv_capabilities.go` |
| Texture usage | `core/texture_usage_validate.go`, `core/track/texture_compat_matrix_test.go` |
| Query gates | `core/query_validate.go`, `query_native.go` |
| Indirect count | `core/command_indirect_count.go`, `renderpass_indirect_count.go`, `hal/indirect_count.go` |
| Wiring | `device_native.go`, `renderpass_native.go` |
| Tests | `core/feature_*_test.go`, `core/indirect_count_test.go`, `hal/indirect_count_test.go`, `renderpass_indirect_count_test.go`, `wgpu_feature_gate_test.go` |

## References

- [Go wiki: Table-Driven Tests](https://go.dev/wiki/TableDrivenTests)
- [Google Go Style: assertion helpers](https://google.github.io/styleguide/go/decisions#assertion-helper-functions)
- Rust wgpu-core validation: `wgpu-core/device/resource.rs`, `wgpu-core/command/render.rs`
- Architecture overview: [`ARCHITECTURE.md`](ARCHITECTURE.md)
