# TimesFM 3.0 MLX Migration

This document tracks the native MLX backend for TimesFM 3.0. PyTorch remains
the numerical reference and the default backend until MLX parity and benchmark
targets are met.

## Goals

- Run TimesFM 3.0 inference natively on Apple silicon through MLX.
- Preserve the existing `TimesFM3Forecaster` and `TimesFM3Evaluator` behavior.
- Load the official Hugging Face safetensors without requiring PyTorch.
- Establish FP32 parity before enabling FP16 or quantized inference.
- Measure cold start, warm latency, throughput, and peak memory on an M4 Mac.

## Milestones

- [x] Create the `venkashank/timesfm-mlx` fork and a dedicated working clone.
- [x] Record architecture, acceptance criteria, and progress in this document.
- [x] Make Torch and MLX independently installable optional backends.
- [x] Extract the shared NumPy forecasting pipeline behind a backend protocol.
- [x] Port MLX utilities, normalization, dense layers, and attention primitives.
- [x] Port the mixing transformer, model forward pass, and CPM RevIN refinement.
- [x] Load Hugging Face configuration and safetensors directly into MLX.
- [x] Add public MLX backend selection without breaking the Torch API.
- [x] Add primitive, model, and end-to-end Torch-versus-MLX parity tests.
- [x] Add FP16 mixed precision while retaining statistical operations in FP32.
- [x] Add shape-bucketed compilation and M4 benchmark tooling.
- [x] Document installation, usage, performance, and known limitations.

## Correctness matrix

Parity coverage must include:

- univariate and multivariate inputs;
- variable and non-patch-aligned context lengths;
- left padding and explicit masks;
- past-only and past-and-future covariates;
- linear detrending, stitching, and CPM iterative RevIN;
- symmetric averaging, positivity clamping, and quantile sorting;
- 1, 8, and 32 variates; and
- short and multi-patch forecast horizons.

Tiny deterministic models are used in normal CI. Tests requiring the official
checkpoint are marked as integration tests because access is license-gated.

## Numerical policy

1. The first implementation is FP32 end to end.
2. Primitive tests compare intermediate tensors, not only final forecasts.
3. FP16 keeps detrending, running statistics, RevIN, CPM refinement, and final
   inverse normalization in FP32.
4. Quantization is considered only after the FP16 quality suite passes.
5. Attention scaling, RMSNorm epsilon, RoPE layout, mask polarity, and quantile
   order are treated as compatibility-sensitive behavior.

## Benchmark matrix

Compare PyTorch CPU, PyTorch MPS, and MLX using both cold and warmed runs:

| Dimension | Values |
| --- | --- |
| Batch | 1, 4, 16 |
| Context | 512, 2,048, 15,360 |
| Horizon | 64, 256, 1,024 |
| Variates | 1, 8, 32 |
| Precision | FP32, FP16 |

Report wall-clock latency, series per second, peak active memory, model-load
time, and compilation time separately. Benchmarks must synchronize MLX before
stopping timers and include sustained runs on fanless MacBook Air hardware.

## Progress log

- 2026-08-31: Created `venkashank/timesfm-mlx` from
  `google-research/timesfm`, cloned it beside the upstream checkout, and opened
  branch `feat/mlx-backend`.
- 2026-08-31: Inspected the TimesFM 3.0 Torch implementation and confirmed the
  public decode path is a non-autoregressive full-sequence pass. KV-cache parity
  is therefore deferred until after the forecast path is complete.
- 2026-08-31: Added lazy optional runtime imports and a shared backend boundary;
  all 42 original TimesFM 3 tests remained green.
- 2026-08-31: Ported the complete non-autoregressive model to MLX. Tiny-model
  Torch parity is within `3e-6`, including covariates, stitching, detrending,
  variate attention, and CPM iterative RevIN.
- 2026-08-31: Loaded the official 0.3B checkpoint directly from its original
  safetensors. An official-checkpoint comparison produced maximum absolute
  error `1.97e-6` and mean absolute error `4.61e-7` versus Torch FP32.
- 2026-08-31: Added FP16 mixed precision and isolated benchmark tooling. On an
  M4 MacBook Air, a warmed batch-1, context-512, horizon-64 forecast measured
  about 52 ms in MLX FP32 and 39 ms in MLX FP16 in an initial smoke benchmark.
- 2026-08-31: Added opt-in static-shape compilation with context buckets. At
  context 512 it reduced warmed FP16 latency from about 41 ms to 35 ms, with a
  roughly 14.6-second first-compilation cost, so compilation remains opt-in.
- 2026-08-31: Expanded M4 measurements. Batch 4 / 8 variates / context 2,048 /
  horizon 256 measured 1.19 s for MLX FP16 versus 1.57 s for Torch MPS. Maximum
  context 15,360 / horizon 1,024 measured 0.89 s for MLX FP16 versus 0.98 s for
  Torch MPS. FP16 versus MLX FP32 had mean absolute output difference `4.0e-4`
  and p99 absolute difference `1.56e-3` on a multivariate quality smoke test.
- 2026-08-31: Final validation passed 45 TimesFM 3 tests (including three new
  cross-framework parity tests) and 63 current-package tests. The official
  checkpoint also passed univariate, multivariate, evaluator, and direct
  backend smoke tests. Archived v1 collection remains blocked by its historical
  package-path mismatch noted below.

## Known limitations

- The MLX backend is inference-only and requires Apple-silicon macOS.
- FP16 is close to FP32 but is not bitwise identical; use FP32 for parity work.
- Compiled graphs have a substantial first-use cost and are cached only within
  the current process.
- KV-cache support is not ported because the public forecast path is currently
  non-autoregressive and does not use it.
- Weight quantization is intentionally deferred until a forecast-quality suite
  can establish safe error bounds.
- Archived `v1/tests` do not collect against the v3 package layout because they
  import the removed `timesfm.data_loader` module; this predates the MLX fork.

## License

The source remains Apache-2.0. TimesFM 3.0 pretrained weights remain governed
by the TimesFM non-commercial license; converted or cached MLX weights do not
change that license.
