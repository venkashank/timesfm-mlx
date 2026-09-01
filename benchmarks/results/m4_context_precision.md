# M4 context and precision benchmark

This benchmark compares eager TimesFM 3 inference through PyTorch MPS and MLX
on a 32 GB, 10-core M4 MacBook Air. It uses the official
`google/timesfm-3.0-pytorch` checkpoint, batch 1, one variate, horizon 64, two
warm-ups per shape, and the median of five synchronized forecasts. Lower is
better; speedup is PyTorch latency divided by MLX latency.

| Precision | Context | PyTorch MPS | MLX | MLX speedup |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 32 | 145.58 ms | 40.20 ms | 3.62x |
| FP32 | 64 | 148.59 ms | 38.96 ms | 3.81x |
| FP32 | 128 | 160.18 ms | 38.33 ms | 4.18x |
| FP32 | 256 | 152.90 ms | 49.13 ms | 3.11x |
| FP32 | 512 | 165.31 ms | 66.89 ms | 2.47x |
| FP32 | 1,024 | 191.85 ms | 106.70 ms | 1.80x |
| FP32 | 2,048 | 235.90 ms | 164.24 ms | 1.44x |
| FP32 | 4,096 | 332.78 ms | 277.93 ms | 1.20x |
| FP32 | 8,192 | 606.81 ms | 523.18 ms | 1.16x |
| FP32 | 15,360 | 1,093.72 ms | 993.92 ms | 1.10x |
| FP16 | 32 | 159.75 ms | 25.99 ms | 6.15x |
| FP16 | 64 | 159.80 ms | 27.51 ms | 5.81x |
| FP16 | 128 | 168.86 ms | 35.63 ms | 4.74x |
| FP16 | 256 | 175.04 ms | 36.86 ms | 4.75x |
| FP16 | 512 | 180.11 ms | 49.18 ms | 3.66x |
| FP16 | 1,024 | 214.61 ms | 63.62 ms | 3.37x |
| FP16 | 2,048 | 261.51 ms | 119.95 ms | 2.18x |
| FP16 | 4,096 | 396.49 ms | 242.92 ms | 1.63x |
| FP16 | 8,192 | 663.48 ms | 489.15 ms | 1.36x |
| FP16 | 15,360 | 1,130.47 ms | 914.92 ms | 1.24x |
| BF16 | 32 | 166.89 ms | 26.09 ms | 6.40x |
| BF16 | 64 | 167.02 ms | 35.61 ms | 4.69x |
| BF16 | 128 | 173.58 ms | 34.99 ms | 4.96x |
| BF16 | 256 | 184.22 ms | 39.44 ms | 4.67x |
| BF16 | 512 | 188.10 ms | 48.97 ms | 3.84x |
| BF16 | 1,024 | 208.74 ms | 70.97 ms | 2.94x |
| BF16 | 2,048 | 245.59 ms | 116.74 ms | 2.10x |
| BF16 | 4,096 | 369.95 ms | 257.48 ms | 1.44x |
| BF16 | 8,192 | 660.41 ms | 513.29 ms | 1.29x |
| BF16 | 15,360 | 1,158.32 ms | 1,057.58 ms | 1.10x |

## Summary

| Precision | MLX wins | Geomean speedup | Range |
| --- | ---: | ---: | ---: |
| FP32 | 10/10 | 2.11x | 1.10–4.18x |
| FP16 | 10/10 | 3.00x | 1.24–6.15x |
| BF16 | 10/10 | 2.83x | 1.10–6.40x |

MLX FP32 differed from the PyTorch FP32 reference by at most `5.37e-6` across
this matrix. Against that same reference, the worst absolute differences were
`4.74e-3` for MLX FP16 and `3.69e-2` for MLX BF16. The synthetic inputs were
standard-normal, so these error magnitudes should not be generalized to every
dataset without an application-level quality check.

PyTorch low precision uses MPS autocast with FP32 weights because converting
the complete TimesFM model to FP16 or BF16 conflicts with the model's explicit
FP32 normalization and positional operations. MLX converts both weights and
matrix compute to the selected precision. Consequently this is a comparison of
each backend's supported low-precision inference path, not identical weight
storage. MLX active model memory was approximately 1.323 GB in FP32 and 0.661 GB
in FP16/BF16; PyTorch retained approximately 1.323 GB of active model memory in
all three modes. Native allocator metrics are backend-specific, and process RSS
does not capture all PyTorch MPS unified memory.

The complete samples, cold timings, allocator metrics, and numerical-error
statistics are in [`m4_context_precision.json`](m4_context_precision.json).
