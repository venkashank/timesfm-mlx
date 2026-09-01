"""Benchmark one TimesFM 3 runtime in an isolated process.

Examples:
  uv run --extra mlx benchmarks/benchmark_mlx.py --backend mlx --dtype float16
  uv run --extra torch benchmarks/benchmark_mlx.py --backend torch --device mps
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from timesfm3 import ModelConfig, TimesFM3Forecaster


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--backend", choices=["torch", "mlx"], required=True)
  parser.add_argument(
    "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
  )
  parser.add_argument("--device", default=None)
  parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--context", type=int, default=512)
  parser.add_argument("--horizon", type=int, default=64)
  parser.add_argument("--variates", type=int, default=1)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--local-files-only", action="store_true")
  parser.add_argument("--compile", action="store_true")
  args = parser.parse_args()

  rng = np.random.default_rng(42)
  contexts = [
    rng.normal(size=(args.variates, args.context)).astype(np.float32)
    for _ in range(args.batch)
  ]
  if args.variates == 1:
    contexts = [value[0] for value in contexts]

  started = time.perf_counter()
  forecaster = TimesFM3Forecaster(
    ModelConfig(
      checkpoint_path=args.checkpoint,
      backend=args.backend,
      dtype=args.dtype,
      device=args.device,
      per_core_batch_size=args.batch,
      local_files_only=args.local_files_only,
      compile=args.compile,
    )
  )
  loaded = time.perf_counter()
  list(forecaster.predict_batch(contexts, horizon=args.horizon))
  warmed = time.perf_counter()
  samples = []
  for _ in range(args.repeats):
    before = time.perf_counter()
    list(forecaster.predict_batch(contexts, horizon=args.horizon))
    samples.append(time.perf_counter() - before)

  result = {
    "backend": args.backend,
    "dtype": args.dtype,
    "device": str(forecaster.device),
    "batch": args.batch,
    "context": args.context,
    "horizon": args.horizon,
    "variates": args.variates,
    "compile": args.compile,
    "load_seconds": loaded - started,
    "first_forecast_seconds": warmed - loaded,
    "warm_median_seconds": statistics.median(samples),
    "warm_min_seconds": min(samples),
    "series_per_second": args.batch / statistics.median(samples),
  }
  if args.backend == "mlx":
    import mlx.core as mx

    result["peak_memory_bytes"] = mx.get_peak_memory()
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
