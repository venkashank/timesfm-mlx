"""Benchmark TimesFM 3 PyTorch MPS and MLX across contexts and precisions.

The matrix runner launches one isolated worker per backend/precision pair, then
compares warmed public-API latency and numerical quality against PyTorch FP32.

Example:
  uv run --with psutil --extra torch --extra mlx \
    benchmarks/benchmark_context_precision.py --local-files-only \
    --output benchmarks/results/m4_context_precision.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Self

import numpy as np

from timesfm3 import ModelConfig, TimesFM3Forecaster

_ALL_CONTEXT_BUCKETS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 15360)
_ALL_PRECISIONS = ("float32", "float16", "bfloat16")
_ALL_BACKENDS = ("torch", "mlx")


class _PeakRssSampler:
  """Samples process RSS without putting polling work in the timed thread."""

  def __init__(self, interval_seconds: float = 0.002):
    try:
      import psutil
    except ImportError as exc:
      raise RuntimeError(
        "psutil is required for memory sampling; run with `uv run --with psutil`"
      ) from exc
    self._process = psutil.Process()
    self._interval_seconds = interval_seconds
    self._done = threading.Event()
    self.peak_bytes = self._process.memory_info().rss
    self._thread = threading.Thread(target=self._sample, daemon=True)

  def _sample(self) -> None:
    while not self._done.wait(self._interval_seconds):
      self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)

  def __enter__(self) -> Self:
    self._thread.start()
    return self

  def __exit__(self, *_: object) -> None:
    self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)
    self._done.set()
    self._thread.join()


def _csv_values(raw: str, cast: Any) -> tuple[Any, ...]:
  return tuple(cast(value.strip()) for value in raw.split(",") if value.strip())


def _synchronize(backend: str) -> None:
  if backend == "mlx":
    import mlx.core as mx

    mx.synchronize()
  else:
    import torch

    torch.mps.synchronize()


def _reset_memory(backend: str) -> None:
  if backend == "mlx":
    import mlx.core as mx

    mx.reset_peak_memory()
  else:
    import torch

    torch.mps.empty_cache()


def _memory_metrics(backend: str) -> dict[str, int]:
  if backend == "mlx":
    import mlx.core as mx

    return {
      "native_active_bytes": int(mx.get_active_memory()),
      "native_cache_bytes": int(mx.get_cache_memory()),
      "native_peak_bytes": int(mx.get_peak_memory()),
    }
  import torch

  return {
    "native_active_bytes": int(torch.mps.current_allocated_memory()),
    "native_driver_bytes": int(torch.mps.driver_allocated_memory()),
  }


def _contexts(context: int, batch: int, variates: int) -> list[np.ndarray]:
  rng = np.random.default_rng(20260831 + context)
  values = [
    rng.normal(size=(variates, context)).astype(np.float32) for _ in range(batch)
  ]
  if variates == 1:
    return [value[0] for value in values]
  return values


def _forecast(
  forecaster: TimesFM3Forecaster,
  contexts: list[np.ndarray],
  horizon: int,
) -> np.ndarray:
  outputs = list(
    forecaster.predict_batch(
      contexts,
      horizon=horizon,
      return_quantiles=True,
      use_symmetric_averaging=False,
      make_positive=False,
      sort_quantiles=False,
      use_znorm=False,
    )
  )
  packed = []
  for output in outputs:
    if output.forecast is None or output.quantiles is None:
      raise AssertionError("Benchmark forecast was incomplete")
    packed.append(
      np.concatenate([output.forecast[..., None], output.quantiles], axis=-1)
    )
  result = np.stack(packed)
  if not np.isfinite(result).all():
    raise AssertionError("Benchmark forecast contained non-finite values")
  return result


def _worker(args: argparse.Namespace) -> None:
  contexts = _csv_values(args.contexts, int)
  if args.backend == "torch" and args.device != "mps":
    raise ValueError("The comparison worker requires PyTorch MPS")

  started = time.perf_counter()
  forecaster = TimesFM3Forecaster(
    ModelConfig(
      checkpoint_path=args.checkpoint,
      backend=args.backend,
      dtype=args.precision,
      device=args.device if args.backend == "torch" else None,
      per_core_batch_size=args.batch,
      local_files_only=args.local_files_only,
      compile=args.compile_mlx and args.backend == "mlx",
    )
  )
  _synchronize(args.backend)
  load_seconds = time.perf_counter() - started

  rows: list[dict[str, Any]] = []
  forecasts: dict[str, np.ndarray] = {}
  for context in contexts:
    inputs = _contexts(context, args.batch, args.variates)
    _reset_memory(args.backend)

    before = time.perf_counter()
    latest = _forecast(forecaster, inputs, args.horizon)
    _synchronize(args.backend)
    first_seconds = time.perf_counter() - before
    for _ in range(max(0, args.warmups - 1)):
      latest = _forecast(forecaster, inputs, args.horizon)
      _synchronize(args.backend)

    samples = []
    with _PeakRssSampler() as rss_sampler:
      for _ in range(args.repeats):
        before = time.perf_counter()
        latest = _forecast(forecaster, inputs, args.horizon)
        _synchronize(args.backend)
        samples.append(time.perf_counter() - before)

    median_seconds = statistics.median(samples)
    sorted_samples = sorted(samples)
    p90_index = math.ceil(0.9 * len(sorted_samples)) - 1
    row = {
      "backend": args.backend,
      "precision": args.precision,
      "precision_mode": (
        "full_float32"
        if args.precision == "float32"
        else (
          "autocast_with_float32_weights"
          if args.backend == "torch"
          else "low_precision_weights_and_compute"
        )
      ),
      "device": str(forecaster.device),
      "compiled": bool(args.compile_mlx and args.backend == "mlx"),
      "context": context,
      "horizon": args.horizon,
      "batch": args.batch,
      "variates": args.variates,
      "load_seconds": load_seconds,
      "first_forecast_seconds": first_seconds,
      "warm_median_seconds": median_seconds,
      "warm_min_seconds": min(samples),
      "warm_p90_seconds": sorted_samples[p90_index],
      "examples_per_second": args.batch / median_seconds,
      "input_points_per_second": (
        args.batch * args.variates * context / median_seconds
      ),
      "process_peak_rss_bytes": rss_sampler.peak_bytes,
      "samples_seconds": samples,
    }
    row.update(_memory_metrics(args.backend))
    rows.append(row)
    forecasts[f"context_{context}"] = latest.astype(np.float32)

  Path(args.worker_output).write_text(json.dumps(rows, indent=2, sort_keys=True))
  np.savez(args.forecast_output, **forecasts)


def _run_worker(
  backend: str,
  precision: str,
  worker_output: Path,
  forecast_output: Path,
  args: argparse.Namespace,
) -> None:
  command = [
    sys.executable,
    str(Path(__file__).resolve()),
    "--worker",
    "--backend",
    backend,
    "--precision",
    precision,
    "--device",
    args.device,
    "--contexts",
    args.contexts,
    "--horizon",
    str(args.horizon),
    "--batch",
    str(args.batch),
    "--variates",
    str(args.variates),
    "--warmups",
    str(args.warmups),
    "--repeats",
    str(args.repeats),
    "--checkpoint",
    args.checkpoint,
    "--worker-output",
    str(worker_output),
    "--forecast-output",
    str(forecast_output),
  ]
  if args.local_files_only:
    command.append("--local-files-only")
  if args.compile_mlx:
    command.append("--compile-mlx")
  environment = os.environ.copy()
  environment["PYTHONHASHSEED"] = "0"
  print(f"benchmarking {backend}/{precision}...", file=sys.stderr, flush=True)
  subprocess.run(command, env=environment, check=True)


def _git_commit() -> str | None:
  result = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=Path(__file__).resolve().parents[1],
    text=True,
    capture_output=True,
    check=False,
  )
  return result.stdout.strip() if result.returncode == 0 else None


def _git_worktree_dirty() -> bool | None:
  result = subprocess.run(
    ["git", "status", "--porcelain"],
    cwd=Path(__file__).resolve().parents[1],
    text=True,
    capture_output=True,
    check=False,
  )
  return bool(result.stdout.strip()) if result.returncode == 0 else None


def _main(args: argparse.Namespace) -> None:
  contexts = _csv_values(args.contexts, int)
  precisions = _csv_values(args.precisions, str)
  backends = _csv_values(args.backends, str)
  invalid_contexts = {
    context for context in contexts if context < 32 or context > 15360
  }
  invalid_precisions = set(precisions) - set(_ALL_PRECISIONS)
  invalid_backends = set(backends) - set(_ALL_BACKENDS)
  if invalid_contexts or invalid_precisions or invalid_backends:
    raise ValueError(
      "Unsupported matrix values: "
      f"contexts={invalid_contexts}, precisions={invalid_precisions}, "
      f"backends={invalid_backends}"
    )
  if set(backends) != set(_ALL_BACKENDS):
    raise ValueError("Both torch and mlx are required for a comparison")
  if "float32" not in precisions:
    raise ValueError("float32 is required as the quality reference")

  all_rows: list[dict[str, Any]] = []
  forecast_paths: dict[tuple[str, str], Path] = {}
  with tempfile.TemporaryDirectory(prefix="timesfm-context-precision-") as directory:
    temporary = Path(directory)
    for precision in precisions:
      for backend in backends:
        label = f"{backend}-{precision}"
        worker_output = temporary / f"{label}.json"
        forecast_output = temporary / f"{label}.npz"
        _run_worker(backend, precision, worker_output, forecast_output, args)
        all_rows.extend(json.loads(worker_output.read_text()))
        forecast_paths[(backend, precision)] = forecast_output

    with np.load(forecast_paths[("torch", "float32")]) as reference:
      for (backend, precision), forecast_path in forecast_paths.items():
        with np.load(forecast_path) as candidate:
          for context in contexts:
            key = f"context_{context}"
            difference = np.abs(reference[key] - candidate[key])
            row = next(
              value
              for value in all_rows
              if value["backend"] == backend
              and value["precision"] == precision
              and value["context"] == context
            )
            row["vs_torch_fp32_max_abs"] = float(difference.max())
            row["vs_torch_fp32_mean_abs"] = float(difference.mean())
            row["vs_torch_fp32_p99_abs"] = float(np.quantile(difference, 0.99))

  comparisons = []
  summaries: dict[str, dict[str, float | int]] = {}
  for precision in precisions:
    speedups = []
    for context in contexts:
      torch_row = next(
        row
        for row in all_rows
        if row["backend"] == "torch"
        and row["precision"] == precision
        and row["context"] == context
      )
      mlx_row = next(
        row
        for row in all_rows
        if row["backend"] == "mlx"
        and row["precision"] == precision
        and row["context"] == context
      )
      speedup = torch_row["warm_median_seconds"] / mlx_row["warm_median_seconds"]
      speedups.append(speedup)
      comparisons.append(
        {
          "precision": precision,
          "context": context,
          "torch_median_seconds": torch_row["warm_median_seconds"],
          "mlx_median_seconds": mlx_row["warm_median_seconds"],
          "mlx_speedup": speedup,
          "winner": "mlx" if speedup > 1 else "torch",
        }
      )
    summaries[precision] = {
      "mlx_wins": sum(speedup > 1 for speedup in speedups),
      "contexts": len(speedups),
      "geomean_mlx_speedup": math.exp(
        statistics.mean(math.log(speedup) for speedup in speedups)
      ),
      "min_mlx_speedup": min(speedups),
      "max_mlx_speedup": max(speedups),
    }

  report = {
    "schema_version": 1,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "checkpoint": args.checkpoint,
    "git_commit": _git_commit(),
    "git_worktree_dirty": _git_worktree_dirty(),
    "hardware": {
      "machine": platform.machine(),
      "platform": platform.platform(),
      "processor": platform.processor(),
    },
    "methodology": {
      "contexts": contexts,
      "precisions": precisions,
      "backends": backends,
      "torch_device": args.device,
      "horizon": args.horizon,
      "batch": args.batch,
      "variates": args.variates,
      "warmups_per_shape": args.warmups,
      "repeats_per_shape": args.repeats,
      "mlx_compiled": args.compile_mlx,
      "timing": "wall clock with explicit backend synchronization",
      "quality_reference": "PyTorch MPS float32 output from the same inputs",
      "precision_modes": {
        "torch": "FP32 weights with MPS autocast for FP16/BF16",
        "mlx": "weights and matrix compute converted to the selected precision",
      },
      "memory_note": (
        "Native allocator counters are backend-specific. Process RSS does not include "
        "all PyTorch MPS unified-memory allocations, so compare memory within a backend."
      ),
    },
    "summary": summaries,
    "comparisons": comparisons,
    "runs": all_rows,
  }
  rendered = json.dumps(report, indent=2, sort_keys=True)
  if args.output:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n")
    print(f"wrote {output}", file=sys.stderr)
  if not args.quiet:
    print(rendered)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--contexts", default=",".join(map(str, _ALL_CONTEXT_BUCKETS)))
  parser.add_argument("--precisions", default=",".join(_ALL_PRECISIONS))
  parser.add_argument("--backends", default=",".join(_ALL_BACKENDS))
  parser.add_argument("--device", default="mps")
  parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
  parser.add_argument("--horizon", type=int, default=64)
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--variates", type=int, default=1)
  parser.add_argument("--warmups", type=int, default=2)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--compile-mlx", action="store_true")
  parser.add_argument("--local-files-only", action="store_true")
  parser.add_argument("--output")
  parser.add_argument("--quiet", action="store_true")
  parser.add_argument("--worker", action="store_true")
  parser.add_argument("--backend", choices=_ALL_BACKENDS)
  parser.add_argument("--precision", choices=_ALL_PRECISIONS)
  parser.add_argument("--worker-output")
  parser.add_argument("--forecast-output")
  args = parser.parse_args()
  if args.worker and not all(
    (args.backend, args.precision, args.worker_output, args.forecast_output)
  ):
    parser.error(
      "worker mode requires --backend, --precision, --worker-output, and "
      "--forecast-output"
    )
  if args.warmups < 1 or args.repeats < 1:
    parser.error("--warmups and --repeats must be positive")
  return args


if __name__ == "__main__":
  parsed = _parse_args()
  if parsed.worker:
    _worker(parsed)
  else:
    _main(parsed)
