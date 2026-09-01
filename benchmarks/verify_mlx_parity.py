"""Verify fork MLX forecasts against the untouched upstream PyTorch checkout.

The verifier launches each implementation in an isolated process so importing
the fork cannot accidentally replace the upstream oracle. Both workers use the
same official checkpoint and deterministically generated inputs.

Example:
  uv run --extra torch --extra mlx benchmarks/verify_mlx_parity.py \
    --upstream-root /path/to/google-research/timesfm --local-files-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_CORE_FILES = (
  "configs.py",
  "model.py",
  "transformer.py",
  "transformations.py",
  "util.py",
  "dense.py",
  "normalization.py",
  "cpm_revin_refine.py",
  "evaluator.py",
)


def _cases() -> list[tuple[str, dict[str, Any]]]:
  rng = np.random.default_rng(20260831)
  variable = [
    rng.normal(size=33).astype(np.float32),
    rng.normal(size=96).astype(np.float32),
    rng.normal(size=513).astype(np.float32),
  ]
  variable[0][7] = np.nan
  variable[1][20:23] = np.nan

  multivariate = [
    rng.normal(size=(4, 257)).astype(np.float32),
    rng.normal(size=(4, 257)).astype(np.float32),
  ]
  past_only = [
    rng.normal(size=(2, 257)).astype(np.float32),
    rng.normal(size=(2, 257)).astype(np.float32),
  ]
  past_future = [
    rng.normal(size=(3, 353)).astype(np.float32),
    rng.normal(size=(3, 353)).astype(np.float32),
  ]

  positive = [
    np.abs(rng.normal(size=127)).astype(np.float32),
    np.abs(rng.normal(size=255)).astype(np.float32),
  ]
  return [
    (
      "variable_univariate",
      {
        "contexts": variable,
        "horizon": 65,
        "return_quantiles": True,
        "use_symmetric_averaging": False,
        "make_positive": False,
      },
    ),
    (
      "multivariate_covariates",
      {
        "contexts": multivariate,
        "horizon": 96,
        "past_only_covariates": past_only,
        "past_future_covariates": past_future,
        "return_quantiles": True,
        "use_symmetric_averaging": False,
        "make_positive": False,
      },
    ),
    (
      "postprocessing",
      {
        "contexts": positive,
        "horizon": 17,
        "return_quantiles": True,
        "use_symmetric_averaging": True,
        "make_positive": True,
        "sort_quantiles": True,
        "use_znorm": True,
        "padding_mode": "edge",
      },
    ),
    (
      "thirty_two_variates",
      {
        "contexts": [rng.normal(size=(32, 64)).astype(np.float32)],
        "horizon": 8,
        "return_quantiles": True,
        "use_symmetric_averaging": False,
        "make_positive": False,
      },
    ),
    (
      "long_context_horizon",
      {
        "contexts": [rng.normal(size=2048).astype(np.float32)],
        "horizon": 256,
        "return_quantiles": True,
        "use_symmetric_averaging": False,
        "make_positive": False,
      },
    ),
  ]


def _worker(args: argparse.Namespace) -> None:
  from timesfm3 import ModelConfig, TimesFM3Evaluator

  config_kwargs: dict[str, Any] = {
    "checkpoint_path": args.checkpoint,
    "per_core_batch_size": 4,
    "local_files_only": args.local_files_only,
  }
  if args.worker == "torch":
    config_kwargs["device"] = "cpu"
  else:
    config_kwargs.update({"backend": "mlx", "dtype": "float32"})
  forecaster = TimesFM3Evaluator(ModelConfig(**config_kwargs))
  arrays: dict[str, np.ndarray] = {}
  for case_name, case_kwargs in _cases():
    outputs = list(forecaster.predict_batch(**case_kwargs))
    for index, output in enumerate(outputs):
      if output.forecast is None or output.quantiles is None:
        raise AssertionError(f"{case_name}[{index}] returned an incomplete forecast")
      arrays[f"{case_name}.{index}.forecast"] = output.forecast
      arrays[f"{case_name}.{index}.quantiles"] = output.quantiles
  np.savez(args.output, **arrays)


def _run_worker(
  worker: str,
  source_root: Path,
  output: Path,
  args: argparse.Namespace,
) -> None:
  environment = os.environ.copy()
  environment["PYTHONPATH"] = str(source_root / "src")
  command = [
    sys.executable,
    str(Path(__file__).resolve()),
    "--worker",
    worker,
    "--output",
    str(output),
    "--checkpoint",
    args.checkpoint,
  ]
  if args.local_files_only:
    command.append("--local-files-only")
  subprocess.run(command, cwd=source_root, env=environment, check=True)


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _main(args: argparse.Namespace) -> None:
  upstream_root = Path(args.upstream_root).resolve()
  fork_root = Path(__file__).resolve().parents[1]
  core_matches = {
    name: _sha256(upstream_root / "src" / "timesfm3" / name)
    == _sha256(fork_root / "src" / "timesfm3" / name)
    for name in _CORE_FILES
  }
  if not all(core_matches.values()):
    raise AssertionError(f"The PyTorch oracle changed in the fork: {core_matches}")

  with tempfile.TemporaryDirectory(prefix="timesfm-mlx-parity-") as directory:
    temporary = Path(directory)
    torch_path = temporary / "torch.npz"
    mlx_path = temporary / "mlx.npz"
    _run_worker("torch", upstream_root, torch_path, args)
    _run_worker("mlx", fork_root, mlx_path, args)
    torch_outputs = np.load(torch_path)
    mlx_outputs = np.load(mlx_path)
    if set(torch_outputs.files) != set(mlx_outputs.files):
      raise AssertionError("Torch and MLX workers returned different output keys")

    case_results: dict[str, dict[str, float | int | bool]] = {}
    passed = True
    for key in sorted(torch_outputs.files):
      expected = torch_outputs[key]
      actual = mlx_outputs[key]
      if expected.shape != actual.shape:
        raise AssertionError(
          f"Shape mismatch for {key}: {expected.shape} != {actual.shape}"
        )
      if not np.isfinite(actual).all():
        raise AssertionError(f"MLX returned non-finite values for {key}")
      difference = np.abs(expected - actual)
      case_name = key.split(".", 1)[0]
      result = case_results.setdefault(
        case_name,
        {"arrays": 0, "max_abs": 0.0, "mean_abs": 0.0, "passed": True},
      )
      result["arrays"] = int(result["arrays"]) + 1
      result["max_abs"] = max(float(result["max_abs"]), float(difference.max()))
      result["mean_abs"] = max(float(result["mean_abs"]), float(difference.mean()))
      close = bool(
        np.allclose(expected, actual, rtol=args.rtol, atol=args.atol, equal_nan=False)
      )
      result["passed"] = bool(result["passed"]) and close
      passed = passed and close

  report = {
    "passed": passed,
    "checkpoint": args.checkpoint,
    "rtol": args.rtol,
    "atol": args.atol,
    "upstream_root": str(upstream_root),
    "fork_root": str(fork_root),
    "torch_core_files_identical": core_matches,
    "cases": case_results,
  }
  print(json.dumps(report, indent=2, sort_keys=True))
  if not passed:
    raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--upstream-root", default="../timesfm")
  parser.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
  parser.add_argument("--rtol", type=float, default=2e-5)
  parser.add_argument("--atol", type=float, default=2e-5)
  parser.add_argument("--local-files-only", action="store_true")
  parser.add_argument("--worker", choices=["torch", "mlx"], default=None)
  parser.add_argument("--output", default=None)
  args = parser.parse_args()
  if args.worker is not None and args.output is None:
    parser.error("--output is required in worker mode")
  return args


if __name__ == "__main__":
  parsed = _parse_args()
  if parsed.worker is not None:
    _worker(parsed)
  else:
    _main(parsed)
