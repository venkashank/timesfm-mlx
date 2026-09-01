"""Runtime backends for TimesFM 3 inference."""

from __future__ import annotations

from typing import Any

from .base import TimesFM3Backend


def create_backend(config: Any) -> TimesFM3Backend:
  """Creates the configured inference backend without eager framework imports."""
  backend = config.backend
  if backend == "torch":
    from .torch_backend import TorchBackend

    return TorchBackend(config)
  if backend == "mlx":
    try:
      from .mlx_backend import MLXBackend
    except ImportError as exc:
      raise ImportError(
        "The MLX backend requires an Apple-silicon macOS environment and the "
        "'mlx' extra: `uv sync --extra mlx`."
      ) from exc
    return MLXBackend(config)
  raise ValueError(f"Unknown TimesFM3 backend: {backend!r}.")


__all__ = ["TimesFM3Backend", "create_backend"]
