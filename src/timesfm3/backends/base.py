"""Backend protocol used by the shared TimesFM 3 forecasting pipeline."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


class TimesFM3Backend(Protocol):
  """Minimal tensor-runtime boundary required by `TimesFM3Forecaster`."""

  model: Any
  device: Any

  def decode(
    self,
    *,
    target: np.ndarray,
    horizon: int,
    past_only_covariates: np.ndarray | None,
    past_future_covariates: np.ndarray | None,
    mask: np.ndarray,
  ) -> np.ndarray:
    """Runs a model decode and returns a host NumPy array."""
    ...

  def cleanup(self) -> None:
    """Releases backend caches when appropriate."""
    ...
