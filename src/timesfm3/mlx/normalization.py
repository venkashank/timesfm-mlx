"""MLX normalization layers for TimesFM 3."""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

_RECIPROCAL_OF_SOFTPLUS_0 = 1.442695041


class PerDimScale(nn.Module):
  def __init__(self, num_dims: int):
    super().__init__()
    self.num_dims = num_dims
    self.per_dim_scale = mx.zeros((num_dims,))

  def __call__(self, x: mx.array) -> mx.array:
    return (
      x
      * _RECIPROCAL_OF_SOFTPLUS_0
      / math.sqrt(self.num_dims)
      * nn.softplus(self.per_dim_scale)
    )
