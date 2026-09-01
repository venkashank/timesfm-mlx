"""MLX dense layers for TimesFM 3."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn

from .. import configs
from . import util

_RMS_EPS = float(np.finfo(np.float32).eps)


class ResidualBlock(nn.Module):
  def __init__(self, config: configs.ResidualBlockConfig, input_dims: int):
    super().__init__()
    self.config = config
    self.hidden_layer = nn.Linear(input_dims, config.hidden_dims, bias=config.use_bias)
    self.output_layer = nn.Linear(
      config.hidden_dims, config.output_dims, bias=config.use_bias
    )
    self.residual_layer = (
      None
      if config.identity_skip
      else nn.Linear(input_dims, config.output_dims, bias=config.use_bias)
    )
    self.activation = util.get_activation_fn(config.activation)
    self.pre_norm = (
      nn.RMSNorm(input_dims, eps=_RMS_EPS) if config.prenorm == "rms" else None
    )

  def __call__(self, x: mx.array) -> mx.array:
    hidden_input = self.pre_norm(x) if self.pre_norm is not None else x
    hidden = self.activation(self.hidden_layer(hidden_input))
    residual = self.residual_layer(x) if self.residual_layer is not None else x
    return self.output_layer(hidden) + residual
