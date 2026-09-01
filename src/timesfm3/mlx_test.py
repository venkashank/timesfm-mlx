"""Torch-versus-MLX parity tests for the TimesFM 3 backend."""

from __future__ import annotations

import unittest

import numpy as np
import torch

try:
  import mlx.core as mx

  from .mlx import util as mlx_util
  from .mlx.model import TimesFM3MLX

  _HAS_MLX = True
except ImportError:
  _HAS_MLX = False

from . import configs, util
from .model import TimesFM3Torch


@unittest.skipUnless(_HAS_MLX, "MLX is available only on supported Apple silicon")
class MLXParityTest(unittest.TestCase):
  def _copy_weights(self, torch_model, mlx_model):
    mlx_model.load_weights(
      [
        (name, mx.array(value.detach().cpu().numpy()))
        for name, value in torch_model.state_dict().items()
      ],
      strict=False,
    )

  def test_running_stats(self):
    rng = np.random.default_rng(1)
    values = rng.normal(size=(2, 3, 5, 4)).astype(np.float32)
    masks = rng.random(size=values.shape) < 0.2
    expected = util.get_running_stats(torch.from_numpy(values), torch.from_numpy(masks))
    actual = mlx_util.get_running_stats(mx.array(values), mx.array(masks))
    mx.eval(actual)
    for torch_value, mlx_value in zip(expected, actual):
      np.testing.assert_allclose(
        torch_value.numpy(), np.asarray(mlx_value), rtol=1e-5, atol=1e-6
      )

  def test_end_to_end_decode_with_cpm(self):
    residual = configs.ResidualBlockConfig(8, 8, False, "relu")
    transformer = configs.StackedTransformersConfig(
      2,
      configs.TransformerConfig(
        8,
        8,
        2,
        "rms",
        "rms",
        "rms",
        False,
        True,
        True,
        "relu",
        True,
        use_sdpa=True,
      ),
    )
    kwargs = {
      "input_patch_len": 4,
      "output_patch_len": 8,
      "quantiles": [0.1, 0.5, 0.9],
      "residual_block_config": residual,
      "transformer_config": transformer,
      "use_variate_attention": True,
      "use_stitching": True,
      "use_linear_detrending": True,
      "use_iterative_cpm_revin": True,
    }
    torch.manual_seed(2)
    torch_model = TimesFM3Torch(**kwargs)
    mlx_model = TimesFM3MLX(**kwargs)
    self._copy_weights(torch_model, mlx_model)
    target = np.random.default_rng(3).normal(size=(2, 2, 12)).astype(np.float32)
    expected = torch_model.decode(torch.from_numpy(target), horizon=11).numpy()
    actual = mlx_model.decode(mx.array(target), horizon=11)
    mx.eval(actual)
    np.testing.assert_allclose(expected, np.asarray(actual), rtol=2e-5, atol=3e-6)

  def test_covariate_decode(self):
    residual = configs.ResidualBlockConfig(8, 8, False, "relu")
    transformer = configs.StackedTransformersConfig(
      1,
      configs.TransformerConfig(
        8, 8, 2, "rms", "rms", "rms", False, True, False, "relu", True
      ),
    )
    kwargs = {
      "input_patch_len": 4,
      "output_patch_len": 8,
      "quantiles": [0.1, 0.5, 0.9],
      "residual_block_config": residual,
      "transformer_config": transformer,
    }
    torch.manual_seed(4)
    torch_model = TimesFM3Torch(**kwargs)
    mlx_model = TimesFM3MLX(**kwargs)
    self._copy_weights(torch_model, mlx_model)
    rng = np.random.default_rng(5)
    target = rng.normal(size=(1, 2, 12)).astype(np.float32)
    past = rng.normal(size=(1, 1, 12)).astype(np.float32)
    future = rng.normal(size=(1, 1, 21)).astype(np.float32)
    global_mask = np.zeros((1, 12), dtype=bool)
    global_mask[:, :2] = True
    target_mask = np.zeros_like(target, dtype=bool)
    target_mask[:, 0, 5] = True
    past_mask = np.zeros_like(past, dtype=bool)
    past_mask[:, :, 7] = True
    future_mask = np.zeros_like(future, dtype=bool)
    future_mask[:, :, 4] = True
    future_mask[:, :, 16] = True
    expected = torch_model.decode(
      torch.from_numpy(target),
      past_only_covariates=torch.from_numpy(past),
      past_future_covariates=torch.from_numpy(future),
      target_mask=torch.from_numpy(target_mask),
      past_only_mask=torch.from_numpy(past_mask),
      past_future_mask=torch.from_numpy(future_mask),
      mask=torch.from_numpy(global_mask),
    ).numpy()
    actual = mlx_model.decode(
      mx.array(target),
      past_only_covariates=mx.array(past),
      past_future_covariates=mx.array(future),
      target_mask=mx.array(target_mask),
      past_only_mask=mx.array(past_mask),
      past_future_mask=mx.array(future_mask),
      mask=mx.array(global_mask),
    )
    mx.eval(actual)
    np.testing.assert_allclose(expected, np.asarray(actual), rtol=2e-5, atol=3e-6)

  def test_bfloat16_decode_is_finite(self):
    residual = configs.ResidualBlockConfig(8, 8, False, "relu")
    transformer = configs.StackedTransformersConfig(
      1,
      configs.TransformerConfig(
        8, 8, 2, "rms", "rms", "rms", False, True, False, "relu", True
      ),
    )
    model = TimesFM3MLX(
      input_patch_len=4,
      output_patch_len=8,
      quantiles=[0.1, 0.5, 0.9],
      residual_block_config=residual,
      transformer_config=transformer,
    )
    model.set_dtype(mx.bfloat16)
    model.compute_dtype = mx.bfloat16
    target = mx.array(
      np.random.default_rng(6).normal(size=(1, 1, 12)).astype(np.float32)
    )
    output = model.decode(target, horizon=8).astype(mx.float32)
    mx.eval(output)
    self.assertTrue(np.isfinite(np.asarray(output)).all())


if __name__ == "__main__":
  unittest.main()
