"""MLX runtime adapter and Hugging Face checkpoint loader."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten

from ..mlx.model import TimesFM3MLX


def _checkpoint_directory(config: Any) -> tuple[Path | None, Path | None]:
  path = Path(os.path.expanduser(config.checkpoint_path))
  if path.is_dir():
    return path, None
  if path.is_file():
    return None, path
  downloaded = snapshot_download(
    repo_id=config.checkpoint_path,
    allow_patterns=["config.json", "*.safetensors"],
    cache_dir=config.cache_dir,
    force_download=config.force_download,
    token=config.token,
    revision=config.revision,
    local_files_only=config.local_files_only,
  )
  return Path(downloaded), None


def load_mlx_model(config: Any) -> TimesFM3MLX:
  directory, weight_file = _checkpoint_directory(config)
  if directory is not None:
    config_path = directory / "config.json"
    if not config_path.exists():
      raise FileNotFoundError(f"Missing TimesFM config: {config_path}")
    model_config = json.loads(config_path.read_text())
    weight_files = sorted(directory.glob("*.safetensors"))
    if not weight_files:
      raise FileNotFoundError(f"No safetensors found in {directory}")
  else:
    assert weight_file is not None
    if weight_file.suffix != ".safetensors":
      raise ValueError("The MLX backend accepts safetensors checkpoints only.")
    model_config = {
      "input_patch_len": config.input_patch_length,
      "output_patch_len": config.output_patch_length,
      "quantiles": config.quantiles,
      "residual_block_config": config.residual_block_config,
      "transformer_config": config.transformer_config,
      "use_variate_attention": config.use_variate_attention,
      "value_clip": config.value_clip,
      "use_stitching": config.use_stitching,
      "use_linear_detrending": config.use_linear_detrending,
      "linear_detrending_threshold": config.linear_detrending_threshold,
      "use_iterative_cpm_revin": config.use_iterative_cpm_revin,
      "use_frozen_running_stats": config.use_frozen_running_stats,
      "input_transform": config.input_transform,
    }
    weight_files = [weight_file]
  model = TimesFM3MLX(**model_config)
  dtype = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
  }[config.dtype]
  model.set_dtype(dtype)
  model.compute_dtype = dtype
  weights: dict[str, mx.array] = {}
  for file in weight_files:
    weights.update(
      {name: value.astype(dtype) for name, value in mx.load(str(file)).items()}
    )
  model_keys = {name for name, _ in tree_flatten(model.parameters())}
  unexpected = set(weights) - model_keys
  missing = model_keys - set(weights)
  allowed_missing = {
    key for key in missing if key.endswith("rotary_position_embedding.timescale")
  }
  if unexpected or missing - allowed_missing:
    raise ValueError(
      "Checkpoint does not match TimesFM3MLX: "
      f"unexpected={sorted(unexpected)}, missing={sorted(missing - allowed_missing)}"
    )
  model.load_weights(list(weights.items()), strict=False)
  mx.eval(model.parameters())
  return model


class MLXBackend:
  def __init__(self, config: Any):
    self.config = config
    self.device = mx.gpu
    self.model = load_mlx_model(config)
    self._compiled_decodes: dict[tuple[Any, ...], Any] = {}

  def _decode_function(
    self,
    target: mx.array,
    horizon: int,
    past_only_covariates: mx.array | None,
    past_future_covariates: mx.array | None,
    mask: mx.array,
  ) -> mx.array:
    if not self.config.compile:
      return self.model.decode(
        target=target,
        horizon=horizon,
        past_only_covariates=past_only_covariates,
        past_future_covariates=past_future_covariates,
        mask=mask,
      )
    key = (
      target.shape,
      horizon,
      None if past_only_covariates is None else past_only_covariates.shape,
      None if past_future_covariates is None else past_future_covariates.shape,
      target.dtype,
    )
    function = self._compiled_decodes.get(key)
    if function is None:
      if past_only_covariates is None and past_future_covariates is None:
        function = mx.compile(
          lambda target_arg, mask_arg: self.model.decode(
            target=target_arg, horizon=horizon, mask=mask_arg
          )
        )
      elif past_future_covariates is None:
        function = mx.compile(
          lambda target_arg, po_arg, mask_arg: self.model.decode(
            target=target_arg,
            horizon=horizon,
            past_only_covariates=po_arg,
            mask=mask_arg,
          )
        )
      elif past_only_covariates is None:
        function = mx.compile(
          lambda target_arg, pf_arg, mask_arg: self.model.decode(
            target=target_arg,
            past_future_covariates=pf_arg,
            mask=mask_arg,
          )
        )
      else:
        function = mx.compile(
          lambda target_arg, po_arg, pf_arg, mask_arg: self.model.decode(
            target=target_arg,
            past_only_covariates=po_arg,
            past_future_covariates=pf_arg,
            mask=mask_arg,
          )
        )
      self._compiled_decodes[key] = function
    if past_only_covariates is None and past_future_covariates is None:
      return function(target, mask)
    if past_future_covariates is None:
      return function(target, past_only_covariates, mask)
    if past_only_covariates is None:
      return function(target, past_future_covariates, mask)
    return function(target, past_only_covariates, past_future_covariates, mask)

  def decode(
    self,
    *,
    target: np.ndarray,
    horizon: int,
    past_only_covariates: np.ndarray | None,
    past_future_covariates: np.ndarray | None,
    mask: np.ndarray,
  ) -> np.ndarray:
    output = self._decode_function(
      mx.array(target, dtype=mx.float32),
      horizon,
      (
        mx.array(past_only_covariates, dtype=mx.float32)
        if past_only_covariates is not None
        else None
      ),
      (
        mx.array(past_future_covariates, dtype=mx.float32)
        if past_future_covariates is not None
        else None
      ),
      mx.array(mask, dtype=mx.bool_),
    )
    mx.eval(output)
    if output.dtype == mx.bfloat16:
      output = output.astype(mx.float32)
      mx.eval(output)
    return np.asarray(output)

  def cleanup(self) -> None:
    mx.clear_cache()
