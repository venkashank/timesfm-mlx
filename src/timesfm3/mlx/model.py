"""Inference-only TimesFM 3 model implemented natively in MLX."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import mlx.core as mx
from mlx import nn

from .. import configs
from . import cpm_revin_refine, dense, transformer, util


class TimesFM3MLX(nn.Module):
  def __init__(
    self,
    input_patch_len: int = 32,
    output_patch_len: int = 64,
    quantiles: list[float] | None = None,
    residual_block_config: configs.ResidualBlockConfig | dict[str, Any] | None = None,
    transformer_config: configs.StackedTransformersConfig
    | dict[str, Any]
    | None = None,
    use_variate_attention: bool = True,
    value_clip: float = 1e20,
    use_stitching: bool = True,
    use_linear_detrending: bool = True,
    linear_detrending_threshold: float = 0.5,
    use_iterative_cpm_revin: bool = True,
    use_frozen_running_stats: bool = False,
    input_transform: str = "identity",
  ):
    super().__init__()
    quantiles = quantiles or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if residual_block_config is None:
      residual_block_config = configs.ResidualBlockConfig(1280, 1280, False, "relu")
    elif isinstance(residual_block_config, dict):
      residual_block_config = configs.ResidualBlockConfig(**residual_block_config)
    if transformer_config is None:
      transformer_config = configs.StackedTransformersConfig(
        20,
        configs.TransformerConfig(
          1280, 1280, 16, "rms", "rms", "rms", False, True, False, "relu", True
        ),
      )
    elif isinstance(transformer_config, dict):
      config_dict = dict(transformer_config)
      if isinstance(config_dict.get("transformer"), dict):
        config_dict["transformer"] = configs.TransformerConfig(
          **config_dict["transformer"]
        )
      transformer_config = configs.StackedTransformersConfig(**config_dict)
    if output_patch_len % input_patch_len:
      raise ValueError("output_patch_len must be a multiple of input_patch_len")
    self.input_patch_len = input_patch_len
    self.output_patch_len = output_patch_len
    self.quantiles = quantiles
    self.num_quantiles = len(quantiles)
    self.rolls = output_patch_len // input_patch_len
    self.residual_block_config = residual_block_config
    self.transformer_config = transformer_config
    self.use_variate_attention = use_variate_attention
    self.value_clip = value_clip
    self.use_stitching = use_stitching
    self.use_linear_detrending = use_linear_detrending
    self.linear_detrending_threshold = linear_detrending_threshold
    self.use_iterative_cpm_revin = use_iterative_cpm_revin
    self.use_frozen_running_stats = use_frozen_running_stats
    self.input_transform = input_transform
    self.compute_dtype = mx.float32
    if use_stitching:
      self._stitching_extract_len = min(2 * input_patch_len, output_patch_len)
    input_dims = 2 * (input_patch_len + output_patch_len)
    self.pre_transformer_resblock = dense.ResidualBlock(
      residual_block_config, input_dims
    )
    self.transformer_stack = transformer.StackedMixingTransformer(
      transformer_config, use_variate_attention
    )
    self.output_head = nn.Linear(
      transformer_config.transformer.model_dims,
      output_patch_len * self.num_quantiles,
      bias=True,
    )

  def to_dict(self) -> dict[str, Any]:
    return {
      "input_patch_len": self.input_patch_len,
      "output_patch_len": self.output_patch_len,
      "quantiles": list(self.quantiles),
      "residual_block_config": dataclasses.asdict(self.residual_block_config),
      "transformer_config": dataclasses.asdict(self.transformer_config),
      "use_variate_attention": self.use_variate_attention,
      "value_clip": self.value_clip,
      "use_stitching": self.use_stitching,
      "use_linear_detrending": self.use_linear_detrending,
      "linear_detrending_threshold": self.linear_detrending_threshold,
      "use_iterative_cpm_revin": self.use_iterative_cpm_revin,
      "use_frozen_running_stats": self.use_frozen_running_stats,
      "input_transform": self.input_transform,
    }

  def _preprocess(
    self,
    values: mx.array,
    masks: mx.array,
    patch_is_target: mx.array,
    freeze_after: int | None = None,
    patch_cpm_mask: mx.array | None = None,
  ) -> tuple[mx.array, mx.array, mx.array, tuple[mx.array, mx.array], mx.array]:
    running_n, running_mean, running_std = util.get_running_stats(values, masks)
    n = values.shape[2]
    if freeze_after is not None and 0 <= freeze_after < n - 1:
      repeat = n - freeze_after - 1
      running_mean = mx.concatenate(
        [
          running_mean[:, :, : freeze_after + 1],
          mx.repeat(
            running_mean[:, :, freeze_after : freeze_after + 1], repeat, axis=2
          ),
        ],
        axis=2,
      )
      running_std = mx.concatenate(
        [
          running_std[:, :, : freeze_after + 1],
          mx.repeat(running_std[:, :, freeze_after : freeze_after + 1], repeat, axis=2),
        ],
        axis=2,
      )
    if patch_cpm_mask is not None:
      target_cpm = patch_cpm_mask[:, None, :, None] & patch_is_target[..., None]
      masks = masks | target_cpm
    values_norm = util.revin(values, running_mean, running_std)
    values_norm = mx.where(masks, 0.0, values_norm)
    future_values, wrap_mask = util.get_output_patch_via_roll(values, self.rolls)
    future_values = util.revin(future_values, running_mean, running_std)
    future_masks, _ = util.get_output_patch_via_roll(masks, self.rolls)
    future_masks = future_masks | patch_is_target[..., None] | wrap_mask
    future_values = mx.where(future_masks, 0.0, future_values)
    values_cat = mx.concatenate([values_norm, future_values], axis=-1)
    masks_cat = mx.concatenate([masks, future_masks], axis=-1)
    block_input = mx.concatenate([values_cat, masks_cat.astype(mx.float32)], axis=-1)
    block_output = self.pre_transformer_resblock(block_input.astype(self.compute_dtype))
    patch_mask = mx.all(masks_cat, axis=3)
    return block_input, block_output, patch_mask, (running_mean, running_std), running_n

  def __call__(
    self,
    inputs: dict[str, mx.array],
    freeze_after: int | None = None,
    patch_cpm_mask: mx.array | None = None,
    return_aux_outputs: bool = False,
  ) -> dict[str, Any]:
    values = mx.clip(
      mx.nan_to_num(inputs["values"], nan=0.0), -self.value_clip, self.value_clip
    )
    masks = inputs["masks"].astype(mx.bool_)
    patch_is_target = inputs["patch_is_target"]
    block_in, transformer_in, patch_mask, stats, running_n = self._preprocess(
      values, masks, patch_is_target, freeze_after, patch_cpm_mask
    )
    effective_mask = mx.cumprod(patch_mask.astype(mx.int32), axis=2).astype(mx.bool_)
    transformer_out, attention_masks = self.transformer_stack(
      transformer_in, effective_mask
    )
    raw_logits = self.output_head(transformer_out).astype(mx.float32)
    revin_mean, revin_std = stats
    if self.use_iterative_cpm_revin and patch_cpm_mask is not None:
      refined_mean, refined_std = cpm_revin_refine.cpm_iterative_revin_refine(
        raw_logits,
        running_n,
        revin_mean,
        revin_std,
        patch_cpm_mask,
        self.num_quantiles // 2,
        self.rolls,
        self.input_patch_len,
        self.num_quantiles,
        self.value_clip,
      )
      cpm_mask = patch_cpm_mask[:, None, :]
      revin_mean = mx.where(cpm_mask, refined_mean, revin_mean)
      revin_std = mx.where(cpm_mask, refined_std, revin_std)
    logits = mx.clip(
      util.revin(raw_logits, revin_mean, revin_std, reverse=True),
      -self.value_clip,
      self.value_clip,
    )
    b, v, n = logits.shape[:3]
    outputs = {
      "logits": logits.reshape(b, v, n, self.output_patch_len, self.num_quantiles),
      "revin_stats": stats,
    }
    if return_aux_outputs:
      outputs.update(
        {
          "__call__:resblock_input": block_in,
          "__call__:transformer_input": transformer_in,
          "__call__:seq_attn_mask": attention_masks,
          "__call__:transformer_output": transformer_out,
        }
      )
    return outputs

  def decode(
    self,
    target: mx.array,
    horizon: int = 0,
    past_only_covariates: mx.array | None = None,
    past_future_covariates: mx.array | None = None,
    target_mask: mx.array | None = None,
    past_only_mask: mx.array | None = None,
    past_future_mask: mx.array | None = None,
    mask: mx.array | None = None,
    return_aux_outputs: bool = False,
  ) -> Any:
    batch, num_target, context = target.shape
    if past_future_covariates is not None:
      horizon = past_future_covariates.shape[-1] - context
    if horizon <= 0:
      raise ValueError("Decode function requires horizon > 0.")
    ctx_padding = (-context) % self.input_patch_len
    if ctx_padding:
      pad3 = [(0, 0), (0, 0), (ctx_padding, 0)]
      target = mx.pad(target, pad3)
      past_only_covariates = (
        mx.pad(past_only_covariates, pad3) if past_only_covariates is not None else None
      )
      past_future_covariates = (
        mx.pad(past_future_covariates, pad3)
        if past_future_covariates is not None
        else None
      )
      target_mask = (
        mx.pad(target_mask, pad3, constant_values=True)
        if target_mask is not None
        else None
      )
      past_only_mask = (
        mx.pad(past_only_mask, pad3, constant_values=True)
        if past_only_mask is not None
        else None
      )
      past_future_mask = (
        mx.pad(past_future_mask, pad3, constant_values=True)
        if past_future_mask is not None
        else None
      )
      mask = (
        mx.pad(mask, [(0, 0), (ctx_padding, 0)], constant_values=True)
        if mask is not None
        else None
      )
      context += ctx_padding
    if mask is None:
      mask = mx.zeros((batch, context), dtype=mx.bool_)
      if ctx_padding:
        mask = mx.concatenate(
          [mx.ones((batch, ctx_padding), dtype=mx.bool_), mask[:, ctx_padding:]], axis=1
        )
    if self.use_stitching:
      overlap = self._stitching_extract_len - self.input_patch_len
      num_forecast_patches = max(
        math.ceil((horizon - overlap) / self.input_patch_len), 1
      )
      num_horizon_patches = num_forecast_patches + self.rolls - 1
      padded_horizon = num_horizon_patches * self.input_patch_len
    else:
      padded_horizon = horizon + (-horizon) % self.output_patch_len
      num_horizon_patches = padded_horizon // self.input_patch_len
    horizon_padding = padded_horizon - horizon
    num_context_patches = context // self.input_patch_len
    target_mask = (
      mx.zeros_like(target, dtype=mx.bool_) if target_mask is None else target_mask
    )
    target_mask = target_mask | mask[:, None, :]
    context_values = [target]
    context_masks = [target_mask]
    num_past_only = 0
    if past_only_covariates is not None:
      num_past_only = past_only_covariates.shape[1]
      past_only_mask = (
        mx.zeros_like(past_only_covariates, dtype=mx.bool_)
        if past_only_mask is None
        else past_only_mask
      )
      context_values.append(past_only_covariates)
      context_masks.append(past_only_mask | mask[:, None, :])
    if past_future_covariates is not None:
      past_future_mask = (
        mx.zeros_like(past_future_covariates, dtype=mx.bool_)
        if past_future_mask is None
        else past_future_mask
      )
      context_values.append(past_future_covariates[..., :context])
      context_masks.append(past_future_mask[..., :context] | mask[:, None, :])
    ctx_values = mx.concatenate(context_values, axis=1)
    ctx_masks = mx.concatenate(context_masks, axis=1)
    valid = ~ctx_masks
    num_variates = ctx_values.shape[1]
    if self.use_linear_detrending:
      t = mx.arange(-(context - 1), 1, dtype=mx.float32)[None, None, :] / context
      count = mx.sum(valid.astype(mx.float32), axis=-1, keepdims=True)
      safe_count = mx.maximum(count, 1.0)
      sum_t = mx.sum(mx.where(valid, t, 0.0), axis=-1, keepdims=True)
      sum_t2 = mx.sum(mx.where(valid, t * t, 0.0), axis=-1, keepdims=True)
      sum_y = mx.sum(mx.where(valid, ctx_values, 0.0), axis=-1, keepdims=True)
      sum_ty = mx.sum(mx.where(valid, t * ctx_values, 0.0), axis=-1, keepdims=True)
      determinant = count * sum_t2 - sum_t * sum_t
      safe_det = mx.where(determinant == 0, 1.0, determinant)
      slope = mx.where(
        determinant == 0, 0.0, (count * sum_ty - sum_t * sum_y) / safe_det
      )
      intercept = mx.where(
        determinant == 0,
        mx.where(count > 0, sum_y / safe_count, 0.0),
        (sum_y - slope * sum_t) / safe_count,
      )
      detrended = ctx_values - (slope * t + intercept)
      mean = sum_y / safe_count
      original_var = mx.maximum(
        mx.sum(mx.where(valid, ctx_values * ctx_values, 0.0), axis=-1, keepdims=True)
        / safe_count
        - mean * mean,
        0.0,
      )
      det_mean = (
        mx.sum(mx.where(valid, detrended, 0.0), axis=-1, keepdims=True) / safe_count
      )
      det_var = mx.maximum(
        mx.sum(mx.where(valid, detrended * detrended, 0.0), axis=-1, keepdims=True)
        / safe_count
        - det_mean * det_mean,
        0.0,
      )
      apply_detrend = mx.sqrt(det_var) < self.linear_detrending_threshold * mx.sqrt(
        original_var
      )
      ctx_values = mx.where(apply_detrend, detrended, ctx_values)
    else:
      slope = mx.zeros((batch, num_variates, 1), dtype=mx.float32)
      intercept = mx.zeros_like(slope)
      apply_detrend = mx.zeros_like(slope, dtype=mx.bool_)
    ctx_values = mx.where(ctx_masks, 0.0, ctx_values)
    horizon_values = [mx.zeros((batch, num_target + num_past_only, padded_horizon))]
    horizon_masks = [
      mx.ones((batch, num_target + num_past_only, padded_horizon), dtype=mx.bool_)
    ]
    if past_future_covariates is not None:
      future_values = past_future_covariates[..., context : context + horizon]
      future_masks = past_future_mask[..., context : context + horizon]
      if self.use_linear_detrending:
        future_t = mx.arange(1, horizon + 1, dtype=mx.float32)[None, None, :] / context
        future_trend = (
          slope[:, num_target + num_past_only :] * future_t
          + intercept[:, num_target + num_past_only :]
        )
        future_values = mx.where(
          apply_detrend[:, num_target + num_past_only :],
          future_values - future_trend,
          future_values,
        )
      future_values = mx.where(future_masks, 0.0, future_values)
      if horizon_padding:
        pad3 = [(0, 0), (0, 0), (0, horizon_padding)]
        future_values = mx.pad(future_values, pad3)
        future_masks = mx.pad(future_masks, pad3, constant_values=True)
      horizon_values.append(future_values)
      horizon_masks.append(future_masks)
    all_values = mx.concatenate(
      [ctx_values, mx.concatenate(horizon_values, axis=1)], axis=-1
    )
    all_masks = mx.concatenate(
      [ctx_masks, mx.concatenate(horizon_masks, axis=1)], axis=-1
    )
    num_variates = all_values.shape[1]
    total_patches = num_context_patches + num_horizon_patches
    patch_is_target = mx.concatenate(
      [
        mx.ones((batch, num_target + num_past_only, total_patches), dtype=mx.bool_),
        mx.zeros(
          (batch, num_variates - num_target - num_past_only, total_patches),
          dtype=mx.bool_,
        ),
      ],
      axis=1,
    )
    inputs = {
      "values": all_values.reshape(batch, num_variates, -1, self.input_patch_len),
      "masks": all_masks.reshape(batch, num_variates, -1, self.input_patch_len),
      "patch_is_target": patch_is_target,
    }
    cpm_mask = mx.concatenate(
      [
        mx.zeros((batch, num_context_patches), dtype=mx.bool_),
        mx.ones((batch, num_horizon_patches), dtype=mx.bool_),
      ],
      axis=1,
    )
    output = self(
      inputs,
      num_context_patches - 1 if self.use_frozen_running_stats else None,
      cpm_mask,
      return_aux_outputs,
    )
    logits = output["logits"]
    if self.use_stitching:
      overlap = self._stitching_extract_len - self.input_patch_len
      count = max(math.ceil((horizon - overlap) / self.input_patch_len), 1)
      indices = mx.arange(count) + num_context_patches - 1
      result = util.stitch_patches(
        logits[:, :, indices, : self._stitching_extract_len, :], self.input_patch_len
      )[:, :, :horizon, :]
    else:
      count = padded_horizon // self.output_patch_len
      indices = mx.arange(count) * self.rolls + num_context_patches - 1
      result = logits[:, :, indices].reshape(
        batch, num_variates, -1, self.num_quantiles
      )[:, :, :horizon]
    if self.use_linear_detrending:
      forecast_t = mx.arange(1, horizon + 1, dtype=mx.float32) / context
      trend = (
        slope[:, :, 0, None] * forecast_t[None, None, :] + intercept[:, :, 0, None]
      )
      result = result + mx.where(apply_detrend[:, :, 0, None], trend, 0.0)[..., None]
    return (result, output) if return_aux_outputs else result
