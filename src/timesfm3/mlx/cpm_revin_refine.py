"""Iterative CPM RevIN refinement implemented with MLX arrays."""

from __future__ import annotations

import mlx.core as mx

from . import util


def cpm_iterative_revin_refine(
  raw_logits: mx.array,
  revin_n: mx.array,
  revin_mu: mx.array,
  revin_sigma: mx.array,
  patch_cpm_mask: mx.array,
  median_q_idx: int,
  rolls: int,
  patch_len: int,
  num_quantiles: int,
  value_clip: float = 1e9,
) -> tuple[mx.array, mx.array]:
  b, v, n_patches, _ = raw_logits.shape
  median_logits = raw_logits.reshape(b, v, n_patches, rolls, patch_len, num_quantiles)[
    :, :, :, :, :, median_q_idx
  ]
  carry_n = mx.zeros((b, v), dtype=mx.float32)
  carry_mu = mx.zeros((b, v), dtype=mx.float32)
  carry_sigma = mx.zeros((b, v), dtype=mx.float32)
  anchor_values = mx.zeros((b, v, rolls, patch_len), dtype=mx.float32)
  block_offset = mx.zeros((b,), dtype=mx.int32)
  step_masks = mx.zeros((b, v, patch_len), dtype=mx.bool_)
  refined_mu: list[mx.array] = []
  refined_sigma: list[mx.array] = []

  for i in range(n_patches):
    actual_n = revin_n[:, :, i]
    actual_mu = revin_mu[:, :, i]
    actual_sigma = revin_sigma[:, :, i]
    current_logits = median_logits[:, :, i]
    is_cpm = patch_cpm_mask[:, i : i + 1]
    offset_onehot = (
      mx.arange(rolls)[None, :] == mx.expand_dims(block_offset, 1)
    ).astype(mx.float32)
    predicted_step = mx.einsum("br,bvrp->bvp", offset_onehot, anchor_values)
    new_n, new_mu, new_sigma = util.update_running_stats(
      carry_n, carry_mu, carry_sigma, predicted_step, step_masks
    )
    out_n = mx.where(is_cpm, new_n, actual_n)
    out_mu = mx.where(is_cpm, new_mu, actual_mu)
    out_sigma = mx.where(is_cpm, new_sigma, actual_sigma)
    new_offset = mx.where(
      mx.squeeze(is_cpm, -1),
      (block_offset + 1) % rolls,
      mx.zeros_like(block_offset),
    )
    should_update = new_offset == 0
    step_values = util.revin(current_logits, out_mu, out_sigma, reverse=True)
    step_values = mx.clip(step_values, -value_clip, value_clip)
    anchor_values = mx.where(
      should_update.reshape(b, 1, 1, 1), step_values, anchor_values
    )
    carry_n, carry_mu, carry_sigma = out_n, out_mu, out_sigma
    block_offset = new_offset
    refined_mu.append(out_mu)
    refined_sigma.append(out_sigma)
  return mx.stack(refined_mu, axis=2), mx.stack(refined_sigma, axis=2)
