"""MLX numerical utilities for TimesFM 3 inference."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
from mlx import nn

_TOLERANCE = 1e-6


def _make_safe_for_division(values: mx.array) -> mx.array:
  return mx.where(values < _TOLERANCE, 1.0, values)


def update_running_stats(
  n: mx.array,
  mu: mx.array,
  sigma: mx.array,
  x: mx.array,
  mask: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
  is_legit = mx.logical_not(mask)
  is_legit_f = is_legit.astype(mx.float32)
  inc_n = mx.sum(is_legit_f, axis=-1)
  x_masked = mx.where(is_legit, x, mx.zeros_like(x))
  inc_sum = mx.sum(x_masked, axis=-1)
  safe_inc_n = mx.where(inc_n == 0, 1.0, inc_n)
  inc_mu = mx.where(inc_n == 0, mx.zeros_like(inc_sum), inc_sum / safe_inc_n)
  x_diff_sq = mx.where(
    is_legit, mx.square(x - mx.expand_dims(inc_mu, -1)), mx.zeros_like(x)
  )
  inc_var = mx.where(
    inc_n == 0,
    mx.zeros_like(inc_sum),
    mx.sum(x_diff_sq, axis=-1) / safe_inc_n,
  )
  inc_sigma = mx.sqrt(inc_var)
  new_n = n + inc_n
  safe_new_n = mx.where(new_n == 0, 1.0, new_n)
  new_mu = mx.where(
    new_n == 0,
    mx.zeros_like(mu),
    (n * mu + inc_mu * inc_n) / safe_new_n,
  )
  new_sigma = mx.sqrt(
    mx.where(
      new_n == 0,
      mx.zeros_like(sigma),
      (
        n * mx.square(sigma)
        + inc_n * mx.square(inc_sigma)
        + n * mx.square(mu - new_mu)
        + inc_n * mx.square(inc_mu - new_mu)
      )
      / safe_new_n,
    )
  )
  return new_n, new_mu, new_sigma


def get_running_stats(
  values: mx.array,
  masks: mx.array,
  *,
  segment_ids: mx.array | None = None,
  initial_stats: tuple[mx.array, mx.array, mx.array] | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
  b, v, n, _ = values.shape
  if initial_stats is None:
    init_n = mx.zeros((b, v), dtype=mx.float32)
    init_mu = mx.zeros((b, v), dtype=mx.float32)
    init_sigma = mx.zeros((b, v), dtype=mx.float32)
  else:
    init_n, init_mu, init_sigma = initial_stats

  if segment_ids is None:
    is_new_segment = mx.zeros((b, n), dtype=mx.bool_)
  else:
    shifted = mx.pad(segment_ids[:, :-1], [(0, 0), (1, 0)], constant_values=-1)
    is_new_segment = segment_ids != shifted

  all_n: list[mx.array] = []
  all_mu: list[mx.array] = []
  all_sigma: list[mx.array] = []
  cur_n, cur_mu, cur_sigma = init_n, init_mu, init_sigma
  for i in range(n):
    reset = mx.expand_dims(is_new_segment[:, i], -1)
    cur_n = mx.where(reset, init_n, cur_n)
    cur_mu = mx.where(reset, init_mu, cur_mu)
    cur_sigma = mx.where(reset, init_sigma, cur_sigma)
    cur_n, cur_mu, cur_sigma = update_running_stats(
      cur_n, cur_mu, cur_sigma, values[:, :, i, :], masks[:, :, i, :]
    )
    all_n.append(cur_n)
    all_mu.append(cur_mu)
    all_sigma.append(cur_sigma)
  return (
    mx.stack(all_n, axis=2),
    mx.stack(all_mu, axis=2),
    mx.stack(all_sigma, axis=2),
  )


def revin(
  x: mx.array,
  mu: mx.array,
  sigma: mx.array,
  reverse: bool = False,
) -> mx.array:
  if mu.ndim == x.ndim - 1:
    mu = mx.expand_dims(mu, -1)
    sigma = mx.expand_dims(sigma, -1)
  elif mu.ndim == x.ndim - 2:
    mu = mx.expand_dims(mu, (-1, -2))
    sigma = mx.expand_dims(sigma, (-1, -2))
  else:
    raise ValueError(f"Unsupported shapes for x and mu: {x.shape}, {mu.shape}.")
  if reverse:
    return x * sigma + mu
  return (x - mu) / _make_safe_for_division(sigma)


def get_output_patch_via_roll(x: mx.array, rolls: int) -> tuple[mx.array, mx.array]:
  b, v, n, p = x.shape
  rolled = [mx.roll(x, shift=-(i + 1), axis=2) for i in range(rolls)]
  result = mx.stack(rolled, axis=3).reshape(b, v, n, rolls * p)
  patch_idx = mx.arange(n)
  point_idx = mx.arange(rolls * p)
  source_patch = patch_idx[:, None] + 1 + point_idx[None, :] // p
  wrap_mask = mx.expand_dims(source_patch >= n, (0, 1))
  return result, wrap_mask


_ACTIVATIONS: dict[str, Callable[[mx.array], mx.array]] = {
  "relu": nn.relu,
  "swish": nn.silu,
  "silu": nn.silu,
  "swiglu": nn.silu,
  "none": lambda x: x,
}


def get_activation_fn(name: str) -> Callable[[mx.array], mx.array]:
  try:
    return _ACTIVATIONS[name]
  except KeyError:
    raise ValueError(
      f"Activation: {name} not supported. Supported activations: "
      f"{list(_ACTIVATIONS.keys())}"
    ) from None


def stitch_patches(patch_preds: mx.array, patch_len: int) -> mx.array:
  b, v, num_patches, total_len, q = patch_preds.shape
  overlap = total_len - patch_len
  if num_patches == 1:
    return patch_preds[:, :, 0, :, :]
  weights = mx.linspace(1.0, 0.0, overlap).astype(patch_preds.dtype)
  weights = weights[None, None, None, :, None]
  first_chunk = patch_preds[:, :, 0, :patch_len, :]
  prev_patches = patch_preds[:, :, :-1, :, :]
  next_patches = patch_preds[:, :, 1:, :, :]
  stitched = (
    weights * prev_patches[:, :, :, patch_len:, :]
    + (1.0 - weights) * next_patches[:, :, :, :overlap, :]
  )
  middles = next_patches[:, :, :, overlap:patch_len, :]
  chunks = mx.concatenate([stitched, middles], axis=3)
  middle = chunks.reshape(b, v, (num_patches - 1) * patch_len, q)
  tail = patch_preds[:, :, -1, patch_len:, :]
  return mx.concatenate([first_chunk, middle, tail], axis=2)
