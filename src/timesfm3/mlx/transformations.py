"""Reversible TimesFM transformations implemented in MLX."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx


def signed_log(x: mx.array, *, reverse: bool = False) -> mx.array:
  return mx.sign(x) * (mx.expm1(mx.abs(x)) if reverse else mx.log1p(mx.abs(x)))


def signed_sqrt(x: mx.array, *, reverse: bool = False) -> mx.array:
  return mx.sign(x) * (mx.square(x) if reverse else mx.sqrt(mx.abs(x)))


def identity(x: mx.array, *, reverse: bool = False) -> mx.array:
  del reverse
  return x


_REGISTRY: dict[str, Callable[..., mx.array]] = {
  "signed_log": signed_log,
  "signed_sqrt": signed_sqrt,
  "identity": identity,
}


def get_transform(name: str) -> Callable[..., mx.array]:
  try:
    return _REGISTRY[name]
  except KeyError:
    raise ValueError(f"Unknown transform name: {name}") from None


def max_output(name: str, value_clip: float) -> mx.array:
  if name == "signed_log":
    return mx.log1p(mx.array(value_clip))
  if name == "signed_sqrt":
    return mx.sqrt(mx.array(value_clip))
  if name == "identity":
    return mx.array(value_clip, dtype=mx.float32)
  raise ValueError(f"Unknown transform name: {name}")
