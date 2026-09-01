"""Native MLX mixing-transformer layers for TimesFM 3 inference."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
from mlx import nn

from .. import configs
from . import normalization, util

_RMS_EPS = float(np.finfo(np.float32).eps)


def make_attn_mask(
  query_length: int,
  num_all_masked_kv: mx.array,
  query_index_offset: mx.array | None = None,
  kv_length: int = 0,
  causal: bool = True,
) -> mx.array:
  if kv_length == 0:
    kv_length = query_length
  q_index = mx.arange(query_length).reshape(1, 1, -1, 1)
  if query_index_offset is not None:
    q_index = q_index + query_index_offset.reshape(-1, 1, 1, 1)
  kv_index = mx.arange(kv_length).reshape(1, 1, 1, -1)
  mask = kv_index >= num_all_masked_kv.reshape(-1, 1, 1, 1)
  return mx.logical_and(q_index >= kv_index, mask) if causal else mask


def make_segment_mask(segment_ids: mx.array) -> mx.array:
  return mx.expand_dims(
    mx.expand_dims(segment_ids, 2) == mx.expand_dims(segment_ids, 1), 1
  )


class RotaryPositionalEmbedding(nn.Module):
  def __init__(
    self,
    embedding_dims: int,
    min_timescale: int = 1,
    max_timescale: int = 10000,
  ):
    super().__init__()
    self.embedding_dims = embedding_dims
    half_dim = embedding_dims // 2
    fraction = 2.0 * mx.arange(half_dim, dtype=mx.float32) / embedding_dims
    self.timescale = min_timescale * (max_timescale / min_timescale) ** fraction

  def __call__(self, inputs: mx.array, position: mx.array | None = None) -> mx.array:
    if self.embedding_dims != inputs.shape[-1]:
      raise ValueError(
        "The embedding dims of the rotary position embedding must match "
        "the hidden dimension of the inputs."
      )
    if position is None:
      position = mx.expand_dims(mx.arange(inputs.shape[1], dtype=mx.float32), 0)
    if inputs.ndim == 4:
      pos = mx.expand_dims(position, (-1, -2))
      timescale = self.timescale.reshape(1, 1, 1, -1)
    elif inputs.ndim == 3:
      pos = mx.expand_dims(position, -1)
      timescale = self.timescale.reshape(1, 1, -1)
    else:
      raise ValueError("Inputs must be of rank 3 or 4.")
    sinusoid = pos.astype(mx.float32) / timescale
    sin_value = mx.sin(sinusoid).astype(inputs.dtype)
    cos_value = mx.cos(sinusoid).astype(inputs.dtype)
    first, second = mx.split(inputs, 2, axis=-1)
    return mx.concatenate(
      [first * cos_value - second * sin_value, second * cos_value + first * sin_value],
      axis=-1,
    )


class _RMSNormNoWeight(nn.Module):
  def __call__(self, x: mx.array) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + _RMS_EPS)


class MultiHeadAttention(nn.Module):
  def __init__(
    self,
    num_heads: int,
    in_features: int,
    use_per_dim_scale: bool = True,
    use_rotary_position_embeddings: bool = True,
    causal_attention: bool = True,
    use_bias: bool = False,
    qk_norm: str = "rms",
    v_norm: str = "none",
    use_sdpa: bool = True,
    rescale_logits: bool = False,
  ):
    super().__init__()
    self.num_heads = num_heads
    self.in_features = in_features
    self.causal_attention = causal_attention
    self.head_dim = in_features // num_heads
    self.use_sdpa = use_sdpa
    self.rescale_logits = rescale_logits
    self.query_proj = nn.Linear(in_features, in_features, bias=use_bias)
    self.key_proj = nn.Linear(in_features, in_features, bias=use_bias)
    self.value_proj = nn.Linear(in_features, in_features, bias=use_bias)
    self.out_proj = nn.Linear(in_features, in_features, bias=use_bias)
    self.query_ln = (
      nn.RMSNorm(self.head_dim, eps=_RMS_EPS) if qk_norm == "rms" else None
    )
    self.key_ln = nn.RMSNorm(self.head_dim, eps=_RMS_EPS) if qk_norm == "rms" else None
    self.value_ln = _RMSNormNoWeight() if v_norm == "rms" else None
    self.rotary_position_embedding = (
      RotaryPositionalEmbedding(self.head_dim)
      if use_rotary_position_embeddings
      else None
    )
    self.per_dim_scale = (
      normalization.PerDimScale(self.head_dim) if use_per_dim_scale else None
    )

  def __call__(
    self,
    inputs_q: mx.array,
    *,
    segment_ids: mx.array | None = None,
    segment_pos: mx.array | None = None,
    patch_mask: mx.array | None = None,
  ) -> tuple[mx.array, mx.array]:
    batch_size, n_patches, _ = inputs_q.shape
    if patch_mask is None:
      patch_mask = mx.zeros((batch_size, n_patches), dtype=mx.bool_)
    query = self.query_proj(inputs_q).reshape(
      batch_size, n_patches, self.num_heads, self.head_dim
    )
    key = self.key_proj(inputs_q).reshape(
      batch_size, n_patches, self.num_heads, self.head_dim
    )
    value = self.value_proj(inputs_q).reshape(
      batch_size, n_patches, self.num_heads, self.head_dim
    )
    if self.rotary_position_embedding is not None:
      position = (
        mx.expand_dims(mx.arange(n_patches, dtype=mx.int32), 0)
        if segment_pos is None
        else segment_pos
      )
      query = self.rotary_position_embedding(query, position)
      key = self.rotary_position_embedding(key, position)
    if self.query_ln is not None:
      query = self.query_ln(query)
      key = self.key_ln(key)
    if self.per_dim_scale is not None:
      query = self.per_dim_scale(query)
    if self.value_ln is not None:
      value = self.value_ln(value)

    zeros = mx.zeros((batch_size,), dtype=mx.int32)
    attn_mask = make_attn_mask(n_patches, zeros, causal=self.causal_attention)
    attn_mask = mx.logical_and(attn_mask, mx.logical_not(patch_mask[:, None, None, :]))
    if segment_ids is not None:
      attn_mask = mx.logical_and(attn_mask, make_segment_mask(segment_ids))
    query = mx.swapaxes(query, 1, 2)
    key = mx.swapaxes(key, 1, 2)
    value = mx.swapaxes(value, 1, 2)
    scale = 1.0 if self.rescale_logits else math.sqrt(self.head_dim)
    if self.use_sdpa:
      output = mx.fast.scaled_dot_product_attention(
        query, key, value, scale=scale, mask=attn_mask
      )
    else:
      additive_mask = mx.where(attn_mask, 0.0, -1e9)
      weights = mx.softmax(
        (query @ mx.swapaxes(key, -2, -1)) * scale + additive_mask, axis=-1
      )
      output = weights @ value
    output = mx.swapaxes(output, 1, 2).reshape(batch_size, n_patches, self.in_features)
    return self.out_proj(output), attn_mask


class MixingTransformer(nn.Module):
  def __init__(
    self, config: configs.TransformerConfig, use_variate_attention: bool = True
  ):
    super().__init__()
    self.config = config
    self.use_variate_attention = use_variate_attention
    self.pre_seq_attn_ln = nn.RMSNorm(config.model_dims, eps=_RMS_EPS)
    self.post_seq_attn_ln = nn.RMSNorm(config.model_dims, eps=_RMS_EPS)
    rescale_logits = not config.use_memory_efficient_attention
    self.seq_attn = MultiHeadAttention(
      config.num_heads,
      config.model_dims,
      use_rotary_position_embeddings=config.use_rope_seq,
      qk_norm=config.qk_norm,
      v_norm=config.v_norm,
      causal_attention=config.causal_attention,
      use_bias=config.use_bias,
      use_sdpa=config.use_sdpa,
      rescale_logits=rescale_logits,
    )
    if use_variate_attention:
      self.pre_var_attn_ln = nn.RMSNorm(config.model_dims, eps=_RMS_EPS)
      self.post_var_attn_ln = nn.RMSNorm(config.model_dims, eps=_RMS_EPS)
      self.var_attn = MultiHeadAttention(
        config.num_heads,
        config.model_dims,
        use_rotary_position_embeddings=config.use_rope_var,
        qk_norm=config.qk_norm,
        v_norm=config.v_norm,
        causal_attention=False,
        use_bias=config.use_bias,
        use_sdpa=config.use_sdpa,
        rescale_logits=rescale_logits,
      )
    self.pre_ff_ln = nn.RMSNorm(config.model_dims, eps=_RMS_EPS)
    self.post_ff_ln = nn.RMSNorm(config.model_dims, eps=_RMS_EPS)
    self.ff0 = nn.Linear(config.model_dims, config.hidden_dims, bias=config.use_bias)
    self.ff1 = nn.Linear(config.hidden_dims, config.model_dims, bias=config.use_bias)
    self.activation = util.get_activation_fn(config.ff_activation)

  def __call__(
    self, input_embeddings: mx.array, patch_mask: mx.array
  ) -> tuple[mx.array, mx.array]:
    b, v, n, d = input_embeddings.shape
    seq_input = self.pre_seq_attn_ln(input_embeddings).reshape(b * v, n, d)
    seq_output, seq_mask = self.seq_attn(
      seq_input, patch_mask=patch_mask.reshape(b * v, n)
    )
    h1 = self.post_seq_attn_ln(seq_output.reshape(b, v, n, d)) + input_embeddings
    if self.use_variate_attention:
      var_input = mx.transpose(self.pre_var_attn_ln(h1), (0, 2, 1, 3)).reshape(
        b * n, v, d
      )
      var_mask = mx.transpose(patch_mask, (0, 2, 1)).reshape(b * n, v)
      var_output, _ = self.var_attn(var_input, patch_mask=var_mask)
      var_output = mx.transpose(var_output.reshape(b, n, v, d), (0, 2, 1, 3))
      h2 = self.post_var_attn_ln(var_output) + h1
    else:
      h2 = h1
    ff_output = self.ff1(self.activation(self.ff0(self.pre_ff_ln(h2))))
    return self.post_ff_ln(ff_output) + h2, seq_mask


class StackedMixingTransformer(nn.Module):
  def __init__(
    self,
    config: configs.StackedTransformersConfig,
    use_variate_attention: bool = True,
  ):
    super().__init__()
    self.layers = [
      MixingTransformer(config.transformer, use_variate_attention)
      for _ in range(config.num_layers)
    ]

  def __call__(
    self, input_embeddings: mx.array, patch_mask: mx.array
  ) -> tuple[mx.array, list[mx.array]]:
    output = input_embeddings
    masks = []
    for layer in self.layers:
      output, mask = layer(output, patch_mask)
      masks.append(mask)
    return output, masks
