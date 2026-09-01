"""PyTorch runtime adapter for the shared TimesFM 3 forecaster."""

from __future__ import annotations

import gc
import os
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from .. import configs, util
from .. import model as torch_model_lib

_GC_MEMORY_THRESHOLD = 0.9


def make_torch_model(config: Any) -> torch_model_lib.TimesFM3Torch:
  """Builds a PyTorch model using a forecaster configuration."""
  resblock_config = (
    config.residual_block_config
    if config.residual_block_config is not None
    else configs.ResidualBlockConfig(
      hidden_dims=1280,
      output_dims=1280,
      use_bias=False,
      activation="relu",
    )
  )
  transformer_config = (
    config.transformer_config
    if config.transformer_config is not None
    else configs.StackedTransformersConfig(
      num_layers=20,
      transformer=configs.TransformerConfig(
        model_dims=1280,
        hidden_dims=1280,
        num_heads=16,
        attention_norm="rms",
        feedforward_norm="rms",
        qk_norm="rms",
        use_rope_seq=True,
        use_rope_var=True,
        use_bias=False,
        ff_activation="relu",
        deterministic=True,
      ),
    )
  )
  model = torch_model_lib.TimesFM3Torch(
    input_patch_len=config.input_patch_length,
    output_patch_len=config.output_patch_length,
    quantiles=config.quantiles,
    use_variate_attention=config.use_variate_attention,
    value_clip=config.value_clip,
    input_transform=config.input_transform,
    use_stitching=config.use_stitching,
    use_linear_detrending=config.use_linear_detrending,
    linear_detrending_threshold=config.linear_detrending_threshold,
    use_iterative_cpm_revin=config.use_iterative_cpm_revin,
    use_frozen_running_stats=config.use_frozen_running_stats,
    residual_block_config=resblock_config,
    transformer_config=transformer_config,
  )
  model.eval()
  input_dim = 2 * (model.input_patch_len + model.output_patch_len)
  model.pre_transformer_resblock.set_input_dims(input_dim)
  return model


class TorchBackend:
  """Owns PyTorch model loading, tensor conversion, and cache cleanup."""

  def __init__(self, config: Any):
    self.config = config
    self.device = (
      torch.device(config.device)
      if config.device is not None
      else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    self.dtype = {
      "float32": torch.float32,
      "float16": torch.float16,
      "bfloat16": torch.bfloat16,
    }[config.dtype]
    checkpoint_path = os.path.expanduser(config.checkpoint_path)
    is_local_dir = os.path.isdir(checkpoint_path)
    is_local_file = os.path.isfile(checkpoint_path)

    if is_local_dir or not is_local_file:
      self.model = torch_model_lib.TimesFM3Torch.from_pretrained(
        checkpoint_path,
        cache_dir=config.cache_dir,
        force_download=config.force_download,
        token=config.token,
        revision=config.revision,
        local_files_only=config.local_files_only,
      )
    else:
      self.model = make_torch_model(config)
      if checkpoint_path.endswith(".safetensors"):
        state_dict = util.load_safetensors(checkpoint_path, device=self.device)
        self.model.load_state_dict(state_dict)
      elif checkpoint_path.endswith((".pth", ".pt")):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
      else:
        raise ValueError(
          f"Unsupported checkpoint path format: {checkpoint_path}. "
          "Expected .safetensors or .pth / .pt file."
        )

    self.model.to(self.device)
    self.model.eval()

  def decode(
    self,
    *,
    target: np.ndarray,
    horizon: int,
    past_only_covariates: np.ndarray | None,
    past_future_covariates: np.ndarray | None,
    mask: np.ndarray,
  ) -> np.ndarray:
    target_tensor = torch.from_numpy(target).to(self.device, dtype=torch.float32)
    mask_tensor = torch.from_numpy(mask).to(self.device, dtype=torch.bool)
    po_tensor = (
      torch.from_numpy(past_only_covariates).to(self.device, dtype=torch.float32)
      if past_only_covariates is not None
      else None
    )
    pf_tensor = (
      torch.from_numpy(past_future_covariates).to(self.device, dtype=torch.float32)
      if past_future_covariates is not None
      else None
    )
    autocast = (
      torch.autocast(device_type=self.device.type, dtype=self.dtype)
      if self.dtype != torch.float32
      else nullcontext()
    )
    with torch.inference_mode(), autocast:
      output = self.model.decode(
        target=target_tensor,
        horizon=horizon,
        past_only_covariates=po_tensor,
        past_future_covariates=pf_tensor,
        mask=mask_tensor,
      )
    return output.float().cpu().numpy()

  def cleanup(self) -> None:
    if self.device.type == "cuda":
      allocated = torch.cuda.memory_allocated(self.device)
      total = torch.cuda.get_device_properties(self.device).total_memory
      if total > 0 and allocated / total > _GC_MEMORY_THRESHOLD:
        gc.collect()
        torch.cuda.empty_cache()
        return
    gc.collect()
