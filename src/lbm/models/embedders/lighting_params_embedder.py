from __future__ import annotations

from pydantic.dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn

from .base import BaseConditioner, BaseConditionerConfig


class GaussianFourierFeatures(nn.Module):
    """Gaussian Fourier feature mapping for [batch, features] inputs."""

    def __init__(self, in_features: int, mapping_size: int = 256, scale: float = 10.0):
        super().__init__()
        b = torch.randn(mapping_size, in_features) * scale
        self.register_buffer("B", b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * torch.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


@dataclass
class LightingParamsEmbedderConfig(BaseConditionerConfig):
    """Configuration for lighting parameter embeddings."""

    mapping_size: int = 256
    scale: float = 10.0
    hidden_dim: int = 512
    out_dim: int = 768
    seq_len: int = 1
    input_key: str = "lighting_params"


class LightingParamsEmbedder(BaseConditioner):
    """Embed lighting parameters into cross-attention conditioning."""

    def __init__(self, config: LightingParamsEmbedderConfig):
        BaseConditioner.__init__(self, config)
        self.fourier = GaussianFourierFeatures(
            in_features=7,
            mapping_size=config.mapping_size,
            scale=config.scale,
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.mapping_size * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.out_dim * config.seq_len),
        )
        self.out_dim = config.out_dim
        self.seq_len = config.seq_len

    def forward(
        self,
        batch: Dict[str, Any],
        force_zero_embedding: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        lighting_params = batch[self.input_key]
        if lighting_params.dim() == 1:
            lighting_params = lighting_params.unsqueeze(0)

        if force_zero_embedding:
            zeros = torch.zeros(
                lighting_params.shape[0],
                self.seq_len,
                self.out_dim,
                device=lighting_params.device,
                dtype=lighting_params.dtype,
            )
            return {self.dim2outputkey[zeros.dim()]: zeros}

        features = self.fourier(lighting_params)
        embedding = self.mlp(features)
        embedding = embedding.view(lighting_params.shape[0], self.seq_len, self.out_dim)
        return {self.dim2outputkey[embedding.dim()]: embedding}
