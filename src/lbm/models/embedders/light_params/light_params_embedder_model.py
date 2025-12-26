from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ..base import BaseConditioner
from .light_params_embedder_config import LightParamsEmbedderConfig


class LightParamsEmbedder(BaseConditioner):
    """
    Encode 7 light parameters via Fourier features and an MLP into cross-attention tokens.
    """

    def __init__(self, config: LightParamsEmbedderConfig):
        super().__init__(config)
        self.config = config
        frequencies = 2 ** torch.arange(config.num_frequencies).float() * torch.pi
        self.register_buffer("frequencies", frequencies)

        input_dim = 7 * (
            (2 * config.num_frequencies) + (1 if config.include_input else 0)
        )
        layers = []
        current_dim = input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        output_dim = config.num_tokens * config.embedding_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def _fourier_features(self, params: torch.Tensor) -> torch.Tensor:
        scaled = params.unsqueeze(-1) * self.frequencies
        encoded = [torch.sin(scaled), torch.cos(scaled)]
        if self.config.include_input:
            encoded.insert(0, params.unsqueeze(-1))
        encoded = torch.cat(encoded, dim=-1)
        return encoded.flatten(start_dim=1)

    def forward(
        self, batch: Dict[str, Any], force_zero_embedding: bool = False, *args, **kwargs
    ) -> dict:
        params = batch[self.input_key]
        if not isinstance(params, torch.Tensor):
            params = torch.tensor(params, device=self.device)
        if params.ndim == 1:
            params = params.unsqueeze(0)
        params = params.to(device=self.device, dtype=self.dtype)

        if force_zero_embedding:
            output = torch.zeros(
                (params.shape[0], self.config.num_tokens, self.config.embedding_dim),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            features = self._fourier_features(params)
            output = self.mlp(features)
            output = output.view(
                params.shape[0], self.config.num_tokens, self.config.embedding_dim
            )

        return {self.dim2outputkey[output.dim()]: output}
