from pydantic.dataclasses import dataclass
import torch
from torch import nn

from .base import BaseConditioner, BaseConditionerConfig


class GaussianFourierFeatures(nn.Module):
    """Gaussian Fourier feature mapping for [batch, features] inputs."""

    def __init__(self, in_features, mapping_size=256, scale=10.0):
        super().__init__()
        b = torch.randn(mapping_size, in_features) * scale
        self.register_buffer("B", b)

    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


@dataclass
class LightingParamsEmbedderConfig(BaseConditionerConfig):
    """Configuration for lighting parameter embeddings."""
    mapping_size: int = 256
    scale: float = 10.0
    hidden_dim: int = 512
    out_dim: int = 768
    seq_len: int = 4
    # param_groups: list = [[0, 1], [2], [3, 4, 5], [6]]
    param_groups: list[list[int]] | None = None
    input_key: str = "lighting_params"
    # param_groups = [[0, 1], [2], [3, 4, 5], [6]]


class LightingParamsEmbedder(BaseConditioner):
    """Embed lighting parameters into cross-attention conditioning."""

    def __init__(self, config: LightingParamsEmbedderConfig):
        BaseConditioner.__init__(self, config)
        self.param_groups = config.param_groups #or [[idx] for idx in range(7)]
        if config.seq_len != len(self.param_groups):
            raise ValueError(
                "lighting_embedder_config.seq_len must match the number of param_groups "
                f"({len(self.param_groups)})."
            )
        self.group_fourier = nn.ModuleList(
            [
                GaussianFourierFeatures(
                    in_features=len(group),
                    mapping_size=config.mapping_size,
                    scale=config.scale,
                )
                for group in self.param_groups
            ]
        )
        self.group_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.mapping_size * 2, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(config.hidden_dim, config.out_dim),
                )
                for _ in self.param_groups
            ]
        )
        self.out_dim = config.out_dim
        self.seq_len = config.seq_len

    def forward(self, batch, force_zero_embedding=False, *args, **kwargs):
        lighting_params = batch[self.input_key]
        if lighting_params.dim() == 1:
            lighting_params = lighting_params.unsqueeze(0)

        embeddings = []
        for group, fourier, mlp in zip(self.param_groups, self.group_fourier, self.group_mlps):
            group_params = lighting_params[:, group]
            features = fourier(group_params)
            embeddings.append(mlp(features))
        embedding = torch.stack(embeddings, dim=1)
        # 1 in 100 chance to print max mean min of embedding for debugging
        if torch.rand(1).item() < 0.01:
            print(
                f"LightingParamsEmbedder embedding stats: max {embedding.max().item():.4f}, "
                f"mean {embedding.mean().item():.4f}, min {embedding.min().item():.4f}"
            )
        return {self.dim2outputkey[embedding.dim()]: embedding}
