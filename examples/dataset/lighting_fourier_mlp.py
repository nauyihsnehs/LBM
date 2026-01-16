import torch
from torch import nn


class GaussianFourierFeatures(nn.Module):
    """Gaussian Fourier feature mapping for 2D [batch, features] inputs."""

    def __init__(self, in_features, mapping_size=256, scale=10.0):
        super().__init__()
        b = torch.randn(mapping_size, in_features) * scale
        self.register_buffer("B", b)

    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class LightingToTextEmbedding(nn.Module):
    """Maps parameters to a text-embedding vector."""

    def __init__(self, mapping_size=256, scale=10.0, hidden_dim=512, out_dim=768):
        super().__init__()
        self.fourier = GaussianFourierFeatures(
            in_features=7, mapping_size=mapping_size, scale=scale
        )
        self.mlp = nn.Sequential(
            nn.Linear(mapping_size * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, lighting_params):
        features = self.fourier(lighting_params)
        print(features.shape)
        return self.mlp(features)


def example():
    model = LightingToTextEmbedding(mapping_size=256, scale=10.0, hidden_dim=512, out_dim=768)
    lighting = torch.tensor(
        [
            [1.0, 0.8, 0.6, 2.5, 0.3, 1.2, -0.5],
            [0.2, 0.4, 0.9, 1.2, 0.6, -0.1, 2.3],
        ]
    )
    out = model(lighting)
    print(out.shape)


if __name__ == "__main__":
    example()
