from typing import Any, Dict

import torch

from ..base import BaseConditioner
from .latents_concat_embedder_config import LatentsConcatEmbedderConfig


class RawConcatEmbedder(BaseConditioner):
    """
    Concatenate raw image/mask tensors for conditioning without VAE encoding.

    Args:
        config (LatentsConcatEmbedderConfig): Configs to create the embedder
    """

    def __init__(self, config: LatentsConcatEmbedderConfig):
        BaseConditioner.__init__(self, config)

    def forward(self, batch: Dict[str, Any], vae=None, *args, **kwargs) -> dict:
        dims_list = []
        for image_key in self.config.image_keys:
            dims_list.append(batch[image_key].shape[-2:])
        for mask_key in self.config.mask_keys:
            dims_list.append(batch[mask_key].shape[-2:])
        assert all(
            dims == dims_list[0] for dims in dims_list
        ), "All images and masks must have the same dimensions."

        outputs = []

        for mask_key in self.config.mask_keys:
            outputs.append(batch[mask_key])

        for image_key in self.config.image_keys:
            outputs.append(batch[image_key])

        outputs = torch.concat(outputs, dim=1)
        outputs = {self.dim2outputkey[outputs.dim()]: outputs}

        return outputs
