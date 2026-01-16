from pathlib import Path
from typing import Dict, Optional, Union

import torch

from .model import MoGeModel


def intrinsics_to_fov(intrinsics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    focal_x = intrinsics[..., 0, 0]
    focal_y = intrinsics[..., 1, 1]
    return 2 * torch.atan(0.5 / focal_x), 2 * torch.atan(0.5 / focal_y)


class MoGeDepthModel:
    def __init__(self, model: torch.nn.Module, default_use_fp16: bool = False) -> None:
        self.model = model.eval()
        self._default_use_fp16 = default_use_fp16

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, Path],
            device: Union[str, torch.device] = "cuda",
            use_fp16: bool = False,
            **hf_kwargs,
    ) -> "MoGeDepthModel":
        model = MoGeModel.from_pretrained(pretrained_model_name_or_path, **hf_kwargs).to(device).eval()
        if use_fp16:
            model.half()
        return cls(model, default_use_fp16=use_fp16)

    @torch.inference_mode()
    def infer_depth(
            self,
            images: torch.Tensor,
            fov_x: Optional[float] = None,
            resolution_level: int = 9,
            num_tokens: Optional[int] = None,
            use_fp16: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        if use_fp16 is None:
            use_fp16 = self._default_use_fp16

        with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=use_fp16 and self.dtype != torch.float16,
        ):
            output = self.model.infer(
                images,
                fov_x=fov_x,
                resolution_level=resolution_level,
                num_tokens=num_tokens,
            )

        if isinstance(output, dict):
            depth = output["depth"]
            intrinsics = output["intrinsics"]
        else:
            depth, intrinsics = output

        fov_x_rad, fov_y_rad = intrinsics_to_fov(intrinsics)
        return {
            "depth": depth,
            "fov_x": torch.rad2deg(fov_x_rad),
            "fov_y": torch.rad2deg(fov_y_rad),
        }


class DepthBatchInferencer:
    def __init__(self, model: MoGeDepthModel) -> None:
        self.model = model

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, Path],
            device: Union[str, torch.device] = "cuda",
            use_fp16: bool = False,
            **hf_kwargs,
    ) -> "DepthBatchInferencer":
        model = MoGeDepthModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            device=device,
            use_fp16=use_fp16,
            **hf_kwargs,
        )
        return cls(model)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def infer_batch(
            self,
            images: torch.Tensor,
            fov_x: Optional[float] = None,
            resolution_level: int = 9,
            num_tokens: Optional[int] = None,
            use_fp16: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.model.infer_depth(
            images,
            fov_x=fov_x,
            resolution_level=resolution_level,
            num_tokens=num_tokens,
            use_fp16=use_fp16,
        )
