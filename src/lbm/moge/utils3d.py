from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np
import torch


def _normalize_tuple(value, length: int) -> Tuple[int, ...]:
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        value = tuple(value)
        if len(value) != length:
            raise ValueError(f"Expected tuple of length {length}, got {len(value)}")
        return value
    return (value,) * length


def _sliding_window_1d(x: np.ndarray, window_size: int, stride: int, axis: int) -> np.ndarray:
    if x.shape[axis] < window_size:
        raise ValueError("window_size larger than axis length")
    axis = axis % x.ndim
    shape = (*x.shape[:axis], (x.shape[axis] - window_size + 1) // stride, *x.shape[axis + 1 :], window_size)
    strides = (*x.strides[:axis], stride * x.strides[axis], *x.strides[axis + 1 :], x.strides[axis])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def _sliding_window_nd(x: np.ndarray, window_size: Tuple[int, ...], stride: Tuple[int, ...], axis: Tuple[int, ...]) -> np.ndarray:
    axis = tuple(a % x.ndim for a in axis)
    for idx, ax in enumerate(axis):
        x = _sliding_window_1d(x, window_size[idx], stride[idx], ax)
    return x


@dataclass(frozen=True)
class _NumpyNamespace:
    def masked_nearest_resize(
        self,
        points: np.ndarray,
        uv: np.ndarray,
        *,
        mask: np.ndarray,
        size: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mask is None:
            raise ValueError("mask is required for masked_nearest_resize")
        height, width = size
        target_size = (width, height)

        points_resized = cv2.resize(points, target_size, interpolation=cv2.INTER_NEAREST)
        uv_resized = cv2.resize(uv, target_size, interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(mask.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST).astype(bool)

        points_filtered = points_resized[mask_resized]
        uv_filtered = uv_resized[mask_resized]
        return points_filtered, uv_filtered, mask_resized

    def sliding_window(
        self,
        x: np.ndarray,
        window_size: Tuple[int, int],
        stride: int,
        *,
        axis: Tuple[int, int] = (-2, -1),
    ) -> np.ndarray:
        window_size = _normalize_tuple(window_size, 2)
        stride = _normalize_tuple(stride, 2)
        axis = _normalize_tuple(axis, 2)
        return _sliding_window_nd(x, window_size, stride, axis)


@dataclass(frozen=True)
class _TorchNamespace:
    def sliding_window(
        self,
        x: torch.Tensor,
        *,
        window_size: int | Tuple[int, int],
        stride: int | Tuple[int, int] = 1,
        dim: int | Tuple[int, int] = -1,
    ) -> torch.Tensor:
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        window_size = _normalize_tuple(window_size, len(dims))
        stride = _normalize_tuple(stride, len(dims))
        for idx, axis in enumerate(dims):
            axis = axis % x.ndim
            x = x.unfold(axis, window_size[idx], stride[idx])
        return x

    def sliding_window_2d(
        self,
        x: torch.Tensor,
        *,
        window_size: int | Tuple[int, int],
        stride: int | Tuple[int, int],
        dim: Tuple[int, int] = (-2, -1),
    ) -> torch.Tensor:
        return self.sliding_window(x, window_size=window_size, stride=stride, dim=dim)

    def uv_map(self, size: Tuple[int, int], *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        height, width = size
        u = torch.linspace(0.5 / width, (width - 0.5) / width, width, device=device, dtype=dtype)
        v = torch.linspace(0.5 / height, (height - 0.5) / height, height, device=device, dtype=dtype)
        u, v = torch.meshgrid(u, v, indexing="xy")
        return torch.stack([u, v], dim=-1)

    def intrinsics_from_focal_center(
        self,
        fx: torch.Tensor | float,
        fy: torch.Tensor | float,
        cx: torch.Tensor | float,
        cy: torch.Tensor | float,
    ) -> torch.Tensor:
        fx_t = torch.as_tensor(fx)
        fy_t = torch.as_tensor(fy, device=fx_t.device, dtype=fx_t.dtype)
        cx_t = torch.as_tensor(cx, device=fx_t.device, dtype=fx_t.dtype)
        cy_t = torch.as_tensor(cy, device=fx_t.device, dtype=fx_t.dtype)
        fx_t, fy_t, cx_t, cy_t = torch.broadcast_tensors(fx_t, fy_t, cx_t, cy_t)
        zeros = torch.zeros_like(fx_t)
        ones = torch.ones_like(fx_t)
        stacked = torch.stack([fx_t, zeros, cx_t, zeros, fy_t, cy_t, zeros, zeros, ones], dim=-1)
        return stacked.reshape(fx_t.shape + (3, 3))

    def depth_map_to_point_map(self, depth: torch.Tensor, *, intrinsics: torch.Tensor) -> torch.Tensor:
        height, width = depth.shape[-2:]
        uv = self.uv_map((height, width), device=depth.device, dtype=depth.dtype)

        leading_shape = depth.shape[:-2]
        uv = uv.expand(*leading_shape, height, width, 2)

        fx = intrinsics[..., 0, 0]
        fy = intrinsics[..., 1, 1]
        cx = intrinsics[..., 0, 2]
        cy = intrinsics[..., 1, 2]

        x = (uv[..., 0] - cx[..., None, None]) / fx[..., None, None] * depth
        y = (uv[..., 1] - cy[..., None, None]) / fy[..., None, None] * depth
        z = depth
        return torch.stack([x, y, z], dim=-1)


np = _NumpyNamespace()
pt = _TorchNamespace()

__all__ = ["np", "pt"]
