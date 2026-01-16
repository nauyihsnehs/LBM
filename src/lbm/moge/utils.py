import importlib
import importlib.util
import math
import os
import time
import warnings
from functools import wraps, partial
from typing import *

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve

from . import utils3d


def catch_exception(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            import traceback
            print(f"Exception in {fn.__name__}", end='r')
            # print({', '.join(repr(arg) for arg in args)}, {', '.join(f'{k}={v!r}' for k, v in kwargs.items())})
            traceback.print_exc(chain=False)
            time.sleep(0.1)
            return None

    return wrapper


class CallbackOnException:
    def __init__(self, callback: Callable, exception: type):
        self.exception = exception
        self.callback = callback

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(exc_val, self.exception):
            self.callback()
            return True
        return False


def traverse_nested_dict_keys(d: Dict[str, Dict]) -> Generator[Tuple[str, ...], None, None]:
    for k, v in d.items():
        if isinstance(v, dict):
            for sub_key in traverse_nested_dict_keys(v):
                yield (k,) + sub_key
        else:
            yield (k,)


def get_nested_dict(d: Dict[str, Dict], keys: Tuple[str, ...], default: Any = None):
    for k in keys:
        d = d.get(k, default)
        if d is None:
            break
    return d


def set_nested_dict(d: Dict[str, Dict], keys: Tuple[str, ...], value: Any):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def key_average(list_of_dicts: list) -> Dict[str, Any]:
    """
    Returns a dictionary with the average value of each key in the input list of dictionaries.
    """
    _nested_dict_keys = set()
    for d in list_of_dicts:
        _nested_dict_keys.update(traverse_nested_dict_keys(d))
    _nested_dict_keys = sorted(_nested_dict_keys)
    result = {}
    for k in _nested_dict_keys:
        values = []
        for d in list_of_dicts:
            v = get_nested_dict(d, k)
            if v is not None and not math.isnan(v):
                values.append(v)
        avg = sum(values) / len(values) if values else float('nan')
        set_nested_dict(result, k, avg)
    return result


def flatten_nested_dict(d: Dict[str, Any], parent_key: Tuple[str, ...] = None) -> Dict[Tuple[str, ...], Any]:
    """
    Flattens a nested dictionary into a single-level dictionary, with keys as tuples.
    """
    items = []
    if parent_key is None:
        parent_key = ()
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, MutableMapping):
            items.extend(flatten_nested_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unflattens a single-level dictionary into a nested dictionary, with keys as tuples.
    """
    result = {}
    for k, v in d.items():
        sub_dict = result
        for k_ in k[:-1]:
            if k_ not in sub_dict:
                sub_dict[k_] = {}
            sub_dict = sub_dict[k_]
        sub_dict[k[-1]] = v
    return result


def read_jsonl(file):
    import json
    with open(file, 'r') as f:
        data = f.readlines()
    return [json.loads(line) for line in data]


def write_jsonl(data: List[dict], file):
    import json
    with open(file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def to_hierachical_dataframe(data: List[Dict[Tuple[str, ...], Any]]):
    import pandas as pd
    data = [flatten_nested_dict(d) for d in data]
    df = pd.DataFrame(data)
    df = df.sort_index(axis=1)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def recursive_replace(d: Union[List, Dict, str], mapping: Dict[str, str]):
    if isinstance(d, str):
        for old, new in mapping.items():
            d = d.replace(old, new)
    elif isinstance(d, list):
        for i, item in enumerate(d):
            d[i] = recursive_replace(item, mapping)
    elif isinstance(d, dict):
        for k, v in d.items():
            d[k] = recursive_replace(v, mapping)
    return d


class timeit:
    _history: Dict[str, List['timeit']] = {}

    def __init__(self, name: str = None, verbose: bool = True, average: bool = False):
        self.name = name
        self.verbose = verbose
        self.start = None
        self.end = None
        self.average = average
        if average and name not in timeit._history:
            timeit._history[name] = []

    def __call__(self, func: Callable):
        import inspect
        if inspect.iscoroutinefunction(func):
            async def wrapper(*args, **kwargs):
                with timeit(self.name or func.__qualname__):
                    ret = await func(*args, **kwargs)
                return ret

            return wrapper
        else:
            def wrapper(*args, **kwargs):
                with timeit(self.name or func.__qualname__):
                    ret = func(*args, **kwargs)
                return ret

            return wrapper

    def __enter__(self):
        self.start = time.time()
        return self

    @property
    def time(self) -> float:
        assert self.start is not None, "Time not yet started."
        assert self.end is not None, "Time not yet ended."
        return self.end - self.start

    @property
    def average_time(self) -> float:
        assert self.average, "Average time not available."
        return sum(t.time for t in timeit._history[self.name]) / len(timeit._history[self.name])

    @property
    def history(self) -> List['timeit']:
        return timeit._history.get(self.name, [])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        if self.average:
            timeit._history[self.name].append(self)
        if self.verbose:
            if self.average:
                avg = self.average_time
                print(f"{self.name or 'It'} took {avg:.6f} seconds in average.")
            else:
                print(f"{self.name or 'It'} took {self.time:.6f} seconds.")


def strip_common_prefix_suffix(strings: List[str]) -> List[str]:
    first = strings[0]

    for start in range(len(first)):
        if any(s[start] != strings[0][start] for s in strings):
            break

    for end in range(1, min(len(s) for s in strings)):
        if any(s[-end] != first[-end] for s in strings):
            break

    return [s[start:len(s) - end + 1] for s in strings]


def multithead_execute(inputs: List[Any], num_workers: int, pbar=None):
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    if pbar is not None:
        pbar.total = len(inputs) if hasattr(inputs, '__len__') else None
    else:
        pbar = tqdm(total=len(inputs) if hasattr(inputs, '__len__') else None)

    def decorator(fn: Callable):
        with (
            ThreadPoolExecutor(max_workers=num_workers) as executor,
            pbar
        ):
            pbar.refresh()

            @catch_exception
            @suppress_traceback
            def _fn(input):
                ret = fn(input)
                pbar.update()
                return ret

            executor.map(_fn, inputs)
            executor.shutdown(wait=True)

    return decorator


def suppress_traceback(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            e.__traceback__ = e.__traceback__.tb_next.tb_next
            raise

    return wrapper


class no_warnings:
    def __init__(self, action: str = 'ignore', **kwargs):
        self.action = action
        self.filter_kwargs = kwargs

    def __call__(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter(self.action, **self.filter_kwargs)
                return fn(*args, **kwargs)

        return wrapper

    def __enter__(self):
        self.warnings_manager = warnings.catch_warnings()
        self.warnings_manager.__enter__()
        warnings.simplefilter(self.action, **self.filter_kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.warnings_manager.__exit__(exc_type, exc_val, exc_tb)


def import_file_as_module(file_path: Union[str, os.PathLike], module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weighted_mean_numpy(x: np.ndarray, w: np.ndarray = None, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False, eps: float = 1e-7) -> np.ndarray:
    if w is None:
        return np.mean(x, axis=axis)
    else:
        w = w.astype(x.dtype)
        return (x * w).mean(axis=axis) / np.clip(w.mean(axis=axis), eps, None)


def harmonic_mean_numpy(x: np.ndarray, w: np.ndarray = None, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False, eps: float = 1e-7) -> np.ndarray:
    if w is None:
        return 1 / (1 / np.clip(x, eps, None)).mean(axis=axis)
    else:
        w = w.astype(x.dtype)
        return 1 / (weighted_mean_numpy(1 / (x + eps), w, axis=axis, keepdims=keepdims, eps=eps) + eps)


def normalized_view_plane_uv_numpy(width: int, height: int, aspect_ratio: float = None, dtype: np.dtype = np.float32) -> np.ndarray:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = np.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype)
    v = np.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype)
    u, v = np.meshgrid(u, v, indexing='xy')
    uv = np.stack([u, v], axis=-1)
    return uv


def focal_to_fov_numpy(focal: np.ndarray):
    return 2 * np.arctan(0.5 / focal)


def fov_to_focal_numpy(fov: np.ndarray):
    return 0.5 / np.tan(fov / 2)


def intrinsics_to_fov_numpy(intrinsics: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fov_x = focal_to_fov_numpy(intrinsics[..., 0, 0])
    fov_y = focal_to_fov_numpy(intrinsics[..., 1, 1])
    return fov_x, fov_y


def point_map_to_depth_legacy_numpy(points: np.ndarray):
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    uv = normalized_view_plane_uv_numpy(width, height, dtype=points.dtype)  # (H, W, 2)
    _, uv = np.broadcast_arrays(points[..., :2], uv)

    # Solve least squares problem
    b = (uv * points[..., 2:]).reshape(*points.shape[:-3], -1)  # (..., H * W * 2)
    A = np.stack([points[..., :2], -uv], axis=-1).reshape(*points.shape[:-3], -1, 2)  # (..., H * W * 2, 2)

    M = A.swapaxes(-2, -1) @ A
    solution = (np.linalg.inv(M + 1e-6 * np.eye(2)) @ (A.swapaxes(-2, -1) @ b[..., None])).squeeze(-1)
    focal, shift = solution

    depth = points[..., 2] + shift[..., None, None]
    fov_x = np.arctan(width / diagonal / focal) * 2
    fov_y = np.arctan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def solve_optimal_focal_shift(uv: np.ndarray, xyz: np.ndarray):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    xy_proj = xy / (z + optim_shift)[:, None]
    optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

    return optim_shift, optim_focal


def solve_optimal_shift(uv: np.ndarray, xyz: np.ndarray, focal: float):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[:, None]
        err = (focal * xy_proj - uv).ravel()
        return err

    solution = least_squares(partial(fn, uv, xy, z), x0=0, ftol=1e-3, method='lm')
    optim_shift = solution['x'].squeeze().astype(np.float32)

    return optim_shift


def recover_focal_shift_numpy(points: np.ndarray, mask: np.ndarray = None, focal: float = None, downsample_size: Tuple[int, int] = (64, 64)):
    import cv2
    assert points.shape[-1] == 3, "Points should (H, W, 3)"

    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    uv = normalized_view_plane_uv_numpy(width=width, height=height)

    if mask is None:
        points_lr = cv2.resize(points, downsample_size, interpolation=cv2.INTER_LINEAR).reshape(-1, 3)
        uv_lr = cv2.resize(uv, downsample_size, interpolation=cv2.INTER_LINEAR).reshape(-1, 2)
    else:
        points_lr, uv_lr, mask_lr = utils3d.np.masked_nearest_resize(points, uv, mask=mask, size=downsample_size)

    if points_lr.size < 2:
        return 1., 0.

    if focal is None:
        shift, focal = solve_optimal_focal_shift(uv_lr, points_lr)
    else:
        shift = solve_optimal_shift(uv_lr, points_lr, focal)

    return focal, shift


def norm3d(x: np.ndarray) -> np.ndarray:
    "Faster `np.linalg.norm(x, axis=-1)` for 3D vectors"
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]) + np.square(x[..., 2]))


def depth_occlusion_edge_numpy(depth: np.ndarray, mask: np.ndarray, thickness: int = 1, tol: float = 0.1):
    disp = np.where(mask, 1 / depth, 0)
    disp_pad = np.pad(disp, (thickness, thickness), constant_values=0)
    mask_pad = np.pad(mask, (thickness, thickness), constant_values=False)
    kernel_size = 2 * thickness + 1
    disp_window = utils3d.np.sliding_window(disp_pad, (kernel_size, kernel_size), 1, axis=(-2, -1))  # [..., H, W, kernel_size ** 2]
    mask_window = utils3d.np.sliding_window(mask_pad, (kernel_size, kernel_size), 1, axis=(-2, -1))  # [..., H, W, kernel_size ** 2]

    disp_mean = weighted_mean_numpy(disp_window, mask_window, axis=(-2, -1))
    fg_edge_mask = mask & (disp > (1 + tol) * disp_mean)
    bg_edge_mask = mask & (disp_mean > (1 + tol) * disp)

    edge_mask = (cv2.dilate(fg_edge_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=thickness) > 0) \
                & (cv2.dilate(bg_edge_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=thickness) > 0)

    return edge_mask


def disk_kernel(radius: int) -> np.ndarray:
    """
    Generate disk kernel with given radius.
    
    Args:
        radius (int): Radius of the disk (in pixels).
    
    Returns:
        np.ndarray: (2*radius+1, 2*radius+1) normalized convolution kernel.
    """
    # Create coordinate grid centered at (0,0)
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    # Generate disk: region inside circle with radius R is 1
    kernel = ((X ** 2 + Y ** 2) <= radius ** 2).astype(np.float32)
    # Normalize the kernel
    kernel /= np.sum(kernel)
    return kernel


def disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """
    Apply disk blur to an image using FFT convolution.

    Args:
        image (np.ndarray): Input image, can be grayscale or color.
        radius (int): Blur radius (in pixels).

    Returns:
        np.ndarray: Blurred image.
    """
    if radius == 0:
        return image
    kernel = disk_kernel(radius)
    if image.ndim == 2:
        blurred = fftconvolve(image, kernel, mode='same')
    elif image.ndim == 3:
        channels = []
        for i in range(image.shape[2]):
            blurred_channel = fftconvolve(image[..., i], kernel, mode='same')
            channels.append(blurred_channel)
        blurred = np.stack(channels, axis=-1)
    else:
        raise ValueError("Image must be 2D or 3D.")
    return blurred


def depth_of_field(
        img: np.ndarray,
        disp: np.ndarray,
        focus_disp: float,
        max_blur_radius: int = 10,
) -> np.ndarray:
    """
    Apply depth of field effect to an image.

    Args:
        img (numpy.ndarray): (H, W, 3) input image.
        depth (numpy.ndarray): (H, W) depth map of the scene.
        focus_depth (float): Focus depth of the lens.
        strength (float): Strength of the depth of field effect.
        max_blur_radius (int): Maximum blur radius (in pixels).
        
    Returns:
        numpy.ndarray: (H, W, 3) output image with depth of field effect applied.
    """
    # Precalculate dialated depth map for each blur radius
    max_disp = np.max(disp)
    disp = disp / max_disp
    focus_disp = focus_disp / max_disp
    dilated_disp = []
    for radius in range(max_blur_radius + 1):
        dilated_disp.append(cv2.dilate(disp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)), iterations=1))

    # Determine the blur radius for each pixel based on the depth map
    blur_radii = np.clip(abs(disp - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
    for radius in range(max_blur_radius + 1):
        dialted_blur_radii = np.clip(abs(dilated_disp[radius] - focus_disp) * max_blur_radius, 0, max_blur_radius).astype(np.int32)
        mask = (dialted_blur_radii >= radius) & (dialted_blur_radii >= blur_radii) & (dilated_disp[radius] > disp)
        blur_radii[mask] = dialted_blur_radii[mask]
    blur_radii = np.clip(blur_radii, 0, max_blur_radius)
    blur_radii = cv2.blur(blur_radii, (5, 5))

    # Precalculate the blured image for each blur radius
    unique_radii = np.unique(blur_radii)
    precomputed = {}
    for radius in range(max_blur_radius + 1):
        if radius not in unique_radii:
            continue
        precomputed[radius] = disk_blur(img, radius)

    # Composit the blured image for each pixel
    output = np.zeros_like(img)
    for r in unique_radii:
        mask = blur_radii == r
        output[mask] = precomputed[r][mask]

    return output


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)


def harmonic_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).reciprocal().mean(dim=dim, keepdim=keepdim).reciprocal()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).reciprocal(), w, dim=dim, keepdim=keepdim, eps=eps).add(eps).reciprocal()


def geometric_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).log().mean(dim=dim).exp()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).log(), w, dim=dim, keepdim=keepdim, eps=eps).exp()


def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def gaussian_blur_2d(input: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    kernel = torch.exp(-(torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=input.dtype, device=input.device) ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :]).reshape(1, 1, kernel_size, kernel_size)
    input = F.pad(input, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode='replicate')
    input = F.conv2d(input, kernel, groups=input.shape[1])
    return input


def focal_to_fov(focal: torch.Tensor):
    return 2 * torch.atan(0.5 / focal)


def fov_to_focal(fov: torch.Tensor):
    return 0.5 / torch.tan(fov / 2)


def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))


def intrinsics_to_fov(intrinsics: torch.Tensor):
    """
    Returns field of view in radians from normalized intrinsics matrix.
    ### Parameters:
    - intrinsics: torch.Tensor of shape (..., 3, 3)

    ### Returns:
    - fov_x: torch.Tensor of shape (...)
    - fov_y: torch.Tensor of shape (...)
    """
    focal_x = intrinsics[..., 0, 0]
    focal_y = intrinsics[..., 1, 1]
    return 2 * torch.atan(0.5 / focal_x), 2 * torch.atan(0.5 / focal_y)


def point_map_to_depth_legacy(points: torch.Tensor):
    height, width = points.shape[-3:-1]
    diagonal = (height ** 2 + width ** 2) ** 0.5
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    # Solve least squares problem
    b = (uv * points[..., 2:]).flatten(-3, -1)  # (..., H * W * 2)
    A = torch.stack([points[..., :2], -uv.expand_as(points[..., :2])], dim=-1).flatten(-4, -2)  # (..., H * W * 2, 2)

    M = A.transpose(-2, -1) @ A
    solution = (torch.inverse(M + 1e-6 * torch.eye(2).to(A)) @ (A.transpose(-2, -1) @ b[..., None])).squeeze(-1)
    focal, shift = solution.unbind(-1)

    depth = points[..., 2] + shift[..., None, None]
    fov_x = torch.atan(width / diagonal / focal) * 2
    fov_y = torch.atan(height / diagonal / focal) * 2
    return depth, fov_x, fov_y, shift


def view_plane_uv_to_focal(uv: torch.Tensor):
    normed_uv = normalized_view_plane_uv(width=uv.shape[-2], height=uv.shape[-3], device=uv.device, dtype=uv.dtype)
    focal = (uv * normed_uv).sum() / uv.square().sum().add(1e-12)
    return focal


def recover_focal_shift(points: torch.Tensor, mask: torch.Tensor = None, focal: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `focal`: torch.Tensor of shape (...) the estimated focal length, relative to the half diagonal of the map
    - `shift`: torch.Tensor of shape (...) Z-axis shift to translate the point map to camera space
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    focal = focal.reshape(-1) if focal is not None else None
    uv = normalized_view_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0

    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    focal_np = focal.cpu().numpy() if focal is not None else None
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        if uv_lr_i_np.shape[0] < 2:
            optim_focal.append(1)
            optim_shift.append(0)
            continue
        if focal is None:
            optim_shift_i, optim_focal_i = solve_optimal_focal_shift(uv_lr_i_np, points_lr_i_np)
            optim_focal.append(float(optim_focal_i))
        else:
            optim_shift_i = solve_optimal_shift(uv_lr_i_np, points_lr_i_np, focal_np[i])
        optim_shift.append(float(optim_shift_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype).reshape(shape[:-3])

    if focal is None:
        optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype).reshape(shape[:-3])
    else:
        optim_focal = focal.reshape(shape[:-3])

    return optim_focal, optim_shift


def theshold_depth_change(depth: torch.Tensor, mask: torch.Tensor, pooler: Literal['min', 'max'], rtol: float = 0.2, kernel_size: int = 3):
    *batch_shape, height, width = depth.shape
    depth = depth.reshape(-1, 1, height, width)
    mask = mask.reshape(-1, 1, height, width)
    if pooler == 'max':
        pooled_depth = F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        output_mask = pooled_depth > depth * (1 + rtol)
    elif pooler == 'min':
        pooled_depth = -F.max_pool2d(-torch.where(mask, depth, torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        output_mask = pooled_depth < depth * (1 - rtol)
    else:
        raise ValueError(f'Unsupported pooler: {pooler}')
    output_mask = output_mask.reshape(*batch_shape, height, width)
    return output_mask


def dilate_with_mask(input: torch.Tensor, mask: torch.BoolTensor, filter: Literal['min', 'max', 'mean', 'median'] = 'mean', iterations: int = 1) -> torch.Tensor:
    kernel = torch.tensor([[False, True, False], [True, True, True], [False, True, False]], device=input.device, dtype=torch.bool)
    for _ in range(iterations):
        input_window = utils3d.pt.sliding_window(F.pad(input, (1, 1, 1, 1), mode='constant', value=0), window_size=3, stride=1, dim=(-2, -1))
        mask_window = kernel & utils3d.pt.sliding_window(F.pad(mask, (1, 1, 1, 1), mode='constant', value=False), window_size=3, stride=1, dim=(-2, -1))
        if filter == 'min':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.inf).min(dim=(-2, -1)).values)
        elif filter == 'max':
            input = torch.where(mask, input, torch.where(mask_window, input_window, -torch.inf).max(dim=(-2, -1)).values)
        elif filter == 'mean':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.nan).nanmean(dim=(-2, -1)))
        elif filter == 'median':
            input = torch.where(mask, input, torch.where(mask_window, input_window, torch.nan).flatten(-2).nanmedian(dim=-1).values)
        mask = mask_window.any(dim=(-2, -1))
    return input, mask


def refine_depth_with_normal(depth: torch.Tensor, normal: torch.Tensor, intrinsics: torch.Tensor, iterations: int = 10, damp: float = 1e-3, eps: float = 1e-12, kernel_size: int = 5) -> torch.Tensor:
    device, dtype = depth.device, depth.dtype
    height, width = depth.shape[-2:]
    radius = kernel_size // 2

    duv = torch.stack(torch.meshgrid(torch.linspace(-radius / width, radius / width, kernel_size, device=device, dtype=dtype), torch.linspace(-radius / height, radius / height, kernel_size, device=device, dtype=dtype), indexing='xy'), dim=-1).to(dtype=dtype, device=device)

    log_depth = depth.clamp_min_(eps).log()
    log_depth_diff = utils3d.pt.sliding_window(log_depth, window_size=kernel_size, stride=1, dim=(-2, -1)) - log_depth[..., radius:-radius, radius:-radius, None, None]

    weight = torch.exp(-(log_depth_diff / duv.norm(dim=-1).clamp_min_(eps) / 10).square())
    tot_weight = weight.sum(dim=(-2, -1)).clamp_min_(eps)

    uv = utils3d.pt.uv_map((height, width), device=device, dtype=dtype)
    K_inv = torch.inverse(intrinsics)

    grad = -(normal[..., None, :2] @ K_inv[..., None, None, :2, :2]).squeeze(-2) \
           / (normal[..., None, 2:] + normal[..., None, :2] @ (K_inv[..., None, None, :2, :2] @ uv[..., :, None] + K_inv[..., None, None, :2, 2:])).squeeze(-2)
    laplacian = (weight * ((utils3d.pt.sliding_window(grad, window_size=kernel_size, stride=1, dim=(-3, -2)) + grad[..., radius:-radius, radius:-radius, :, None, None]) * (duv.permute(2, 0, 1) / 2)).sum(dim=-3)).sum(dim=(-2, -1))

    laplacian = laplacian.clamp(-0.1, 0.1)
    log_depth_refine = log_depth.clone()

    for _ in range(iterations):
        log_depth_refine[..., radius:-radius, radius:-radius] = 0.1 * log_depth_refine[..., radius:-radius, radius:-radius] + 0.9 * (damp * log_depth[..., radius:-radius, radius:-radius] - laplacian + (weight * utils3d.pt.sliding_window_2d(log_depth_refine, window_size=kernel_size, stride=1, dim=(-2, -1))).sum(dim=(-2, -1))) / (tot_weight + damp)

    depth_refine = log_depth_refine.exp()

    return depth_refine
