import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import h5py
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode, ToPILImage

from lbm.inference.relight import (
    build_filllight_model,
    build_filllight_refine_model,
    build_relight_model,
)
from lbm.moge import DepthBatchInferencer

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
# MANUAL_LIGHTING_PARAMS = [[0.1, -0.2, 120.0, 1.0, 1.0, 1.0, 0.03, 1.0]]
MANUAL_LIGHTING_PARAMS = [
    [-45, 0, 50, 255, 255, 255, 0.03, 1.0],  # left
    [45, 0, 50, 255, 255, 255, 0.03, 1.0],  # right
    [0, -10, 50, 255, 255, 255, 0.03, 1.0],  # front
    # [-90, 1, 25, 255, 255, 255, 0.05, 1.],
    # [-45, 1, 25, 255, 255, 255, 0.05, 1.],
    # [0, 1, 25, 255, 255, 255, 0.05, 1.],
    # [45, 1, 25, 255, 255, 255, 0.05, 1.],
    # [90, 1, 25, 255, 255, 255, 0.05, 1.],
    # [0, -60, 25, 255, 255, 255, 0.05, 1.],
    # [0, -30, 25, 255, 255, 255, 0.05, 1.],
    # [0, 0, 25, 255, 255, 255, 0.05, 1.],
    # [0, 30, 25, 255, 255, 255, 0.05, 1.],
    # [0, 60, 25, 255, 255, 255, 0.05, 1.],
    # [45, 10, 1, 255, 255, 255, 0.05, 1.],
    # [45, 10, 5, 255, 255, 255, 0.05, 1.],
    # [45, 10, 10, 255, 255, 255, 0.05, 1.],
    # [45, 10, 20, 255, 255, 255, 0.05, 1.],
    # [45, 10, 50, 255, 255, 255, 0.05, 1.],
    # [-45, 10, 1, 255, 255, 255, 0.05, 1.],#
    # [-45, 10, 5, 255, 255, 255, 0.05, 1.],#
    # [-45, 10, 10, 255, 255, 255, 0.05, 1.],#
    # [-45, 10, 20, 255, 255, 255, 0.05, 1.],#
    # [-45, 10, 50, 255, 255, 255, 0.05, 1.],#
    # [-45, 5, 20, 255, 255, 255, 0.05, 1.],#
    # [-45, -5, 20, 255, 255, 255, 0.05, 1.],#
    # [45, -5, 20, 255, 255, 255, 0.05, 1.],#
    # [45, 5, 20, 255, 255, 255, 0.05, 1.],#
    # [0, 45, 20, 255, 255, 255, 0.05, 1.],#
    # [0, -45, 20, 255, 255, 255, 0.05, 1.],#
    # [45, 10, 50, 255, 255, 255, 0.005, 1.],
    # [45, 10, 50, 255, 255, 255, 0.02, 1.],
    # [45, 10, 50, 255, 255, 255, 0.04, 1.],
    # [45, 10, 50, 255, 255, 255, 0.06, 1.],
    # [45, 10, 50, 255, 255, 255, 0.08, 1.],
    # [-45, -10, 100, 255, 0, 0, 0.05, 1.],
    # [-45, -10, 100, 0, 255, 0, 0.05, 1.],
    # [-45, -10, 100, 0, 0, 255, 0.05, 1.],
    # [-45, -10, 100, 0, 255, 255, 0.05, 1.],
    # [-45, -10, 100, 255, 0, 255, 0.05, 1.],
    # [-45, -10, 100, 255, 255, 0, 0.05, 1.],
    # [-45, 10, 20, 255, 255, 255, 0.05, 0.],# re
    # [-45, 10, 20, 255, 255, 255, 0.05, 0.25],# re
    # [-45, 10, 20, 255, 255, 255, 0.05, 0.5],# re
    # [-45, 10, 20, 255, 255, 255, 0.05, 0.75],# re
    # [-45, 10, 20, 255, 255, 255, 0.05, 1.0],# re
    # [0, 5, 5, 255, 255, 255, 0.05, 1.],#
    # [0, 5, 6, 255, 255, 255, 0.05, 1.],#
    # [0, 5, 7, 255, 255, 255, 0.05, 1.],#
    # [0, 5, 8, 255, 255, 255, 0.05, 1.],#
    # [0, 5, 9, 255, 255, 255, 0.05, 1.],#
    # [0, 5, 10, 255, 255, 255, 0.05, 1.],#
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_yaml(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r") as file:
        return yaml.safe_load(file) or {}


def _merge_config(*configs: dict) -> dict:
    merged: dict = {}
    for config in configs:
        merged.update({k: v for k, v in config.items() if v is not None})
    return merged


def _list_images(folder: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
        ],
        key=lambda path: path.name,
    )


def _prepare_inputs(source_path: str) -> List[Path]:
    source = Path(source_path)
    if source.is_dir():
        files = _list_images(source)
        if not files:
            raise ValueError("Source folder has no images.")
        return files
    if not source.exists():
        raise FileNotFoundError(f"Missing source image: {source}")
    return [source]


def _load_rgb_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.NEAREST_EXACT,
            ),
            transforms.ToTensor(),
        ]
    )
    return transform(image)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    filtered_state = {
        key.replace("model.", ""): value
        for key, value in state_dict.items()
        if key.startswith("model.")
    }
    model.load_state_dict(filtered_state, strict=False)


def _normalize_lighting_params(raw_params: torch.Tensor) -> torch.Tensor:
    raw_params = raw_params.flatten()
    position = raw_params[:2]
    intensity = (raw_params[2:3] / 200) * 2 - 1
    color = raw_params[3:6] * 2 - 1
    if raw_params.numel() == 8:
        area = (raw_params[6:7] * 5) * 2 - 1
        shading_scale = raw_params[-1:] * 2 - 1
    else:
        area = (raw_params[-1:] * 5) * 2 - 1
        shading_scale = torch.ones_like(area)
    return torch.cat([position, intensity, color, area, shading_scale], dim=0)


def _parse_manual_lighting_params(raw: Optional[str]) -> List[torch.Tensor]:
    if not raw:
        return []
    data = yaml.safe_load(raw)
    if not data:
        return []
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], (int, float)):
        data = [data]
    if not isinstance(data, (list, tuple)):
        raise ValueError("manual_lighting_params must be a list or list of lists.")
    tensors = []
    for params in data:
        if not isinstance(params, (list, tuple)):
            raise ValueError("Each manual lighting params entry must be a list.")
        if len(params) not in (7, 8):
            raise ValueError("Lighting params must have 7 or 8 values.")
        params = list(params)
        params[0] = float(params[0]) / 180.0
        params[1] = float(params[1]) / 90.0
        params[3] = float(params[3]) / 255.0
        params[4] = float(params[4]) / 255.0
        params[5] = float(params[5]) / 255.0
        tensors.append(_normalize_lighting_params(torch.tensor(params, dtype=torch.float32)))
    return tensors


def _load_lighting_params_h5(path: Path) -> List[torch.Tensor]:
    with h5py.File(path, "r") as file:
        if "lighting_params" in file:
            data = file["lighting_params"][()]
        else:
            first_key = next(iter(file.keys()))
            data = file[first_key][()]
    params = torch.tensor(data, dtype=torch.float32)
    if params.ndim == 1:
        params = params.unsqueeze(0)
    return [_normalize_lighting_params(row) for row in params]


def _collect_lighting_params(
        manual_params: Optional[Sequence[Sequence[float]]],
        lighting_paths: Optional[Sequence[str]],
) -> List[Tuple[str, torch.Tensor]]:
    lighting_params: List[Tuple[str, torch.Tensor]] = []
    manual_list = []
    if manual_params is not None:
        manual_list = _parse_manual_lighting_params(yaml.safe_dump(manual_params))
    for idx, tensor in enumerate(manual_list):
        lighting_params.append((f"manual_{idx:03d}", tensor))

    if not lighting_paths:
        return lighting_params

    h5_files: List[Path] = []
    for raw_path in lighting_paths:
        if raw_path is None:
            continue
        path = Path(raw_path)
        if path.is_dir():
            h5_files.extend(sorted(path.glob("*.h5")))
        elif path.is_file():
            h5_files.append(path)
        else:
            raise FileNotFoundError(f"Missing lighting params path: {path}")

    for h5_path in h5_files:
        params_list = _load_lighting_params_h5(h5_path)
        for idx, tensor in enumerate(params_list):
            lighting_params.append((f"{h5_path.stem}_{idx:03d}", tensor))
    return lighting_params


def _save_lighting_params(path: Path, lighting_params: List[Tuple[str, torch.Tensor]]) -> None:
    payload = [
        {"id": light_id, "params": tensor.detach().cpu().tolist()}
        for light_id, tensor in lighting_params
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def _save_depth(path: Path, depth: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    depth_np = depth.cpu().numpy().astype("float32")
    cv2.imwrite(
        str(path),
        depth_np,
        [
            cv2.IMWRITE_EXR_TYPE,
            cv2.IMWRITE_EXR_TYPE_HALF,
            cv2.IMWRITE_EXR_COMPRESSION,
            cv2.IMWRITE_EXR_COMPRESSION_ZIP,
        ],
    )


def _load_depth_exr(path: Path) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    return torch.from_numpy(depth.astype("float32"))


def _normalize_depth(depth: torch.Tensor, image_size: int) -> torch.Tensor:
    depth = depth.clone()
    depth[torch.isinf(depth)] = 0.0
    depth[torch.isnan(depth)] = 0.0
    depth[depth >= 100.0] = 0.0
    max_val = float(torch.quantile(depth, 0.99).item()) if depth.numel() else 0.0
    if max_val > 0:
        depth = depth.clamp(min=0.0, max=max_val)
        depth = depth / max_val
    depth_tensor = depth.unsqueeze(0).unsqueeze(0)
    depth_tensor = torch.nn.functional.interpolate(
        depth_tensor,
        size=(image_size, image_size),
        mode="nearest",
    ).squeeze(0)
    return depth_tensor * 2 - 1


def _infer_albedo(
        model: torch.nn.Module,
        source_tensor: torch.Tensor,
        num_steps: int,
        source_key: str,
        device: torch.device,
        torch_dtype: torch.dtype,
) -> torch.Tensor:
    batch = {source_key: (source_tensor * 2 - 1).unsqueeze(0).to(device=device)}
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                z_source = model.vae.encode(batch[source_key])
                output = model.sample(
                    z=z_source,
                    num_steps=num_steps,
                    conditioner_inputs=batch,
                    max_samples=1,
                )
        else:
            z_source = model.vae.encode(batch[source_key])
            output = model.sample(
                z=z_source,
                num_steps=num_steps,
                conditioner_inputs=batch,
                max_samples=1,
            )
    return (output[0].float().cpu() + 1) / 2


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _timestamped_output(root: Path) -> Path:
    return root / datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )
    return transform(image)


def _infer_fill_lighting(
        model: torch.nn.Module,
        albedo: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        lighting_params: torch.Tensor,
        num_steps: int,
        source_key: str,
        device: torch.device,
        torch_dtype: torch.dtype,
) -> torch.Tensor:
    shading = (rgb / albedo.clamp(1e-3, 1.0)).clamp(0.0, 1.0)
    batch = {
        source_key: (albedo * 2 - 1).unsqueeze(0).to(device),
        "shading": (shading * 2 - 1).unsqueeze(0).to(device),
        "depth": depth.unsqueeze(0).to(device),
        "lighting_params": lighting_params.unsqueeze(0).to(device),
    }
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                z_source = model.vae.encode(batch[source_key])
                output = model.sample(
                    z=z_source,
                    num_steps=num_steps,
                    conditioner_inputs=batch,
                    max_samples=1,
                )
        else:
            z_source = model.vae.encode(batch[source_key])
            output = model.sample(
                z=z_source,
                num_steps=num_steps,
                conditioner_inputs=batch,
                max_samples=1,
            )
    return (output[0].float().cpu() + 1) / 2


def _infer_refine(
        model: torch.nn.Module,
        fill_tensor: torch.Tensor,
        rgb_tensor: torch.Tensor,
        num_steps: int,
        source_key: str,
        device: torch.device,
        torch_dtype: torch.dtype,
) -> torch.Tensor:
    batch = {
        source_key: (fill_tensor * 2 - 1).unsqueeze(0).to(device),
        "rgb": (rgb_tensor * 2 - 1).unsqueeze(0).to(device),
    }
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                output = model.sample(
                    z=batch[source_key],
                    num_steps=num_steps,
                    conditioner_inputs=batch,
                    max_samples=1,
                )
        else:
            output = model.sample(
                z=batch[source_key],
                num_steps=num_steps,
                conditioner_inputs=batch,
                max_samples=1,
            )
    return (output[0].float().cpu() + 1) / 2


def run_pipeline(config: dict) -> None:
    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = (
        torch.bfloat16
        if str(config.get("torch_dtype", "bfloat16")) == "bfloat16"
        else torch.float16
    )
    device = torch.device(device_name)

    inputs = _prepare_inputs(config["source_path"])
    lighting_params = _collect_lighting_params(
        MANUAL_LIGHTING_PARAMS,
        config.get("lighting_params_paths"),
    )
    if not lighting_params:
        raise ValueError("No lighting params provided via manual list or .h5 files.")

    albedo_config = _merge_config(
        _load_yaml(config.get("albedo_train_config")),
    )
    fill_config = _merge_config(_load_yaml(config.get("fill_train_config")))
    refine_config = _merge_config(_load_yaml(config.get("refine_train_config")))

    albedo_model = build_relight_model(
        backbone_signature=albedo_config.get(
            "backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        ),
        vae_num_channels=int(albedo_config.get("vae_num_channels", 4)),
        unet_input_channels=int(albedo_config.get("unet_input_channels", 4)),
        source_key=albedo_config.get("source_key", "source"),
        target_key=albedo_config.get("target_key", "target"),
        mask_key=albedo_config.get("mask_key"),
        timestep_sampling=albedo_config.get("timestep_sampling", "log_normal"),
        logit_mean=float(albedo_config.get("logit_mean", 0.0)),
        logit_std=float(albedo_config.get("logit_std", 1.0)),
        pixel_loss_type=albedo_config.get("pixel_loss_type", "lpips"),
        latent_loss_type=albedo_config.get("latent_loss_type", "l2"),
        latent_loss_weight=float(albedo_config.get("latent_loss_weight", 1.0)),
        pixel_loss_weight=float(albedo_config.get("pixel_loss_weight", 0.0)),
        selected_timesteps=albedo_config.get("selected_timesteps"),
        prob=albedo_config.get("prob"),
        conditioning_images_keys=albedo_config.get("conditioning_images_keys"),
        conditioning_masks_keys=albedo_config.get("conditioning_masks_keys"),
        bridge_noise_sigma=float(albedo_config.get("bridge_noise_sigma", 0.0)),
    )
    albedo_model.to(device).to(torch_dtype)
    albedo_model.eval()
    _load_checkpoint(albedo_model, config["albedo_checkpoint"])

    fill_model = build_filllight_model(
        backbone_signature=fill_config.get(
            "backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        ),
        vae_num_channels=int(fill_config.get("vae_num_channels", 4)),
        unet_input_channels=int(fill_config.get("unet_input_channels", 9)),
        source_key=fill_config.get("source_key", "source"),
        mask_key=fill_config.get("mask_key"),
        timestep_sampling=fill_config.get("timestep_sampling", "log_normal"),
        logit_mean=float(fill_config.get("logit_mean", 0.0)),
        logit_std=float(fill_config.get("logit_std", 1.0)),
        pixel_loss_type=fill_config.get("pixel_loss_type", "lpips"),
        latent_loss_type=fill_config.get("latent_loss_type", "l2"),
        latent_loss_weight=float(fill_config.get("latent_loss_weight", 1.0)),
        pixel_loss_weight=float(fill_config.get("pixel_loss_weight", 0.0)),
        selected_timesteps=fill_config.get("selected_timesteps"),
        prob=fill_config.get("prob"),
        conditioning_images_keys=fill_config.get("conditioning_images_keys"),
        conditioning_masks_keys=fill_config.get("conditioning_masks_keys"),
        lighting_embedder_config=fill_config.get("lighting_embedder_config"),
        lighting_condition_weight=float(fill_config.get("lighting_condition_weight", 1.0)),
        concat_condition_weight=float(fill_config.get("concat_condition_weight", 1.0)),
        bridge_noise_sigma=float(fill_config.get("bridge_noise_sigma", 0.0)),
    )
    fill_model.to(device).to(torch_dtype)
    fill_model.eval()
    _load_checkpoint(fill_model, config["fill_checkpoint"])

    refine_model = build_filllight_refine_model(
        backbone_signature=refine_config.get(
            "backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        ),
        unet_input_channels=int(refine_config.get("unet_input_channels", 6)),
        unet_output_channels=int(refine_config.get("unet_output_channels", 3)),
        block_out_channels=refine_config.get("block_out_channels"),
        layers_per_block=int(refine_config.get("layers_per_block", 2)),
        source_key=refine_config.get("source_key", "source"),
        target_key=refine_config.get("target_key", "target"),
        mask_key=refine_config.get("mask_key"),
        timestep_sampling=refine_config.get("timestep_sampling", "log_normal"),
        logit_mean=float(refine_config.get("logit_mean", 0.0)),
        logit_std=float(refine_config.get("logit_std", 1.0)),
        latent_loss_type=refine_config.get("latent_loss_type", "l2"),
        latent_loss_weight=float(refine_config.get("latent_loss_weight", 1.0)),
        selected_timesteps=refine_config.get("selected_timesteps"),
        prob=refine_config.get("prob"),
        conditioning_images_keys=refine_config.get("conditioning_images_keys"),
        conditioning_masks_keys=refine_config.get("conditioning_masks_keys"),
        bridge_noise_sigma=float(refine_config.get("bridge_noise_sigma", 0.0)),
    )
    refine_model.to(device).to(torch_dtype)
    refine_model.eval()
    _load_checkpoint(refine_model, config["refine_checkpoint"])

    depth_device = torch.device(config.get("depth_device", device_name))
    depth_inferencer = DepthBatchInferencer.from_pretrained(
        pretrained_model_name_or_path=config.get("depth_model", "Ruicheng/moge-2-vitl"),
        device=depth_device,
        use_fp16=bool(config.get("depth_use_fp16", False)),
    )

    output_root = _timestamped_output(Path(config.get("output_root", "./outputs/full_pipeline")))
    albedo_dir = output_root / "albedo"
    depth_dir = output_root / "depth"
    fill_dir = output_root / "fill"
    refine_dir = output_root / "refine"
    albedo_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    fill_dir.mkdir(parents=True, exist_ok=True)
    refine_dir.mkdir(parents=True, exist_ok=True)
    _save_lighting_params(output_root / "lighting_params.yaml", lighting_params)

    image_size = 512
    refine_image_size = 1024
    albedo_steps = int(config.get("albedo_steps", 4))
    fill_steps = int(config.get("fill_steps", 4))
    refine_steps = int(config.get("refine_steps", 4))
    stop_at = config.get("stop_at", "refine")
    to_pil = ToPILImage()

    albedo_input_dir = Path(config["albedo_input_dir"]) if config.get("albedo_input_dir") else None
    depth_input_dir = Path(config["depth_input_dir"]) if config.get("depth_input_dir") else None
    fill_input_dir = Path(config["fill_input_dir"]) if config.get("fill_input_dir") else None

    logger.info("Output run folder: %s", output_root)

    albedo_source_dir = albedo_input_dir or albedo_dir
    depth_source_dir = depth_input_dir or depth_dir
    fill_source_dir = fill_input_dir or fill_dir

    logger.info("Step 1/4: Albedo")

    for source_path in inputs:
        albedo_path = albedo_dir / f"{source_path.stem}_alb.png"
        if albedo_input_dir is not None:
            input_path = albedo_input_dir / f"{source_path.stem}_alb.png"
            albedo_tensor = _load_image_tensor(input_path, image_size)
        else:
            rgb_tensor = _load_rgb_tensor(source_path, image_size)
            albedo_tensor = _infer_albedo(
                albedo_model,
                rgb_tensor,
                albedo_steps,
                albedo_config.get("source_key", "source"),
                device,
                torch_dtype,
            )
        albedo_tensor = albedo_tensor.clamp(0.0, 1.0)
        _ensure_parent(albedo_path)
        to_pil(albedo_tensor).save(albedo_path)

    if stop_at == "albedo":
        return

    logger.info("Step 2/4: Depth")
    for source_path in inputs:
        depth_path = depth_dir / f"{source_path.stem}_dpt.exr"
        if depth_input_dir is not None:
            depth_raw = _load_depth_exr(depth_input_dir / f"{source_path.stem}_dpt.exr")
        else:
            rgb_np = cv2.imread(str(source_path))
            if rgb_np is None:
                raise ValueError(f"Failed to read RGB image: {source_path}")
            rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
            rgb_np = rgb_np.astype("float32") / 255.0
            rgb_tensor_depth = torch.from_numpy(rgb_np).permute(2, 0, 1)
            depth_output = depth_inferencer.infer_batch(
                rgb_tensor_depth.unsqueeze(0).to(device=depth_device),
                fov_x=config.get("depth_fov_x"),
                resolution_level=config.get("depth_resolution_level", 9),
                num_tokens=config.get("depth_num_tokens"),
                use_fp16=bool(config.get("depth_use_fp16", False)),
            )
            depth_raw = depth_output["depth"][0].detach().cpu()
        _save_depth(depth_path, depth_raw)

    if stop_at == "depth":
        return

    logger.info("Step 3/4: Fill lighting")
    for source_path in inputs:
        rgb_tensor = _load_rgb_tensor(source_path, image_size)
        albedo_tensor = _load_image_tensor(
            albedo_source_dir / f"{source_path.stem}_alb.png",
            image_size,
        )
        depth_raw = _load_depth_exr(depth_source_dir / f"{source_path.stem}_dpt.exr")
        depth_norm = _normalize_depth(depth_raw, image_size)

        for light_id, lighting_tensor in lighting_params:
            fill_output_dir = fill_dir / source_path.stem
            fill_path = fill_output_dir / f"{source_path.stem}_{light_id}_fill.png"
            if fill_input_dir is not None:
                fill_tensor = _load_image_tensor(
                    fill_input_dir / source_path.stem / f"{source_path.stem}_{light_id}_fill.png",
                    image_size,
                )
            else:
                fill_tensor = _infer_fill_lighting(
                    fill_model,
                    albedo_tensor,
                    rgb_tensor,
                    depth_norm,
                    lighting_tensor,
                    fill_steps,
                    fill_config.get("source_key", "source"),
                    device,
                    torch_dtype,
                )
            fill_tensor = fill_tensor.clamp(0.0, 1.0)
            _ensure_parent(fill_path)
            to_pil(fill_tensor).save(fill_path)

    if stop_at == "fill":
        return

    logger.info("Step 4/4: Refine")
    for source_path in inputs:
        rgb_tensor = _load_rgb_tensor(source_path, refine_image_size)
        for light_id, _ in lighting_params:
            fill_path = fill_source_dir / source_path.stem / f"{source_path.stem}_{light_id}_fill.png"
            fill_tensor = _load_image_tensor(fill_path, refine_image_size)

            refine_tensor = _infer_refine(
                refine_model,
                fill_tensor,
                rgb_tensor,
                refine_steps,
                refine_config.get("source_key", "source"),
                device,
                torch_dtype,
            ).clamp(0.0, 1.0)
            refine_output_dir = refine_dir / source_path.stem
            refine_output_dir.mkdir(parents=True, exist_ok=True)
            refine_path = refine_output_dir / f"{source_path.stem}_{light_id}_refine.png"
            _ensure_parent(refine_path)
            to_pil(refine_tensor).save(refine_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full LBM inference pipeline.")
    parser.add_argument("--source_path", type=str,
                        # default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/huawei-mini')
                        # default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/in-the-wild-1024')
                        # default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/object-based')
                        # default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/huawei-1024')
                        # default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/shadow')
                        default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/testset')
    parser.add_argument("--output_root", type=str,
                        # default="/mnt/data1/ssy/render_people/fill-light-dataset/infer/huawei-mini-results")
                        # default="/mnt/data1/ssy/render_people/fill-light-dataset/infer/in-the-wild-1024-results")
                        # default="/mnt/data1/ssy/render_people/fill-light-dataset/infer/object-based-results")
                        # default="/mnt/data1/ssy/render_people/fill-light-dataset/infer/huawei-1024-results")
                        # default="/mnt/data1/ssy/render_people/fill-light-dataset/infer/shadow-results")
                        default="/mnt/data1/ssy/render_people/fill-light-dataset/infer/testset-results")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--stop_at", type=str, choices=["albedo", "depth", "fill", "refine"], default="refine")

    parser.add_argument("--albedo_train_config", type=str,
                        default='/mnt/data1/ssy/render_people/LBM/examples/training/config/albedo.yaml')
    parser.add_argument("--albedo_checkpoint", type=str,
                        default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/albedo/epoch=1-step=50000.ckpt')
    parser.add_argument("--albedo_steps", type=int, default=4)
    parser.add_argument("--albedo_input_dir", type=str, default=None)

    parser.add_argument("--fill_train_config", type=str,
                        default='/mnt/data1/ssy/render_people/LBM/examples/training/config/fill_lighting.yaml')
    parser.add_argument("--fill_checkpoint", type=str,
                        default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=3-step=40000.ckpt') # fill light
                        # default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=3-step=50000-v1.ckpt') # old
                        # default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=2-step=30000-v1.ckpt') # fill light
                        # default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=2-step=30000-v2.ckpt')  # relight
    # default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=4-step=32000.ckpt') # real
    parser.add_argument("--fill_steps", type=int, default=4)
    parser.add_argument("--fill_input_dir", type=str, default=None)

    parser.add_argument("--refine_train_config", type=str,
                        default='/mnt/data1/ssy/render_people/LBM/examples/training/config/fill_lighting_refine.yaml')
    parser.add_argument("--refine_checkpoint", type=str,
                        default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting_refine/epoch=0-step=20000.ckpt')
    parser.add_argument("--refine_steps", type=int, default=4)

    # parser.add_argument("--lighting_params_paths", type=str, default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/testset/lighting')
    parser.add_argument("--lighting_params_paths", type=str, default=None)

    parser.add_argument("--depth_model", type=str, default="Ruicheng/moge-2-vitl")
    parser.add_argument("--depth_device", type=str, default='cuda')
    parser.add_argument("--depth_use_fp16", action="store_true")
    parser.add_argument("--depth_resolution_level", type=int, default=9)
    parser.add_argument("--depth_num_tokens", type=int, default=None)
    parser.add_argument("--depth_fov_x", type=float, default=None)
    parser.add_argument(
        "--depth_input_dir",
        type=str,
        default=None,
        help="Optional folder of precomputed depth EXR files.",
    )

    args = parser.parse_args()
    run_pipeline(vars(args))


if __name__ == "__main__":
    main()
