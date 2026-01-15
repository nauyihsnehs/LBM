import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode, ToPILImage

from lbm.inference.fill_lighting import build_fill_lighting_model

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LIGHT_PARAMS_SUFFIXES = {".npy", ".npz", ".pt", ".pth", ".txt"}

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


def _list_light_params(folder: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in LIGHT_PARAMS_SUFFIXES
        ],
        key=lambda path: path.name,
    )


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


def _load_depth_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
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


def _load_light_params(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        params = torch.load(path)
        params = torch.as_tensor(params, dtype=torch.float32)
    elif suffix in {".npy", ".npz"}:
        params = np.load(path)
        if isinstance(params, np.lib.npyio.NpzFile):
            params = params[sorted(params.files)[0]]
        params = torch.tensor(params, dtype=torch.float32)
    elif suffix == ".txt":
        params = np.loadtxt(path, dtype=np.float32)
        params = torch.tensor(params, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported light params file: {path}")

    params = params.flatten()
    if params.numel() != 7:
        raise ValueError(
            f"Expected 7 light parameters, got {params.numel()} from {path}"
        )
    return params


def _compute_shading(rgb: torch.Tensor, albedo: torch.Tensor) -> torch.Tensor:
    return torch.clamp(rgb / torch.clamp(albedo, 1e-3, 1.0), 0.0, 1.0)


def _prepare_pairs(
    source_path: str,
    depth_path: str,
    light_params_path: str,
    shading_path: Optional[str] = None,
    rgb_path: Optional[str] = None,
    light_scale_path: Optional[str] = None,
) -> List[Dict[str, Optional[Path]]]:
    source = Path(source_path)
    depth = Path(depth_path)
    light_params = Path(light_params_path)
    shading = Path(shading_path) if shading_path else None
    rgb = Path(rgb_path) if rgb_path else None
    light_scale = Path(light_scale_path) if light_scale_path else None

    paths = [source, depth, light_params]
    if shading:
        paths.append(shading)
    if rgb:
        paths.append(rgb)
    if light_scale:
        paths.append(light_scale)

    any_dir = any(path.is_dir() for path in paths)
    if any_dir:
        if not all(path.is_dir() for path in paths if path is not None):
            raise ValueError("When using folder inputs, all provided paths must be folders.")

        source_files = _list_images(source)
        depth_files = _list_images(depth)
        light_params_files = _list_light_params(light_params)
        shading_files = _list_images(shading) if shading else []
        rgb_files = _list_images(rgb) if rgb else []
        light_scale_files = _list_images(light_scale) if light_scale else []

        source_map = {path.stem: path for path in source_files}
        depth_map = {path.stem: path for path in depth_files}
        light_params_map = {path.stem: path for path in light_params_files}
        shading_map = {path.stem: path for path in shading_files}
        rgb_map = {path.stem: path for path in rgb_files}
        light_scale_map = {path.stem: path for path in light_scale_files}

        stems = set(source_map) & set(depth_map) & set(light_params_map)
        if shading_map:
            stems &= set(shading_map)
        if rgb_map:
            stems &= set(rgb_map)
        if light_scale_map:
            stems &= set(light_scale_map)

        if not stems:
            raise ValueError("No matching filenames found across input folders.")

        return [
            {
                "source": source_map[stem],
                "depth": depth_map[stem],
                "light_params": light_params_map[stem],
                "shading": shading_map.get(stem),
                "rgb": rgb_map.get(stem),
                "light_scale": light_scale_map.get(stem),
            }
            for stem in sorted(stems)
        ]

    return [
        {
            "source": source,
            "depth": depth,
            "light_params": light_params,
            "shading": shading,
            "rgb": rgb,
            "light_scale": light_scale,
        }
    ]


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


def _build_light_scale(depth: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(depth)


def _prepare_shading(
    source_tensor: torch.Tensor,
    shading_path: Optional[Path],
    rgb_path: Optional[Path],
    image_size: int,
) -> torch.Tensor:
    if shading_path is not None:
        shading = _load_rgb_tensor(shading_path, image_size)
        return shading
    if rgb_path is None:
        raise ValueError("Either shading_image or rgb_image must be provided.")
    rgb = _load_rgb_tensor(rgb_path, image_size)
    return _compute_shading(rgb, source_tensor)


def run_inference(config: dict) -> None:
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = (
        torch.bfloat16
        if str(config.get("torch_dtype", "bfloat16")) == "bfloat16"
        else torch.float16
    )

    model = build_fill_lighting_model(
        backbone_signature=config.get(
            "backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        ),
        vae_num_channels=int(config.get("vae_num_channels", 4)),
        unet_input_channels=int(config.get("unet_input_channels", 10)),
        source_key=config.get("source_key", "source"),
        target_key=config.get("target_key", "target"),
        mask_key=config.get("mask_key"),
        timestep_sampling=config.get("timestep_sampling", "log_normal"),
        logit_mean=float(config.get("logit_mean", 0.0)),
        logit_std=float(config.get("logit_std", 1.0)),
        pixel_loss_type=config.get("pixel_loss_type", "lpips"),
        latent_loss_type=config.get("latent_loss_type", "l2"),
        latent_loss_weight=float(config.get("latent_loss_weight", 1.0)),
        pixel_loss_weight=float(config.get("pixel_loss_weight", 0.0)),
        selected_timesteps=config.get("selected_timesteps"),
        prob=config.get("prob"),
        conditioning_images_keys=config.get("conditioning_images_keys"),
        conditioning_masks_keys=config.get("conditioning_masks_keys"),
        bridge_noise_sigma=float(config.get("bridge_noise_sigma", 0.0)),
        light_params_input_key=config.get("light_params_input_key", "light_params"),
        light_params_num_frequencies=int(config.get("light_params_num_frequencies", 6)),
        light_params_include_input=bool(config.get("light_params_include_input", True)),
        light_params_hidden_dims=config.get("light_params_hidden_dims"),
        light_params_num_tokens=int(config.get("light_params_num_tokens", 77)),
        light_params_embedding_dim=int(config.get("light_params_embedding_dim", 768)),
        light_params_ucg_rate=float(config.get("light_params_ucg_rate", 0.0)),
    )
    model.to(device).to(torch_dtype)
    model.eval()

    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        _load_checkpoint(model, checkpoint_path)

    pairs = _prepare_pairs(
        config.get("source_image"),
        config.get("depth_image"),
        config.get("light_params"),
        shading_path=config.get("shading_image"),
        rgb_path=config.get("rgb_image"),
        light_scale_path=config.get("light_scale"),
    )

    output_dir = Path(config.get("output_path", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(config.get("image_size", 512))
    num_steps = int(config.get("num_inference_steps", 1))

    total_time = 0.0
    max_memory_mb = 0.0

    for item in pairs:
        source_tensor = _load_rgb_tensor(item["source"], image_size)
        shading_tensor = _prepare_shading(
            source_tensor,
            shading_path=item.get("shading"),
            rgb_path=item.get("rgb"),
            image_size=image_size,
        )
        depth_tensor = _load_depth_tensor(item["depth"], image_size)
        light_scale_tensor = (
            _load_depth_tensor(item["light_scale"], image_size)
            if item.get("light_scale") is not None
            else _build_light_scale(depth_tensor)
        )
        light_params = _load_light_params(item["light_params"])

        batch = {
            config.get("source_key", "source"): (source_tensor * 2 - 1)
            .unsqueeze(0)
            .to(device),
            "shading": (shading_tensor * 2 - 1).unsqueeze(0).to(device),
            "depth": depth_tensor.unsqueeze(0).to(device),
            "light_scale": light_scale_tensor.unsqueeze(0).to(device),
            "light_params": light_params.unsqueeze(0).to(device),
        }

        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    z_source = model.vae.encode(batch[model.source_key])
                    output = model.sample(
                        z=z_source,
                        num_steps=num_steps,
                        conditioner_inputs=batch,
                        max_samples=1,
                    )
            else:
                z_source = model.vae.encode(batch[model.source_key])
                output = model.sample(
                    z=z_source,
                    num_steps=num_steps,
                    conditioner_inputs=batch,
                    max_samples=1,
                )

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        total_time += elapsed

        if device.startswith("cuda"):
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
            max_memory_mb = max(max_memory_mb, peak_memory)

        output_image = (output[0].float().cpu() + 1) / 2
        output_pil = ToPILImage()(output_image)
        output_name = f"{item['source'].stem}_fill.png"
        output_pil.save(output_dir / output_name)

    avg_time = total_time / len(pairs)
    logger.info("Processed %d samples", len(pairs))
    logger.info("Average inference time: %.4fs", avg_time)
    if device.startswith("cuda"):
        logger.info("Peak GPU memory: %.2f MB", max_memory_mb)
    else:
        logger.info("GPU memory stats unavailable on CPU device.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--source_image", type=str, default=None)
    parser.add_argument("--shading_image", type=str, default=None)
    parser.add_argument("--rgb_image", type=str, default=None)
    parser.add_argument("--depth_image", type=str, default=None)
    parser.add_argument("--light_params", type=str, default=None)
    parser.add_argument("--light_scale", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default=None)

    args = parser.parse_args()

    inference_config = _load_yaml(args.inference_config)
    train_config_path = args.train_config or inference_config.get("train_config")
    train_config = _load_yaml(train_config_path)
    merged = _merge_config(train_config, inference_config, vars(args))

    required_keys = ["source_image", "depth_image", "light_params"]
    if not all(merged.get(key) for key in required_keys):
        raise ValueError(
            "source_image, depth_image, and light_params must be provided via args or config."
        )

    run_inference(merged)


if __name__ == "__main__":
    main()
