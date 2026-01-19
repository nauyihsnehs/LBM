import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode, ToPILImage

from lbm.inference.relight import build_filllight_model

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

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


def _load_rgb(path: Path, image_size: int) -> torch.Tensor:
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


def _load_gray(path: Path, image_size: int) -> torch.Tensor:
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


def _load_lighting_params(path: Optional[str]) -> torch.Tensor:
    if path is None:
        raise ValueError("lighting_params must be provided via file or config list.")
    param_path = Path(path)
    if param_path.suffix.lower() == ".json":
        import json

        with open(param_path, "r") as file:
            values = json.load(file)
        params = torch.tensor(values, dtype=torch.float32)
    elif param_path.suffix.lower() in {".pt", ".pth"}:
        params = torch.load(param_path, map_location="cpu")
        params = torch.tensor(params, dtype=torch.float32)
    elif param_path.suffix.lower() == ".npy":
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("numpy is required to load .npy lighting params") from exc
        params = torch.tensor(np.load(param_path), dtype=torch.float32)
    else:
        with open(param_path, "r") as file:
            content = file.read().replace(",", " ").split()
        params = torch.tensor([float(value) for value in content], dtype=torch.float32)

    params = params.flatten()
    if params.numel() != 7:
        raise ValueError(
            f"Expected 7 lighting params, got {params.numel()} from {param_path}"
        )
    return params


def _prepare_pairs(
        source_path: str,
        depth_path: str,
        lighting_scale_path: Optional[str],
) -> List[dict]:
    source = Path(source_path)
    depth = Path(depth_path)
    lighting_scale = Path(lighting_scale_path)

    if source.is_dir() or depth.is_dir() or lighting_scale.is_dir():
        if not (source.is_dir() and depth.is_dir() and lighting_scale.is_dir()):
            raise ValueError(
                "When using folder inputs, source, depth, and lighting_scale must all be folders."
            )
        source_files = _list_images(source)
        depth_files = _list_images(depth)
        lighting_scale_files = _list_images(lighting_scale)
        if not (source_files and depth_files and lighting_scale_files):
            raise ValueError("One or more input folders are empty.")
        if not (len(source_files) == len(depth_files) == len(lighting_scale_files)):
            raise ValueError(
                "Folder inputs must contain the same number of images for pairing."
            )
        return [
            {
                "source": source_files[i],
                "depth": depth_files[i],
                "lighting_scale": lighting_scale_files[i],
            }
            for i in range(len(source_files))
        ]

    return [
        {
            "source": source,
            "depth": depth,
            "lighting_scale": lighting_scale,
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


def run_inference(config: dict) -> None:
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = (
        torch.bfloat16
        if str(config.get("torch_dtype", "bfloat16")) == "bfloat16"
        else torch.float16
    )

    model = build_filllight_model(
        backbone_signature=config.get(
            "backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        ),
        vae_num_channels=int(config.get("vae_num_channels", 4)),
        unet_input_channels=int(config.get("unet_input_channels", 6)),
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
        lighting_conditioning=True,
        lighting_embedder_config=config.get("lighting_embedder_config"),
        bridge_noise_sigma=float(config.get("bridge_noise_sigma", 0.0)),
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
        config.get("lighting_scale_image"),
    )

    lighting_params = config.get("lighting_params")
    if isinstance(lighting_params, list):
        lighting_params_tensor = torch.tensor(lighting_params, dtype=torch.float32)
    else:
        lighting_params_tensor = _load_lighting_params(config.get("lighting_params_path"))

    output_dir = Path(config.get("output_path", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(config.get("image_size", 512))
    num_steps = int(config.get("num_inference_steps", 1))

    total_time = 0.0
    max_memory_mb = 0.0

    for item in pairs:
        source_tensor = _load_rgb(item["source"], image_size)
        depth_tensor = _load_gray(item["depth"], image_size)
        lighting_scale_tensor = _load_gray(item["lighting_scale"], image_size)
        batch = {
            config.get("source_key", "source"): (source_tensor * 2 - 1)
            .unsqueeze(0)
            .to(device),
            "depth": depth_tensor.unsqueeze(0).to(device),
            "lighting_scale": lighting_scale_tensor.unsqueeze(0).to(device),
            "lighting_params": lighting_params_tensor.unsqueeze(0).to(device),
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
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            max_memory_mb = max(max_memory_mb, peak_memory)

        output_image = (output[0].float().cpu() + 1) / 2
        output_pil = ToPILImage()(output_image)
        output_name = f"{item['source'].stem}_fill_lighting_simple.png"
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
    parser.add_argument("--depth_image", type=str, default=None)
    parser.add_argument("--lighting_scale_image", type=str, default=None)
    parser.add_argument("--lighting_params", type=float, nargs="+", default=None)
    parser.add_argument("--lighting_params_path", type=str, default=None)
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

    required_keys = ["source_image", "depth_image", "lighting_scale_image"]
    if not all(merged.get(key) for key in required_keys):
        raise ValueError(
            "source_image, depth_image, and lighting_scale_image must be provided via args or config."
        )

    if not merged.get("lighting_params") and not merged.get("lighting_params_path"):
        raise ValueError(
            "lighting_params (list) or lighting_params_path must be provided via args or config."
        )

    run_inference(merged)


if __name__ == "__main__":
    main()
