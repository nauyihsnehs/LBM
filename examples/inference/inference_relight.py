import argparse
import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from lbm.inference.relight import build_relight_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class RelightInputs:
    name: str
    source_path: str
    shading_path: str
    normal_path: str
    original_size: Tuple[int, int]


def _list_images(directory: str) -> List[str]:
    return sorted(
        [
            entry
            for entry in os.listdir(directory)
            if os.path.splitext(entry)[1].lower() in ALLOWED_EXTENSIONS
        ]
    )


def _resolve_inputs(
    root_dir: str,
    source_subdir: str,
    shading_subdir: str,
    normal_subdir: str,
) -> List[RelightInputs]:
    source_dir = os.path.join(root_dir, source_subdir)
    shading_dir = os.path.join(root_dir, shading_subdir)
    normal_dir = os.path.join(root_dir, normal_subdir)

    if not os.path.isdir(source_dir):
        raise FileNotFoundError(f"Missing source directory: {source_dir}")
    if not os.path.isdir(shading_dir):
        raise FileNotFoundError(f"Missing shading directory: {shading_dir}")
    if not os.path.isdir(normal_dir):
        raise FileNotFoundError(f"Missing normal directory: {normal_dir}")

    inputs = []
    for name in _list_images(source_dir):
        source_path = os.path.join(source_dir, name)
        shading_path = os.path.join(shading_dir, name)
        normal_path = os.path.join(normal_dir, name)
        if not os.path.exists(shading_path) or not os.path.exists(normal_path):
            logger.warning(
                "Skipping %s because shading or normal is missing.", source_path
            )
            continue
        with Image.open(source_path) as img:
            original_size = img.size
        inputs.append(
            RelightInputs(
                name=name,
                source_path=source_path,
                shading_path=shading_path,
                normal_path=normal_path,
                original_size=original_size,
            )
        )
    return inputs


def _load_image(path: str, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((size, size), Image.Resampling.BICUBIC)


def _batch_iter(items: Sequence[RelightInputs], batch_size: int) -> Iterable[List[RelightInputs]]:
    for idx in range(0, len(items), batch_size):
        yield list(items[idx : idx + batch_size])


def _load_state_dict(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = model.state_dict()
    trimmed_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            trimmed_key = key[len("model.") :]
        else:
            trimmed_key = key
        if trimmed_key in model_state:
            trimmed_state[trimmed_key] = value

    if not trimmed_state:
        raise ValueError(
            "No matching weights found in checkpoint. "
            "Ensure the checkpoint is from TrainingPipeline."
        )
    missing, unexpected = model.load_state_dict(trimmed_state, strict=False)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(device: torch.device, dtype: str) -> torch.dtype:
    if dtype == "bf16" and device.type == "cuda":
        return torch.bfloat16
    if dtype == "fp16" and device.type == "cuda":
        return torch.float16
    return torch.float32


def _apply_config_overrides(config_path: Optional[str]) -> dict:
    if not config_path:
        return {}
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    if config is None:
        return {}
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch inference for LBM relighting.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--source_subdir", type=str, default="source")
    parser.add_argument("--shading_subdir", type=str, default="shading")
    parser.add_argument("--normal_subdir", type=str, default="normal")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--config", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    inputs = _resolve_inputs(
        args.input_dir, args.source_subdir, args.shading_subdir, args.normal_subdir
    )
    if not inputs:
        raise ValueError("No valid relight inputs found in the input directory.")

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(device, args.dtype)

    config_overrides = _apply_config_overrides(args.config)
    model = build_relight_model(
        backbone_signature=config_overrides.get(
            "backbone_signature", "runwayml/stable-diffusion-v1-5"
        ),
        vae_num_channels=config_overrides.get("vae_num_channels", 4),
        unet_input_channels=config_overrides.get("unet_input_channels", 12),
        source_key=config_overrides.get("source_key", "source"),
        target_key=config_overrides.get("target_key", "target"),
        mask_key=config_overrides.get("mask_key"),
        timestep_sampling=config_overrides.get("timestep_sampling", "uniform"),
        logit_mean=config_overrides.get("logit_mean", 0.0),
        logit_std=config_overrides.get("logit_std", 1.0),
        latent_loss_type=config_overrides.get("latent_loss_type", "l2"),
        latent_loss_weight=config_overrides.get("latent_loss_weight", 1.0),
        pixel_loss_type=config_overrides.get("pixel_loss_type", "lpips"),
        pixel_loss_weight=config_overrides.get("pixel_loss_weight", 0.0),
        selected_timesteps=config_overrides.get("selected_timesteps"),
        prob=config_overrides.get("prob"),
        conditioning_images_keys=config_overrides.get("conditioning_images_keys"),
        conditioning_masks_keys=config_overrides.get("conditioning_masks_keys"),
        bridge_noise_sigma=config_overrides.get("bridge_noise_sigma", 0.0),
    )
    _load_state_dict(model, args.checkpoint_path)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    to_pil = ToPILImage()
    with torch.no_grad():
        for batch_inputs in _batch_iter(inputs, args.batch_size):
            source_images = []
            shading_images = []
            normal_images = []
            for item in batch_inputs:
                source_images.append(_load_image(item.source_path, args.image_size))
                shading_images.append(_load_image(item.shading_path, args.image_size))
                normal_images.append(_load_image(item.normal_path, args.image_size))

            source_tensor = torch.stack([ToTensor()(img) for img in source_images]) * 2 - 1
            shading_tensor = torch.stack([ToTensor()(img) for img in shading_images]) * 2 - 1
            normal_tensor = torch.stack([ToTensor()(img) for img in normal_images]) * 2 - 1

            batch = {
                model.source_key: source_tensor.to(device=device, dtype=dtype),
                "shading": shading_tensor.to(device=device, dtype=dtype),
                "normal": normal_tensor.to(device=device, dtype=dtype),
            }

            z_source = model.vae.encode(batch[model.source_key])
            output = model.sample(
                z=z_source,
                num_steps=args.num_steps,
                conditioner_inputs=batch,
                max_samples=len(batch_inputs),
            ).clamp(-1, 1)

            output = (output.float().cpu() + 1) / 2
            for item, output_image in zip(batch_inputs, output):
                pil_image = to_pil(output_image)
                pil_image = pil_image.resize(item.original_size, Image.Resampling.BICUBIC)
                output_path = os.path.join(args.output_dir, item.name)
                pil_image.save(output_path)
                logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()