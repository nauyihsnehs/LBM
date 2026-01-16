import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode, ToPILImage

from lbm.inference.refine import build_filllight_refine_model

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


def _load_rgb(path: Path, size: Optional[int] = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if size is None:
        transform = transforms.ToTensor()
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (size, size),
                    interpolation=InterpolationMode.BILINEAR,
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


def run_inference(config: dict) -> None:
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = (
        torch.bfloat16
        if str(config.get("torch_dtype", "bfloat16")) == "bfloat16"
        else torch.float16
    )

    model = build_filllight_refine_model(
        backbone_signature=config.get(
            "backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        ),
        unet_input_channels=int(config.get("unet_input_channels", 6)),
        unet_output_channels=int(config.get("unet_output_channels", 3)),
        source_key=config.get("source_key", "source"),
        target_key=config.get("target_key", "target"),
        mask_key=config.get("mask_key"),
        timestep_sampling=config.get("timestep_sampling", "log_normal"),
        logit_mean=float(config.get("logit_mean", 0.0)),
        logit_std=float(config.get("logit_std", 1.0)),
        pixel_loss_type=config.get("pixel_loss_type", "l2"),
        latent_loss_type=config.get("latent_loss_type", "l2"),
        latent_loss_weight=float(config.get("latent_loss_weight", 1.0)),
        pixel_loss_weight=float(config.get("pixel_loss_weight", 0.0)),
        selected_timesteps=config.get("selected_timesteps"),
        prob=config.get("prob"),
        bridge_noise_sigma=float(config.get("bridge_noise_sigma", 0.0)),
        block_out_channels=config.get("block_out_channels"),
        layers_per_block=int(config.get("layers_per_block", 2)),
    )
    model.to(device).to(torch_dtype)
    model.eval()

    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        _load_checkpoint(model, checkpoint_path)

    filllight_path = Path(config.get("filllight_image"))
    rgb_path = Path(config.get("rgb_image"))

    if not (filllight_path.exists() and rgb_path.exists()):
        raise ValueError("filllight_image and rgb_image must be valid files.")

    if filllight_path.suffix.lower() not in VALID_SUFFIXES:
        raise ValueError("filllight_image must be an image file.")
    if rgb_path.suffix.lower() not in VALID_SUFFIXES:
        raise ValueError("rgb_image must be an image file.")

    output_dir = Path(config.get("output_path", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = config.get("image_size")
    rgb_tensor = _load_rgb(rgb_path, image_size)
    filllight_tensor = _load_rgb(filllight_path, image_size)

    source = torch.cat([filllight_tensor, rgb_tensor], dim=0)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.no_grad():
        if device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch_dtype):
                output = model.sample(
                    z=(source * 2 - 1).unsqueeze(0).to(device),
                    num_steps=int(config.get("num_inference_steps", 1)),
                    conditioner_inputs=None,
                    max_samples=1,
                )
        else:
            output = model.sample(
                z=(source * 2 - 1).unsqueeze(0).to(device),
                num_steps=int(config.get("num_inference_steps", 1)),
                conditioner_inputs=None,
                max_samples=1,
            )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    max_memory_mb = 0.0
    if device.startswith("cuda"):
        max_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

    output_image = (output[0].float().cpu() + 1) / 2
    output_pil = ToPILImage()(output_image.clamp(0, 1))
    output_name = f"{filllight_path.stem}_fill_lighting_refine.png"
    output_pil.save(output_dir / output_name)

    logger.info("Processed 1 sample")
    logger.info("Inference time: %.4fs", elapsed)
    if device.startswith("cuda"):
        logger.info("Peak GPU memory: %.2f MB", max_memory_mb)
    else:
        logger.info("GPU memory stats unavailable on CPU device.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--filllight_image", type=str, default=None)
    parser.add_argument("--rgb_image", type=str, default=None)
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

    required_keys = ["filllight_image", "rgb_image"]
    if not all(merged.get(key) for key in required_keys):
        raise ValueError(
            "filllight_image and rgb_image must be provided via args or config."
        )

    run_inference(merged)


if __name__ == "__main__":
    main()
