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

from lbm.inference.relight import build_relight_model

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


def _load_tensor(path: Path, image_size: int) -> torch.Tensor:
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
    return transform(image) * 2 - 1


def _prepare_pairs(
    source_path: str,
    shading_path: str,
    normal_path: str,
) -> List[dict]:
    source = Path(source_path)
    shading = Path(shading_path)
    normal = Path(normal_path)

    if source.is_dir() or shading.is_dir() or normal.is_dir():
        if not (source.is_dir() and shading.is_dir() and normal.is_dir()):
            raise ValueError(
                "When using folder inputs, source, shading, and normal must all be folders."
            )
        source_files = _list_images(source)
        shading_files = _list_images(shading)
        normal_files = _list_images(normal)
        if not (source_files and shading_files and normal_files):
            raise ValueError("One or more input folders are empty.")
        if not (
            len(source_files) == len(shading_files) == len(normal_files)
        ):
            raise ValueError(
                "Folder inputs must contain the same number of images for pairing."
            )
        return [
            {
                "source": source_files[i],
                "shading": shading_files[i],
                "normal": normal_files[i],
            }
            for i in range(len(source_files))
        ]

    return [
        {
            "source": source,
            "shading": shading,
            "normal": normal,
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

    model = build_relight_model(
        backbone_signature=config.get(
            "backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        ),
        vae_num_channels=int(config.get("vae_num_channels", 4)),
        unet_input_channels=int(config.get("unet_input_channels", 12)),
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
    )
    model.to(device).to(torch_dtype)
    model.eval()

    checkpoint_path = config.get("checkpoint_path")
    if checkpoint_path:
        logger.info("Loading checkpoint from %s", checkpoint_path)
        _load_checkpoint(model, checkpoint_path)

    pairs = _prepare_pairs(
        config.get("source_image"),
        config.get("shading_image"),
        config.get("normal_image"),
    )

    output_dir = Path(config.get("output_path", "../inference/outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = int(config.get("image_size", 512))
    num_steps = int(config.get("num_inference_steps", 1))

    total_time = 0.0
    max_memory_mb = 0.0

    for item in pairs:
        source_tensor = _load_tensor(item["source"], image_size)
        shading_tensor = _load_tensor(item["shading"], image_size)
        normal_tensor = _load_tensor(item["normal"], image_size)
        batch = {
            config.get("source_key", "source"): source_tensor.unsqueeze(0).to(device),
            "shading": shading_tensor.unsqueeze(0).to(device),
            "normal": normal_tensor.unsqueeze(0).to(device),
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
        output_name = f"{item['source'].stem}_relight.png"
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
    parser.add_argument("--normal_image", type=str, default=None)
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

    required_keys = ["source_image", "shading_image", "normal_image"]
    if not all(merged.get(key) for key in required_keys):
        raise ValueError(
            "source_image, shading_image, and normal_image must be provided via args or config."
        )

    run_inference(merged)


if __name__ == "__main__":
    main()
