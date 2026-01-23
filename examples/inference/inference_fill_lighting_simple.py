import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode, ToPILImage

from lbm.inference.relight import build_filllight_model

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

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


def _load_depth(path: Path, image_size: int) -> torch.Tensor:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read depth image: {path}")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.astype("float32")
    depth[depth >= 100.0] = 0.0
    max_val = float(torch.tensor(depth).quantile(0.99).item())
    depth = depth.clip(min=0.0, max=max_val if max_val > 0 else 0.0)
    if max_val > 0:
        depth = depth / max_val
    depth_tensor = torch.from_numpy(depth).unsqueeze(0)
    depth_tensor = torch.nn.functional.interpolate(
        depth_tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="nearest",
    ).squeeze(0)
    return depth_tensor * 2 - 1


def _load_lighting_params(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing lighting params file: {path}")
    with open(path, "r") as file:
        payload = yaml.safe_load(file) or []
    if not isinstance(payload, list):
        raise ValueError("lighting_params.yaml must be a list of items.")
    return payload


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
        unet_input_channels=int(config.get("unet_input_channels", 5)),
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
        conditioning_images_keys=config.get("conditioning_images_keys"),
        conditioning_masks_keys=config.get("conditioning_masks_keys"),
        lighting_embedder_config=config.get("lighting_embedder_config"),
        lighting_condition_weight=float(config.get("lighting_condition_weight", 1.0)),
        concat_condition_weight=float(config.get("concat_condition_weight", 1.0)),
        bridge_noise_sigma=float(config.get("bridge_noise_sigma", 0.0)),
    )
    model.to(device).to(torch_dtype)
    model.eval()

    checkpoint_path = config.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be provided.")
    _load_checkpoint(model, checkpoint_path)

    pipeline_root = Path(config["pipeline_output_root"])
    depth_dir = pipeline_root / "depth"
    lighting_params_path = pipeline_root / "lighting_params.yaml"

    rgb_root = Path(config["rgb_path"])
    rgb_images = _list_images(rgb_root) if rgb_root.is_dir() else [rgb_root]
    if not rgb_images:
        raise ValueError(f"No RGB images found under {rgb_root}")

    if not depth_dir.exists():
        raise FileNotFoundError(f"Missing depth folder: {depth_dir}")

    lighting_params = _load_lighting_params(lighting_params_path)
    if not lighting_params:
        raise ValueError(f"No lighting params found in {lighting_params_path}")

    output_dir = Path(config.get("output_path") or pipeline_root / "fill_simple")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_size = 512
    num_steps = int(config.get("num_inference_steps", 1))
    to_pil = ToPILImage()

    for rgb_path in rgb_images:
        stem = rgb_path.stem
        depth_path = depth_dir / f"{stem}_dpt.exr"
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth image: {depth_path}")

        source_tensor = _load_rgb(rgb_path, image_size)
        depth_tensor = _load_depth(depth_path, image_size)
        lighting_scale = torch.ones_like(depth_tensor)

        for entry in lighting_params:
            light_id = entry.get("id", "light")
            lighting_tensor = torch.tensor(entry["params"], dtype=torch.float32)
            batch = {
                config.get("source_key", "source"): (source_tensor * 2 - 1)
                .unsqueeze(0)
                .to(device),
                "depth": depth_tensor.unsqueeze(0).to(device),
                "lighting_scale": lighting_scale.unsqueeze(0).to(device),
                "lighting_params": lighting_tensor.unsqueeze(0).to(device),
            }

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

            output_image = (output[0].float().cpu() + 1) / 2
            output_pil = to_pil(output_image.clamp(0.0, 1.0))
            output_name = f"{stem}_{light_id}_fill_simple.png"
            output_pil.save(output_dir / output_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal fill lighting simple inference.")
    parser.add_argument("--pipeline_output_root", type=str, default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/in-the-wild-1024-results/20260123_002054')
    parser.add_argument("--checkpoint_path", type=str, default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting_simple/epoch=1-step=10000.ckpt')
    parser.add_argument("--train_config", type=str, default='/mnt/data1/ssy/render_people/LBM/examples/training/config/fill_lighting_simple.yaml')
    parser.add_argument("--output_path", type=str, default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/in-the-wild-1024-results/simple')
    parser.add_argument("--rgb_path", type=str, default='/mnt/data1/ssy/render_people/fill-light-dataset/infer/in-the-wild-1024')
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--torch_dtype", type=str, default=None)

    args = parser.parse_args()

    train_config = _load_yaml(args.train_config)
    config = _merge_config(train_config, vars(args))
    run_inference(config)


if __name__ == "__main__":
    main()
