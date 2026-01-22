import argparse
import os
import random
import re
from pathlib import Path
from typing import List, Optional

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import h5py
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode, ToPILImage

from lbm.inference.relight import build_filllight_model

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LIGHTING_PARAM_SUFFIXES = ".h5"
MANUAL_LIGHTING_PARAMS = None


# Example:
# MANUAL_LIGHTING_PARAMS = [0.1, -0.2, 120.0, 1.0, 1.0, 1.0, 0.03]


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


def _build_items(root_dir: Path) -> List[dict]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Missing dataset root: {root_dir}")

    items = []
    # alb_pattern = re.compile(r"^(?P<pos>\d{3})_alb$")
    alb_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_elb$")
    rgb_pattern = re.compile(r"^(?P<pos>\d{3})_999_rgb$")
    depth_pattern = re.compile(r"^(?P<pos>\d{3})_dpt$")
    lighting_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_lgt$")

    base_dirs = [path for path in root_dir.iterdir() if path.is_dir()]
    for base_dir in sorted(base_dirs):
        files = list(base_dir.iterdir())
        alb_files, rgb_files, depth_files, lighting_params_files = {}, {}, {}, {}
        for path in files:
            if not path.is_file():
                continue
            stem = path.stem
            suffix = path.suffix.lower()
            if suffix in VALID_SUFFIXES:
                alb_match = alb_pattern.match(stem)
                if alb_match:
                    alb_files[alb_match.group("pos")] = path
                    continue
                rgb_match = rgb_pattern.match(stem)
                if rgb_match:
                    rgb_files[rgb_match.group("pos")] = path
                    continue
            if suffix == ".exr":
                depth_match = depth_pattern.match(stem)
                if depth_match:
                    depth_files[depth_match.group("pos")] = path
                    continue
            if suffix == LIGHTING_PARAM_SUFFIXES:
                lighting_match = lighting_pattern.match(stem)
                if lighting_match:
                    pos_id = lighting_match.group("pos")
                    light_id = lighting_match.group("light")
                    lighting_params_files.setdefault(pos_id, {})[light_id] = path

        # valid_pos_ids = set(target_files.keys())
        # valid_pos_ids &= set(rgb_files.keys())
        # valid_pos_ids &= set(alb_files.keys())
        # valid_pos_ids &= set(depth_files.keys())
        for pos_id in sorted(set(lighting_params_files.keys())):
            light_map = lighting_params_files[pos_id]
            alb_path = alb_files.get(pos_id)
            rgb_path = rgb_files.get(pos_id)
            depth_path = depth_files.get(pos_id)
            for light_id, light_path in light_map.items():
                if int(light_id) >= 100:
                    continue
                lighting_params_path = lighting_params_files.get(pos_id, {}).get(light_id)
                items.append({"alb_source": alb_path,
                              "rgb": rgb_path,
                              "depth": depth_path,
                              "lighting_params": lighting_params_path,
                              "id": f"{pos_id}_{light_id}"})

    if not items:
        raise ValueError("No matching samples found across albedo/fill/rgb/depth/lighting params folders.")

    return items


def _load_rgb(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST_EXACT),
         transforms.ToTensor()])
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
    depth_tensor = torch.nn.functional.interpolate(depth_tensor.unsqueeze(0),
                                                   size=(image_size, image_size),
                                                   mode="nearest").squeeze(0)
    return depth_tensor * 2 - 1


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


def _load_lighting_params_h5(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as file:
        if "lighting_params" in file:
            data = file["lighting_params"][()]
        else:
            first_key = next(iter(file.keys()))
            data = file[first_key][()]
    params = torch.tensor(data, dtype=torch.float32)
    return _normalize_lighting_params(params)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    filtered_state = {key.replace("model.", ""): value
                      for key, value in state_dict.items()
                      if key.startswith("model.")}
    model.load_state_dict(filtered_state, strict=False)


def _format_test_output_dir(config: dict) -> Path:
    base_output = Path(config.get("output_path", "./fill_lighting_outputs"))
    checkpoint_path = config.get("checkpoint_path", "")
    match = re.search(r"epoch=(\d+)-step=(\d+)", checkpoint_path)
    if match:
        epoch_id = int(match.group(1))
        step_id = int(match.group(2))
        suffix = f"{epoch_id:02d}_{step_id:06d}"
    else:
        suffix = "unknown"
    return base_output / "test" / suffix


def run_inference(config: dict) -> None:
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = (torch.bfloat16 if str(config.get("torch_dtype", "bfloat16")) == "bfloat16" else torch.float16)

    model = build_filllight_model(
        backbone_signature=config.get("backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        vae_num_channels=int(config.get("vae_num_channels", 4)),
        unet_input_channels=int(config.get("unet_input_channels", 9)),
        source_key=config.get("source_key", "source"),
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
        lighting_embedder_config=config.get("lighting_embedder_config"),
        lighting_condition_weight=float(config.get("lighting_condition_weight", 1.0)),
        concat_condition_weight=float(config.get("concat_condition_weight", 1.0)),
        bridge_noise_sigma=float(config.get("bridge_noise_sigma", 0.0)),
    )

    model.to(device)
    model.eval()

    checkpoint_path = config.get("checkpoint_path")
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be provided.")
    _load_checkpoint(model, checkpoint_path)

    data_root = config.get("data_root")
    if not data_root:
        raise ValueError("data_root must be provided.")
    data_root = Path(data_root)
    items = _build_items(data_root)

    manual_lighting_params = config.get("lighting_params")
    if manual_lighting_params is not None:
        lighting_params_tensor = _normalize_lighting_params(torch.tensor(manual_lighting_params, dtype=torch.float32))
    else:
        lighting_params_tensor = None

    image_size = int(config.get("image_size", 512))
    num_steps = int(config.get("num_inference_steps", 4))
    output_dir = _format_test_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_samples = config.get("max_samples")
    if max_samples is not None:
        items = items[: int(max_samples)]

    for item in items:
        albedo_tensor = _load_rgb(item["alb_source"], image_size)
        rgb_tensor = _load_rgb(item["rgb"], image_size)
        depth_tensor = _load_depth(item["depth"], image_size)
        shading_tensor = (rgb_tensor / albedo_tensor.clamp(1e-3, 1.0)).clamp(0.0, 1.0)
        if lighting_params_tensor is None and item["lighting_params"] is None:
            raise ValueError(f"Missing lighting params; provide MANUAL_LIGHTING_PARAMS.")
        lighting_tensor = (lighting_params_tensor
                           if lighting_params_tensor is not None
                           else _load_lighting_params_h5(item["lighting_params"]))

        batch = {config.get("source_key", "source"): (albedo_tensor * 2 - 1).unsqueeze(0).to(device),
                 "shading": (shading_tensor * 2 - 1).unsqueeze(0).to(device),
                 "depth": depth_tensor.unsqueeze(0).to(device),
                 "lighting_params": lighting_tensor.unsqueeze(0).to(device)}

        with torch.no_grad():
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=torch_dtype):
                    z_source = model.vae.encode(batch[model.source_key])
                    output = model.sample(z=z_source,
                                          num_steps=num_steps,
                                          conditioner_inputs=batch,
                                          max_samples=1)
            else:
                z_source = model.vae.encode(batch[model.source_key])
                output = model.sample(z=z_source,
                                      num_steps=num_steps,
                                      conditioner_inputs=batch,
                                      max_samples=1)

        output_image = (output[0].float().cpu() + 1) / 2
        output_image = torch.clamp(output_image, 0.0, 1.0)
        output_pil = ToPILImage()(output_image)
        relative_parent = item["lighting_params"].parent.relative_to(data_root)
        output_subdir = output_dir / relative_parent
        output_subdir.mkdir(parents=True, exist_ok=True)
        output_name = f"{item['id']}_fill.png"
        output_pil.save(output_subdir / output_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config", type=str,
                        default='/mnt/data1/ssy/render_people/LBM/examples/training/config/fill_lighting.yaml')
    parser.add_argument("--checkpoint_path", type=str,
                        # default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=2-step=30000.ckpt')
                        # default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=3-step=50000.ckpt')
                        # default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=2-step=40000-v1.ckpt')
                        # default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=3-step=50000-v1.ckpt')
                        # default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=0-step=10000-v1.ckpt')
                        default='/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=1-step=20000-v1.ckpt')
    parser.add_argument("--data_root", type=str, default='/mnt/data1/ssy/render_people/fill-light-dataset/test')
    parser.add_argument("--output_path", type=str,
                        default='/mnt/data1/ssy/render_people/LBM/examples/inference/outputs/fill_lighting')
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=3407)

    args = parser.parse_args()

    config = _merge_config(_load_yaml(args.train_config), vars(args))

    random.seed(int(config.get("seed", 0)))
    run_inference(config)


if __name__ == "__main__":
    main()
