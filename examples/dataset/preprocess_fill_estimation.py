import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

import fire
import h5py
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, ToPILImage

from lbm.inference.relight import build_filllight_model

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEPTH_SUFFIXES = {".exr"}
LIGHTING_PARAM_SUFFIX = ".h5"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _FillLightingItem:
    def __init__(
            self,
            albedo_path: Path,
            elb_path: Optional[Path],
            rgb_path: Path,
            depth_path: Path,
            lighting_params_path: Path,
            output_path: Path,
    ) -> None:
        self.albedo_path = albedo_path
        self.elb_path = elb_path
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.lighting_params_path = lighting_params_path
        self.output_path = output_path


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


def _fill_output_path(
        lighting_params_path: Path,
        output_root: Optional[Path],
        output_suffix: str,
        output_extension: str,
        lighting_suffix: str,
) -> Path:
    stem = lighting_params_path.stem
    if lighting_suffix and stem.endswith(lighting_suffix):
        stem = stem[: -len(lighting_suffix)] + output_suffix
    else:
        stem = f"{stem}{output_suffix}"
    output_name = f"{stem}{output_extension}"
    if output_root is None:
        return lighting_params_path.with_name(output_name)
    output_dir = output_root / lighting_params_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / output_name


def _load_rgb(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST_EXACT),
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
    return depth_tensor


def _load_lighting_params(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as file:
        if "lighting_params" in file:
            data = file["lighting_params"][()]
        else:
            first_key = next(iter(file.keys()))
            data = file[first_key][()]
    params = torch.tensor(data, dtype=torch.float32)

    params = params.flatten()
    position = params[:2]
    intensity = (params[2:3] / 200) * 2 - 1
    color = params[3:6] * 2 - 1
    area = (params[-1:] * 5) * 2 - 1
    shading_scale = torch.ones_like(area)
    params = torch.cat([position, intensity, color, area, shading_scale], dim=0)
    return params


def _collect_items(
        root: Path,
        output_root: Optional[Path],
        albedo_suffix: str,
        albedo_extension: str,
        elb_suffix: str,
        output_suffix: str,
        output_extension: str,
        lighting_suffix: str,
        overwrite: bool,
) -> List[_FillLightingItem]:
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    rgb_pattern = re.compile(r"^(?P<pos>\d{3})_999_rgb$")
    alb_pattern = re.compile(rf"^(?P<pos>\d{{3}}){re.escape(albedo_suffix)}$")
    elb_pattern = re.compile(rf"^(?P<pos>\d{{3}}){re.escape(elb_suffix)}$")
    depth_pattern = re.compile(r"^(?P<pos>\d{3})_dpt$")
    lighting_pattern = re.compile(rf"^(?P<pos>\d{{3}})_(?P<light>\d{{3}}){lighting_suffix}$")

    items: List[_FillLightingItem] = []
    person_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not person_dirs:
        raise ValueError(f"No subdirectories found under {root}")

    for person_dir in sorted(person_dirs):
        rgb_files: dict[str, Path] = {}
        albedo_files: dict[str, Path] = {}
        elb_files: dict[str, Path] = {}
        depth_files: dict[str, Path] = {}
        lighting_files: dict[str, dict[str, Path]] = {}

        for path in person_dir.iterdir():
            if not path.is_file():
                continue
            stem = path.stem
            suffix = path.suffix.lower()
            if suffix == LIGHTING_PARAM_SUFFIX:
                lighting_match = lighting_pattern.match(stem)
                if lighting_match:
                    pos_id = lighting_match.group("pos")
                    light_id = lighting_match.group("light")
                    lighting_files.setdefault(pos_id, {})[light_id] = path
                continue
            if suffix in VALID_SUFFIXES:
                rgb_match = rgb_pattern.match(stem)
                if rgb_match:
                    rgb_files[rgb_match.group("pos")] = path
                    continue
                if suffix == albedo_extension:
                    alb_match = alb_pattern.match(stem)
                    if alb_match:
                        albedo_files[alb_match.group("pos")] = path
                        continue
                    elb_match = elb_pattern.match(stem)
                    if elb_match:
                        elb_files[elb_match.group("pos")] = path
                        continue
                continue
            if suffix in DEPTH_SUFFIXES:
                depth_match = depth_pattern.match(stem)
                if depth_match:
                    depth_files[depth_match.group("pos")] = path
                    continue

        valid_pos_ids = set(lighting_files.keys())
        valid_pos_ids &= set(rgb_files.keys())
        valid_pos_ids &= set(albedo_files.keys())
        valid_pos_ids &= set(depth_files.keys())

        for pos_id in sorted(valid_pos_ids):
            light_map = lighting_files[pos_id]
            rgb_path = rgb_files.get(pos_id)
            albedo_path = albedo_files.get(pos_id)
            depth_path = depth_files.get(pos_id)
            elb_path = elb_files.get(pos_id)
            if rgb_path is None or albedo_path is None or depth_path is None:
                continue
            for light_id, lighting_path in light_map.items():
                if int(light_id) >= 100:
                    continue
                output_path = _fill_output_path(
                    lighting_path,
                    output_root,
                    output_suffix,
                    output_extension,
                    lighting_suffix,
                )
                if output_path.exists() and not overwrite:
                    continue
                items.append(
                    _FillLightingItem(
                        albedo_path=albedo_path,
                        elb_path=elb_path,
                        rgb_path=rgb_path,
                        depth_path=depth_path,
                        lighting_params_path=lighting_path,
                        output_path=output_path,
                    )
                )

    return items


class _FillLightingDataset(Dataset):
    def __init__(self, items: Sequence[_FillLightingItem], image_size: int, elb_ratio: float):
        self._items = list(items)
        self._image_size = image_size
        self._elb_ratio = elb_ratio

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Path]:
        item = self._items[index]
        source_path = item.albedo_path
        if item.elb_path is not None and torch.rand(1).item() < self._elb_ratio:
            source_path = item.elb_path
        albedo = _load_rgb(source_path, self._image_size)
        rgb = _load_rgb(item.rgb_path, self._image_size)
        depth = _load_depth(item.depth_path, self._image_size)
        lighting_params = _load_lighting_params(item.lighting_params_path)
        return albedo, rgb, depth, lighting_params, item.output_path


def _collate_batch(
        items: List[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Path]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Path]]:
    albedo, rgb, depth, lighting_params, output_paths = zip(*items)
    return (
        torch.stack(list(albedo), dim=0),
        torch.stack(list(rgb), dim=0),
        torch.stack(list(depth), dim=0),
        torch.stack(list(lighting_params), dim=0),
        list(output_paths),
    )


def _run_batch(
        model: torch.nn.Module,
        albedo: torch.Tensor,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        lighting_params: torch.Tensor,
        output_paths: List[Path],
        to_pil: ToPILImage,
        num_steps: int,
        source_key: str,
        device: torch.device,
        dtype: torch.dtype,
) -> int:
    source = albedo.to(device=device, dtype=dtype) * 2 - 1
    shading = (rgb / albedo.clamp(1e-3, 1.0)).clamp(0.0, 1.0)
    shading = shading.to(device=device, dtype=dtype) * 2 - 1
    depth = depth.to(device=device, dtype=dtype)
    lighting_scale = torch.ones_like(depth)
    lighting_params = lighting_params.to(device=device, dtype=dtype)

    batch = {
        source_key: source,
        "shading": shading,
        "depth": depth,
        "lighting_scale": lighting_scale,
        "lighting_params": lighting_params,
    }

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                z_source = model.vae.encode(batch[source_key])
                output = model.sample(
                    z=z_source,
                    num_steps=num_steps,
                    conditioner_inputs=batch,
                    max_samples=albedo.shape[0],
                )
        else:
            z_source = model.vae.encode(batch[source_key])
            output = model.sample(
                z=z_source,
                num_steps=num_steps,
                conditioner_inputs=batch,
                max_samples=albedo.shape[0],
            )

    output_batch = (output.float().detach().cpu() + 1) / 2
    output_batch = output_batch.clamp(0.0, 1.0)
    processed = 0
    for output_tensor, output_path in zip(output_batch, output_paths):
        output_image = to_pil(output_tensor)
        output_image.save(output_path)
        processed += 1
    return processed


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


def main(
        # data_root: str = "/mnt/data1/ssy/render_people/fill-light-dataset/train",
        data_root: str = "/mnt/data1/ssy/render_people/fill-light-dataset/val",
        train_config: Optional[str] = "/mnt/data1/ssy/render_people/LBM/examples/training/config/fill_lighting.yaml",
        checkpoint_path: Optional[
            str] = "/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/fill_lighting/epoch=2-step=40000-v1.ckpt",
        output_root: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        num_inference_steps: int = 1,
        image_size: int = 512,
        torch_dtype: str = "bfloat16",
        albedo_suffix: str = "_alb",
        albedo_extension: str = ".png",
        elb_suffix: str = "_999_elb",
        elb_ratio: float = 0.8,
        output_suffix: str = "_fill",
        output_extension: str = ".png",
        lighting_suffix: str = "_lgt",
        overwrite: bool = False,
):
    config = _merge_config(
        _load_yaml(train_config),
        {
            "data_root": data_root,
            "checkpoint_path": checkpoint_path,
            "device": device,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "num_inference_steps": num_inference_steps,
            "image_size": image_size,
            "torch_dtype": torch_dtype,
            "output_root": output_root,
            "albedo_suffix": albedo_suffix,
            "albedo_extension": albedo_extension,
            "elb_suffix": elb_suffix,
            "elb_ratio": elb_ratio,
            "output_suffix": output_suffix,
            "output_extension": output_extension,
            "lighting_suffix": lighting_suffix,
            "overwrite": overwrite,
        },
    )

    root = Path(config["data_root"])
    output_root_path = Path(config["output_root"]) if config.get("output_root") else None
    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = str(config.get("torch_dtype", "bfloat16"))
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    device = torch.device(device_name)

    model = build_filllight_model(
        backbone_signature=config.get("backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
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
        lighting_conditioning=True,
        lighting_embedder_config=config.get("lighting_embedder_config"),
        bridge_noise_sigma=float(config.get("bridge_noise_sigma", 0.0)),
    )
    model.to(device).to(dtype)
    model.eval()

    if config.get("checkpoint_path"):
        _load_checkpoint(model, config["checkpoint_path"])

    items = _collect_items(
        root,
        output_root_path,
        config["albedo_suffix"],
        config["albedo_extension"],
        config["elb_suffix"],
        config["output_suffix"],
        config["output_extension"],
        config["lighting_suffix"],
        config["overwrite"],
    )
    if not items:
        raise ValueError(f"No matching samples found under {root}.")

    dataset = _FillLightingDataset(
        items,
        int(config["image_size"]),
        float(config.get("elb_ratio", 0.8)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_batch,
    )

    to_pil = ToPILImage()
    processed = 0
    for albedo, rgb, depth, lighting_params, output_paths in loader:
        processed += _run_batch(
            model,
            albedo,
            rgb,
            depth,
            lighting_params,
            output_paths,
            to_pil,
            int(config["num_inference_steps"]),
            config.get("source_key", "source"),
            device,
            dtype,
        )

    logger.info("Generated %d estimated fill lighting images.", processed)


if __name__ == "__main__":
    fire.Fire(main)
