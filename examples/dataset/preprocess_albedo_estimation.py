import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import fire
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import ToPILImage

from lbm.inference.relight import build_relight_model

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _list_rgb_images(folder: Path) -> Iterable[Path]:
    for path in sorted(folder.iterdir()):
        if (
                path.is_file()
                and path.suffix.lower() in VALID_SUFFIXES
                and path.stem.endswith("999_rgb")
        ):
            yield path


def _load_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image_size > 0:
        image = transforms.Resize((image_size, image_size))(image)
    return transforms.ToTensor()(image)


def _albedo_path(
        source_path: Path,
        output_root: Optional[Path],
        albedo_suffix: str,
        albedo_extension: str,
) -> Path:
    albedo_stem = source_path.stem[: -len("_rgb")] + albedo_suffix
    albedo_name = f"{albedo_stem}{albedo_extension}"
    if output_root is None:
        return source_path.with_name(albedo_name)
    relative = source_path.parent.name
    output_dir = output_root / relative
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / albedo_name


class _AlbedoDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], image_size: int):
        self._image_paths = list(image_paths)
        self._image_size = image_size

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Path]:
        path = self._image_paths[index]
        return _load_tensor(path, self._image_size), path


def _collate_batch(items: List[tuple[torch.Tensor, Path]]) -> tuple[torch.Tensor, List[Path]]:
    images, paths = zip(*items)
    return torch.stack(list(images), dim=0), list(paths)


def _run_batch(
        model: torch.nn.Module,
        images: torch.Tensor,
        batch_paths: List[Path],
        output_root: Optional[Path],
        albedo_suffix: str,
        albedo_extension: str,
        to_pil: ToPILImage,
        num_steps: int,
        source_key: str,
        device: torch.device,
        dtype: torch.dtype,
        overwrite: bool = False,
) -> int:
    source = images.to(device=device, dtype=dtype) * 2 - 1
    batch = {source_key: source}
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                z_source = model.vae.encode(batch[source_key])
                output = model.sample(
                    z=z_source,
                    num_steps=num_steps,
                    conditioner_inputs=batch,
                    max_samples=images.shape[0],
                )
        else:
            z_source = model.vae.encode(batch[source_key])
            output = model.sample(
                z=z_source,
                num_steps=num_steps,
                conditioner_inputs=batch,
                max_samples=images.shape[0],
            )
    albedo_batch = (output.float().detach().cpu() + 1) / 2
    albedo_batch = albedo_batch.clamp(0.0, 1.0)
    processed = 0
    for albedo_tensor, source_path in zip(albedo_batch, batch_paths):
        albedo_path = _albedo_path(
            source_path,
            output_root,
            albedo_suffix,
            albedo_extension,
        )
        if albedo_path.exists() and not overwrite:
            continue
        albedo_image = to_pil(albedo_tensor)
        albedo_image.save(albedo_path)
        processed += 1
    return processed


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
        # data_root: str = '/mnt/data1/ssy/render_people/fill-light-dataset/test',
        data_root: str = '/mnt/data1/ssy/render_people/fill-light-dataset/train',
        train_config: Optional[str] = '/mnt/data1/ssy/render_people/LBM/examples/training/config/albedo.yaml',
        inference_config: Optional[
            str] = '/mnt/data1/ssy/render_people/LBM/examples/inference/config/albedo_infer.yaml',
        checkpoint_path: Optional[
            str] = '/mnt/data1/ssy/render_people/LBM/examples/training/checkpoints/albedo/epoch=1-step=50000.ckpt',
        output_root: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        num_inference_steps: int = 4,
        image_size: int = 512,
        torch_dtype: str = "bfloat16",
        albedo_suffix: str = "_elb",
        albedo_extension: str = ".png",
        overwrite: bool = False,
):
    config = _merge_config(
        _load_yaml(train_config),
        _load_yaml(inference_config),
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
            "overwrite": overwrite,
        },
    )

    root = Path(config["data_root"])
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    output_root_path = Path(config["output_root"]) if config.get("output_root") else None
    device_name = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = str(config.get("torch_dtype", "bfloat16"))
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    device = torch.device(device_name)

    model = build_relight_model(
        backbone_signature=config.get("backbone_signature", "stable-diffusion-v1-5/stable-diffusion-v1-5"),
        vae_num_channels=int(config.get("vae_num_channels", 4)),
        unet_input_channels=int(config.get("unet_input_channels", 4)),
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
    model.to(device).to(dtype)
    model.eval()

    if config.get("checkpoint_path"):
        _load_checkpoint(model, config["checkpoint_path"])

    to_pil = ToPILImage()

    person_dirs = [path for path in root.iterdir() if path.is_dir()]
    if not person_dirs:
        raise ValueError(f"No subdirectories found under {root}")

    image_paths: List[Path] = []
    for person_dir in sorted(person_dirs):
        for rgb_path in _list_rgb_images(person_dir):
            albedo_path = _albedo_path(
                rgb_path,
                output_root_path,
                config["albedo_suffix"],
                config["albedo_extension"],
            )
            if albedo_path.exists() and not config["overwrite"]:
                continue
            image_paths.append(rgb_path)

    dataset = _AlbedoDataset(image_paths, int(config["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
        collate_fn=_collate_batch,
    )
    processed = 0
    for images, batch_paths in tqdm(loader, desc="Estimating albedo", unit="batch"):
        processed += _run_batch(
            model,
            images,
            batch_paths,
            output_root_path,
            config["albedo_suffix"],
            config["albedo_extension"],
            to_pil,
            int(config["num_inference_steps"]),
            config.get("source_key", "source"),
            device,
            dtype,
            config["overwrite"],
        )

    logger.info("Generated %d estimated albedo images.", processed)


if __name__ == "__main__":
    fire.Fire(main)
