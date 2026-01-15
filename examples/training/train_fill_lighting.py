import datetime
import logging
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import fire
import torch
import yaml
from PIL import Image
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from lbm.inference.relight import build_relight_model
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LIGHTING_PARAM_SUFFIXES = {".txt", ".json", ".pt", ".pth", ".npy", ".h5", ".hdf5"}


class FillLightingFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        source_folder: str = "albedo",
        target_folder: str = "fill",
        rgb_folder: str = "rgb",
        depth_folder: str = "depth",
        lighting_params_folder: str = "lighting_params",
        lighting_scale_folder: str = "lighting_scale",
        image_size: int = 512,
        random_flip: bool = False,
        random_scale_min: float = 1.0,
        random_scale_max: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.lighting_params_folder = lighting_params_folder
        self.lighting_scale_folder = lighting_scale_folder
        self.image_size = image_size
        self.random_flip = random_flip
        self.random_scale_min = random_scale_min
        self.random_scale_max = random_scale_max

        self._target_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_rgb$")
        self._source_pattern = re.compile(r"^(?P<pos>\d{3})_alb$")
        self._rgb_pattern = re.compile(r"^(?P<pos>\d{3})_999_rgb$")
        self._depth_pattern = re.compile(r"^(?P<pos>\d{3})_dpt$")
        self._lighting_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_lgt$")

        self.items = self._build_items()
        if not self.items:
            raise ValueError(
                "No matching samples found across albedo/fill/rgb/depth/lighting params folders."
            )

        self.rgb_transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.NEAREST_EXACT,
                ),
                transforms.ToTensor(),
            ]
        )

    def _index_files(self, directory: Path, suffixes: Set[str]) -> Dict[str, Path]:
        files = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        ]
        return {path.stem: path for path in files}

    def _build_items(self) -> List[dict]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing dataset root: {self.root_dir}")

        items: List[dict] = []
        base_dirs = [path for path in self.root_dir.iterdir() if path.is_dir()]
        for base_dir in sorted(base_dirs):
            files = list(base_dir.iterdir())
            source_files: Dict[str, Path] = {}
            rgb_files: Dict[str, Path] = {}
            depth_files: Dict[str, Path] = {}
            lighting_params_files: Dict[str, Dict[str, Path]] = {}
            target_files: Dict[str, Dict[str, Path]] = {}

            for path in files:
                if not path.is_file():
                    continue
                stem = path.stem
                suffix = path.suffix.lower()
                if suffix in VALID_SUFFIXES:
                    source_match = self._source_pattern.match(stem)
                    if source_match:
                        source_files[source_match.group("pos")] = path
                        continue
                    rgb_match = self._rgb_pattern.match(stem)
                    if rgb_match:
                        rgb_files[rgb_match.group("pos")] = path
                        continue
                    target_match = self._target_pattern.match(stem)
                    if target_match:
                        pos_id = target_match.group("pos")
                        light_id = target_match.group("light")
                        target_files.setdefault(pos_id, {})[light_id] = path
                        continue
                if suffix == ".exr":
                    depth_match = self._depth_pattern.match(stem)
                    if depth_match:
                        depth_files[depth_match.group("pos")] = path
                        continue
                if suffix in LIGHTING_PARAM_SUFFIXES:
                    lighting_match = self._lighting_pattern.match(stem)
                    if lighting_match:
                        pos_id = lighting_match.group("pos")
                        light_id = lighting_match.group("light")
                        lighting_params_files.setdefault(pos_id, {})[light_id] = path

            for pos_id, light_map in target_files.items():
                source_path = source_files.get(pos_id)
                rgb_path = rgb_files.get(pos_id)
                depth_path = depth_files.get(pos_id)
                if source_path is None or rgb_path is None or depth_path is None:
                    continue
                for light_id, target_path in light_map.items():
                    if int(light_id) >= 100:
                        continue
                    lighting_params_path = lighting_params_files.get(pos_id, {}).get(
                        light_id
                    )
                    if lighting_params_path is None:
                        continue
                    items.append(
                        {
                            "source": source_path,
                            "target": target_path,
                            "rgb": rgb_path,
                            "depth": depth_path,
                            "lighting_params": lighting_params_path,
                        }
                    )
        return items

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.rgb_transform(image)

    def _load_depth(self, path: Path) -> torch.Tensor:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to read depth image: {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype("float32")
        depth = depth / 100.0
        max_val = float(torch.tensor(depth).quantile(0.99).item())
        depth = depth.clip(min=0.0, max=max_val if max_val > 0 else 0.0)
        if max_val > 0:
            depth = depth / max_val
        depth_tensor = torch.from_numpy(depth).unsqueeze(0)
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)
        return depth_tensor

    def _load_lighting_params(self, path: Path) -> torch.Tensor:
        if path.suffix.lower() == ".json":
            import json

            with open(path, "r") as file:
                values = json.load(file)
            params = torch.tensor(values, dtype=torch.float32)
        elif path.suffix.lower() in {".pt", ".pth"}:
            params = torch.load(path, map_location="cpu")
            params = torch.tensor(params, dtype=torch.float32)
        elif path.suffix.lower() == ".npy":
            try:
                import numpy as np
            except ImportError as exc:
                raise ImportError("numpy is required to load .npy lighting params") from exc
            params = torch.tensor(np.load(path), dtype=torch.float32)
        elif path.suffix.lower() in {".h5", ".hdf5"}:
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("h5py is required to load .h5 lighting params") from exc
            with h5py.File(path, "r") as file:
                if "lighting_params" in file:
                    data = file["lighting_params"][()]
                else:
                    first_key = next(iter(file.keys()))
                    data = file[first_key][()]
            params = torch.tensor(data, dtype=torch.float32)
        else:
            with open(path, "r") as file:
                content = file.read().replace(",", " ").split()
            params = torch.tensor([float(value) for value in content], dtype=torch.float32)

        params = params.flatten()
        if params.numel() != 7:
            raise ValueError(
                f"Expected 7 lighting params, got {params.numel()} from {path}"
            )
        return params

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        albedo = self._load_rgb(item["source"])
        target = self._load_rgb(item["target"])
        rgb = self._load_rgb(item["rgb"])
        depth = self._load_depth(item["depth"])
        lighting_params = self._load_lighting_params(item["lighting_params"])
        lighting_scale = torch.ones_like(depth)

        shading = (rgb / albedo.clamp(1e-3, 1.0)).clamp(0.0, 1.0)

        if self.random_scale_min != 1.0 or self.random_scale_max != 1.0:
            scale = torch.empty(1).uniform_(self.random_scale_min, self.random_scale_max)
            albedo = torch.clamp(albedo * scale, 0.0, 1.0)
            target = torch.clamp(target * scale, 0.0, 1.0)
            shading = torch.clamp(shading * scale, 0.0, 1.0)

        if self.random_flip and torch.rand(1).item() < 0.5:
            albedo = torch.flip(albedo, dims=[2])
            target = torch.flip(target, dims=[2])
            shading = torch.flip(shading, dims=[2])
            depth = torch.flip(depth, dims=[2])
            lighting_scale = torch.flip(lighting_scale, dims=[2])

        sample = {
            "source": albedo * 2 - 1,
            "target": target * 2 - 1,
            "shading": shading * 2 - 1,
            "depth": depth,
            "lighting_scale": lighting_scale,
            "lighting_params": lighting_params,
        }
        return sample


def get_dataloaders(
    train_data_root: str,
    validation_data_root: str,
    batch_size: int,
    source_folder: str = "albedo",
    target_folder: str = "fill",
    rgb_folder: str = "rgb",
    depth_folder: str = "depth",
    lighting_params_folder: str = "lighting_params",
    lighting_scale_folder: str = "lighting_scale",
    image_size: int = 512,
    num_workers: int = 4,
    train_random_flip: bool = True,
    train_random_scale_min: float = 1.0,
    train_random_scale_max: float = 1.0,
):
    train_dataset = FillLightingFolderDataset(
        root_dir=train_data_root,
        source_folder=source_folder,
        target_folder=target_folder,
        rgb_folder=rgb_folder,
        depth_folder=depth_folder,
        lighting_params_folder=lighting_params_folder,
        lighting_scale_folder=lighting_scale_folder,
        image_size=image_size,
        random_flip=train_random_flip,
        random_scale_min=train_random_scale_min,
        random_scale_max=train_random_scale_max,
    )
    validation_dataset = FillLightingFolderDataset(
        root_dir=validation_data_root,
        source_folder=source_folder,
        target_folder=target_folder,
        rgb_folder=rgb_folder,
        depth_folder=depth_folder,
        lighting_params_folder=lighting_params_folder,
        lighting_scale_folder=lighting_scale_folder,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, validation_loader


def main(
    train_data_root: str = "path/to/train",
    validation_data_root: str = "path/to/validation",
    backbone_signature: str = "runwayml/stable-diffusion-v1-5",
    vae_num_channels: int = 4,
    unet_input_channels: int = 10,
    source_key: str = "source",
    target_key: str = "target",
    mask_key: Optional[str] = None,
    wandb_project: str = "lbm-fill-lighting",
    batch_size: int = 8,
    num_steps: List[int] = [1, 2, 4],
    learning_rate: float = 5e-5,
    learning_rate_scheduler: str = None,
    learning_rate_scheduler_kwargs: dict = {},
    optimizer: str = "AdamW",
    optimizer_kwargs: dict = {},
    timestep_sampling: str = "uniform",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    selected_timesteps: List[float] = None,
    prob: List[float] = None,
    conditioning_images_keys: Optional[List[str]] = None,
    conditioning_masks_keys: Optional[List[str]] = None,
    lighting_embedder_config: Optional[dict] = None,
    source_folder: str = "albedo",
    target_folder: str = "fill",
    rgb_folder: str = "rgb",
    depth_folder: str = "depth",
    lighting_params_folder: str = "lighting_params",
    lighting_scale_folder: str = "lighting_scale",
    image_size: int = 512,
    num_workers: int = 4,
    train_random_flip: bool = True,
    train_random_scale_min: float = 1.0,
    train_random_scale_max: float = 1.0,
    config_yaml: dict = None,
    save_ckpt_path: str = "./checkpoints",
    log_interval: int = 100,
    resume_from_checkpoint: bool = True,
    max_epochs: int = 100,
    bridge_noise_sigma: float = 0.005,
    save_interval: int = 1000,
    devices: Optional[int] = None,
    num_nodes: int = 1,
    path_config: str = None,
):
    if conditioning_images_keys is None:
        conditioning_images_keys = ["shading"]
    if conditioning_masks_keys is None:
        conditioning_masks_keys = ["depth", "lighting_scale"]

    model = build_relight_model(
        backbone_signature=backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
        lighting_conditioning=True,
        lighting_embedder_config=lighting_embedder_config,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    train_loader, validation_loader = get_dataloaders(
        train_data_root=train_data_root,
        validation_data_root=validation_data_root,
        batch_size=batch_size,
        source_folder=source_folder,
        target_folder=target_folder,
        rgb_folder=rgb_folder,
        depth_folder=depth_folder,
        lighting_params_folder=lighting_params_folder,
        lighting_scale_folder=lighting_scale_folder,
        image_size=image_size,
        num_workers=num_workers,
        train_random_flip=train_random_flip,
        train_random_scale_min=train_random_scale_min,
        train_random_scale_max=train_random_scale_max,
    )

    train_parameters = ["denoiser.*"]

    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=[
            "source",
            "target",
            "shading",
            "depth",
            "lighting_scale",
        ],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": num_steps,
        },
    )
    if (
        os.path.exists(save_ckpt_path)
        and resume_from_checkpoint
        and "last.ckpt" in os.listdir(save_ckpt_path)
    ):
        start_ckpt = f"{save_ckpt_path}/last.ckpt"
        print(f"Resuming from checkpoint: {start_ckpt}")
        last_model = torch.load(start_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(last_model["state_dict"], strict=False)
    else:
        start_ckpt = None

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    pipeline.save_hyperparameters(
        {
            f"embedder_{i}": embedder.config.to_dict()
            for i, embedder in enumerate(model.conditioner.conditioners)
        }
    )

    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "vae": model.vae.config.to_dict(),
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )

    training_signature = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-LBM-Fill-Lighting"
    )
    run_name = training_signature

    import platform

    if platform.system() == "Windows":
        strategy = DDPStrategy(find_unused_parameters=True, process_group_backend="gloo")
    else:
        strategy = "ddp_find_unused_parameters_true"
    trainer = Trainer(
        accelerator="gpu",
        devices=devices if devices is not None else max(torch.cuda.device_count(), 1),
        num_nodes=num_nodes,
        strategy=strategy,
        default_root_dir="logs",
        logger=loggers.WandbLogger(
            project=wandb_project, offline=True, name=run_name, save_dir=save_ckpt_path
        ),
        callbacks=[
            WandbSampleLogger(log_batch_freq=log_interval),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                every_n_epochs=100,
                save_last=True,
                save_top_k=-1,
                save_weights_only=False,
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=2,
        check_val_every_n_epoch=1,
        max_epochs=max_epochs,
    )

    trainer.fit(
        pipeline, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
