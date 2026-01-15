import datetime
import logging
import os
from pathlib import Path
from typing import List, Optional

import fire
import numpy as np
import torch
import yaml
from PIL import Image
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from lbm.inference.fill_lighting import build_fill_lighting_model
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger


class FillLightingFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        albedo_folder: str = "albedo",
        rgb_folder: str = "rgb",
        target_folder: str = "target",
        depth_folder: str = "depth",
        light_params_folder: str = "light_params",
        light_scale_folder: Optional[str] = "light_scale",
        image_size: int = 512,
        random_flip: bool = False,
        random_scale_min: float = 1.0,
        random_scale_max: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.albedo_folder = albedo_folder
        self.rgb_folder = rgb_folder
        self.target_folder = target_folder
        self.depth_folder = depth_folder
        self.light_params_folder = light_params_folder
        self.light_scale_folder = light_scale_folder
        self.random_flip = random_flip
        self.random_scale_min = random_scale_min
        self.random_scale_max = random_scale_max

        self.items = self._build_items()
        if not self.items:
            raise ValueError(
                "No matching samples found across albedo/rgb/target/depth/light params folders."
            )

        self.rgb_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.NEAREST_EXACT,
                ),
                transforms.ToTensor(),
            ]
        )
        self.depth_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.NEAREST_EXACT,
                ),
                transforms.ToTensor(),
            ]
        )

    def _validate_directories(self, directories: List[Path]) -> None:
        for directory in directories:
            if directory is None:
                continue
            if not directory.exists():
                raise FileNotFoundError(f"Missing directory: {directory}")

    def _index_files(self, directory: Path, suffixes: set[str]) -> dict:
        files = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in suffixes
        ]
        return {path.stem: path for path in files}

    def _index_light_params(self, directory: Path) -> dict:
        suffixes = {".npy", ".npz", ".pt", ".pth", ".txt"}
        return self._index_files(directory, suffixes)

    def _build_items(self) -> List[dict]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing dataset root: {self.root_dir}")

        items: List[dict] = []
        person_dirs = [path for path in self.root_dir.iterdir() if path.is_dir()]
        valid_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

        for person_dir in sorted(person_dirs):
            albedo_dir = person_dir / self.albedo_folder
            rgb_dir = person_dir / self.rgb_folder
            target_dir = person_dir / self.target_folder
            depth_dir = person_dir / self.depth_folder
            light_params_dir = person_dir / self.light_params_folder
            light_scale_dir = (
                person_dir / self.light_scale_folder if self.light_scale_folder else None
            )

            self._validate_directories(
                [
                    albedo_dir,
                    rgb_dir,
                    target_dir,
                    depth_dir,
                    light_params_dir,
                    light_scale_dir,
                ]
            )

            albedo_files = self._index_files(albedo_dir, valid_suffixes)
            rgb_files = self._index_files(rgb_dir, valid_suffixes)
            depth_files = self._index_files(depth_dir, valid_suffixes)
            light_params_files = self._index_light_params(light_params_dir)
            light_scale_files = (
                self._index_files(light_scale_dir, valid_suffixes)
                if light_scale_dir is not None
                else {}
            )

            target_files = [
                path
                for path in target_dir.iterdir()
                if path.is_file() and path.suffix.lower() in valid_suffixes
            ]

            for target_path in target_files:
                stem = target_path.stem
                albedo_path = albedo_files.get(stem)
                rgb_path = rgb_files.get(stem)
                depth_path = depth_files.get(stem)
                light_params_path = light_params_files.get(stem)
                light_scale_path = light_scale_files.get(stem)

                if (
                    albedo_path is None
                    or rgb_path is None
                    or depth_path is None
                    or light_params_path is None
                ):
                    continue

                items.append(
                    {
                        "source": albedo_path,
                        "target": target_path,
                        "rgb": rgb_path,
                        "depth": depth_path,
                        "light_params": light_params_path,
                        "light_scale": light_scale_path,
                    }
                )
        return items

    def _load_image_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.rgb_transforms(image)

    def _load_image_depth(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        return self.depth_transforms(image)

    def _load_light_params(self, path: Path) -> torch.Tensor:
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

    def _compute_shading(self, rgb: torch.Tensor, albedo: torch.Tensor) -> torch.Tensor:
        return torch.clamp(rgb / torch.clamp(albedo, 1e-3, 1.0), 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        source = self._load_image_rgb(item["source"])
        rgb = self._load_image_rgb(item["rgb"])
        target = self._load_image_rgb(item["target"])
        depth = self._load_image_depth(item["depth"])
        light_params = self._load_light_params(item["light_params"])

        shading = self._compute_shading(rgb, source)

        if item["light_scale"] is None:
            light_scale = torch.ones_like(depth)
        else:
            light_scale = self._load_image_depth(item["light_scale"])

        sample = {
            "source": source,
            "target": target,
            "shading": shading,
            "depth": depth,
            "light_scale": light_scale,
            "light_params": light_params,
        }

        if self.random_scale_min != 1.0 or self.random_scale_max != 1.0:
            scale = torch.empty(1).uniform_(self.random_scale_min, self.random_scale_max)
            for key in ["source", "target", "shading"]:
                sample[key] = torch.clamp(sample[key] * scale, 0.0, 1.0)

        if self.random_flip and torch.rand(1).item() < 0.5:
            for key in ["source", "target", "shading", "depth", "light_scale"]:
                sample[key] = torch.flip(sample[key], dims=[2])

        sample["source"] = sample["source"] * 2 - 1
        sample["target"] = sample["target"] * 2 - 1
        sample["shading"] = sample["shading"] * 2 - 1
        return sample


def get_dataloaders(
    train_data_root: str,
    validation_data_root: str,
    batch_size: int,
    albedo_folder: str = "albedo",
    rgb_folder: str = "rgb",
    target_folder: str = "target",
    depth_folder: str = "depth",
    light_params_folder: str = "light_params",
    light_scale_folder: Optional[str] = "light_scale",
    image_size: int = 512,
    num_workers: int = 4,
    train_random_flip: bool = True,
    train_random_scale_min: float = 1.0,
    train_random_scale_max: float = 1.0,
):
    train_dataset = FillLightingFolderDataset(
        root_dir=train_data_root,
        albedo_folder=albedo_folder,
        rgb_folder=rgb_folder,
        target_folder=target_folder,
        depth_folder=depth_folder,
        light_params_folder=light_params_folder,
        light_scale_folder=light_scale_folder,
        image_size=image_size,
        random_flip=train_random_flip,
        random_scale_min=train_random_scale_min,
        random_scale_max=train_random_scale_max,
    )
    validation_dataset = FillLightingFolderDataset(
        root_dir=validation_data_root,
        albedo_folder=albedo_folder,
        rgb_folder=rgb_folder,
        target_folder=target_folder,
        depth_folder=depth_folder,
        light_params_folder=light_params_folder,
        light_scale_folder=light_scale_folder,
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
    albedo_folder: str = "albedo",
    rgb_folder: str = "rgb",
    target_folder: str = "target",
    depth_folder: str = "depth",
    light_params_folder: str = "light_params",
    light_scale_folder: Optional[str] = "light_scale",
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
    light_params_input_key: str = "light_params",
    light_params_num_frequencies: int = 6,
    light_params_include_input: bool = True,
    light_params_hidden_dims: Optional[List[int]] = None,
    light_params_num_tokens: int = 77,
    light_params_embedding_dim: int = 768,
    light_params_ucg_rate: float = 0.0,
):
    if conditioning_images_keys is None:
        conditioning_images_keys = ["shading"]

    if conditioning_masks_keys is None:
        conditioning_masks_keys = ["depth", "light_scale"]

    model = build_fill_lighting_model(
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
        bridge_noise_sigma=bridge_noise_sigma,
        light_params_input_key=light_params_input_key,
        light_params_num_frequencies=light_params_num_frequencies,
        light_params_include_input=light_params_include_input,
        light_params_hidden_dims=light_params_hidden_dims,
        light_params_num_tokens=light_params_num_tokens,
        light_params_embedding_dim=light_params_embedding_dim,
        light_params_ucg_rate=light_params_ucg_rate,
    )

    train_loader, validation_loader = get_dataloaders(
        train_data_root=train_data_root,
        validation_data_root=validation_data_root,
        batch_size=batch_size,
        albedo_folder=albedo_folder,
        rgb_folder=rgb_folder,
        target_folder=target_folder,
        depth_folder=depth_folder,
        light_params_folder=light_params_folder,
        light_scale_folder=light_scale_folder,
        image_size=image_size,
        num_workers=num_workers,
        train_random_flip=train_random_flip,
        train_random_scale_min=train_random_scale_min,
        train_random_scale_max=train_random_scale_max,
    )

    train_parameters = ["denoiser.*", "conditioner.conditioners.1.*"]

    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["source", "target", "shading"],
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
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "-LBM-Fill-Lighting"
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
                every_n_epochs=1,
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
