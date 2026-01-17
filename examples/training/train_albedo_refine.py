import datetime
import logging
import os
from pathlib import Path
from typing import List, Optional

import fire
import torch
import yaml
from PIL import Image
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from lbm.inference.relight import build_relight_model
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger


class AlbedoRefineFolderDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            robedo_suffix: str = "_rlb",
            robedo_extension: str = ".png",
            image_size: int = 512,
            random_flip: bool = False,
            random_scale_min: float = 1.0,
            random_scale_max: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.robedo_suffix = robedo_suffix
        self.robedo_extension = robedo_extension
        self.random_flip = random_flip
        self.random_scale_min = random_scale_min
        self.random_scale_max = random_scale_max

        self.items = self._build_items()
        if not self.items:
            raise ValueError("No matching samples found for albedo training.")

        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.NEAREST_EXACT,
                ),
                transforms.ToTensor(),
            ]
        )

    def _index_albedo_files(self, directory: Path) -> dict:
        valid_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        albedo_files = [
            path
            for path in directory.iterdir()
            if path.is_file()
               and path.suffix.lower() in valid_suffixes
               and path.stem.endswith("_alb")
        ]
        camera_to_path = {}
        for path in albedo_files:
            camera_id = path.stem.split("_")[0]
            camera_to_path[camera_id] = path
        return camera_to_path

    def _build_items(self) -> List[dict]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing dataset root: {self.root_dir}")

        valid_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        items: List[dict] = []
        person_dirs = [path for path in self.root_dir.iterdir() if path.is_dir()]
        for person_dir in sorted(person_dirs):
            albedo_by_camera = self._index_albedo_files(person_dir)
            rgb_files = [
                path
                for path in person_dir.iterdir()
                if path.is_file()
                   and path.suffix.lower() in valid_suffixes
                   and path.stem.endswith("_rgb")
            ]

            for rgb_path in rgb_files:
                stem_parts = rgb_path.stem.split("_")
                if len(stem_parts) < 2:
                    continue
                camera_id = stem_parts[0]
                albedo_path = albedo_by_camera.get(camera_id)
                if albedo_path is None:
                    continue
                robedo_path = self._get_robedo_path(rgb_path)
                if not robedo_path.exists():
                    continue
                items.append(
                    {"source": rgb_path, "target": albedo_path, "robedo": robedo_path}
                )
        return items

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.transforms(image)

    def _get_robedo_path(self, rgb_path: Path) -> Path:
        if not rgb_path.stem.endswith("_rgb"):
            raise ValueError(f"Expected rgb filename to end with _rgb: {rgb_path.name}")
        robedo_stem = rgb_path.stem[: -len("_rgb")] + self.robedo_suffix
        return rgb_path.with_name(f"{robedo_stem}{self.robedo_extension}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        source = self._load_image(item["source"])
        target = self._load_image(item["target"])
        robedo = self._load_image(item["robedo"])
        sample = {
            "source": source,
            "target": target,
            "robedo": robedo,
        }
        if self.random_scale_min != 1.0 or self.random_scale_max != 1.0:
            scale = torch.empty(1).uniform_(self.random_scale_min, self.random_scale_max)
            sample = {
                key: torch.clamp(value * scale, 0.0, 1.0)
                for key, value in sample.items()
            }
        if self.random_flip and torch.rand(1).item() < 0.5:
            sample = {key: torch.flip(value, dims=[2]) for key, value in sample.items()}
        return {key: value * 2 - 1 for key, value in sample.items()}


def get_dataloaders(
        train_data_root: str,
        validation_data_root: str,
        batch_size: int,
        robedo_suffix: str = "_rlb",
        robedo_extension: str = ".png",
        image_size: int = 512,
        num_workers: int = 4,
        train_random_flip: bool = True,
        train_random_scale_min: float = 1.0,
        train_random_scale_max: float = 1.0,
):
    train_dataset = AlbedoRefineFolderDataset(
        root_dir=train_data_root,
        robedo_suffix=robedo_suffix,
        robedo_extension=robedo_extension,
        image_size=image_size,
        random_flip=train_random_flip,
        random_scale_min=train_random_scale_min,
        random_scale_max=train_random_scale_max,
    )
    validation_dataset = AlbedoRefineFolderDataset(
        root_dir=validation_data_root,
        robedo_suffix=robedo_suffix,
        robedo_extension=robedo_extension,
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
        unet_input_channels: int = 8,
        source_key: str = "source",
        target_key: str = "target",
        mask_key: Optional[str] = None,
        wandb_project: str = "lbm-albedo-refine",
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
        conditioning_masks_keys: Optional[List[str]] = [],
        robedo_suffix: str = "_rlb",
        robedo_extension: str = ".png",
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
        # save_interval: int = 1000,
        devices: Optional[int] = None,
        num_nodes: int = 1,
        # path_config: str = None,
):
    os.mkdirs(save_ckpt_path, exist_ok=True)
    if conditioning_images_keys is None:
        conditioning_images_keys = ["robedo"]

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
        bridge_noise_sigma=bridge_noise_sigma,
    )

    train_loader, validation_loader = get_dataloaders(
        train_data_root=train_data_root,
        validation_data_root=validation_data_root,
        batch_size=batch_size,
        robedo_suffix=robedo_suffix,
        robedo_extension=robedo_extension,
        image_size=image_size,
        num_workers=num_workers,
        train_random_flip=train_random_flip,
        train_random_scale_min=train_random_scale_min,
        train_random_scale_max=train_random_scale_max,
    )

    train_parameters = ["denoiser.*"]

    # Training Config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["source", "target", "robedo"],
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
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-LBM-Albedo-Refine"
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
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                every_n_epochs=1,
                save_last=False,
                save_top_k=-1,
                save_weights_only=False,
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=2,
        check_val_every_n_epoch=1,
        max_epochs=max_epochs,
        enable_progress_bar=True,
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
    main(**config, config_yaml=config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
