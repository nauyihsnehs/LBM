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
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from lbm.inference.relight import build_relight_model
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger


class RelightFolderDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        source_folder: str = "source",
        target_folder: str = "target",
        shading_folder: str = "shading",
        normal_folder: str = "normal",
        image_size: int = 512,
    ):
        self.root_dir = Path(root_dir)
        self.source_dir = self.root_dir / source_folder
        self.target_dir = self.root_dir / target_folder
        self.shading_dir = self.root_dir / shading_folder
        self.normal_dir = self.root_dir / normal_folder

        self._validate_directories()

        self.source_files = self._index_files(self.source_dir)
        self.target_files = self._index_files(self.target_dir)
        self.shading_files = self._index_files(self.shading_dir)
        self.normal_files = self._index_files(self.normal_dir)

        self.file_stems = sorted(
            set(self.source_files)
            & set(self.target_files)
            & set(self.shading_files)
            & set(self.normal_files)
        )
        if not self.file_stems:
            raise ValueError(
                "No matching files found across source/target/shading/normal folders."
            )

        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.NEAREST_EXACT,
                ),
                transforms.ToTensor(),
            ]
        )

    def _validate_directories(self) -> None:
        for directory in [
            self.source_dir,
            self.target_dir,
            self.shading_dir,
            self.normal_dir,
        ]:
            if not directory.exists():
                raise FileNotFoundError(f"Missing directory: {directory}")

    def _index_files(self, directory: Path) -> dict:
        valid_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        files = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in valid_suffixes
        ]
        return {path.stem: path for path in files}

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        tensor = self.transforms(image)
        return tensor * 2 - 1

    def __len__(self) -> int:
        return len(self.file_stems)

    def __getitem__(self, index: int) -> dict:
        stem = self.file_stems[index]
        return {
            "source": self._load_image(self.source_files[stem]),
            "target": self._load_image(self.target_files[stem]),
            "shading": self._load_image(self.shading_files[stem]),
            "normal": self._load_image(self.normal_files[stem]),
        }


def get_dataloaders(
    train_data_root: str,
    validation_data_root: str,
    batch_size: int,
    source_folder: str = "source",
    target_folder: str = "target",
    shading_folder: str = "shading",
    normal_folder: str = "normal",
    image_size: int = 512,
    num_workers: int = 4,
):
    train_dataset = RelightFolderDataset(
        root_dir=train_data_root,
        source_folder=source_folder,
        target_folder=target_folder,
        shading_folder=shading_folder,
        normal_folder=normal_folder,
        image_size=image_size,
    )
    validation_dataset = RelightFolderDataset(
        root_dir=validation_data_root,
        source_folder=source_folder,
        target_folder=target_folder,
        shading_folder=shading_folder,
        normal_folder=normal_folder,
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
    unet_input_channels: int = 12,
    source_key: str = "source",
    target_key: str = "target",
    mask_key: Optional[str] = None,
    wandb_project: str = "lbm-relight",
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
    source_folder: str = "source",
    target_folder: str = "target",
    shading_folder: str = "shading",
    normal_folder: str = "normal",
    image_size: int = 512,
    num_workers: int = 4,
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
        source_folder=source_folder,
        target_folder=target_folder,
        shading_folder=shading_folder,
        normal_folder=normal_folder,
        image_size=image_size,
        num_workers=num_workers,
    )

    train_parameters = ["denoiser.*"]

    # Training Config
    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["source", "target", "shading", "normal"],
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
        # import pathlib
        # p = pathlib.Path.cwd()
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
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-LBM-Relight"
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
        # strategy='ddp_find_unused_parameters_true',
        default_root_dir="logs",
        logger=loggers.WandbLogger(
            project=wandb_project, offline=True, name=run_name, save_dir=save_ckpt_path
        ),
        callbacks=[
            WandbSampleLogger(log_batch_freq=log_interval),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=save_ckpt_path,
                # every_n_train_steps=save_interval,
                every_n_epochs=1,
                save_last=False,
                save_top_k=-1,
                save_weights_only=False,
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=2,
        val_check_interval=1000,
        max_epochs=max_epochs,
    )

    trainer.fit(pipeline, train_dataloaders=train_loader, val_dataloaders=validation_loader)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
