import logging
from pathlib import Path
from typing import List, Optional

import fire
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from lbm.inference.relight import build_relight_model
from lbm.trainer import TrainingConfig

from train_base import (
    build_trainer,
    build_concat_keys,
    create_run_name,
    fit_trainer,
    resolve_resume_checkpoint,
    resolve_task_dir,
    setup_pipeline,
)


class RelightFolderDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            source_folder: str = "source",
            target_folder: str = "target",
            shading_folder: str = "shading",
            normal_folder: str = "normal",
            image_size: int = 512,
            random_flip: bool = False,
            random_scale_min: float = 1.0,
            random_scale_max: float = 1.0,
    ):
        self.root_dir = Path(root_dir)
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.shading_folder = shading_folder
        self.normal_folder = normal_folder
        self.random_flip = random_flip
        self.random_scale_min = random_scale_min
        self.random_scale_max = random_scale_max

        self.items = self._build_items()
        if not self.items:
            raise ValueError(
                "No matching samples found across source/target/shading/normal folders."
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

    def _validate_directories(self, directories: List[Path]) -> None:
        for directory in directories:
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

    def _build_items(self) -> List[dict]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing dataset root: {self.root_dir}")

        items: List[dict] = []
        person_dirs = [path for path in self.root_dir.iterdir() if path.is_dir()]
        for person_dir in sorted(person_dirs):
            source_dir = person_dir / self.source_folder
            target_dir = person_dir / self.target_folder
            shading_dir = person_dir / self.shading_folder
            normal_dir = person_dir / self.normal_folder

            self._validate_directories(
                [source_dir, target_dir, shading_dir, normal_dir]
            )

            source_files = self._index_files(source_dir)
            normal_files = self._index_files(normal_dir)

            target_files = [
                path
                for path in target_dir.iterdir()
                if path.is_file()
                   and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
            ]

            for target_path in target_files:
                stem = target_path.stem
                frame_id = stem.split("_")[0]
                shading_path = shading_dir / target_path.name
                source_path = source_files.get(frame_id)
                normal_path = normal_files.get(frame_id)
                if (
                        shading_path.exists()
                        and source_path is not None
                        and normal_path is not None
                ):
                    items.append(
                        {
                            "source": source_path,
                            "target": target_path,
                            "shading": shading_path,
                            "normal": normal_path,
                        }
                    )
        return items

    def _load_image(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.transforms(image)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        sample = {
            "source": self._load_image(item["source"]),
            "target": self._load_image(item["target"]),
            "shading": self._load_image(item["shading"]),
            "normal": self._load_image(item["normal"]),
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
        source_folder: str = "source",
        target_folder: str = "target",
        shading_folder: str = "shading",
        normal_folder: str = "normal",
        image_size: int = 512,
        num_workers: int = 4,
        train_random_flip: bool = True,
        train_random_scale_min: float = 1.0,
        train_random_scale_max: float = 1.0,
):
    train_dataset = RelightFolderDataset(
        root_dir=train_data_root,
        source_folder=source_folder,
        target_folder=target_folder,
        shading_folder=shading_folder,
        normal_folder=normal_folder,
        image_size=image_size,
        random_flip=train_random_flip,
        random_scale_min=train_random_scale_min,
        random_scale_max=train_random_scale_max,
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
        resume_ckpt_path: Optional[str] = None,
        devices: Optional[int] = None,
        num_nodes: int = 1,
        path_config: str = None,
):
    task_dir = resolve_task_dir(save_ckpt_path, "relight")
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
        log_keys=["source", "target", "shading", "normal"],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": None,
            "num_steps": num_steps,
        },
    )
    pipeline = setup_pipeline(model, training_config, config_yaml)
    run_name = create_run_name("Relight")
    trainer = build_trainer(
        wandb_project=wandb_project,
        run_name=run_name,
        save_dir=task_dir,
        log_interval=log_interval,
        save_interval=save_interval,
        max_epochs=max_epochs,
        concat_keys=build_concat_keys(training_config),
        devices=devices,
        num_nodes=num_nodes,
    )
    ckpt_path = resolve_resume_checkpoint(
        task_dir, resume_from_checkpoint, resume_ckpt_path
    )
    fit_trainer(trainer, pipeline, train_loader, validation_loader, ckpt_path)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
