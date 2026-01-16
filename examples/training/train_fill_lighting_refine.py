import datetime
import os
import platform
import re
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

from lbm.inference.refine import build_filllight_refine_model
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class FillLightingRefineFolderDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: int = 1024,
            random_flip: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.random_flip = random_flip

        self._target_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_rgb$")
        self._rgb_pattern = re.compile(r"^(?P<pos>\d{3})_999_rgb$")

        self.items = self._build_items()
        if not self.items:
            raise ValueError("No matching samples found across filllight/rgb folders.")

        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size),
                    interpolation=InterpolationMode.NEAREST_EXACT,
                ),
                transforms.ToTensor(),
            ]
        )

    def _index_files(self, directory: Path, pattern: re.Pattern) -> dict:
        files = [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
        ]
        results = {}
        for path in files:
            match = pattern.match(path.stem)
            if not match:
                continue
            results[match.group("pos")] = path
        return results

    def _build_items(self) -> List[dict]:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing dataset root: {self.root_dir}")

        items: List[dict] = []
        base_dirs = [path for path in self.root_dir.iterdir() if path.is_dir()]
        for base_dir in sorted(base_dirs):
            rgb_files = self._index_files(base_dir, self._rgb_pattern)
            files = [
                path
                for path in base_dir.iterdir()
                if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
            ]

            for path in files:
                match = self._target_pattern.match(path.stem)
                if not match:
                    continue
                pos_id = match.group("pos")
                light_id = match.group("light")
                if int(light_id) >= 100:
                    continue
                rgb_path = rgb_files.get(pos_id)
                if rgb_path is None:
                    continue
                items.append({"target": path, "rgb": rgb_path})
        return items

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.transforms(image)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        target = self._load_rgb(item["target"])
        rgb = self._load_rgb(item["rgb"])

        filllight_input = target
        rgb_input = rgb

        if self.random_flip and torch.rand(1).item() < 0.5:
            target = torch.flip(target, dims=[2])
            filllight_input = torch.flip(filllight_input, dims=[2])
            rgb_input = torch.flip(rgb_input, dims=[2])

        source = torch.cat([filllight_input, rgb_input], dim=0)

        return {
            "source": source * 2 - 1,
            "target": target * 2 - 1,
            "filllight_input": filllight_input * 2 - 1,
            "rgb": rgb_input * 2 - 1,
        }


def get_dataloaders(
        train_data_root: str,
        validation_data_root: str,
        batch_size: int,
        image_size: int = 1024,
        num_workers: int = 4,
        train_random_flip: bool = True,
):
    train_dataset = FillLightingRefineFolderDataset(
        root_dir=train_data_root,
        image_size=image_size,
        random_flip=train_random_flip,
    )
    validation_dataset = FillLightingRefineFolderDataset(
        root_dir=validation_data_root,
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
        backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        unet_input_channels: int = 6,
        unet_output_channels: int = 3,
        source_key: str = "source",
        target_key: str = "target",
        mask_key: Optional[str] = None,
        wandb_project: str = "lbm-fill-lighting-refine",
        batch_size: int = 1,
        num_steps: List[int] = [1, 2, 4],
        learning_rate: float = 5e-5,
        learning_rate_scheduler: Optional[str] = None,
        learning_rate_scheduler_kwargs: dict = {},
        optimizer: str = "AdamW",
        optimizer_kwargs: dict = {},
        timestep_sampling: str = "log_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        pixel_loss_type: str = "l2",
        latent_loss_type: str = "l2",
        latent_loss_weight: float = 1.0,
        pixel_loss_weight: float = 0.0,
        selected_timesteps: Optional[List[float]] = None,
        prob: Optional[List[float]] = None,
        image_size: int = 1024,
        num_workers: int = 4,
        train_random_flip: bool = True,
        config_yaml: Optional[dict] = None,
        save_ckpt_path: str = "./checkpoints",
        log_interval: int = 100,
        resume_from_checkpoint: bool = True,
        max_epochs: int = 50,
        bridge_noise_sigma: float = 0.005,
        block_out_channels: Optional[List[int]] = None,
        layers_per_block: int = 2,
        devices: Optional[int] = None,
        num_nodes: int = 1,
):
    model = build_filllight_refine_model(
        backbone_signature=backbone_signature,
        unet_input_channels=unet_input_channels,
        unet_output_channels=unet_output_channels,
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
        bridge_noise_sigma=bridge_noise_sigma,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
    )

    train_loader, validation_loader = get_dataloaders(
        train_data_root=train_data_root,
        validation_data_root=validation_data_root,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
        train_random_flip=train_random_flip,
    )

    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["source", "target", "filllight_input", "rgb"],
        trainable_params=["denoiser.*"],
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={"input_shape": (3, image_size, image_size), "num_steps": num_steps},
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

    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )

    run_name = (
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + "-LBM-Fill-Lighting-Refine"
    )

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

    trainer.fit(pipeline, train_dataloaders=train_loader, val_dataloaders=validation_loader)


def main_from_config(path_config: str):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    print(f"Running main with config: {yaml.dump(config, default_flow_style=False)}")
    main(**config, config_yaml=config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
