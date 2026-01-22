import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import re
from pathlib import Path
from typing import Optional

import fire
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from lbm.inference.relight import build_filllight_refine_model
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

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class FillLightingRefineFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=1024, random_flip=False):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.random_flip = random_flip

        self._target_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_rgb$")
        self._source_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_fill$")
        self._rgb_pattern = re.compile(r"^(?P<pos>\d{3})_999_rgb$")

        self.items = self._build_items()
        if not self.items:
            raise ValueError("No matching samples found across filllight/rgb folders.")

        self.target_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.source_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )

    def _build_items(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing dataset root: {self.root_dir}")

        items = []
        base_dirs = [path for path in self.root_dir.iterdir() if path.is_dir()]
        for base_dir in sorted(base_dirs):
            files = list(base_dir.iterdir())
            rgb_files, source_files, target_files = {}, {}, {}

            for path in files:
                if not path.is_file():
                    continue
                stem = path.stem
                suffix = path.suffix.lower()
                if suffix in VALID_SUFFIXES:
                    rgb_match = self._rgb_pattern.match(stem)
                    if rgb_match:
                        rgb_files[rgb_match.group("pos")] = path
                        continue
                    source_match = self._source_pattern.match(stem)
                    if source_match:
                        pos_id = source_match.group("pos")
                        light_id = source_match.group("light")
                        source_files.setdefault(pos_id, {})[light_id] = path
                        continue
                    target_match = self._target_pattern.match(stem)
                    if target_match:
                        pos_id = target_match.group("pos")
                        light_id = target_match.group("light")
                        target_files.setdefault(pos_id, {})[light_id] = path
                        continue

            for pos_id, light_map in target_files.items():
                rgb_path = rgb_files.get(pos_id)
                if rgb_path is None:
                    continue
                for light_id, target_path in light_map.items():
                    if int(light_id) >= 100:
                        continue
                    source_path = source_files.get(pos_id, {}).get(light_id)
                    if source_path is None:
                        continue
                    items.append(
                        {
                            "source": source_path,
                            "target": target_path,
                            "rgb": rgb_path,
                        }
                    )
        return items

    def _load_rgb(self, path, transform):
        image = Image.open(path).convert("RGB")
        return transform(image)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        target = self._load_rgb(item["target"], self.target_transform)
        source = self._load_rgb(item["source"], self.source_transform)
        rgb = self._load_rgb(item["rgb"], self.target_transform)

        if self.random_flip and torch.rand(1).item() < 0.5:
            target = torch.flip(target, dims=[2])
            source = torch.flip(source, dims=[2])
            rgb = torch.flip(rgb, dims=[2])

        sample = {
            "source": source * 2 - 1,
            "target": target * 2 - 1,
            "rgb": rgb * 2 - 1,
        }
        return sample


def get_dataloaders(
        train_data_root,
        validation_data_root,
        batch_size,
        image_size=1024,
        num_workers=4,
        train_random_flip=True,
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
        train_data_root="path/to/train",
        validation_data_root="path/to/validation",
        backbone_signature="stable-diffusion-v1-5/stable-diffusion-v1-5",
        unet_input_channels=6,
        unet_output_channels=3,
        block_out_channels=None,
        layers_per_block=2,
        source_key="source",
        target_key="target",
        mask_key=None,
        wandb_project="lbm-fill-lighting-refine",
        batch_size=1,
        num_steps=(1, 2, 4),
        learning_rate=5e-5,
        learning_rate_scheduler=None,
        learning_rate_scheduler_kwargs={},
        optimizer="AdamW",
        optimizer_kwargs={},
        timestep_sampling="uniform",
        logit_mean=0.0,
        logit_std=1.0,
        pixel_loss_type="lpips",
        latent_loss_type="l2",
        latent_loss_weight=1.0,
        pixel_loss_weight=1.0,
        selected_timesteps=None,
        prob=None,
        conditioning_images_keys=None,
        conditioning_masks_keys=None,
        image_size=1024,
        num_workers=4,
        train_random_flip=True,
        config_yaml=None,
        save_ckpt_path="./checkpoints",
        log_interval=100,
        resume_from_checkpoint=True,
        max_epochs=100,
        bridge_noise_sigma=0.005,
        save_interval: int = 1000,
        resume_ckpt_path: Optional[str] = None,
        devices=None,
        num_nodes=1,
):
    task_dir = resolve_task_dir(save_ckpt_path, "fill_lighting_refine")
    if conditioning_images_keys is None:
        conditioning_images_keys = ["rgb"]
    if conditioning_masks_keys is None:
        conditioning_masks_keys = []

    if block_out_channels is None:
        block_out_channels = [64, 128, 256, 256]

    model = build_filllight_refine_model(
        backbone_signature=backbone_signature,
        unet_input_channels=unet_input_channels,
        unet_output_channels=unet_output_channels,
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
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
        image_size=image_size,
        num_workers=num_workers,
        train_random_flip=train_random_flip,
    )

    train_parameters = ["denoiser.*"]

    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["source", "target", "rgb"],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={
            "input_shape": (3, image_size, image_size),
            "num_steps": num_steps,
        },
    )
    pipeline = setup_pipeline(model, training_config, config_yaml)
    run_name = create_run_name("Fill-Lighting-Refine")
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


def main_from_config(path_config="/mnt/data1/ssy/render_people/LBM/examples/training/config/fill_lighting_refine.yaml"):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    print(f"Running main with config: {yaml.dump(config, default_flow_style=False)}")
    main(**config, config_yaml=config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
