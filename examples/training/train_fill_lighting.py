import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import random
import re
from pathlib import Path
from typing import Optional

import cv2
import fire
import h5py
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from lbm.inference.relight import build_filllight_model
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
LIGHTING_PARAM_SUFFIXES = ".h5"


class FillLightingFolderDataset(Dataset):
    def __init__(self, root_dir, image_size=512, random_flip=False):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.random_flip = random_flip

        self._target_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_rgb$")
        # self._source_pattern = re.compile(r"^(?P<pos>\d{3})_alb$")
        self._alb_pattern = re.compile(r"^(?P<pos>\d{3})_alb$")
        self._elb_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_elb$")
        self._rgb_pattern = re.compile(r"^(?P<pos>\d{3})_999_rgb$")
        self._depth_pattern = re.compile(r"^(?P<pos>\d{3})_dpt$")
        self._lighting_pattern = re.compile(r"^(?P<pos>\d{3})_(?P<light>\d{3})_lgt$")

        self.items = self._build_items()
        if not self.items:
            raise ValueError("No matching samples found across albedo/fill/rgb/depth/lighting params folders.")

        self.rgb_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST_EXACT),
             transforms.ToTensor()])

    def _index_files(self, directory, suffixes):
        files = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes]
        return {path.stem: path for path in files}

    def _build_items(self):
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Missing dataset root: {self.root_dir}")

        items = []
        base_dirs = [path for path in self.root_dir.iterdir() if path.is_dir()]
        for base_dir in sorted(base_dirs):
            files = list(base_dir.iterdir())
            # source_files, rgb_files, depth_files, lighting_params_files, target_files = {}, {}, {}, {}, {}
            alb_files, elb_files, rgb_files, depth_files, lighting_params_files, target_files = {}, {}, {}, {}, {}, {}
            for path in files:
                if not path.is_file():
                    continue
                stem = path.stem
                suffix = path.suffix.lower()
                if suffix in VALID_SUFFIXES:
                    # source_match = self._source_pattern.match(stem)
                    # if source_match:
                    #     source_files[source_match.group("pos")] = path
                    alb_match = self._alb_pattern.match(stem)
                    if alb_match:
                        alb_files[alb_match.group("pos")] = path
                        continue
                    elb_match = self._elb_pattern.match(stem)
                    if elb_match:
                        pos_id = elb_match.group("pos")
                        light_id = elb_match.group("light")
                        elb_files.setdefault(pos_id, {})[light_id] = path
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
                if suffix == LIGHTING_PARAM_SUFFIXES:
                    lighting_match = self._lighting_pattern.match(stem)
                    if lighting_match:
                        pos_id = lighting_match.group("pos")
                        light_id = lighting_match.group("light")
                        lighting_params_files.setdefault(pos_id, {})[light_id] = path

            for pos_id, light_map in target_files.items():
                # source_path = source_files.get(pos_id)
                alb_path = alb_files.get(pos_id)
                rgb_path = rgb_files.get(pos_id)
                depth_path = depth_files.get(pos_id)
                # if source_path is None or rgb_path is None or depth_path is None:
                if alb_path is None or rgb_path is None or depth_path is None:
                    continue
                elb_path = elb_files.get(pos_id, {}).get(light_id)
                for light_id, target_path in light_map.items():
                    if int(light_id) >= 100:
                        continue
                    lighting_params_path = lighting_params_files.get(pos_id, {}).get(light_id)
                    if lighting_params_path is None:
                        continue
                    items.append(
                        {
                            # "source": source_path,
                            "alb_source": alb_path,
                            "elb_source": elb_path,
                            "target": target_path,
                            "rgb": rgb_path,
                            "depth": depth_path,
                            "lighting_params": lighting_params_path,
                        }
                    )
        return items

    def _load_rgb(self, path):
        image = Image.open(path).convert("RGB")
        return self.rgb_transform(image)

    def _load_depth(self, path):
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Failed to read depth image: {path}")
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype("float32")
        # depth = depth / 100.0  # cm to m
        depth[depth >= 100.0] = 0.0
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
        return depth_tensor * 2 - 1

    def _load_lighting_params(self, path):
        with h5py.File(path, "r") as file:
            if "lighting_params" in file:
                data = file["lighting_params"][()]
            else:
                first_key = next(iter(file.keys()))
                data = file[first_key][()]
        params = torch.tensor(data, dtype=torch.float32)

        params = params.flatten()
        if params.numel() != 7:
            raise ValueError(f"Expected 7 lighting params, got {params.numel()} from {path}")
        return params

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        # albedo = self._load_rgb(item["source"])
        elb_path = item["elb_source"]
        if elb_path is not None and random.random() < 0.5:
            source_path = elb_path
        else:
            source_path = item["alb_source"]
        albedo = self._load_rgb(source_path)
        target = self._load_rgb(item["target"])
        rgb = self._load_rgb(item["rgb"])
        depth = self._load_depth(item["depth"])
        lighting_params = self._load_lighting_params(item["lighting_params"])
        lighting_scale = torch.ones_like(depth)

        shading = (rgb / albedo.clamp(1e-3, 1.0)).clamp(0.0, 1.0)

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


def get_dataloaders(train_data_root, validation_data_root, batch_size, image_size=512, num_workers=4,
                    train_random_flip=True):
    train_dataset = FillLightingFolderDataset(root_dir=train_data_root, image_size=image_size,
                                              random_flip=train_random_flip)
    validation_dataset = FillLightingFolderDataset(root_dir=validation_data_root, image_size=image_size)

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
        backbone_signature="runwayml/stable-diffusion-v1-5",
        vae_num_channels=4,
        unet_input_channels=10,
        source_key="source",
        target_key="target",
        mask_key=None,
        wandb_project="lbm-fill-lighting",
        batch_size=8,
        num_steps=[1, 2, 4],
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
        pixel_loss_weight=0.0,
        selected_timesteps=None,
        prob=None,
        conditioning_images_keys=None,
        conditioning_masks_keys=None,
        lighting_embedder_config=None,
        image_size=512,
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
    task_dir = resolve_task_dir(save_ckpt_path, "fill_lighting")
    if conditioning_images_keys is None:
        conditioning_images_keys = ["shading"]
    if conditioning_masks_keys is None:
        conditioning_masks_keys = ["depth", "lighting_scale"]

    model = build_filllight_model(
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
        image_size=image_size,
        num_workers=num_workers,
        train_random_flip=train_random_flip,
    )

    train_parameters = ["denoiser.*"]

    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["source", "target", "shading", "depth", "lighting_scale"],
        trainable_params=train_parameters,
        optimizer_name=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        log_samples_model_kwargs={"input_shape": None, "num_steps": num_steps},
    )
    pipeline = setup_pipeline(model, training_config, config_yaml)
    run_name = create_run_name("Fill-Lighting")
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


def main_from_config(path_config):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    print(f"Running main with config: {yaml.dump(config, default_flow_style=False)}")
    main(**config, config_yaml=config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
