import datetime
import logging
import os
import random
from typing import List, Optional

import braceexpand
import fire
import torch
import yaml
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision.transforms import InterpolationMode

from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    LightParamsMapper,
    LightParamsMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from lbm.inference.relight import build_relight_params_model
from lbm.models.embedders import LightParamsEmbedderConfig
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger


def get_filter_mappers(
    source_image_key: str = "source.png",
    target_image_key: str = "target.png",
    depth_key: str = "depth.png",
    light_params_key: str = "light_params.json",
):
    filters_mappers = [
        KeyFilter(
            KeyFilterConfig(
                keys=[source_image_key, target_image_key, depth_key, light_params_key]
            )
        ),
        MapperWrapper(
            [
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            source_image_key: "source",
                            target_image_key: "target",
                            depth_key: "depth",
                            light_params_key: "light_params",
                        }
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="source",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (512, 512),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="target",
                        transforms=["ToTensor", "Resize"],
                        transforms_kwargs=[
                            {},
                            {
                                "size": (512, 512),
                                "interpolation": InterpolationMode.NEAREST_EXACT,
                            },
                        ],
                    )
                ),
                TorchvisionMapper(
                    TorchvisionMapperConfig(
                        key="depth",
                        transforms=["Grayscale", "ToTensor", "Resize"],
                        transforms_kwargs=[
                            {"num_output_channels": 1},
                            {},
                            {
                                "size": (512, 512),
                                "interpolation": InterpolationMode.BILINEAR,
                            },
                        ],
                    )
                ),
                RescaleMapper(RescaleMapperConfig(key="source")),
                RescaleMapper(RescaleMapperConfig(key="target")),
                RescaleMapper(RescaleMapperConfig(key="depth")),
                LightParamsMapper(
                    LightParamsMapperConfig(
                        key="light_params",
                        output_key="light_params",
                    )
                ),
            ],
        ),
    ]

    return filters_mappers


def get_data_module(
    train_shards: List[str],
    validation_shards: List[str],
    batch_size: int,
    source_image_key: str = "source.png",
    target_image_key: str = "target.png",
    depth_key: str = "depth.png",
    light_params_key: str = "light_params.json",
):
    # TRAIN
    train_filters_mappers = get_filter_mappers(
        source_image_key=source_image_key,
        target_image_key=target_image_key,
        depth_key=depth_key,
        light_params_key=light_params_key,
    )

    train_shards_path_or_urls_unbraced = []
    for train_shards_path_or_url in train_shards:
        train_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(train_shards_path_or_url)
        )

    random.shuffle(train_shards_path_or_urls_unbraced)

    train_data_config = DataModuleConfig(
        shards_path_or_urls=train_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=20,
        shuffle_before_split_by_workers_buffer_size=20,
        shuffle_before_filter_mappers_buffer_size=20,
        shuffle_after_filter_mappers_buffer_size=20,
        per_worker_batch_size=batch_size,
        num_workers=min(2, len(train_shards_path_or_urls_unbraced)),
    )

    # VALIDATION
    validation_filters_mappers = get_filter_mappers(
        source_image_key=source_image_key,
        target_image_key=target_image_key,
        depth_key=depth_key,
        light_params_key=light_params_key,
    )

    validation_shards_path_or_urls_unbraced = []
    for validation_shards_path_or_url in validation_shards:
        validation_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(validation_shards_path_or_url)
        )

    validation_data_config = DataModuleConfig(
        shards_path_or_urls=validation_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=10,
        shuffle_before_split_by_workers_buffer_size=10,
        shuffle_before_filter_mappers_buffer_size=10,
        shuffle_after_filter_mappers_buffer_size=10,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    return DataModule(
        train_config=train_data_config,
        train_filters_mappers=train_filters_mappers,
        eval_config=validation_data_config,
        eval_filters_mappers=validation_filters_mappers,
    )


def main(
    train_shards: List[str] = ["pipe:cat path/to/train/shards"],
    validation_shards: List[str] = ["pipe:cat path/to/validation/shards"],
    backbone_signature: str = "runwayml/stable-diffusion-v1-5",
    vae_num_channels: int = 4,
    unet_input_channels: int = 8,
    source_key: str = "source",
    target_key: str = "target",
    mask_key: Optional[str] = None,
    wandb_project: str = "lbm-relight-params",
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
    source_image_key: str = "source.png",
    target_image_key: str = "target.png",
    depth_key: str = "depth.png",
    light_params_key: str = "light_params.json",
    light_params_num_frequencies: int = 6,
    light_params_hidden_dims: List[int] = [256, 512],
    light_params_num_tokens: int = 77,
    light_params_embedding_dim: int = 768,
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
    light_params_config = LightParamsEmbedderConfig(
        input_key="light_params",
        num_frequencies=light_params_num_frequencies,
        hidden_dims=light_params_hidden_dims,
        num_tokens=light_params_num_tokens,
        embedding_dim=light_params_embedding_dim,
    )
    model = build_relight_params_model(
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
        bridge_noise_sigma=bridge_noise_sigma,
        light_params_config=light_params_config,
    )

    data_module = get_data_module(
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size,
        source_image_key=source_image_key,
        target_image_key=target_image_key,
        depth_key=depth_key,
        light_params_key=light_params_key,
    )

    train_parameters = ["denoiser.*", "conditioner.*"]

    training_config = TrainingConfig(
        learning_rate=learning_rate,
        lr_scheduler_name=learning_rate_scheduler,
        lr_scheduler_kwargs=learning_rate_scheduler_kwargs,
        log_keys=["source", "target", "depth"],
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
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "-LBM-Relight-Params"
    )
    run_name = training_signature

    trainer = Trainer(
        accelerator="gpu",
        devices=devices if devices is not None else max(torch.cuda.device_count(), 1),
        num_nodes=num_nodes,
        strategy="ddp_find_unused_parameters_true",
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
                save_last=False,
                save_top_k=-1,
            ),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=2,
        val_check_interval=1000,
        max_epochs=max_epochs,
    )

    trainer.fit(pipeline, data_module, ckpt_path=start_ckpt)


def main_from_config(path_config: str = None):
    with open(path_config, "r") as file:
        config = yaml.safe_load(file)
    logging.info(
        f"Running main with config: {yaml.dump(config, default_flow_style=False)}"
    )
    main(**config, config_yaml=config, path_config=path_config)


if __name__ == "__main__":
    fire.Fire(main_from_config)
