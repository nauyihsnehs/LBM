import datetime
import logging
import os
import random
import re
import shutil
from typing import List, Optional

import braceexpand
import fire
import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torchvision.transforms import InterpolationMode

from lbm.data.datasets import DataModule, DataModuleConfig
from lbm.data.filters import KeyFilter, KeyFilterConfig
from lbm.data.mappers import (
    KeyRenameMapper,
    KeyRenameMapperConfig,
    MapperWrapper,
    RescaleMapper,
    RescaleMapperConfig,
    TorchvisionMapper,
    TorchvisionMapperConfig,
)
from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger
from lbm.trainer.utils import StateDictAdapter


def get_model(
        backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        vae_num_channels: int = 4,
        unet_input_channels: int = 12,
        timestep_sampling: str = "log_normal",
        selected_timesteps: Optional[List[float]] = None,
        prob: Optional[List[float]] = None,
        conditioning_images_keys: Optional[List[str]] = None,
        conditioning_masks_keys: Optional[List[str]] = None,
        source_key: str = "source",
        target_key: str = "target",
        mask_key: Optional[str] = None,
        bridge_noise_sigma: float = 0.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        pixel_loss_type: str = "lpips",
        latent_loss_type: str = "l2",
        latent_loss_weight: float = 1.0,
        pixel_loss_weight: float = 0.0,
):
    conditioners = []

    if conditioning_images_keys is None:
        conditioning_images_keys = ["shading", "normal"]

    if conditioning_masks_keys is None:
        conditioning_masks_keys = []

    pipe = StableDiffusionPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch.bfloat16,
    )

    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=[
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280, 1280],
        transformer_layers_per_block=1,
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=8,
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=False,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=False,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
        # sample_size=64,
    ).to(torch.bfloat16)

    state_dict = pipe.unet.state_dict()

    # Remove SDXL-specific layers
    keys_to_remove = [
        "add_embedding.linear_1.weight",
        "add_embedding.linear_1.bias",
        "add_embedding.linear_2.weight",
        "add_embedding.linear_2.bias",
    ]
    for key in keys_to_remove:
        if key in state_dict:
            del state_dict[key]

    # Adapt the shapes for SD1.5
    state_dict_adapter = StateDictAdapter()
    state_dict = state_dict_adapter(
        model_state_dict=denoiser.state_dict(),
        checkpoint_state_dict=state_dict,
        regex_keys=[
            r"conv_in.weight",
            r"(down_blocks|up_blocks)\.\d+\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
            r"mid_block\.attentions\.\d+\.transformer_blocks\.\d+\.attn\d+\.(to_k|to_v)\.weight",
        ],
        strategy="zeros",
    )

    denoiser.load_state_dict(state_dict, strict=True)

    del pipe

    if conditioning_images_keys != [] or conditioning_masks_keys != []:
        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

    # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config)
    vae.freeze()
    vae.to(torch.bfloat16)

    # LBM Config
    config = LBMConfig(
        ucg_keys=None,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    training_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )
    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    # LBM Model
    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch.bfloat16)

    return model


def get_filter_mappers(
        source_image_key: str = "source.png",
        target_image_key: str = "target.png",
        shading_key: str = "shading.png",
        normal_key: str = "normal.png",
):
    filters_mappers = [
        KeyFilter(
            KeyFilterConfig(
                keys=[source_image_key, target_image_key, shading_key, normal_key]
            )
        ),
        MapperWrapper(
            [
                KeyRenameMapper(
                    KeyRenameMapperConfig(
                        key_map={
                            source_image_key: "source",
                            target_image_key: "target",
                            shading_key: "shading",
                            normal_key: "normal",
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
                        key="shading",
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
                        key="normal",
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
                RescaleMapper(RescaleMapperConfig(key="source")),
                RescaleMapper(RescaleMapperConfig(key="target")),
                RescaleMapper(RescaleMapperConfig(key="shading")),
                RescaleMapper(RescaleMapperConfig(key="normal")),
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
        shading_key: str = "shading.png",
        normal_key: str = "normal.png",
):
    # TRAIN
    train_filters_mappers = get_filter_mappers(
        source_image_key=source_image_key,
        target_image_key=target_image_key,
        shading_key=shading_key,
        normal_key=normal_key,
    )

    # unbrace urls
    train_shards_path_or_urls_unbraced = []
    for train_shards_path_or_url in train_shards:
        train_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(train_shards_path_or_url)
        )

    # shuffle shards
    random.shuffle(train_shards_path_or_urls_unbraced)

    # data config
    data_config = DataModuleConfig(
        shards_path_or_urls=train_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=20,
        shuffle_before_split_by_workers_buffer_size=20,
        shuffle_before_filter_mappers_buffer_size=20,
        shuffle_after_filter_mappers_buffer_size=20,
        per_worker_batch_size=batch_size,
        num_workers=min(2, len(train_shards_path_or_urls_unbraced)),
    )

    train_data_config = data_config

    # VALIDATION
    validation_filters_mappers = get_filter_mappers(
        source_image_key=source_image_key,
        target_image_key=target_image_key,
        shading_key=shading_key,
        normal_key=normal_key,
    )

    # unbrace urls
    validation_shards_path_or_urls_unbraced = []
    for validation_shards_path_or_url in validation_shards:
        validation_shards_path_or_urls_unbraced.extend(
            braceexpand.braceexpand(validation_shards_path_or_url)
        )

    data_config = DataModuleConfig(
        shards_path_or_urls=validation_shards_path_or_urls_unbraced,
        decoder="pil",
        shuffle_before_split_by_node_buffer_size=10,
        shuffle_before_split_by_workers_buffer_size=10,
        shuffle_before_filter_mappers_buffer_size=10,
        shuffle_after_filter_mappers_buffer_size=10,
        per_worker_batch_size=batch_size,
        num_workers=min(10, len(train_shards_path_or_urls_unbraced)),
    )

    validation_data_config = data_config

    # data module
    data_module = DataModule(
        train_config=train_data_config,
        train_filters_mappers=train_filters_mappers,
        eval_config=validation_data_config,
        eval_filters_mappers=validation_filters_mappers,
    )

    return data_module


def main(
        train_shards: List[str] = ["pipe:cat path/to/train/shards"],
        validation_shards: List[str] = ["pipe:cat path/to/validation/shards"],
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
        source_image_key: str = "source.png",
        target_image_key: str = "target.png",
        shading_key: str = "shading.png",
        normal_key: str = "normal.png",
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
    model = get_model(
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

    data_module = get_data_module(
        train_shards=train_shards,
        validation_shards=validation_shards,
        batch_size=batch_size,
        source_image_key=source_image_key,
        target_image_key=target_image_key,
        shading_key=shading_key,
        normal_key=normal_key,
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
            + "-LBM-Relight"
    )
    run_name = training_signature


    trainer = Trainer(
        accelerator="gpu",
        devices=devices if devices is not None else max(torch.cuda.device_count(), 1),
        num_nodes=num_nodes,
        # strategy=strategy,
        strategy='ddp_find_unused_parameters_true',
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
                save_top_k=-1
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
