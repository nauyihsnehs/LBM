from typing import List, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DWrapper


def build_filllight_refine_model(
    backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    unet_input_channels: int = 6,
    unet_output_channels: int = 3,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    source_key: str = "source",
    target_key: str = "target",
    mask_key: Optional[str] = None,
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "l2",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    block_out_channels: Optional[List[int]] = None,
    layers_per_block: int = 2,
) -> LBMModel:
    if block_out_channels is None:
        block_out_channels = [64, 128, 256, 256]

    denoiser = DiffusersUNet2DWrapper(
        in_channels=unet_input_channels,
        out_channels=unet_output_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
        ],
        up_block_types=[
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ],
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        attention_head_dim=None,
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
        class_embed_type=None,
        num_class_embeds=None,
    ).to(torch.bfloat16)

    config = LBMConfig(
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

    model = LBMModel(
        config,
        denoiser=denoiser,
        training_noise_scheduler=training_noise_scheduler,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=None,
        conditioner=None,
    ).to(torch.bfloat16)

    return model
