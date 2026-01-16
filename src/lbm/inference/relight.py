from typing import List, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusionPipeline

from lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
    LightingParamsEmbedder,
    LightingParamsEmbedderConfig,
)
from lbm.models.lbm import LBMConfig, LBMModel
from lbm.models.unets import DiffusersUNet2DCondWrapper
from lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig
from lbm.trainer.utils import StateDictAdapter


def _build_relight_model(
    backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    vae_num_channels: int = 4,
    unet_input_channels: int = 12,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = None,
    conditioning_masks_keys: Optional[List[str]] = None,
    lighting_conditioning: bool = False,
    lighting_embedder_config: Optional[dict] = None,
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
    cross_attention_dim: Optional[List[int] | int] = None,
) -> LBMModel:
    conditioners = []

    if conditioning_images_keys is None:
        conditioning_images_keys = ["shading", "normal"]

    if conditioning_masks_keys is None:
        conditioning_masks_keys = []

    pipe = StableDiffusionPipeline.from_pretrained(
        backbone_signature,
        torch_dtype=torch.bfloat16,
    )

    if cross_attention_dim is None:
        cross_attention_dim = pipe.unet.config.cross_attention_dim

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
        cross_attention_dim=cross_attention_dim,
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

    if lighting_conditioning:
        embedder_kwargs = lighting_embedder_config or {}
        lighting_embedder_config_obj = LightingParamsEmbedderConfig(**embedder_kwargs)
        lighting_embedder = LightingParamsEmbedder(lighting_embedder_config_obj)
        conditioners.append(lighting_embedder)

    # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

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


def build_relight_model(
    backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    vae_num_channels: int = 4,
    unet_input_channels: int = 12,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = None,
    conditioning_masks_keys: Optional[List[str]] = None,
    lighting_conditioning: bool = False,
    lighting_embedder_config: Optional[dict] = None,
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
) -> LBMModel:
    return _build_relight_model(
        backbone_signature=backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        timestep_sampling=timestep_sampling,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
        lighting_conditioning=lighting_conditioning,
        lighting_embedder_config=lighting_embedder_config,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        bridge_noise_sigma=bridge_noise_sigma,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
        cross_attention_dim=[320, 640, 1280, 1280],
    )


def build_filllight_model(
    backbone_signature: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
    vae_num_channels: int = 4,
    unet_input_channels: int = 12,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = None,
    conditioning_masks_keys: Optional[List[str]] = None,
    lighting_conditioning: bool = True,
    lighting_embedder_config: Optional[dict] = None,
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
) -> LBMModel:
    return _build_relight_model(
        backbone_signature=backbone_signature,
        vae_num_channels=vae_num_channels,
        unet_input_channels=unet_input_channels,
        timestep_sampling=timestep_sampling,
        selected_timesteps=selected_timesteps,
        prob=prob,
        conditioning_images_keys=conditioning_images_keys,
        conditioning_masks_keys=conditioning_masks_keys,
        lighting_conditioning=lighting_conditioning,
        lighting_embedder_config=lighting_embedder_config,
        source_key=source_key,
        target_key=target_key,
        mask_key=mask_key,
        bridge_noise_sigma=bridge_noise_sigma,
        logit_mean=logit_mean,
        logit_std=logit_std,
        pixel_loss_type=pixel_loss_type,
        latent_loss_type=latent_loss_type,
        latent_loss_weight=latent_loss_weight,
        pixel_loss_weight=pixel_loss_weight,
    )
