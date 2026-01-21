import datetime
import logging
import platform
from pathlib import Path
from typing import List, Optional

import torch
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

from lbm.trainer import TrainingConfig, TrainingPipeline
from lbm.trainer.loggers import WandbSampleLogger


def create_run_name(task_label: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{timestamp}-LBM-{task_label}"


def resolve_task_dir(save_ckpt_path: str, task_name: str) -> Path:
    task_dir = Path(save_ckpt_path) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    return task_dir


def resolve_resume_checkpoint(
        task_dir: Path,
        resume_from_checkpoint: bool,
        resume_ckpt_path: Optional[str] = None,
) -> Optional[str]:
    if not resume_from_checkpoint:
        return None

    if resume_ckpt_path:
        resume_path = Path(resume_ckpt_path)
        if resume_path.exists():
            return str(resume_path)
        logging.warning("Resume checkpoint not found: %s", resume_ckpt_path)

    last_ckpt = task_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)

    ckpt_files = sorted(
        task_dir.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if ckpt_files:
        return str(ckpt_files[0])

    return None


def setup_pipeline(
        model,
        training_config: TrainingConfig,
        config_yaml: Optional[dict],
) -> TrainingPipeline:
    pipeline = TrainingPipeline(model=model, pipeline_config=training_config)

    if hasattr(model, "conditioner") and getattr(model.conditioner, "conditioners", None):
        pipeline.save_hyperparameters(
            {
                f"embedder_{i}": embedder.config.to_dict()
                for i, embedder in enumerate(model.conditioner.conditioners)
            }
        )

    pipeline.save_hyperparameters(
        {
            "denoiser": model.denoiser.config,
            "config_yaml": config_yaml,
            "training": training_config.to_dict(),
            "training_noise_scheduler": model.training_noise_scheduler.config,
            "sampling_noise_scheduler": model.sampling_noise_scheduler.config,
        }
    )
    if getattr(model, "vae", None) is not None:
        pipeline.save_hyperparameters({"vae": model.vae.config.to_dict()})
    return pipeline


def build_concat_keys(training_config: TrainingConfig) -> List[str]:
    keys: List[str] = []
    if training_config.log_keys:
        keys.extend(training_config.log_keys)
    num_steps = training_config.log_samples_model_kwargs.get("num_steps", [])
    if isinstance(num_steps, int):
        num_steps = [num_steps]
    for step in num_steps:
        keys.append(f"samples_{step}_steps")
    seen = set()
    deduped = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def build_trainer(
        wandb_project: str,
        run_name: str,
        save_dir: Path,
        log_interval: int,
        save_interval: Optional[int],
        max_epochs: int,
        concat_keys: Optional[List[str]] = None,
        devices: Optional[int] = None,
        num_nodes: int = 1,
) -> Trainer:
    if platform.system() == "Windows":
        strategy = DDPStrategy(find_unused_parameters=True, process_group_backend="gloo")
    else:
        strategy = "ddp_find_unused_parameters_true"

    checkpoint_kwargs = {
        "dirpath": str(save_dir),
        "save_last": False,
        "save_top_k": -1,
        "save_weights_only": False,
    }
    if save_interval is not None:
        checkpoint_kwargs["every_n_train_steps"] = save_interval

    return Trainer(
        accelerator="gpu",
        devices=devices if devices is not None else max(torch.cuda.device_count(), 1),
        num_nodes=num_nodes,
        strategy=strategy,
        default_root_dir="logs",
        logger=loggers.WandbLogger(
            project=wandb_project, offline=True, name=run_name, save_dir=str(save_dir)
        ),
        callbacks=[
            WandbSampleLogger(log_batch_freq=log_interval, concat_keys=concat_keys),
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=1),
            ModelCheckpoint(**checkpoint_kwargs),
        ],
        num_sanity_val_steps=0,
        precision="bf16-mixed",
        limit_val_batches=2,
        check_val_every_n_epoch=1,
        max_epochs=max_epochs,
        enable_progress_bar=True,
    )


def fit_trainer(
        trainer: Trainer,
        pipeline: TrainingPipeline,
        train_loader,
        validation_loader,
        ckpt_path: Optional[str],
) -> None:
    # torch.serialization.add_safe_globals([TrainingConfig])
    if ckpt_path:
        logging.info("Resuming training from checkpoint: %s", ckpt_path)
        trainer.fit(
            pipeline,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
            ckpt_path=ckpt_path,
            weights_only=False,
        )
    else:
        trainer.fit(
            pipeline, train_dataloaders=train_loader, val_dataloaders=validation_loader
        )
