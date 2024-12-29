import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
import torch
from segmentation_datamodule import SegmentationDataModule
from unet_lightning_module import UNetLightningModule

if __name__ == "__main__":
    """
    Entrypoint for training using Lightning CLI.

    Configuration should be provided through a YAML file, e.g.:
    lightning run train --config=config.yaml
    """

    # Set matrix multiplication precision to high
    torch.set_float32_matmul_precision("high")

    # Initialize the Wandb logger
    wandb_logger = WandbLogger(project="unet_segmentation", log_model=True)

    LightningCLI(
        model_class=UNetLightningModule,
        datamodule_class=SegmentationDataModule,
        trainer_defaults={
            "logger": wandb_logger,
            "strategy": "ddp",  # DistributedDataParallel for multi-GPU
            "precision": 16,  # Use mixed precision training,
            "log_every_n_steps": 10,  # Log metrics every 10 steps
        },
    )
