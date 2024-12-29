import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torchmetrics import (
    MetricCollection,
    Accuracy,
    Precision,
    Recall,
    Dice,
    JaccardIndex,
)
from model import (
    AbstractUNet,
)  # Replace with the actual class name from your model file
from losses import get_loss_function  # A utility to dynamically load loss functions
from torch.optim.lr_scheduler import StepLR  # Example of a scheduler


class UNetLightningModule(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)  # Save the config for reproducibility

        # Model configuration
        model_params = config.get("model", {})
        self.model = AbstractUNet(**model_params)

        # Loss function configuration
        self.loss_fn = get_loss_function(config.get("loss", "CrossEntropyLoss"))

        # Optimizer configuration
        self.optimizer_name = config.get("optimizer", "Adam")
        self.optimizer_params = config.get("optimizer_params", {"lr": 1e-3})

        # Scheduler configuration
        self.scheduler_params = config.get(
            "scheduler_params", {"step_size": 10, "gamma": 0.1}
        )

        # Metrics
        num_classes = config.get("num_classes", 2)
        metrics = MetricCollection(
            {
                "accuracy": Accuracy(),
                "precision": Precision(),
                "recall": Recall(),
                "dice": Dice(num_classes=num_classes),
                "iou": JaccardIndex(num_classes=num_classes),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, step: str):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        preds = torch.argmax(outputs, dim=1)

        metrics = getattr(self, f"{step}_metrics")
        metrics.update(preds, targets)

        self.log(f"{step}/loss", loss)
        self.log_dict(metrics, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        # Instantiate optimizer
        if self.optimizer_name == "Adam":
            optimizer = Adam(self.parameters(), **self.optimizer_params)
        elif self.optimizer_name == "SGD":
            optimizer = SGD(self.parameters(), **self.optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        # Optionally add a scheduler
        scheduler = StepLR(optimizer, **self.scheduler_params)
        return [optimizer], [scheduler]
