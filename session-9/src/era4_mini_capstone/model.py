import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchmetrics
import torchvision.models as M
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor


class ImageNetModel(L.LightningModule):
    pass


class ResNet50WithMixUp(ImageNetModel):
    def __init__(
        self,
        lr=0.1,
        num_classes=1000,
        warmup_pct: float = 0.05,
        mixup_alpha: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = M.resnet50(weights=None)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)

    def mixup_data(self, x: Tensor, y: Tensor, alpha: float = 0.2):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred: Tensor, y_a: Tensor, y_b: Tensor, lam: float):
        """Mixup loss calculation"""
        return lam * self.loss_fn(pred, y_a) + (1 - lam) * self.loss_fn(pred, y_b)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        input, target = batch

        # Save original target for accuracy calculation
        original_target = target.clone()

        # Apply mixup
        input, target_a, target_b, lam = self.mixup_data(
            input, target, alpha=self.hparams.mixup_alpha  # type: ignore
        )

        output = self.forward(input)
        loss = self.mixup_criterion(output, target_a, target_b, lam)

        self.log("metric.train.loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc.update(output, original_target)
        self.log(
            "metric.train.acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        input, target = batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)
        self.log("metric.val.loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_acc.update(output, target)
        self.log(
            "metric.val.acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,  # type: ignore
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
        )

        max_epochs = self.trainer.max_epochs
        if max_epochs is None:
            raise ValueError("max epochs is none")
        warmup_epochs = int(max_epochs * self.hparams.warmup_pct)  # type: ignore

        scheduler = lrs.SequentialLR(
            optimizer=optimizer,
            schedulers=[
                lrs.LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                lrs.CosineAnnealingLR(
                    optimizer,
                    T_max=max_epochs - warmup_epochs,
                    eta_min=1e-5,
                ),
            ],
            milestones=[
                warmup_epochs,
            ],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


class ResNet50OneCycle(ImageNetModel):
    def __init__(
        self,
        lr=0.1,
        num_classes=1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = M.resnet50(weights=None)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model.forward(x)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        input, target = batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)

        self.log("metric.train.loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc.update(output, target)
        self.log(
            "metric.train.acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        input, target = batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)
        self.log("metric.val.loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_acc.update(output, target)
        self.log(
            "metric.val.acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True
        )

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,  # type: ignore
            momentum=0.9,
            weight_decay=1e-4,
            nesterov=True,
        )

        scheduler = lrs.OneCycleLR(
            optimizer=optimizer,
            max_lr=self.hparams.lr,  # type: ignore
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.3,  # 30% warmup
            div_factor=25,  # Start at max_lr/25
            final_div_factor=1e4,  # End very low
            anneal_strategy="cos",  # Cosine annealing
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
