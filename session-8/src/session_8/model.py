import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torchmetrics
import torchvision.models as M
from lightning.pytorch.utilities.types import OptimizerLRScheduler


class Resnet50Cifar100(L.LightningModule):
    def __init__(
        self,
        lr=0.1,
        max_epochs=100,
        use_onecycle=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        num_classes = 100
        self.model = M.resnet50(weights=None, num_classes=num_classes)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # type: ignore

        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        input, target = batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc.update(output, target)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        input, target = batch
        output = self.forward(input)
        loss = self.loss_fn(output, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_acc.update(output, target)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,  # type: ignore
            momentum=0.9,
            weight_decay=5e-4,
            nesterov=True,
        )

        if self.hparams.use_onecycle:  # type: ignore
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.lr,  # type: ignore
                total_steps=int(self.trainer.estimated_stepping_batches),
                pct_start=0.3,  # 30% of training for warmup
                anneal_strategy="cos",
                div_factor=25.0,  # initial_lr = max_lr/25
                final_div_factor=1e4,  # min_lr = initial_lr/1e4
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            warmup_epochs = 5
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,  # Start at 1% of base LR
                end_factor=1.0,  # End at 100% of base LR
                total_iters=warmup_epochs,
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs - warmup_epochs, eta_min=0  # type: ignore
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
