import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from session_7.common import Conv2dWithBN


class Model3(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            Conv2dWithBN(3, 16, kernel_size=3, padding=1),
            Conv2dWithBN(16, 32, kernel_size=3, padding=1),
            Conv2dWithBN(32, 32, kernel_size=2, stride=2, padding=0),
        )

        self.block2 = nn.Sequential(
            Conv2dWithBN(32, 32, kernel_size=3, padding=1),
            Conv2dWithBN(32, 48, kernel_size=3, padding=1),
            Conv2dWithBN(48, 48, kernel_size=2, stride=2, padding=0),
            Conv2dWithBN(48, 32, kernel_size=1, padding=0),
            nn.Dropout2d(0.05),
        )

        self.block3 = nn.Sequential(
            Conv2dWithBN(32, 32, kernel_size=3, padding=1),
            Conv2dWithBN(32, 48, kernel_size=3, padding=1),
            Conv2dWithBN(48, 48, kernel_size=2, stride=2, padding=0),
            Conv2dWithBN(48, 32, kernel_size=1, padding=0),
            nn.Dropout2d(0.05),
        )

        self.block4 = nn.Sequential(
            Conv2dWithBN(32, 32, kernel_size=3, padding=1),
            Conv2dWithBN(32, 10, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        input, target = batch
        output = self.forward(input)
        loss = F.nll_loss(output, target)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_acc.update(output, target)
        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        input, target = batch
        output = self.forward(input)
        loss = F.nll_loss(output, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_acc.update(output, target)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.SGD(self.parameters(), lr=0.01)
