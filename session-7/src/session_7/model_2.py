import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics

from session_7.common import Conv2dWithBN


class Model2(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            Conv2dWithBN(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            Conv2dWithBN(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            Conv2dWithBN(in_channels=16, out_channels=16, kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            Conv2dWithBN(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            Conv2dWithBN(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            Conv2dWithBN(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            Conv2dWithBN(in_channels=32, out_channels=32, kernel_size=2, stride=2),
            nn.Dropout2d(0.1),
        )

        self.block3 = nn.Sequential(
            Conv2dWithBN(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            Conv2dWithBN(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
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
