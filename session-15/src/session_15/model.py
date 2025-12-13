from typing import Callable

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import Tensor

BlockFactory = Callable[[int, int], nn.Module]


class MaxPoolDownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.ds = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.b1(x)
        x = self.b2(x)
        skip = x
        x = self.ds(x)
        return x, skip


class StridedConvDownsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.ds = nn.Conv2d(out_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.b1(x)
        x = self.b2(x)
        skip = x
        x = self.ds(x)
        return x, skip


class TransposeConvUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.us = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.b1 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.us(x)
        x = torch.cat((x, skip), dim=1)
        x = self.b1(x)
        x = self.b2(x)
        return x


class InterpUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.us = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.b1 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.us(x)
        x = self.conv(x)
        x = torch.cat((x, skip), dim=1)
        x = self.b1(x)
        x = self.b2(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_block: BlockFactory,
        upsample_block: BlockFactory,
    ):
        super().__init__()

        self.d1 = downsample_block(in_channels, 64)
        self.d2 = downsample_block(64, 128)
        self.d3 = downsample_block(128, 256)
        self.d4 = downsample_block(256, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.u1 = upsample_block(1024, 512)
        self.u2 = upsample_block(512, 256)
        self.u3 = upsample_block(256, 128)
        self.u4 = upsample_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x, s1 = self.d1(x)
        x, s2 = self.d2(x)
        x, s3 = self.d3(x)
        x, s4 = self.d4(x)

        x = self.bottleneck(x)

        x = self.u1(x, s4)
        x = self.u2(x, s3)
        x = self.u3(x, s2)
        x = self.u4(x, s1)

        return self.final_conv(x)


class UNetModule(L.LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_block: BlockFactory,
        upsample_block: BlockFactory,
        loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        max_lr: float = 1e-3,
        div_factor: float = 25,
        final_div_factor: float = 1e4,
        pct_start: float = 0.3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("loss_fn"))

        self.model = UNet(in_channels, out_channels, downsample_block, upsample_block)
        self.loss_fn = loss_fn
        self.max_lr = max_lr
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.pct_start = pct_start

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        if self.loss_fn is None:
            raise ValueError("cannot train when loss function was not given")

        images, masks = batch
        output = self(images)
        loss = self.loss_fn(output, masks)
        self.log("metric.train.loss", loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        if self.loss_fn is None:
            raise ValueError("cannot validate when loss function was not given")

        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)

        self.log("metric.val.loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.max_lr / self.div_factor
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=self.pct_start,
            anneal_strategy="cos",
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
