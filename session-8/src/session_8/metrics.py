from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from matplotlib.axes import Axes
from torch.types import Number


class MetricsTracker(Callback):
    def __init__(self):
        self.train_loss: list[Number] = []
        self.train_acc: list[Number] = []
        self.val_loss: list[Number] = []
        self.val_acc: list[Number] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.train_loss.append(trainer.callback_metrics["train_loss"].item())
        self.train_acc.append(trainer.callback_metrics["train_acc"].item())

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule):
        self.val_loss.append(trainer.callback_metrics["val_loss"].item())
        self.val_acc.append(trainer.callback_metrics["val_acc"].item())

    def plot_loss(self, ax: Axes):
        ax.plot(self.train_loss, label="Train Loss")
        ax.plot(self.val_loss, label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

    def plot_accuracy(self, ax: Axes):
        ax.plot(self.train_acc, label="Train Accuracy")
        ax.plot(self.val_acc, label="Val Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
