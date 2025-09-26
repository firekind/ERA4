from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from session_6 import utils

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def train(
    model: nn.Module,
    epochs: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: LossFn,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
):
    model = model.to(utils.device())
    train_step_scheduler: OneCycleLR | None = None
    if isinstance(scheduler, OneCycleLR):
        train_step_scheduler = scheduler

    for epoch in range(1, epochs + 1):
        train_step(epoch, model, train_loader, optimizer, loss_fn, train_step_scheduler)
        test_loss = test_step(model, test_loader, loss_fn)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()


def train_step(
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: LossFn,
    scheduler: OneCycleLR | None = None,
):
    model.train()
    pbar = tqdm(train_loader)
    device = utils.device()
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"epoch={epoch:02d} loss={loss.item():.4f} batch_id={batch_idx:04d} accuracy={100*correct/processed:.2f}%"
        )


def test_step(model: nn.Module, test_loader: DataLoader, loss_fn: LossFn) -> float:
    model.eval()
    device = utils.device()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)  # Average loss across batches
    total = len(test_loader.dataset)  # type: ignore

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            total,
            100.0 * correct / total,
        )
    )

    return test_loss


def create_mnist_data_loaders(
    batch_size: int,
    data_path: str,
    train_transform: Callable | None = None,
    test_transform: Callable | None = None,
) -> tuple[DataLoader, DataLoader]:
    # Download and create datasets
    train_dataset = datasets.MNIST(
        root=data_path, train=True, download=True, transform=train_transform
    )

    test_dataset = datasets.MNIST(
        root=data_path, train=False, download=True, transform=test_transform
    )

    loader_args = {}
    if utils.device() == "cuda":
        loader_args = {"num_workers": 1, "pin_memory": True}

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_args,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        **loader_args,
    )

    return (train_loader, test_loader)
    return (train_loader, test_loader)
    return (train_loader, test_loader)
