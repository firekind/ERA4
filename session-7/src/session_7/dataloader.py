from typing import Any, Callable

import albumentations as A
import numpy as np
from albumentations.core.bbox_utils import BboxParams
from albumentations.core.composition import TransformsSeqType
from albumentations.core.keypoints_utils import KeypointParams
from lightning.fabric.utilities.data import suggested_max_num_workers
from torch.utils.data import DataLoader
from torchvision import datasets

from session_7 import utils


def create_cifar10_dataloaders(
    batch_size: int,
    data_path: str,
    train_transform: Callable | None = None,
    test_transform: Callable | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    # Download and create datasets
    train_dataset = datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=test_transform
    )

    loader_args: dict[str, Any] = {
        "num_workers": (
            num_workers if num_workers is not None else suggested_max_num_workers(1)
        ),
        "persistent_workers": True,
    }
    if utils.device() == "cuda":
        loader_args = loader_args | {"pin_memory": True}

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


class AImageCompose:
    def __init__(
        self,
        transforms: TransformsSeqType,
        bbox_params: dict[str, Any] | BboxParams | None = None,
        keypoint_params: dict[str, Any] | KeypointParams | None = None,
        additional_targets: dict[str, str] | None = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        strict: bool = False,
        mask_interpolation: int | None = None,
        seed: int | None = None,
        save_applied_params: bool = False,
    ):
        self.t = A.Compose(
            transforms,
            bbox_params,
            keypoint_params,
            additional_targets,
            p,
            is_check_shapes,
            strict,
            mask_interpolation,
            seed,
            save_applied_params,
        )

    def __call__(self, img):
        img = np.array(img)
        augmented = self.t(image=img)
        return augmented["image"]
