import csv
from pathlib import Path
from typing import Any, Callable

import lightning as L
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import make_dataset


class ImageNet(L.LightningDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()

        self.data_dir = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.08, 1.0)),
                T.RandomHorizontalFlip(p=0.5),
                T.AutoAugment(policy=T.AutoAugmentPolicy.IMAGENET),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.val_transform = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = ImageNetDataset(
                self.data_dir, train=True, transform=self.train_transform
            )
            self.val_dataset = ImageNetDataset(
                self.data_dir, train=False, transform=self.val_transform
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=_should_pin_memory(),
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=_should_pin_memory(),
            persistent_workers=True if self.num_workers > 0 else False,
        )


class ImageNetDataset(VisionDataset):
    def __init__(
        self, root: str | Path, train: bool = True, transform: Callable | None = None
    ):
        super().__init__(root, transform=transform)

        root_dir = Path(self.root)
        mapping_file = root_dir / "LOC_synset_mapping.txt"
        synset_to_idx, self.idx_to_name = self._load_synset_mapping(mapping_file)

        self.samples: list[tuple[Path, int]]
        if train:
            img_dir = root_dir / "ILSVRC" / "Data" / "CLS-LOC" / "train"
            self.samples = self._load_train_samples(img_dir, synset_to_idx)
        else:
            val_solution_csv = root_dir / "LOC_val_solution.csv"
            img_dir = root_dir / "ILSVRC" / "Data" / "CLS-LOC" / "val"
            self.samples = self._load_validation_samples(
                img_dir, val_solution_csv, synset_to_idx
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        path, label = self.samples[idx]
        sample = Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def get_name_for_class(self, idx: int) -> str:
        return self.idx_to_name.get(idx, "")

    @staticmethod
    def _load_synset_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
        synset_to_idx = {}
        idx_to_name = {}
        with open(path) as f:
            for idx, line in enumerate(f):
                synset, name = line.strip().split(" ", 1)
                synset_to_idx[synset] = idx
                idx_to_name[idx] = name

        return synset_to_idx, idx_to_name

    @staticmethod
    def _load_train_samples(
        img_dir: Path, synset_to_idx: dict[str, int]
    ) -> list[tuple[Path, int]]:
        samples = make_dataset(img_dir, synset_to_idx, extensions=("jpeg",))
        return [(Path(p), idx) for (p, idx) in samples]

    @staticmethod
    def _load_validation_samples(
        img_dir: Path, solutions_file: Path, synset_to_idx: dict[str, int]
    ) -> list[tuple[Path, int]]:
        samples = []
        with open(solutions_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_id = row["ImageId"]
                pred_string = row["PredictionString"]

                synset = pred_string.split(" ", 1)[0]
                class_idx = synset_to_idx[synset]

                samples.append((img_dir / f"{img_id}.JPEG", class_idx))

        return samples


def _should_pin_memory() -> bool:
    return torch.cuda.is_available()
