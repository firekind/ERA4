import lightning as L
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets.oxford_iiit_pet import OxfordIIITPet
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode


class OxfordPetDataModule(L.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        image_size: list[int] = [256, 256],
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.transform = OxfordDatasetTransform(image_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = OxfordIIITPet(
                self.root_dir,
                target_types="segmentation",
                transforms=self.transform,
                split="trainval",
                download=True,
            )
            self.val_dataset = OxfordIIITPet(
                self.root_dir,
                target_types="segmentation",
                transforms=self.transform,
                split="test",
                download=True,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if self.num_workers > 0 else False,
        )


class OxfordDatasetTransform:
    def __init__(self, image_size: list[int] = [256, 256]):
        self.image_size = image_size

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __call__(self, image, target):
        image = self.image_transform(image)

        if isinstance(target, list):
            mask = target[0]
        else:
            mask = target

        mask = np.array(mask)
        # Convert trimap to binary BEFORE resizing (1 for foreground, 0 for others)
        mask = (mask == 1).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)
        mask = F.resize(mask, self.image_size, interpolation=InterpolationMode.NEAREST)

        return image, mask
