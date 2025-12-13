import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_lr_finder import LRFinder

from session_15 import (
    DiceLoss,
    InterpUpsampleBlock,
    MaxPoolDownsampleBlock,
    OxfordPetDataModule,
    StridedConvDownsampleBlock,
    TransposeConvUpsampleBlock,
    UNet,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def find_learning_rate(
    model: nn.Module,
    datamodule: OxfordPetDataModule,
    loss_fn: nn.Module,
    start_lr: float = 1e-7,
    end_lr: int = 1,
    num_iter: int = 100,
    save_path: str = "lr_finder_plot.png",
):
    # Setup datamodule
    datamodule.setup("fit")
    train_loader = datamodule.train_dataloader()

    device: str
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)

    lr_finder = LRFinder(model, optimizer, loss_fn, device=device)
    lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter)

    # Create figure and axes
    fig, ax = plt.subplots()

    # Plot and save
    _, suggested_lr = lr_finder.plot(ax=ax, suggest_lr=True)  # type: ignore
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close to free memory
    print(f"LR finder plot saved to {save_path}")

    # Get the suggested learning rate
    print(f"Suggested learning rate: {suggested_lr}")

    # Reset the model and optimizer
    lr_finder.reset()

    return suggested_lr


def main():
    parser = argparse.ArgumentParser(description="Find optimal learning rate for UNet")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/dataset",
        help="Root directory for dataset",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--image-size", type=int, default=256, help="Image size (square)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
    )
    parser.add_argument(
        "--start-lr", type=float, default=1e-7, help="Starting learning rate"
    )
    parser.add_argument("--end-lr", type=float, default=1, help="Ending learning rate")
    parser.add_argument(
        "--num-iter", type=int, default=200, help="Number of iterations"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./data/lr_finder_plot.png",
        help="Path to save plot",
    )
    parser.add_argument(
        "--downsample",
        type=str,
        default="maxpool",
        choices=["maxpool", "strided"],
        help="Downsample block type",
    )
    parser.add_argument(
        "--upsample",
        type=str,
        default="transpose",
        choices=["transpose", "interp"],
        help="Upsample block type",
    )
    parser.add_argument(
        "--loss", type=str, default="bce", choices=["bce", "dice"], help="Loss function"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup dataset
    datamodule = OxfordPetDataModule(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Select blocks
    downsample_block = (
        MaxPoolDownsampleBlock
        if args.downsample == "maxpool"
        else StridedConvDownsampleBlock
    )
    upsample_block = (
        TransposeConvUpsampleBlock
        if args.upsample == "transpose"
        else InterpUpsampleBlock
    )

    # Select loss
    if args.loss == "bce":
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = DiceLoss()

    # Create model (just the UNet, not the LightningModule)
    model = UNet(
        in_channels=3,
        out_channels=1,
        downsample_block=downsample_block,
        upsample_block=upsample_block,
    )

    # Find LR
    suggested_lr = find_learning_rate(
        model,
        datamodule,
        loss_fn,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter,
        save_path=args.save_path,
    )

    print(f"\nUse this learning rate for training: {suggested_lr}")


if __name__ == "__main__":
    main()
