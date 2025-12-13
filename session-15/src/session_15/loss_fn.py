import torch
import torch.nn as nn
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Calculate Dice coefficient
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss (1 - Dice coefficient)
        return 1 - dice.mean()
