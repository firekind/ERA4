"""
Target:
Target was to build the skeleton, add all proper layers, and get a model below 8k parameters

Result:
The skeleton was taken from session 5 assignment and modified a bit. Instead of a direct jump from 8 channels to
16 channels between block 1 and 2, it is first increased to 12 channels and then 16 (in block 3).

Parameter Count: 6174
train accuracy = 87.67%
test accuracy = 99.20%

Analysis:
Good structure, parameter count is below target. But model is underfitting. Need to reduce drop out.
Both test and train accuracies are not stable.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.20),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.20),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(12, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, 3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.20),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

    def summarize(self):
        summary(self, input_size=(1, 28, 28))
