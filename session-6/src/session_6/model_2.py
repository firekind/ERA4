"""
Target:
Target was to take the previous model, and minimize train accuracy - test accuracy gap.

Result:
The base model was taken from Model1 (model_1.py). Probability of drop out in each block is set to 10% (compared to 20% in Model1)

Parameter Count: 6174
Train Accuracy: 95.47%
Test Accuracy: 99.17%

Analysis:
Gap between train and test accuracy has reduced a lot, which is good. Overall accuracy still not reached 99.4%,
and accuracies are not that stable in the last few epochs. Can try out LR schedulers to reduce the learning rate
in the last few epochs, which should stablize the model.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.10),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.10),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(12, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 10, 3),
            nn.ReLU(),
            nn.BatchNorm2d(10),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.10),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=1)

    def summarize(self):
        summary(self, input_size=(1, 28, 28))
