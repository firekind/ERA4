"""
Target:
Target was to take the previous model, and improve and stablize the accuracies using lr schedulers.

Result:
The base model was taken from Model2 (model_2.py). No other changes were made to the model. During training,
StepLR was applied with step size as 10 and gamma as 0.1

Parameter Count: 6174
Train Accuracy: 95.79%
Test Accuracy: 99.39%

Analysis:
StepLR helped! Although test accuracy of epoch 15 is 99.39%, test accuracies of epoch 11 to 14 were always above 99.4%. The model
can do better if pushed more. Since the data is MNIST, some random rotation transform can be applied. Only a little should be
applied, too much would cause the model to underfit again.
"""

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
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
