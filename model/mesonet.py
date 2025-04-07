# /MesoXAI/model/mesonet.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Meso4(nn.Module):
    def __init__(self):
        super(Meso4, self).__init__()

        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )

        # Fully connected
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 8 * 8, 16),  # 256x256 input results in (B, 16, 8, 8)
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Linear(16, 1)  # Binary classification

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x  # Use BCEWithLogitsLoss for binary classification


if __name__ == '__main__':
    model = Meso4()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print("Output shape:", y.shape)  # Should be (2, 1)