# /MesoXAI/model/mesonet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Meso4(nn.Module):
    def __init__(self, num_classes=1):
        super(Meso4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, 5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, 5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(4, 4)
        )

        # Dynamically determine fc1 input size
        self._initialize_fc_layers()

        self.fc2 = nn.Linear(16, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _initialize_fc_layers(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)  # assuming input size = 256x256
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            self.flatten_dim = x.view(1, -1).shape[1]
            self.fc1 = nn.Linear(self.flatten_dim, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)
