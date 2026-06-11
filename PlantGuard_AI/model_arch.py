# model_arch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CustomCNN(nn.Module):
    """
    Baseline CNN model for comparison
    """

    def __init__(self, num_classes=23):
        super(CustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x


class EfficientNetB3Model(nn.Module):
    """
    EfficientNet-B3 for plant disease classification
    """

    def __init__(self, num_classes):

        super().__init__()

        self.model = models.efficientnet_b3(weights="IMAGENET1K_V1")

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):

        return self.model(x)