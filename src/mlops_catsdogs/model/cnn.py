"""SimpleCNN: a small baseline convolutional network for 224x224 RGB binary classification."""
from __future__ import annotations

import torch
import torch.nn as nn

from mlops_catsdogs.config import CLASS_NAMES


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = len(CLASS_NAMES)):
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),     # 224 -> 112
            block(16, 32),    # 112 -> 56
            block(32, 64),    # 56 -> 28
            block(64, 128),   # 28 -> 14
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_model(num_classes: int = len(CLASS_NAMES)) -> SimpleCNN:
    return SimpleCNN(num_classes=num_classes)
