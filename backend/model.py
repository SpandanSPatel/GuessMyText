"""
GuessMyText — CNN Model Definition

Reproduces the exact CNN architecture used during training.
All three checkpoints (32×32, 64×64, 128×128) share this architecture:
  - Conv2d(1, 32, 3, padding=1) → ReLU → MaxPool2d(2)
  - Conv2d(32, 64, 3, padding=1) → ReLU → MaxPool2d(2)
  - AdaptiveAvgPool2d((8, 8))
  - Flatten → Linear(64*8*8, 128) → ReLU → Dropout(0.2) → Linear(128, num_classes)
"""

import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, num_classes: int = 67):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
