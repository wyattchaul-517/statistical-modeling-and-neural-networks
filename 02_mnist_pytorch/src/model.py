"""CNN model definition for MNIST classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A simple convolutional neural network for MNIST.

    Architecture:
        Conv1 (1 -> 32, 3x3) -> ReLU -> MaxPool (2x2)
        Conv2 (32 -> 64, 3x3) -> ReLU -> MaxPool (2x2)
        Flatten -> FC (64*7*7 -> 128) -> ReLU -> Dropout -> FC (128 -> 10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))   # (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)              # (batch, 64*7*7)
        x = F.relu(self.fc1(x))                # (batch, 128)
        x = self.dropout(x)
        x = self.fc2(x)                        # (batch, 10)
        return x