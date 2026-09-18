# Author: Kai Fan kf5601
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/model.py

# Required imports
import torch
from torch import nn


class DigitMLP(nn.Module):
    """Multilayer perceptron for handwritten digit classification."""

    def __init__(self, image_size: int = 28, num_classes: int = 10):
        super().__init__()

        input_size = image_size * image_size

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(input_size, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of images."""
        return self.network(x)