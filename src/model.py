# Author: Kai Fan kf5601 
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/model.py

# Required imports
import torch
from torch import nn


class DigitCNN(nn.Module):
    """Convolutional neural network for handwritten digit classification."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Extract spatial features from 1x28x28 grayscale images.
        # Each pooling layer halves the image dimensions:
        # 28x28 -> 14x14 -> 7x7.
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Convert the 64x7x7 feature maps into class scores for digits 0-9.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the CNN."""
        x = self.features(x)
        x = self.classifier(x)
        return x


# TODO: Accuracy is not good, but kept here for future reference, using CNN instead... I thought MLP would be better
# class DigitMLP(nn.Module):
#     """Multilayer perceptron for handwritten digit classification."""

#     def __init__(self, image_size: int = 28, num_classes: int = 10):
#         super().__init__()

#         input_size = image_size * image_size

#         self.network = nn.Sequential(
#             nn.Flatten(),

#             nn.Linear(input_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.2),

#             nn.Linear(256, 128),
#             nn.ReLU(),

#             nn.Linear(128, num_classes),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.network(x)