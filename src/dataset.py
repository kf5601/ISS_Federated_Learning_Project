# Author: Kai Fan kf5601 
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/dataset.py

import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class FEMNISTClientDataset(Dataset):
    """FEMNIST dataset belonging to a single federated client."""

    def __init__(self, root_dir: str, client_id: int, poison: bool = False, flip_rate: float = 1.0):
        self.client_dir = Path(root_dir) / f"client_{client_id}"

        if not self.client_dir.exists():
            raise FileNotFoundError(f"Client directory not found: {self.client_dir}")

        self.samples = []

        # Check subdirectories (handles both 0-9 and 1-10 folder naming)
        has_zero_folder = (self.client_dir / "0").exists()
        subdirs = [d for d in self.client_dir.iterdir() if d.is_dir()]

        for sdir in sorted(subdirs, key=lambda x: int(x.name) if x.name.isdigit() else 0):
            if sdir.name.isdigit():
                raw_val = int(sdir.name)
                # Map 1-10 folder range to 0-9 zero-indexed class labels if '0' folder doesn't exist
                class_label = (raw_val - 1) if (not has_zero_folder and raw_val in range(1, 11)) else raw_val

                for img_path in sorted(sdir.glob("*.png")):
                    target_label = class_label
                    # Static label flipping if poisoned
                    if poison and torch.rand(1).item() < flip_rate:
                        target_label = (class_label + 1) % 10

                    self.samples.append((str(img_path), target_label))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as img:
            image = self.transform(img)

        return image, label


def get_client_dataloaders(root_dir: str, client_id: int, batch_size: int = 32, poison: bool = False, flip_rate: float = 1.0, test_split: float = 0.2):
    """Generates local train and evaluation dataloaders for a given client."""
    dataset = FEMNISTClientDataset(root_dir, client_id, poison=poison, flip_rate=flip_rate)
    if len(dataset) == 0:
        return None, None

    test_size = max(1, int(len(dataset) * test_split))
    train_size = len(dataset) - test_size

    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader