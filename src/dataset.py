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
    """
    PyTorch Dataset representing the local data stored on a single client node.
    Handles directory parsing, label mapping, PyTorch transformations, and 
    deterministic label-flipping data poisoning.
    """

    def __init__(self, root_dir: str, client_id: int, poison: bool = False):
        """
        Args:
            root_dir (str): Base path to the dataset folder (e.g., 'femnist_dataset')
            client_id (int): Index of the client (0 through 19)
            poison (bool): If True, applies targeted label flipping to corrupt local data
        """
        # Construct the directory path for this specific client (e.g., 'femnist_dataset/client_0')
        self.client_dir = Path(root_dir) / f"client_{client_id}"

        # Raise an explicit error if the client folder does not exist
        if not self.client_dir.exists():
            raise FileNotFoundError(f"Client directory not found: {self.client_dir}")

        self.samples = []

        # Check if client folders use 0-indexed (0-9) or 1-indexed (1-10) folder names
        has_zero_folder = (self.client_dir / "0").exists()
        subdirs = [d for d in self.client_dir.iterdir() if d.is_dir()]

        # Sort subdirectories numerically to process labels sequentially
        for sdir in sorted(subdirs, key=lambda x: int(x.name) if x.name.isdigit() else 0):
            if sdir.name.isdigit():
                raw_val = int(sdir.name)

                # Normalize 1-10 folder naming to standard 0-9 zero-indexed class labels
                class_label = (raw_val - 1) if (not has_zero_folder and raw_val in range(1, 11)) else raw_val

                # DATA POISONING ATTACK:
                # If client is compromised (poison=True), deterministically shift its label: target = (label + 1) mod 10
                # E.g., label 0 becomes 1, label 9 becomes 0.
                target_label = (class_label + 1) % 10 if poison else class_label

                # Collect all image file paths for this class label
                for img_path in sorted(sdir.glob("*.png")):
                    # Store image path as a string primitive (str) rather than Path object.
                    # This prevents Ray pickle serialization crashes during multi-threaded client simulation.
                    self.samples.append((str(img_path), target_label))

        # Standard PyTorch image pipeline: convert image to single-channel Grayscale tensor
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        """Returns the total number of images stored on this client."""
        return len(self.samples)

    def __getitem__(self, index):
        """
        Loads and transforms a single image-label pair on demand.
        
        Args:
            index (int): Index of the sample to fetch
            
        Returns:
            tuple: (transformed_image_tensor, integer_label)
        """
        image_path, label = self.samples[index]

        # Context manager ensures image handles are safely opened and closed in memory
        with Image.open(image_path) as img:
            image = self.transform(img)

        return image, label


def get_client_dataloaders(root_dir: str, client_id: int, batch_size: int = 32, poison: bool = False, test_split: float = 0.2):
    """
    Helper function to load a client's dataset and partition it locally into 
    train (80%) and evaluation (20%) PyTorch DataLoaders.

    Args:
        root_dir (str): Path to root dataset directory
        client_id (int): Client index
        batch_size (int): Batch size for training/evaluation
        poison (bool): Whether this client is compromised by data poisoning
        test_split (float): Ratio of local data dedicated to local testing/evaluation

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Instantiate client dataset
    dataset = FEMNISTClientDataset(root_dir, client_id, poison=poison)
    if len(dataset) == 0:
        return None, None

    # Calculate train/test dataset split sizes
    test_size = max(1, int(len(dataset) * test_split))
    train_size = len(dataset) - test_size

    # Use a fixed generator seed (42) so dataset splits remain reproducible
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=generator)

    # Wrap split datasets into PyTorch DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader