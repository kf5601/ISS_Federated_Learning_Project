# Author: Kai Fan kf5601 
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/dataset.py

# Required imports
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class FEMNISTClientDataset(Dataset):
    """Load the FEMNIST samples belonging to one federated client."""

    def __init__(self, root_dir: str, client_id: int, poison: bool = False, flip_rate: float = 1.0):
        self.client_dir = Path(root_dir) / f"client_{client_id}"

        if not self.client_dir.exists():
            raise FileNotFoundError(
                f"Client directory not found: {self.client_dir}"
            )

        self.samples = []

        # Each client contains directories 0-9. The directory name is
        # the class label for every PNG stored inside that directory.
        for label in range(10):
            label_dir = self.client_dir / str(label)

            if label_dir.exists():
                for image_path in sorted(label_dir.glob("*.png")):
                    # Apply static label flipping if client is poisoned
                    if poison and torch.rand(1).item() < flip_rate:
                        target_label = (label + 1) % 10
                    else:
                        target_label = label
                    self.samples.append((image_path, target_label))

        # Ensure every image has one channel and convert pixel values
        # from image values into PyTorch tensors in the range [0, 1].
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        """Return the number of samples belonging to this client."""
        return len(self.samples)

    def __getitem__(self, index):
        """Load and return one (image, label) pair."""
        image_path, label = self.samples[index]

        image = Image.open(image_path)
        image = self.transform(image)

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


# DO NOT TOUCH THIS, UNLESS DIRECTORY CHANGES. DATA EXTRACTION AND STATISTICS ANALYSIS
if __name__ == "__main__":
    root = "femnist_dataset"

    total_images = 0

    for client_id in range(20):
        dataset = FEMNISTClientDataset(root, client_id)

        class_counts = [0] * 10

        for _, label in dataset:
            class_counts[label] += 1

        total_images += len(dataset)

        print(
            f"Client {client_id:2d}: "
            f"{len(dataset):4d} images | "
            f"{class_counts}"
        )

    print(f"\nTotal images: {total_images}")