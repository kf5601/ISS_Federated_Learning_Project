# Author: Kai Fan kf5601 
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/dataset.py

# Required imports
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FEMNISTClientDataset(Dataset):
    """FEMNIST dataset belonging to a single federated client."""

    def __init__(self, root_dir: str, client_id: int):
        self.client_dir = Path(root_dir) / f"client_{client_id}"

        if not self.client_dir.exists():
            raise FileNotFoundError(
                f"Client directory not found: {self.client_dir}"
            )

        self.samples = []

        for label in range(10):
            label_dir = self.client_dir / str(label)

            for image_path in sorted(label_dir.glob("*.png")):
                self.samples.append((image_path, label))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path)
        image = self.transform(image)

        return image, label


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