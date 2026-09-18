"""Dataset utilities for the FEMNIST federated learning dataset."""

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


if __name__ == "__main__":
    dataset = FEMNISTClientDataset(
        root_dir="../femnist_dataset",
        client_id=0
    )
    
    print("Number of images:", len(dataset))

    image, label = dataset[0]

    print("Image shape:", image.shape)
    print("Label:", label)
    print("Min pixel value:", image.min().item())
    print("Max pixel value:", image.max().item())