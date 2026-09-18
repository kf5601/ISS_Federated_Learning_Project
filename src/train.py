"""Centralized training for the FEMNIST MLP."""

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, random_split

from dataset import FEMNISTClientDataset

# TODO: leave this for now, but will be using CNN instead of MLP, don't delete please
# from model import DigitMLP 

from model import DigitCNN


DATASET_PATH = "femnist_dataset"

# TODO: Hyperparameters to tune
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10


def load_dataset():
    """Combine all client datasets for centralized model testing."""

    clients = [
        FEMNISTClientDataset(DATASET_PATH, client_id)
        for client_id in range(20)
    ]

    dataset = ConcatDataset(clients)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    return train_dataset, test_dataset


def train(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate classification accuracy."""

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_dataset, test_dataset = load_dataset()

    print(f"Training images: {len(train_dataset)}")
    print(f"Testing images:  {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = DigitCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_accuracy = 0.0

    for epoch in range(EPOCHS):

        loss = train(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        accuracy = evaluate(
            model,
            test_loader,
            device,
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")

        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"Loss: {loss:.4f} | "
            f"Test accuracy: {accuracy * 100:.2f}%"
        )
    print(f"\nBest test accuracy: {best_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()