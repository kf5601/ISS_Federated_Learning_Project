# Author: Kai Fan kf5601 
# Author: Tyriz Newton tn1207
# Class: CSCI 532 Introduction to Intelligent Security Systems
# Professor: Dr.Leon Reznik
# File: src/train.py

# Required imports
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, random_split

from dataset import FEMNISTClientDataset
from model import DigitCNN


DATASET_PATH = "femnist_dataset"

# Centralized model-selection hyperparameters.
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10


def load_dataset():
    """
    Create a temporary centralized train/test dataset.

    All 20 federated clients are combined and split 80/20 so that the
    candidate model can be tested before implementing federated learning.
    Part 2 will instead preserve each client and create local train/evaluation
    splits for federated training.
    """

    # Load each of the 20 professor-provided client datasets.
    clients = [
        FEMNISTClientDataset(DATASET_PATH, client_id)
        for client_id in range(20)
    ]

    # Temporarily combine all clients for centralized model selection.
    dataset = ConcatDataset(clients)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # Use a fixed random seed so the same samples are assigned to the
    # train/test sets every time the experiment is run.
    generator = torch.Generator().manual_seed(42)

    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    return train_dataset, test_dataset


def train(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch and return its average batch loss."""

    # Enable training behavior such as Dropout.
    model.train()

    total_loss = 0.0

    for images, labels in dataloader:
        # Move each batch to the selected CPU/GPU.
        images = images.to(device)
        labels = labels.to(device)

        # Gradients accumulate in PyTorch, so clear the previous batch's
        # gradients before computing the next update.
        optimizer.zero_grad()

        # Forward pass: produce class scores and calculate classification loss.
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagate the loss and update the model parameters.
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate and return classification accuracy on a dataset."""

    # Switch to evaluation behavior, disabling Dropout.
    model.eval()

    correct = 0
    total = 0

    # Evaluation does not require gradients because model weights
    # are not being updated.
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            # The output contains one score for each class.
            # Select the class with the highest score as the prediction.
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total


def main():
    # Use an NVIDIA CUDA GPU when available; otherwise train on the CPU.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    train_dataset, test_dataset = load_dataset()

    print(f"Training images: {len(train_dataset)}")
    print(f"Testing images:  {len(test_dataset)}")

    # Shuffle training samples each epoch so batches are presented
    # to the model in a different order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # Testing does not update the model, so shuffling is unnecessary.
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = DigitCNN().to(device)

    # Cross-entropy loss is used for the 10-class digit classification task.
    criterion = nn.CrossEntropyLoss()

    # Adam adjusts the CNN parameters using gradients calculated
    # during backpropagation.
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

        # Preserve the weights from the best-performing epoch instead
        # of simply keeping the model from the final epoch.
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