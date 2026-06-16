import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    log_interval: int,
) -> tuple[float, float]:
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()

        # Log per-batch loss at a fixed interval for TensorBoard
        step = (epoch - 1) * len(loader) + batch_idx
        if batch_idx % log_interval == 0:
            writer.add_scalar("loss/train_batch", loss.item(), step)

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) # type: ignore
    writer.add_scalar("loss/train_epoch", avg_loss, epoch)
    writer.add_scalar("accuracy/train", accuracy, epoch)
    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on a dataset without gradient updates."""
    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset) # type: ignore
    return avg_loss, accuracy