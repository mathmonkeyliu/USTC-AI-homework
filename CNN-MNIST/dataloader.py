from typing import Any
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_dataloaders(
    config: dict[str, Any]
) -> tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders from a YAML config."""
    data_config = config["data"]
    data_root = data_config["root"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
    )
    
    return train_loader, test_loader