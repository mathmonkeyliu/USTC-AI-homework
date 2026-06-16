import torch
from torch import nn


class MNISTCNN1(nn.Module):
    """Two-layer CNN with a single fully-connected classifier."""
    def __init__(
        self,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        hidden_size: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv2_channels * 7 * 7, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class MNISTCNN2(nn.Module):
    """Deeper CNN with batch normalization and three conv blocks."""

    def __init__(
        self,
        conv1_channels: int = 32,
        conv2_channels: int = 64,
        conv3_channels: int = 128,
        hidden_size: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(),
            nn.Conv2d(conv1_channels, conv1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(),
            nn.Conv2d(conv2_channels, conv2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(conv2_channels, conv3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv3_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv3_channels * 3 * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# Map config model names to their class definitions
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "MNISTCNN1": MNISTCNN1,
    "MNISTCNN2": MNISTCNN2,
}


def get_model_by_name(name: str) -> type[nn.Module]:
    """Look up a model class by its registry name."""
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]

