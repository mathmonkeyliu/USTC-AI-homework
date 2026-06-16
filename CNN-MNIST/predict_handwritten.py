import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms
import yaml

from model import get_model_by_name


def preprocess_image(path: Path, invert: bool) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if invert:
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)
    image = image.resize((28, 28))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    
    return transform(image).unsqueeze(0) # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict handwritten digit images.")
    parser.add_argument("--config", default="config/config_1_model_1.yaml")
    parser.add_argument("--image-dir", default="numbersbyHAND")
    parser.add_argument("--checkpoint", default="model/mnist_cnn_config_1_model_1.pth")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = base_dir / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    predict_config = config["predict"]

    image_dir = Path(args.image_dir or predict_config["image_dir"])
    if not image_dir.is_absolute():
        image_dir = base_dir / image_dir

    checkpoint_path = Path(args.checkpoint or predict_config["checkpoint"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = base_dir / checkpoint_path
    image_paths = sorted(
        [
            *image_dir.glob("*.png"),
            *image_dir.glob("*.jpg"),
            *image_dir.glob("*.jpeg"),
        ]
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    if not image_paths:
        raise FileNotFoundError(f"No png/jpg images found in: {image_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = dict(config["model"])
    model_name = model_config.pop("model_name", "MNISTCNN1")
    model = get_model_by_name(model_name)(**model_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    for image_path in image_paths:
        image = preprocess_image(image_path, invert=predict_config["invert"]).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(image), dim=1)
            confidence, prediction = probabilities.max(dim=1)

        print(f"{image_path.name}: {prediction.item()}  confidence={confidence.item():.4f}")


if __name__ == "__main__":
    main()
