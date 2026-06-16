import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import yaml

from log import setup_logging, get_model_logger
from dataloader import build_dataloaders
from train import train_one_epoch, evaluate
from model import get_model_by_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_1_model_1.yaml")
    args = parser.parse_args()

    config_path = args.config

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Separate model name from constructor kwargs
    model_config = dict(config["model"])
    model_name = model_config.pop("model_name", "MNISTCNN1")

    logger = get_model_logger(model_name)
    log_path = Path(config["train"]["log_path"])
    setup_logging(log_path, logger)

    logger.info(f"Experiment started")
    logger.info(f"Model: {model_name}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Log file: {log_path}")
    logger.info("-" * 60)

    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = build_dataloaders(config)

    model_cls = get_model_by_name(model_name)
    model = model_cls(**model_config).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
    )

    log_dir = config["train"]["tensorboard_dir"]

    save_dir = Path(config["train"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = save_dir / config["train"]["save_name"]
    best_accuracy = 0.0

    with SummaryWriter(log_dir=log_dir) as writer:
        for epoch in range(1, config["train"]["epochs"] + 1):
            train_loss, train_accuracy = train_one_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                writer=writer,
                epoch=epoch,
                log_interval=config["train"]["log_interval"],
            )
            test_loss, test_accuracy = evaluate(model, test_loader, loss_fn, device)

            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("accuracy/test", test_accuracy, epoch)

            logger.info(
                f"Epoch {epoch:02d}/{config['train']['epochs']} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_accuracy:.4f} "
                f"test_loss={test_loss:.4f} "
                f"test_acc={test_accuracy:.4f}"
            )

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "accuracy": best_accuracy,
                    },
                    checkpoint_path,
                )

    logger.info(f"Best test accuracy: {best_accuracy:.4f}")
    logger.info(f"Model saved to: {checkpoint_path}")
    logger.info(f"TensorBoard logs: {log_dir}")


if __name__ == "__main__":
    main()
