import argparse
from pathlib import Path

import yaml
from torch.utils.tensorboard import SummaryWriter

from log import get_logger, setup_logging
from train import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DQN experiments from a YAML config.")
    parser.add_argument("--config", default="config/dqn_all.yaml")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_dir / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_name = config["run_name"]
    log_path = project_dir / config["paths"]["logs"] / f"{run_name}.log"
    tensorboard_dir = project_dir / config["paths"]["runs"] / run_name / "tensorboard"

    logger = get_logger(run_name)
    setup_logging(log_path, logger)
    logger.info("Experiment started")
    logger.info(f"Config: {config_path}")
    logger.info(f"Run name: {run_name}")
    logger.info("-" * 60)

    with SummaryWriter(log_dir=str(tensorboard_dir)) as writer:
        rows, results_path = train_all(config, project_dir, logger, writer)

    print(f"Results saved to {results_path}")
    print(f"Log saved to {log_path}")
    print(f"TensorBoard runs saved to {tensorboard_dir}")
    for row in rows:
        print(f"{row['env_name']}: success={row['success_rate']:.2%}, eval_return={row['mean_eval_return']:.1f}")


if __name__ == "__main__":
    main()
