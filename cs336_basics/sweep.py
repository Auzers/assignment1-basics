"""Run a sequential learning-rate sweep for the CS336 TinyStories experiment."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from cs336_basics.train import TrainConfig, train


def _lr_tag(learning_rate: float) -> str:
    return (
        f"{learning_rate:.0e}"
        .replace("e-0", "e-")
        .replace("e+0", "e+")
    )


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(
        description="Run several CS336 learning-rate experiments sequentially."
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[1e-4, 3e-4, 1e-3, 3e-3],
    )
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("artifacts/tokenizers/tinystories_train.npy"),
    )
    parser.add_argument(
        "--valid-data",
        type=Path,
        default=Path("artifacts/tokenizers/tinystories_valid.npy"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=2_000)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--wandb-project", default="cs336-assignment1")
    parser.add_argument("--run-prefix", default="lr-sweep")
    args = parser.parse_args()

    train_tokens = np.load(args.train_data, mmap_mode="r")
    valid_tokens = np.load(args.valid_data, mmap_mode="r")

    base_config = TrainConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        cosine_cycle_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        device=args.device,
        wandb_project=args.wandb_project,
    )

    for learning_rate in args.learning_rates:
        if learning_rate <= 0:
            raise ValueError(f"Learning rates must be positive: {learning_rate}")

        tag = _lr_tag(learning_rate)
        config = replace(
            base_config,
            max_lr=learning_rate,
            checkpoint_path=Path(f"checkpoints/{args.run_prefix}_{tag}.pt"),
            log_path=Path(f"logs/{args.run_prefix}_{tag}.csv"),
            wandb_run_name=f"{args.run_prefix}-{tag}",
        )
        print(
            f"Starting {config.wandb_run_name}: "
            f"max_lr={learning_rate:g}, max_steps={config.max_steps}, "
            f"batch_size={config.batch_size}",
            flush=True,
        )
        train(train_tokens, valid_tokens, config)


if __name__ == "__main__":
    main()
