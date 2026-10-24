"""Train the decoder language model on token ID arrays saved with np.save."""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, clip_gradient, get_lr_consine_schdule


@dataclass
class TrainConfig:
    vocab_size: int = 10_000 # 只是为了让大数字更易读
    context_length: int = 256
    num_layers: int = 4
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    theta: float = 10_000.0
    batch_size: int = 16
    max_steps: int = 1000
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    cosine_cycle_steps: int | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 10
    checkpoint_interval: int = 500
    checkpoint_path: Path = Path("checkpoints/tinystories.pt")
    resume: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def _validation_loss(model, tokens: np.ndarray, config: TrainConfig) -> float:
    model.eval()
    losses = []
    for _ in range(config.eval_batches):
        x, y = get_batch(tokens, config.batch_size, config.context_length, config.device)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(train_tokens: np.ndarray, valid_tokens: np.ndarray, config: TrainConfig) -> int:
    """Run optimization and return the number of completed gradient steps."""
    if len(train_tokens) <= config.context_length or len(valid_tokens) <= config.context_length:
        raise ValueError("Each token array must contain more than context_length tokens")
    if config.eval_interval <= 0 or config.eval_batches <= 0 or config.checkpoint_interval <= 0:
        raise ValueError("Evaluation and checkpoint intervals and eval_batches must be positive")

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        theta=config.theta,
        device=config.device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    start_step = 0
    if config.resume:
        start_step = load_checkpoint(config.checkpoint_path, model, optimizer)

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    cycle_steps = config.cosine_cycle_steps or config.max_steps

    for step in range(start_step + 1, config.max_steps + 1):
        lr = get_lr_consine_schdule(
            step,
            config.max_lr,
            config.min_lr,
            config.warmup_steps,
            cycle_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_tokens, config.batch_size, config.context_length, config.device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, config.vocab_size), y.reshape(-1))
        loss.backward()
        clip_gradient(model.parameters(), config.max_grad_norm)
        optimizer.step()

        if step % config.eval_interval == 0 or step == config.max_steps:
            valid_loss = _validation_loss(model, valid_tokens, config)
            elapsed = time.perf_counter() - start_time
            print(
                f"step={step} elapsed_s={elapsed:.2f} train_loss={loss.item():.4f} "
                f"val_loss={valid_loss:.4f} lr={lr:.6g}",
                flush=True,
            )

        if step % config.checkpoint_interval == 0 or step == config.max_steps:
            save_checkpoint(model, optimizer, step, config.checkpoint_path)

    return config.max_steps


def main() -> None:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train a CS336 TransformerLM")
    parser.add_argument("--train-data", type=Path, default=Path("artifacts/tokenizers/tinystories_train.npy"))
    parser.add_argument("--valid-data", type=Path, default=Path("artifacts/tokenizers/tinystories_valid.npy"))
    parser.add_argument("--checkpoint", type=Path, default=defaults.checkpoint_path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--vocab-size", type=int, default=defaults.vocab_size)
    parser.add_argument("--context-length", type=int, default=defaults.context_length)
    parser.add_argument("--num-layers", type=int, default=defaults.num_layers)
    parser.add_argument("--d-model", type=int, default=defaults.d_model)
    parser.add_argument("--num-heads", type=int, default=defaults.num_heads)
    parser.add_argument("--d-ff", type=int, default=defaults.d_ff)
    parser.add_argument("--theta", type=float, default=defaults.theta)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--max-lr", type=float, default=defaults.max_lr)
    parser.add_argument("--min-lr", type=float, default=defaults.min_lr)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--cosine-cycle-steps", type=int)
    parser.add_argument("--beta1", type=float, default=defaults.beta1)
    parser.add_argument("--beta2", type=float, default=defaults.beta2)
    parser.add_argument("--eps", type=float, default=defaults.eps)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--eval-batches", type=int, default=defaults.eval_batches)
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval)
    args = parser.parse_args()

    train_tokens = np.load(args.train_data, mmap_mode="r") # 不预先全部加载，而是按需加载
    valid_tokens = np.load(args.valid_data, mmap_mode="r")
    config = TrainConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        cosine_cycle_steps=args.cosine_cycle_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
        device=args.device,
    )
    train(train_tokens, valid_tokens, config)


if __name__ == "__main__":  # python -m cs336_basics.train 使用这个命令的时候， __name__会被赋值成"__main__",别的时候，__name__ = cs336_basics.train
    main()
