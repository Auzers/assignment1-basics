import torch
from torch import Tensor, nn

from cs336_basics.model import softmax
from cs336_basics.tokenizer import Tokenizer

@torch.no_grad()
def sample_next_token(
    logits:Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    if logits.ndim != 1:
        raise ValueError("logits must have shape (vocab_size,)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    probabilities = softmax(logits.float() / temperature, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)

    cumulative_probs = sorted_probs.cumsum(dim=-1)
    keep = cumulative_probs - sorted_probs < top_p

    filtered_probs = sorted_probs * keep
    filtered_probs = filtered_probs / filtered_probs.sum()

    sampled_rank = torch.multinomial(filtered_probs, num_samples=1).item() # 按照概率抽一个，返回下标，转成整数类型
    return sorted_ids[sampled_rank].item() # 返回被选中的下标


@torch.inference_mode()
def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    *,
    context_length: int,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    """接续 prompt，返回包含 prompt 的完整文本。"""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt must contain at least one token")

    eos_id = next(
        (token_id for token_id, token in tokenizer.vocab.items() if token == b"<|endoftext|>"),
        None,
    )
    if eos_id is None:
        raise ValueError("tokenizer vocabulary has no <|endoftext|> token")

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(
                token_ids[-context_length:],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)

            next_token_logits = model(input_ids)[0, -1, :]
            next_id = sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
            )

            if next_id == eos_id:
                break

            token_ids.append(next_id)
    finally:
        model.train(was_training)  # nn.Module.train(mode) 接受一个布尔参数：model.train(True) 等价于 model.train()，切训练模式。model.train(False) 等价于 model.eval()，切推理模式。

    return tokenizer.decode(token_ids)
    




