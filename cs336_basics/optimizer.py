import math

import torch

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }
        super().__init__(params, defaults)

    @torch.no_grad() # 让被装饰函数内部所有张量都不构建计算图，本身就没被注册进参数，不会参与反向传播
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure() # 一般用不到

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None: # 由于某些需求，可能会把部分参数冻结
                    continue
                
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                t = state["step"]
                m = state["m"]
                v = state["v"]

                adjusted_lr = (
                    lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                )

                p *= 1 - lr * weight_decay

                m = beta1 * state["m"] + (1 - beta1) * grad
                v = beta2 * state["v"] + (1 - beta2) * grad * grad
                state["m"] = m
                state["v"] = v

                p -= adjusted_lr * m / (v**0.5 + eps)
            
        return loss
                    

def get_lr_consine_schdule(
    t,
    max_learning_rate,
    min_learing_rate,
    warmup_iters,
    cosine_cycle_iters,
) -> float:
    if t < warmup_iters:
        return max_learning_rate * t / warmup_iters
    
    if t <= cosine_cycle_iters:
        progress = (t - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine_weight = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learing_rate + cosine_weight * (max_learning_rate - min_learing_rate)
    
    return min_learing_rate
        

    

            
            

