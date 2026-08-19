"""Optimizers and LR schedule (CS336 A1 §4.2–§4.4): SGD (toy), AdamW, cosine schedule with warmup."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    """The decaying-LR SGD from the handout:  theta <- theta - lr/sqrt(t+1) * grad."""

    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):
    """AdamW (Loshchilov & Hutter 2019), Algorithm 1 of the handout.

    Per step t = 1, 2, ...:
        alpha_t = lr * sqrt(1 - beta2^t) / (1 - beta1^t)
        theta  <- theta - lr * weight_decay * theta          (decoupled weight decay, uses raw lr)
        m      <- beta1 * m + (1 - beta1) * g
        v      <- beta2 * v + (1 - beta2) * g^2
        theta  <- theta - alpha_t * m / (sqrt(v) + eps)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["t"] += 1
                t = state["t"]
                m, v = state["m"], state["v"]

                alpha_t = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)  # theta <- theta - lr*lambda*theta
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Cosine annealing with linear warmup (LLaMA-style).

    it < T_w          : alpha_max * it / T_w
    T_w <= it <= T_c  : alpha_min + 0.5*(1 + cos(pi*(it-T_w)/(T_c-T_w)))*(alpha_max-alpha_min)
    it > T_c          : alpha_min
    """
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)
    return min_learning_rate
