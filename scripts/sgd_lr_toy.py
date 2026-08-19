"""learning_rate_tuning: run the handout's SGD toy loop for lr in {1e0, 1e1, 1e2, 1e3}, 10 iterations each.

Usage: .venv/bin/python scripts/sgd_lr_toy.py
"""

import torch

from cs336_basics.optimizer import SGD


def run(lr: float, iters: int = 10, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)  # same init for every lr so the curves are comparable
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses = []
    for _ in range(iters):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses


if __name__ == "__main__":
    for lr in (1e0, 1e1, 1e2, 1e3):
        losses = run(lr)
        print(f"lr={lr:g}: " + ", ".join(f"{v:.4g}" for v in losses))
