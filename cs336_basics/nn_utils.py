"""Training utilities: cross-entropy loss and gradient clipping (CS336 A1 §4.1, §4.5)."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Average cross-entropy loss  -log softmax(logits)[target].

    Args:
        logits: Float tensor of shape (..., vocab_size). Any number of leading
            batch-like dimensions; the vocabulary dimension is last.
        targets: Int tensor of shape (...) matching the leading dims of ``logits``.

    Returns:
        Scalar tensor: mean of the per-example losses over all batch dims.

    Numerics: we never materialize softmax. With m = max_a o[a],
        -log softmax(o)[y] = -(o[y] - m) + log sum_a exp(o[a] - m),
    so the exp is applied only to non-positive values (no overflow) and the
    log/exp on the target term cancel analytically.
    """
    if targets.shape != logits.shape[:-1]:
        raise ValueError(f"targets shape {tuple(targets.shape)} must equal logits.shape[:-1] {tuple(logits.shape[:-1])}")
    logits = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    max_logit = logits.amax(dim=-1, keepdim=True).detach()
    shifted = logits - max_logit  # (..., V), all <= 0
    log_norm = torch.log(torch.exp(shifted).sum(dim=-1))  # (...)
    target_logit = shifted.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)  # (...)
    losses = log_norm - target_logit
    return losses.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip the *combined* gradient of ``parameters`` to L2 norm at most ``max_l2_norm``, in place.

    Computes ||g||_2 over all parameters that have a gradient. If it exceeds
    ``max_l2_norm``, every gradient is scaled by max_l2_norm / (||g||_2 + eps).
    Parameters with ``grad is None`` are skipped.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(torch.sum(g.detach().float() ** 2) for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale.to(g.dtype))
