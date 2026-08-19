"""Text generation / decoding (assignment §6).

One decoding step takes the prefix x_{1..t}, runs the LM, keeps the last row of
logits v = TransformerLM(x)_t and samples

    P(x_{t+1} = i) = softmax(v / tau)_i                     (temperature, eq. 23)

optionally restricted to the nucleus V(p) — the smallest set of tokens whose
probability mass is >= p — and renormalised over it (top-p, eq. 24).  Sampling
repeats until ``<|endoftext|>`` or ``max_new_tokens``.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class _TokenizerLike(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


EOT = "<|endoftext|>"


def softmax_with_temperature(logits: Tensor, temperature: float = 1.0, dim: int = -1) -> Tensor:
    """Numerically stable softmax(v / tau).  ``temperature`` must be > 0."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use temperature=0 via greedy path in sample_next_token)")
    z = logits.float() / temperature
    z = z - z.amax(dim=dim, keepdim=True)
    e = torch.exp(z)
    return e / e.sum(dim=dim, keepdim=True)


def top_p_filter(probs: Tensor, top_p: float) -> Tensor:
    """Zero out everything outside the nucleus V(p) and renormalise (eq. 24).

    ``probs`` is ``(..., vocab)`` and already a probability distribution.  Tokens are
    sorted by decreasing probability; the kept set is the shortest prefix whose
    cumulative mass reaches ``top_p`` (the token that crosses the threshold is kept).
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_p >= 1.0:
        return probs
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cum = torch.cumsum(sorted_probs, dim=-1)
    # Drop token j if the mass *before* it already reached p (so the crossing token stays).
    drop = (cum - sorted_probs) >= top_p
    drop[..., 0] = False  # always keep the most likely token
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)


def sample_next_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token id per row from ``logits`` ``(..., vocab)``.

    ``temperature == 0`` means greedy (argmax), matching the tau -> 0 limit of eq. 23.
    Returns an int64 tensor of shape ``logits.shape[:-1]``.
    """
    if temperature == 0:
        return logits.argmax(dim=-1)
    probs = softmax_with_temperature(logits, temperature)
    probs = top_p_filter(probs, top_p)
    flat = probs.reshape(-1, probs.shape[-1])
    nxt = torch.multinomial(flat, num_samples=1, generator=generator)
    return nxt.reshape(probs.shape[:-1])


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: list[int] | Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
    device: str | torch.device | None = None,
    context_length: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """Autoregressively sample a continuation of ``prompt_ids``.

    Returns only the *newly generated* ids (the EOS token, if hit, is not included).
    ``context_length`` (default: ``model.context_length`` if the model exposes it) bounds
    the window fed to the model — the prompt is left-truncated to the last
    ``context_length`` tokens at each step.
    """
    if context_length is None:
        context_length = getattr(model, "context_length", None)
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).reshape(1, -1)
    if ids.numel() == 0:
        raise ValueError("prompt must contain at least one token")
    out: list[int] = []
    for _ in range(max_new_tokens):
        window = ids if context_length is None else ids[:, -context_length:]
        logits = model(window)  # (1, T, vocab)
        nxt = sample_next_token(logits[:, -1, :], temperature=temperature, top_p=top_p, generator=generator)
        tok = int(nxt.item())
        if eos_token_id is not None and tok == eos_token_id:
            break
        out.append(tok)
        ids = torch.cat([ids, nxt.reshape(1, 1)], dim=1)

    if was_training:
        model.train()
    return out


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: _TokenizerLike,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device | None = None,
    context_length: int | None = None,
    generator: torch.Generator | None = None,
    eos_token: str | None = EOT,
) -> str:
    """String-in / string-out wrapper around :func:`generate` (stops at ``<|endoftext|>``)."""
    prompt_ids = tokenizer.encode(prompt)
    eos_id = None
    if eos_token is not None:
        eos_ids = tokenizer.encode(eos_token)
        if len(eos_ids) == 1:
            eos_id = eos_ids[0]
    new_ids = generate(
        model,
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_id,
        device=device,
        context_length=context_length,
        generator=generator,
    )
    return tokenizer.decode(new_ids)
