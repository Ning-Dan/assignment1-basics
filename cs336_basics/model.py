"""Transformer language model built from scratch (CS336 assignment 1, section 3).

Everything here is implemented on top of raw torch tensor ops + ``nn.Parameter`` /
``nn.Module`` containers only: no ``torch.nn.Linear``, ``nn.Embedding``,
``nn.functional`` etc.  Parameter / submodule names follow the reference
state-dict layout used by the tests::

    token_embeddings.weight
    layers.{i}.ln1.weight
    layers.{i}.attn.{q,k,v,output}_proj.weight
    layers.{i}.ln2.weight
    layers.{i}.ffn.{w1,w2,w3}.weight
    ln_final.weight
    lm_head.weight

Notation follows the handout: a Linear stores W with shape (d_out, d_in) and
computes y = W x, i.e. ``einsum(x, W, "... d_in, d_out d_in -> ... d_out")``.
"""

from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from torch import Tensor, nn


# --------------------------------------------------------------------------- #
# Basic building blocks
# --------------------------------------------------------------------------- #
class Linear(nn.Module):
    """y = W x, no bias.  ``weight`` has shape (out_features, in_features)."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # N(0, 2/(d_in+d_out)) truncated at +-3 sigma
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


class Embedding(nn.Module):
    """Token-id -> vector lookup.  ``weight`` has shape (num_embeddings, embedding_dim)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # N(0, 1) truncated at +-3
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        # Advanced indexing broadcasts over any leading batch dims: (...,) -> (..., d)
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class RMSNorm(nn.Module):
    """RMSNorm(a)_i = a_i / RMS(a) * g_i,  RMS(a) = sqrt(mean(a^2) + eps).

    Computation is done in float32 regardless of the input dtype and cast back.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = x / rms * self.weight.to(torch.float32)
        return result.to(in_dtype)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


# --------------------------------------------------------------------------- #
# Feed-forward
# --------------------------------------------------------------------------- #
def silu(x: Tensor) -> Tensor:
    """SiLU / Swish: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def swiglu_d_ff(d_model: int, multiple_of: int = 64) -> int:
    """Canonical d_ff = 8/3 * d_model rounded to the nearest multiple of ``multiple_of``."""
    raw = 8.0 * d_model / 3.0
    return max(multiple_of, int(round(raw / multiple_of)) * multiple_of)


class SwiGLU(nn.Module):
    """FFN(x) = W2 (SiLU(W1 x) * W3 x)."""

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = swiglu_d_ff(d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    """Non-gated FFN(x) = W2 SiLU(W1 x)  (ablation; typically d_ff = 4 * d_model)."""

    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class Identity(nn.Module):
    """No-op stand-in for a norm layer (ablation: norm="none")."""

    def forward(self, x: Tensor) -> Tensor:
        return x


def make_norm(norm: str, d_model: int, eps: float, device=None, dtype=None) -> nn.Module:
    if norm == "rmsnorm":
        return RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
    if norm == "none":
        return Identity()
    raise ValueError(f"unknown norm {norm!r}; expected 'rmsnorm' or 'none'")


def make_ffn(ffn_type: str, d_model: int, d_ff: int | None, device=None, dtype=None) -> nn.Module:
    if ffn_type == "swiglu":
        return SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    if ffn_type == "silu":
        return SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
    raise ValueError(f"unknown ffn_type {ffn_type!r}; expected 'swiglu' or 'silu'")


# --------------------------------------------------------------------------- #
# Rotary positional embeddings
# --------------------------------------------------------------------------- #
class RotaryPositionalEmbedding(nn.Module):
    """RoPE: rotate adjacent pairs (x_{2k-1}, x_{2k}) of the last dim by angle
    theta_{i,k} = i / Theta^{(2k-2)/d}, i = token position.

    cos/sin tables of shape (max_seq_len, d_k // 2) are precomputed once and stored as
    non-persistent buffers, so one module can be shared by every layer.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE requires an even d_k, got {d_k}")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        # inv_freq[j] = Theta^{-2j/d},  j = 0 .. d/2-1   (k = j+1 in the handout's indexing)
        inv_freq = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = einsum(positions, inv_freq, "seq, half -> seq half")
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """x: (..., seq_len, d_k); token_positions: (..., seq_len) broadcastable to x's
        leading dims (missing leading dims — e.g. a head dim — are inserted before seq)."""
        cos = self.cos[token_positions].to(x.dtype)  # (..., seq, half)
        sin = self.sin[token_positions].to(x.dtype)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        x_pairs = rearrange(x, "... (half two) -> ... half two", two=2)
        x1, x2 = x_pairs[..., 0], x_pairs[..., 1]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rearrange(out, "... half two -> ... (half two)")

    def extra_repr(self) -> str:
        return f"theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len}"


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax along ``dim`` (subtracts the max first)."""
    x = x - x.amax(dim=dim, keepdim=True)
    e = torch.exp(x)
    return e / e.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """softmax(Q K^T / sqrt(d_k)) V with an optional boolean mask (True = may attend).

    Q: (..., queries, d_k), K: (..., keys, d_k), V: (..., keys, d_v),
    mask: (..., queries, keys) broadcastable to the score matrix.
    """
    d_k = Q.shape[-1]
    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probs = softmax(scores, dim=-1)
    return einsum(probs, V, "... q k, ... k d -> ... q d")


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, optionally with RoPE on Q and K.

    Parameters: q_proj / k_proj / v_proj (h*d_k, d_model) and output_proj (d_model, h*d_v).
    Rows of q/k/v_proj are ordered head-major, i.e. weight == cat([head_0, ..., head_{h-1}], 0).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        # RoPE: pass a shared module, or (theta, max_seq_len) to build a private one, or neither.
        if rope is None and theta is not None and max_seq_len is not None:
            rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        self.rope = rope

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """x: (..., seq_len, d_model) -> (..., seq_len, d_model)."""
        seq_len = x.shape[-2]
        # One matmul each for Q, K, V over all heads, then split heads into a batch dim.
        q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        k = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)
        v = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Causal mask: query i may attend to keys j <= i.
        causal = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask=causal)  # (..., h, s, d)
        out = rearrange(out, "... h s d -> ... s (h d)")
        return self.output_proj(out)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, num_heads={self.num_heads}, rope={self.rope is not None}"


# --------------------------------------------------------------------------- #
# Transformer block + LM
# --------------------------------------------------------------------------- #
class TransformerBlock(nn.Module):
    """Pre-norm block (default): x + Attn(RMSNorm(x)); then x + FFN(RMSNorm(x)).

    Ablation switches (section 7; defaults reproduce the standard architecture):
      norm:          "rmsnorm" | "none"   -- "none" removes ln1/ln2 entirely
      norm_position: "pre" | "post"       -- post: z = Norm(x + Attn(x)); y = Norm(z + FFN(z))
      use_rope:      True | False         -- False = no positional embedding (NoPE)
      ffn_type:      "swiglu" | "silu"    -- silu: FFN(x) = W2 SiLU(W1 x) (caller picks d_ff, e.g. 4*d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        theta: float | None = None,
        max_seq_len: int | None = None,
        eps: float = 1e-5,
        norm: str = "rmsnorm",
        norm_position: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if norm_position not in ("pre", "post"):
            raise ValueError(f"unknown norm_position {norm_position!r}; expected 'pre' or 'post'")
        self.norm_position = norm_position
        if not use_rope:
            rope, theta, max_seq_len = None, None, None
        self.ln1 = make_norm(norm, d_model, eps, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, rope=rope, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        self.ln2 = make_norm(norm, d_model, eps, device=device, dtype=dtype)
        self.ffn = make_ffn(ffn_type, d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.norm_position == "pre":
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
        else:  # post-norm
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            x = self.ln2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    """Embedding -> num_layers pre-norm blocks -> RMSNorm -> LM head (untied) -> logits.

    Ablation switches (norm / norm_position / use_rope / ffn_type) are forwarded to every
    TransformerBlock; see TransformerBlock for their meaning. Defaults = standard architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int | None = None,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        norm: str = "rmsnorm",
        norm_position: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        if d_ff is None:
            d_ff = swiglu_d_ff(d_model) if ffn_type == "swiglu" else 4 * d_model
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.norm = norm
        self.norm_position = norm_position
        self.use_rope = use_rope
        self.ffn_type = ffn_type

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        # A single RoPE table shared by every layer (buffers are non-persistent).
        rope = (
            RotaryPositionalEmbedding(rope_theta, d_model // num_heads, context_length, device=device)
            if use_rope
            else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    self.d_ff,
                    rope=rope,
                    eps=eps,
                    norm=norm,
                    norm_position=norm_position,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        # Final norm is kept for pre-norm; post-norm blocks already end with a norm, so it is skipped there.
        # norm="none" removes it as well.
        self.ln_final = (
            make_norm(norm, d_model, eps, device=device, dtype=dtype) if norm_position == "pre" else Identity()
        )
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """token_ids: (batch, seq_len) -> logits (batch, seq_len, vocab_size)."""
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds context_length {self.context_length}")
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self,
        token_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> Tensor:
        """Autoregressive decoding helper (used later in the assignment).

        token_ids: (batch, seq_len) prompt.  Returns the prompt with up to ``max_new_tokens``
        appended (stops early once every sequence has emitted ``eos_token_id``).
        """
        was_training = self.training
        self.eval()
        finished = torch.zeros(token_ids.shape[0], dtype=torch.bool, device=token_ids.device)
        for _ in range(max_new_tokens):
            ctx = token_ids[:, -self.context_length :]
            logits = self(ctx)[:, -1, :]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1)
            else:
                probs = softmax(logits / temperature, dim=-1)
                if top_p is not None and 0 < top_p < 1:
                    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    # keep the smallest prefix whose mass >= top_p (always keep the first token)
                    remove = cumulative - sorted_probs > top_p
                    sorted_probs = sorted_probs.masked_fill(remove, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    choice = torch.multinomial(sorted_probs, num_samples=1)
                    next_token = torch.gather(sorted_idx, -1, choice).squeeze(-1)
                else:
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            if eos_token_id is not None:
                next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)
                finished = finished | (next_token == eos_token_id)
            token_ids = torch.cat([token_ids, next_token[:, None]], dim=-1)
            if eos_token_id is not None and bool(finished.all()):
                break
        if was_training:
            self.train()
        return token_ids
