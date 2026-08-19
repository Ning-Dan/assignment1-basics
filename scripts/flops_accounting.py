"""Parameter / FLOPs accounting for the assignment-1 Transformer LM (problem transformer_accounting).

Architecture assumptions (this assignment): no biases anywhere, SwiGLU FFN with three
matrices (W1, W3: d_ff x d_model; W2: d_model x d_ff), RoPE has no parameters, RMSNorm has one
gain vector per norm, LM head is NOT tied to the token embedding.

FLOPs rule: A (m x n) @ B (n x p) costs 2*m*n*p FLOPs.  Only matrix multiplies are counted
(softmax, RMSNorm, SiLU, RoPE and residual adds are ignored, as instructed).

Usage:  python scripts/flops_accounting.py            # tables for all models
        python scripts/flops_accounting.py --md       # same, markdown tables
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    name: str
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int


def swiglu_d_ff(d_model: int, multiple_of: int = 64) -> int:
    return int(round(8.0 * d_model / 3.0 / multiple_of)) * multiple_of


def gpt2(name: str, layers: int, d_model: int, heads: int, ctx: int = 1024) -> Config:
    return Config(name, 50_257, ctx, layers, d_model, heads, swiglu_d_ff(d_model))


MODELS = [
    gpt2("GPT-2 small", 12, 768, 12),
    gpt2("GPT-2 medium", 24, 1024, 16),
    gpt2("GPT-2 large", 36, 1280, 20),
    gpt2("GPT-2 XL", 48, 1600, 25),
    gpt2("GPT-2 XL (ctx 16384)", 48, 1600, 25, ctx=16_384),
]


def param_count(c: Config) -> dict[str, int]:
    d, f, V, L = c.d_model, c.d_ff, c.vocab_size, c.num_layers
    per_layer = {
        "attn q/k/v/output proj (4 d^2)": 4 * d * d,
        "ffn w1/w2/w3 (3 d d_ff)": 3 * d * f,
        "ln1 + ln2 (2 d)": 2 * d,
    }
    out = {
        "token_embeddings (V d)": V * d,
        **{f"{L} layers x {k}": v * L for k, v in per_layer.items()},
        "ln_final (d)": d,
        "lm_head (V d)": V * d,
    }
    out["TOTAL"] = sum(out.values())
    return out


def matmul_flops(c: Config, seq_len: int | None = None) -> list[tuple[str, str, int]]:
    """Return [(component, description, flops)] for one forward pass on `seq_len` tokens."""
    T = c.context_length if seq_len is None else seq_len
    d, f, V, L, h = c.d_model, c.d_ff, c.vocab_size, c.num_layers, c.num_heads
    dk = d // h
    per_layer = [
        ("attn: QKV projections", f"3 x [({T} x {d}) @ ({d} x {d})]", 3 * 2 * T * d * d),
        ("attn: Q K^T scores", f"{h} heads x [({T} x {dk}) @ ({dk} x {T})]", h * 2 * T * dk * T),
        ("attn: P V", f"{h} heads x [({T} x {T}) @ ({T} x {dk})]", h * 2 * T * T * dk),
        ("attn: output projection", f"({T} x {d}) @ ({d} x {d})", 2 * T * d * d),
        ("ffn: W1 x and W3 x", f"2 x [({T} x {d}) @ ({d} x {f})]", 2 * 2 * T * d * f),
        ("ffn: W2 (.)", f"({T} x {f}) @ ({f} x {d})", 2 * T * f * d),
    ]
    rows = [(k, f"{L} layers x " + desc, v * L) for k, desc, v in per_layer]
    rows.append(("lm_head", f"({T} x {d}) @ ({d} x {V})", 2 * T * d * V))
    return rows


def grouped(rows: list[tuple[str, str, int]]) -> dict[str, int]:
    g = {"attention projections (QKV+out)": 0, "attention scores (QK^T + PV)": 0, "feed-forward (SwiGLU)": 0, "lm_head": 0}
    for name, _, fl in rows:
        if name.startswith("attn:") and ("projection" in name):
            g["attention projections (QKV+out)"] += fl
        elif name.startswith("attn:"):
            g["attention scores (QK^T + PV)"] += fl
        elif name.startswith("ffn:"):
            g["feed-forward (SwiGLU)"] += fl
        else:
            g["lm_head"] += fl
    return g


def fmt(n: float) -> str:
    for unit, div in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.3f} {unit}"
    return f"{n:.0f}"


def report(c: Config, md: bool) -> None:
    print(f"\n{'##' if md else '=='} {c.name}: L={c.num_layers} d_model={c.d_model} heads={c.num_heads} "
          f"d_ff={c.d_ff} ctx={c.context_length} vocab={c.vocab_size}")
    params = param_count(c)
    total_params = params["TOTAL"]
    print(f"\nParameters: {total_params:,} ({total_params / 1e9:.4f} B); fp32 memory = "
          f"{total_params * 4:,} bytes = {total_params * 4 / 2**30:.3f} GiB ({total_params * 4 / 1e9:.3f} GB)")
    if md:
        print("\n| parameter group | count |\n|---|---:|")
        for k, v in params.items():
            print(f"| {k} | {v:,} |")
    else:
        for k, v in params.items():
            print(f"  {k:45s} {v:>16,}")

    rows = matmul_flops(c)
    total = sum(r[2] for r in rows)
    print(f"\nForward-pass matmul FLOPs on {c.context_length} tokens: {total:,} = {fmt(total)}FLOPs")
    if md:
        print("\n| matmul | shape | FLOPs | share |\n|---|---|---:|---:|")
        for name, desc, fl in rows:
            print(f"| {name} | {desc} | {fl:.3e} | {100 * fl / total:.1f}% |")
        print("\n| component | FLOPs | share |\n|---|---:|---:|")
        for k, v in grouped(rows).items():
            print(f"| {k} | {v:.3e} | {100 * v / total:.1f}% |")
    else:
        for name, desc, fl in rows:
            print(f"  {name:28s} {desc:60s} {fl:>12.3e} {100 * fl / total:6.1f}%")
        for k, v in grouped(rows).items():
            print(f"  {k:45s} {v:>12.3e} {100 * v / total:6.1f}%")


def summary_table(models: list[Config], md: bool) -> None:
    keys = list(grouped(matmul_flops(models[0])).keys())
    print("\n" + ("### " if md else "== ") + "Share of forward FLOPs per component")
    print("| model | params | total FLOPs | " + " | ".join(keys) + " |")
    print("|---|---:|---:|" + "---:|" * len(keys))
    for c in models:
        rows = matmul_flops(c)
        total = sum(r[2] for r in rows)
        g = grouped(rows)
        cells = " | ".join(f"{100 * g[k] / total:.1f}%" for k in keys)
        print(f"| {c.name} | {param_count(c)['TOTAL'] / 1e9:.3f} B | {total:.3e} | {cells} |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="markdown output")
    args = ap.parse_args()
    for c in MODELS:
        report(c, args.md)
    summary_table(MODELS, args.md)
