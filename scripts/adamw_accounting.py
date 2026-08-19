"""adamw_accounting: memory / FLOPs accounting for training a Transformer LM with AdamW in fp32.

Usage: .venv/bin/python scripts/adamw_accounting.py

Symbols: B=batch_size, L=context_length, N=num_layers, d=d_model, h=num_heads, V=vocab_size,
d_ff = 8/3 d (as instructed). All tensors float32 (4 bytes).
"""

from fractions import Fraction

BYTES = 4


def n_params(V, L, N, d, h):
    d_ff = Fraction(8, 3) * d
    per_layer = 4 * d * d + 3 * d * d_ff + 2 * d  # Wq,Wk,Wv,Wo + W1,W2,W3 + two RMSNorm gains
    return V * d + N * per_layer + d + d * V  # tok-emb + layers + final norm + LM head (untied)


def activations_per_example(V, L, N, d, h):
    """Floats stored per batch element for the components enumerated in the problem."""
    d_ff = Fraction(8, 3) * d
    rmsnorm = 2 * L * d
    attn = 3 * L * d + h * L * L + h * L * L + L * d + L * d  # QKV, QK^T, softmax, AV, Wo
    ffn = L * d_ff + L * d_ff + L * d_ff + L * d_ff + L * d  # W1x, W3x, SiLU, product, W2
    per_layer = rmsnorm + attn + ffn
    return N * per_layer + L * d + L * V + L * V  # final RMSNorm, logits, cross-entropy


def forward_flops_per_token(V, L, N, d, h):
    """Matmul FLOPs (2 per MAC) per token in the forward pass."""
    d_ff = Fraction(8, 3) * d
    per_layer = 2 * 3 * d * d + 2 * L * d + 2 * L * d + 2 * d * d + 2 * 3 * d * d_ff  # QKV, QK^T, AV, Wo, FFN
    return N * per_layer + 2 * d * V  # + LM head


if __name__ == "__main__":
    # GPT-2 XL
    V, L, N, d, h = 50257, 1024, 48, 1600, 25
    P = n_params(V, L, N, d, h)
    A = activations_per_example(V, L, N, d, h)
    print(f"GPT-2 XL params P = {float(P):.4e}  ({float(P)/1e9:.3f} B)")
    print(f"activations per example A = {float(A):.4e} floats")
    a = float(A) * BYTES
    b = float(4 * P) * BYTES  # params + grads + m + v
    print(f"memory(bytes) = {a:.4e} * B + {b:.4e}")
    print(f"memory(GB, 1e9) = {a/1e9:.3f} * B + {b/1e9:.3f}")
    print(f"memory(GiB)     = {a/2**30:.3f} * B + {b/2**30:.3f}")
    for name, cap in (("80 GB (1e9)", 80e9), ("80 GiB", 80 * 2**30)):
        bmax = int((cap - b) // a)
        print(f"max batch under {name}: {bmax}  (B={bmax}: {(a*bmax+b)/1e9:.2f} GB; B={bmax+1}: {(a*(bmax+1)+b)/1e9:.2f} GB)")

    # AdamW step FLOPs (per parameter: ~14 elementwise ops, see writeup)
    adamw_flops = 14 * P
    print(f"AdamW step FLOPs ~ 14P = {float(adamw_flops):.3e}")

    # (d) training time
    f_tok = forward_flops_per_token(V, L, N, d, h)
    B, steps, peak, mfu = 1024, 400_000, 495e12, 0.5
    tokens_per_step = B * L
    step_flops = 3 * float(f_tok) * tokens_per_step + float(adamw_flops)  # fwd + 2x fwd for bwd + optimizer
    total = step_flops * steps
    secs = total / (peak * mfu)
    print(f"forward FLOPs/token = {float(f_tok):.4e}; per-step (fwd+bwd+adamw) = {step_flops:.4e}")
    print(f"total FLOPs = {total:.4e}; time = {secs:.4e} s = {secs/3600:.1f} h = {secs/86400:.1f} days")
