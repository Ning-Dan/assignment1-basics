# Section 4 — Training a Transformer LM

Code: `cs336_basics/nn_utils.py` (`cross_entropy`, `gradient_clipping`), `cs336_basics/optimizer.py` (`SGD`, `AdamW`, `get_lr_cosine_schedule`). Scripts: `scripts/sgd_lr_toy.py`, `scripts/adamw_accounting.py`.

## Problem (learning_rate_tuning)

Ran the handout's toy loop (`weights = Parameter(5*randn(10,10))`, loss = `(weights**2).mean()`, decaying SGD `lr/sqrt(t+1)`) for 10 iterations with a fixed seed (`scripts/sgd_lr_toy.py`):

| lr | loss at t = 0 … 9 |
|---|---|
| 1e0 (baseline) | 26.27, 25.23, 24.52, 23.96, 23.48, 23.06, 22.69, 22.35, 22.03, 21.74 |
| 1e1 | 26.27, 16.81, 12.39, 9.697, 7.855, 6.513, 5.492, 4.693, 4.053, 3.531 |
| 1e2 | 26.27, 26.27, 4.507, 0.1079, 1.1e-16, 1.2e-18, 4.1e-20, 2.5e-21, 2.1e-22, 2.4e-23 |
| 1e3 | 26.27, 9484, 1.64e6, 1.82e8, 1.48e10, 9.32e11, 4.78e13, 2.06e15, 7.58e16, 2.44e18 |

**Answer.** lr=1e1 decays clearly faster than lr=1 (26 → 3.5 vs 26 → 21.7 in 10 steps). lr=1e2 overshoots on the first step (the update `θ ← θ − 100·(2θ/100) = −θ` leaves the loss unchanged) and then decays *extremely* fast to ~1e-23 once the `1/√(t+1)` decay shrinks the effective step below 2. lr=1e3 diverges: the loss grows by ~two orders of magnitude per step (each update multiplies θ by `1 − 20/√(t+1)`, whose magnitude stays > 1 for all 10 steps).

## Problem (adamw_accounting)

Notation: `B` = batch_size, `L` = context_length, `N` = num_layers, `d` = d_model, `h` = num_heads, `V` = vocab_size, `d_ff = 8/3·d`. Every tensor is float32 (4 bytes). Numbers computed by `scripts/adamw_accounting.py`.

### (a) Peak memory

**Parameters** (untied token embedding and LM head, RMSNorm gains, no biases, RoPE has no parameters):
- embedding `V·d`; per layer `4d²` (Q,K,V,O) + `3·d·d_ff = 8d²` (W1,W2,W3) + `2d` (two RMSNorm gains); final RMSNorm `d`; LM head `d·V`.
- `P = 2Vd + N(12d² + 2d) + d` floats → **4P bytes**.

**Activations** (only the components listed in the problem, counted as floats saved for backward, per batch element and then ×B):
- Transformer block, per layer:
  - RMSNorms: 2 × `L·d`
  - Attention: QKV projections `3·L·d`; `QKᵀ` scores `h·L²`; softmax `h·L²`; weighted sum of values `L·d`; output projection `L·d` → `5Ld + 2hL²`
  - SwiGLU FFN: `W1x` `L·d_ff`; `W3x` `L·d_ff`; SiLU `L·d_ff`; element-wise product `L·d_ff`; `W2` output `L·d` → `4L·d_ff + Ld = (35/3)Ld`
  - per layer: `(2 + 5 + 35/3)·L·d + 2hL² = (56/3)·L·d + 2hL²`
- final RMSNorm `L·d`; output embedding (logits) `L·V`; cross-entropy on logits (log-softmax / probabilities kept for backward) `L·V`.
- `A = B·[ N·((56/3)·L·d + 2hL²) + L·d + 2L·V ]` floats → **4A bytes**.

**Gradients**: one float per parameter → **4P bytes**.
**Optimizer state**: AdamW keeps `m` and `v`, each the size of the parameters → **8P bytes**.

**Total** peak memory (bytes):
`M = 4·(P + A + P + 2P) = 4·(4P + A) = 16·[2Vd + N(12d²+2d) + d] + 4B·[N((56/3)Ld + 2hL²) + Ld + 2LV]`.

### (b) GPT-2 XL (V=50257, L=1024, N=48, d=1600, h=25, d_ff=8/3·d)

- `P = 1.6355e9` (1.64 B parameters); `4P` in fp32 = 26.17 GB for params+grads+m+v.
- Activations per example: `A/B = 4.089e9` floats = 16.36 GB.
- **`memory ≈ 16.36 GB · batch_size + 26.17 GB`** (`1.6357e10·B + 2.6169e10` bytes; in GiB: `15.23·B + 24.37`).
- 80 GB budget: `B ≤ (80 − 26.17)/16.36 = 3.29` → **maximum batch size = 3** (B=3 uses 75.2 GB, B=4 needs 91.6 GB; the same B=3 holds if "80 GB" means 80 GiB).

The activation term dominates: with fp32 activations and no activation checkpointing, GPT-2 XL barely fits three 1024-token sequences on an 80 GB card even though the static state is only ~26 GB.

### (c) FLOPs of one AdamW step

AdamW is purely element-wise over the parameter vector, so cost is `Θ(P)`. Per parameter, Algorithm 1 does: weight decay `θ ← θ − αλθ` (2 ops), `m ← β₁m + (1−β₁)g` (3), `v ← β₂v + (1−β₂)g²` (4), and `θ ← θ − α_t·m/(√v + ε)` (5: sqrt, add, div, mul, sub); `α_t` is a scalar computed once. So roughly **`≈14·P` FLOPs per step** (`≈ 2.3e10` for GPT-2 XL) — a small constant times P, i.e. negligible next to the `≈6·P·(B·L)` FLOPs of forward+backward, since it does not scale with the number of tokens.

### (d) Training time for GPT-2 XL, 400K steps, batch 1024, one H100 at 50% MFU

Forward matmul FLOPs per token (2 FLOPs per multiply-add):
- per layer: QKV `6d²`, `QKᵀ` `2Ld`, `AV` `2Ld`, output proj `2d²`, SwiGLU `3·2·d·d_ff = 16d²` → `24d² + 4Ld`
- LM head `2dV`
- `F_fwd/token = N(24d² + 4Ld) + 2dV = 3.4245e9` FLOPs for GPT-2 XL.

Per step: tokens = `1024 × 1024 = 1.05e6`; forward+backward = `3 × F_fwd` (backward = 2× forward) → `3 × 3.4245e9 × 1.0486e6 = 1.077e16` FLOPs (+ `1.4e10·… ≈ 2.3e10` for AdamW, negligible).
Total for 400K steps: `4.31e21` FLOPs. Sustained throughput at 50% MFU: `0.5 × 495e12 = 2.475e14` FLOP/s.
`time = 4.31e21 / 2.475e14 = 1.74e7 s ≈ **4,840 hours ≈ 202 days**` on a single H100.

(Non-matmul FLOPs — softmax, RMSNorm, SiLU, RoPE — are ignored, as is standard for this estimate.)
