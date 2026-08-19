# Section 3 — Transformer Language Model Architecture

Implementation: `cs336_basics/model.py` (Linear, Embedding, RMSNorm, silu/SwiGLU,
RotaryPositionalEmbedding, softmax, scaled_dot_product_attention, MultiHeadSelfAttention,
TransformerBlock, TransformerLM). Tests: `tests/test_model.py` (13) + `test_softmax_matches_pytorch`
— 14 passed. Accounting numbers below are produced by `scripts/flops_accounting.py`
(`.venv/bin/python scripts/flops_accounting.py --md`); the GPT-2 XL parameter count was also
cross-checked against `sum(p.numel() for p in TransformerLM(...).parameters())` on a meta device.

## Implementation notes

- **Linear** stores `W ∈ R^{d_out×d_in}` and computes `einsum(x, W, "... d_in, d_out d_in -> ... d_out")`; init
  `trunc_normal(0, σ² = 2/(d_in+d_out))` clipped at ±3σ. **Embedding** `N(0,1)` clipped at ±3, lookup by
  advanced indexing. **RMSNorm** upcasts to fp32, `x / sqrt(mean(x²)+eps) * g`, casts back.
- **SwiGLU** = `W2 (SiLU(W1 x) ⊙ W3 x)`; `d_ff` defaults to `8/3·d_model` rounded to a multiple of 64.
- **RoPE** rotates adjacent pairs `(x_{2k-1}, x_{2k})` by `θ_{i,k} = i·Θ^{-(2k-2)/d}`; `cos`/`sin` tables of
  shape `(max_seq_len, d_k/2)` are non-persistent buffers, indexed by `token_positions`; one module is shared by
  all layers of the LM. Missing leading dims of `token_positions` (e.g. the head dim) are broadcast.
- **softmax** subtracts the max along `dim`. **SDPA** builds scores with einsum over arbitrary leading dims,
  `masked_fill(~mask, -inf)`, softmax, einsum with V. **MHSA** does one matmul each for Q/K/V (all heads at
  once), splits heads into a batch dim with `rearrange`, applies RoPE to Q and K only, uses a `tril` causal mask.
- **TransformerBlock**: `x + Attn(RMSNorm(x))`, then `x + FFN(RMSNorm(x))`. **TransformerLM**:
  embedding → blocks → `ln_final` → untied `lm_head` → logits. Parameter names match the reference state
  dict (`token_embeddings`, `layers.i.attn.{q,k,v,output}_proj`, `ln1/ln2`, `ffn.w1/w2/w3`, `ln_final`,
  `lm_head`), so the adapters call `load_state_dict` directly.

- **Section-7 ablation switches** on `TransformerBlock` / `TransformerLM` (defaults = standard architecture,
  the 14 tests above still pass): `norm="rmsnorm"|"none"`, `norm_position="pre"|"post"`
  (post: `z = Norm(x + MHA(x)); y = Norm(z + FFN(z))`, and `ln_final` is dropped since every block already
  ends with a norm), `use_rope=True|False` (False = NoPE), `ffn_type="swiglu"|"silu"`
  (`FFN(x) = W2 SiLU(W1 x)`; `d_ff` defaults to `4·d_model` when not given).

## Problem (transformer_accounting)

Architecture as implemented in this assignment: no biases, SwiGLU with three matrices, RoPE has no
parameters, RMSNorm has one gain vector, LM head not tied to the embedding. Only matmuls are counted for
FLOPs, using the rule `(m×n)@(n×p) = 2mnp`.

### (a) GPT-2 XL parameters and memory

Per layer: `4·d²` (Q,K,V,O) + `3·d·d_ff` (W1,W2,W3) + `2·d` (two RMSNorm gains)
= 10,240,000 + 20,582,400 + 3,200 = 30,825,600. Plus token embedding `V·d` = 80,411,200, `ln_final` 1,600,
`lm_head` `V·d` = 80,411,200.

**Total = 48·30,825,600 + 2·80,411,200 + 1,600 = 1,640,452,800 ≈ 1.64 B parameters.** In fp32 that is
`4 × 1,640,452,800 = 6,561,811,200 bytes ≈ 6.56 GB (6.11 GiB)` just to hold the weights (no
gradients/optimizer state/activations).

### (b) Matrix multiplies in one forward pass (T = 1024 tokens)

| matmul | shape | FLOPs | share |
|---|---|---:|---:|
| attn: QKV projections | 48 layers × 3 × [(1024×1600)@(1600×1600)] | 7.550e+11 | 21.5% |
| attn: Q Kᵀ scores | 48 layers × 25 heads × [(1024×64)@(64×1024)] | 1.611e+11 | 4.6% |
| attn: P V | 48 layers × 25 heads × [(1024×1024)@(1024×64)] | 1.611e+11 | 4.6% |
| attn: output projection | 48 layers × (1024×1600)@(1600×1600) | 2.517e+11 | 7.2% |
| ffn: W1 x and W3 x | 48 layers × 2 × [(1024×1600)@(1600×4288)] | 1.349e+12 | 38.4% |
| ffn: W2 (·) | 48 layers × (1024×4288)@(4288×1600) | 6.744e+11 | 19.2% |
| lm_head | (1024×1600)@(1600×50257) | 1.647e+11 | 4.7% |

Closed form: per layer `2T·(4d² + 3d·d_ff) + 4T²d`, plus `2T·d·V` for the LM head.

**Total ≈ 3.517 × 10¹² FLOPs (3,516,769,894,400) per forward pass.**

### (c) Most expensive parts

The SwiGLU feed-forward network dominates (57.5 % of all FLOPs; W1/W3 alone are 38 %), followed by the
attention Q/K/V/output projections (28.6 %). The actual attention score computation (QKᵀ and PV) is only
9.2 % at context 1024, and the LM head 4.7 %.

### (d) Other GPT-2 sizes

`d_ff` = 8/3·d_model rounded to a multiple of 64: small 2048, medium 2752, large 3392, XL 4288.

| model | params | total FLOPs | attn projections (QKV+out) | attn scores (QKᵀ+PV) | feed-forward (SwiGLU) | lm_head |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 small (12L, 768, 12h) | 0.162 B | 2.916e+11 | 19.9% | 13.3% | 39.8% | 27.1% |
| GPT-2 medium (24L, 1024, 16h) | 0.407 B | 8.302e+11 | 24.8% | 12.4% | 50.1% | 12.7% |
| GPT-2 large (36L, 1280, 20h) | 0.834 B | 1.769e+12 | 27.3% | 10.9% | 54.3% | 7.4% |
| GPT-2 XL (48L, 1600, 25h) | 1.640 B | 3.517e+12 | 28.6% | 9.2% | 57.5% | 4.7% |

Per-matmul breakdowns for every model are printed by the script. As the model grows, the layer-internal
matmuls that scale as `L·d²` (FFN and attention projections) take a proportionally larger share, while the
LM head (`d·V`, one copy, grows only linearly in `d`) drops from 27 % to under 5 %, and the attention-score
matmuls (`L·T²·d`, linear in `d`) shrink from 13 % to 9 % because at fixed `T=1024` they grow slower than the
`d²` terms.

### (e) GPT-2 XL with context length 16,384

Total forward FLOPs go from 3.52e12 to 1.336e14 — a 38× increase for a 16× longer sequence. Every matmul
except QKᵀ/PV is linear in T (16×), whereas the attention-score matmuls are quadratic (256×), so they go from
9.2 % to 61.7 % of the total and become the dominant cost; the FFN drops to 24.2 %, projections to 12.1 %, LM
head to 2.0 %.
