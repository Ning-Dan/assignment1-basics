# Experiment log (assignment §7)

One row per training run.  All runs: RTX 5080 (shared, ≈6 GB VRAM cap, one run at a
time), `bf16` autocast + TF32 + `torch.compile`, `python -m cs336_basics.train`
(§5 CLI; the SDPA fp32-softmax fix in `model.py` is required for compiled bf16 — see
`section7_experiments.md`).  Unless the *variant* column says otherwise a run uses the
§7.2 baseline: TinyStories 10k vocab, ctx 256, d_model 512, d_ff 1344, 4 layers, 16
heads, RoPE Θ=1e4, AdamW β=(0.9,0.95) ε=1e-8 wd 0.1, grad-clip 1.0, warmup 5% of steps
+ cosine to lr/10 at the last step, batch 32, seed 0.  *final val loss* = mean CE on
the 40 fixed validation batches (327,680 tokens; `--eval-batches` scaled with the batch
so the token count is the same for every run) at the last step; *min val loss* = best
evaluation during the run; *final train loss* = mean of the last 10 logged train
losses; *tok/s* = median training throughput (excludes eval time; wall-clock includes
it).  Full-validation-set numbers for the headline checkpoints (`scripts/eval_checkpoint.py`,
5,465,856 tokens):

| checkpoint | val loss (40 fixed batches) | val loss (full validation set) |
|---|---|---|
| `ts_full_lr2e-3/ckpt_final.pt` (best TinyStories model) | 1.3796 | **1.3909** (ppl 4.02) |
| `ts_full_lr3e-3/ckpt_final.pt` | 1.3945 | 1.4066 (ppl 4.08) |
| `lr_sweep/lr3e-3/ckpt_final.pt` (40.96M tokens) | 1.6485 | 1.6597 (ppl 5.26) |
| `owt/owt_full_lr3e-3/ckpt_final.pt` | 4.1651 | 4.1572 (ppl 63.9; first 25.6M of 66.4M val tokens) |
| `leaderboard/lb_L4_d512/ckpt_final.pt` | 4.1398 | 4.1310 (ppl 62.2; first 25.6M val tokens) |

Regenerate the table / figures: `scripts/plot_runs.py --table <run dirs>` and
`scripts/plot_runs.py --out writeup/figures/<name> <label:run dir> …` (all figure commands: `scripts/sec7_make_figures.sh`; table assembly: run-workspace `make_explog.sh`).  Run directories
(`log.jsonl`, `config.json`, `stdout.log`, checkpoints) live in
`/home/c12/workspace/claude_workspace/cs336_a1/sec7-exp/runs/<group>/<run>` and are not
committed (≈270 MB per checkpoint).

Figures (each exists as `_steps.png` and `_wallclock.png` in `writeup/figures/`):
`lr_sweep`, `lr_divergence`, `ts_full`, `batch_size`, `ablation_rmsnorm`,
`ablation_postnorm`, `ablation_nope`, `ablation_swiglu`, `owt`, `lb_probe`, `leaderboard`.

## Runs

### Learning-rate sweep and divergence (§learning_rate) — 40.96M tokens

| run | variant | lr | batch | steps | tokens | wall-clock (min) | tok/s | final train loss | final val loss | min val loss | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lr1e-4 | baseline | 0.0001 | 32 | 5000 | 40,960,000 | 2.7 | 283k | 2.287 | 2.291 | 2.291 |  |
| lr3e-4 | baseline | 0.0003 | 32 | 5000 | 40,960,000 | 2.3 | 335k | 1.867 | 1.876 | 1.876 |  |
| lr1e-3 | baseline | 0.001 | 32 | 5000 | 40,960,000 | 2.4 | 321k | 1.660 | 1.669 | 1.669 |  |
| lr2e-3 | baseline | 0.002 | 32 | 5000 | 40,960,000 | 2.6 | 291k | 1.631 | 1.639 | 1.639 |  |
| lr3e-3 | baseline | 0.003 | 32 | 5000 | 40,960,000 | 2.7 | 289k | 1.642 | 1.648 | 1.648 |  |
| lr5e-3 | baseline | 0.005 | 32 | 5000 | 40,960,000 | 2.8 | 283k | 1.720 | 1.727 | 1.727 |  |
| lr1e-2 | baseline | 0.01 | 32 | 5000 | 40,960,000 | 2.7 | 326k | 1.794 | 1.803 | 1.803 |  |
| lr3e-2 | baseline | 0.03 | 32 | 5000 | 40,960,000 | 2.6 | 323k | 2.374 | 2.359 | 2.359 |  |
| lr1e-1 | baseline | 0.1 | 32 | 5000 | 40,960,000 | 2.8 | 282k | 3.091 | 3.049 | 3.049 |  |
| lr3e-2_noclip | grad_clip=0.0 | 0.03 | 32 | 5000 | 40,960,000 | 2.2 | 343k | 2.393 | 2.378 | 2.378 |  |
| lr1e-1_noclip | grad_clip=0.0 | 0.1 | 32 | 5000 | 40,960,000 | 2.2 | 348k | 3.062 | 3.020 | 3.020 |  |

Notes: 3e-2 / 1e-1 (with or without clipping) never NaN but the loss blows up to 4.6–5.7 during warmup and only partly recovers (see figures/lr_divergence). Warmup 250 steps for all short runs.

### Full-budget TinyStories runs (327.68M tokens) and second seeds

| run | variant | lr | batch | steps | tokens | wall-clock (min) | tok/s | final train loss | final val loss | min val loss | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ts_full_lr3e-3 | baseline | 0.003 | 32 | 40000 | 327,680,000 | 17.4 | 334k | 1.367 | 1.394 | 1.394 |  |
| ts_full_lr2e-3 | baseline | 0.002 | 32 | 40000 | 327,680,000 | 17.1 | 331k | 1.355 | 1.380 | 1.380 |  |
| lr3e-3_seed1 | baseline | 0.003 | 32 | 5000 | 40,960,000 | 3.8 | 158k | 1.625 | 1.634 | 1.634 |  |
| lr2e-3_seed1 | baseline | 0.002 | 32 | 5000 | 40,960,000 | 2.2 | 336k | 1.620 | 1.628 | 1.628 |  |

Notes: full runs use warmup 2,000, eval every 1,000 steps, checkpoints every 10,000; seed-1 runs repeat lr_sweep/lr3e-3 and lr2e-3 with `--seed 1` (lr3e-3_seed1 overlapped with a validation-set evaluation on the same GPU, hence its lower tok/s / longer wall-clock).

### Ablations (§7.3) — 40.96M tokens, B=32

| run | variant | lr | batch | steps | tokens | wall-clock (min) | tok/s | final train loss | final val loss | min val loss | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no_rmsnorm_lr3e-3 | no_rmsnorm=True | 0.003 | 32 | 5000 | 40,960,000 | 2.7 | 319k | 4040.024 | nan | nan | diverged (NaN at step 220) |
| no_rmsnorm_lr1e-3 | no_rmsnorm=True | 0.001 | 32 | 5000 | 40,960,000 | 2.3 | 321k | 1.681 | 1.690 | 1.690 |  |
| no_rmsnorm_lr3e-4 | no_rmsnorm=True | 0.0003 | 32 | 5000 | 40,960,000 | 2.3 | 327k | 1.875 | 1.879 | 1.879 |  |
| post_norm_lr3e-3 | post_norm=True | 0.003 | 32 | 5000 | 40,960,000 | 2.4 | 327k | 1.695 | 1.701 | 1.701 |  |
| nope_lr3e-3 | no_rope=True | 0.003 | 32 | 5000 | 40,960,000 | 2.3 | 339k | 1.792 | 1.798 | 1.798 |  |
| silu_dff2048_lr3e-3 | d_ff=2048, ffn=silu | 0.003 | 32 | 5000 | 40,960,000 | 2.3 | 339k | 1.626 | 1.635 | 1.635 |  |
| silu_dff2048_lr3e-3_seed1 | d_ff=2048, ffn=silu | 0.003 | 32 | 5000 | 40,960,000 | 2.2 | 338k | 1.639 | 1.646 | 1.646 |  |

Baselines for these are lr_sweep/lr3e-3 (1.648), lr1e-3 (1.669), lr3e-4 (1.876) and seeds/lr3e-3_seed1 (1.634).

### Batch size (§batch_size_experiment) — 40.96M tokens each, LR re-tuned

| run | variant | lr | batch | steps | tokens | wall-clock (min) | tok/s | final train loss | final val loss | min val loss | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bs1_lr3e-4 | baseline | 0.0003 | 1 | 160000 | 40,960,000 | 14.5 | 47k | 2.222 | 2.099 | 2.099 |  |
| bs1_lr1e-3 | baseline | 0.001 | 1 | 160000 | 40,960,000 | 13.9 | 52k | 2.312 | 2.188 | 2.188 |  |
| bs16_lr2e-3 | baseline | 0.002 | 16 | 10000 | 40,960,000 | 2.4 | 305k | 1.678 | 1.679 | 1.679 |  |
| bs16_lr3e-3 | baseline | 0.003 | 16 | 10000 | 40,960,000 | 2.5 | 304k | 1.696 | 1.695 | 1.695 |  |
| lr2e-3 | baseline | 0.002 | 32 | 5000 | 40,960,000 | 2.6 | 291k | 1.631 | 1.639 | 1.639 |  |
| lr3e-3 | baseline | 0.003 | 32 | 5000 | 40,960,000 | 2.7 | 289k | 1.642 | 1.648 | 1.648 |  |
| bs64_lr3e-3 | baseline | 0.003 | 64 | 2500 | 40,960,000 | 2.3 | 350k | 1.631 | 1.625 | 1.625 |  |
| bs64_lr4e-3 | baseline | 0.004 | 64 | 2500 | 40,960,000 | 2.1 | 351k | 1.644 | 1.639 | 1.639 |  |

Notes: warmup = 5% of steps; `--eval-batches 1280/B` keeps the evaluated token count constant. B=128 / 256 were not run: peak allocated memory 0.47 / 1.56 / 2.72 / 5.05 GiB for B = 1 / 16 / 32 / 64 and OOM (>8.5 GiB) at B=128 under our ≈6 GB share of the 16 GB card.

### OpenWebText (§main_experiment) — 32k vocab, same architecture

| run | variant | lr | batch | steps | tokens | wall-clock (min) | tok/s | final train loss | final val loss | min val loss | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| owt40M_lr3e-3 | vocab_size=32000, data=OWT | 0.003 | 32 | 5000 | 40,960,000 | 4.2 | 181k | 4.554 | 4.572 | 4.572 |  |
| owt40M_lr1e-3 | vocab_size=32000, data=OWT | 0.001 | 32 | 5000 | 40,960,000 | 4.0 | 182k | 4.692 | 4.708 | 4.708 |  |
| owt_full_lr3e-3 | vocab_size=32000, data=OWT | 0.003 | 32 | 40000 | 327,680,000 | 31.3 | 178k | 4.120 | 4.165 | 4.165 |  |

Notes: OWT model has 45.22M parameters (32k-vocab embedding + LM head); full-validation-set loss of owt_full_lr3e-3 = 4.157 over the first 25.6M validation tokens.

### Leaderboard probes (equal 4.5-min wall-clock, OWT, B=32, lr 3e-3) and the 45-min run

| run | variant | lr | batch | steps | tokens | wall-clock (min) | tok/s | final train loss | final val loss | min val loss | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| L4_d512 | vocab_size=32000, data=OWT | 0.003 | 32 | 4875 | 39,936,000 | 4.0 | 178k | 4.576 | 4.579 | 4.578 |  |
| L8_d512 | vocab_size=32000, num_layers=8, data=OWT | 0.003 | 32 | 3698 | 30,294,016 | 4.1 | 134k | 4.752 | 4.755 | 4.755 |  |
| L6_d768 | vocab_size=32000, d_model=768, num_layers=6, num_heads=12, d_ff=2048, data=OWT | 0.003 | 32 | 3060 | 25,067,520 | 4.0 | 112k | 5.265 | 5.311 | 5.311 |  |
| lb_L4_d512 | vocab_size=32000, data=OWT | 0.003 | 32 | 55961 | 458,432,512 | 43.1 | 181k | 4.108 | 4.140 | 4.140 |  |

Notes: probe step counts (4,875 / 3,698 / 3,060) were set from a 100-step throughput measurement so that each cosine schedule ends at ≈4 min; the leaderboard run's step count (55,961) was set the same way for 43 min, with a hard `--max-wallclock-min 44.5` stop.

### Not in the tables

Smoke / debugging runs (`smoke_tput`, `dbg_*`: throughput measurement and the compile+bf16 NaN bisection on TinyStories-valid as training data), the 100-step `lb_probe/tput_*` throughput probes, and a VRAM probe (`vram_probe.py`).
