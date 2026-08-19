#!/bin/bash
# Regenerate all §7 figures from the run logs (assignment §7).  RUNS = directory holding the run groups
# (lr_sweep/, ablation/, batch/, owt/, lb_probe/, leaderboard/, seeds/, ts_full_*), PY = a python with matplotlib.
REPO=/home/c12/Documents/doc/private/cs336/assignment1-basics
RUNS=${RUNS:-/home/c12/workspace/claude_workspace/cs336_a1/sec7-exp/runs}
PY=${PY:-/home/c12/miniconda3/bin/python3}   # any python with matplotlib
F=$REPO/writeup/figures
mkdir -p $F
cd $REPO
P="$PY scripts/plot_runs.py"
S=$RUNS/lr_sweep
$P --out $F/lr_sweep --title "LR sweep, TinyStories, 40.96M tokens (B=32, 5000 steps)" --ymax 4.0 --ymin 1.4 --no-train \
  lr=1e-4:$S/lr1e-4 lr=3e-4:$S/lr3e-4 lr=1e-3:$S/lr1e-3 lr=2e-3:$S/lr2e-3 lr=3e-3:$S/lr3e-3 lr=5e-3:$S/lr5e-3 lr=1e-2:$S/lr1e-2
$P --out $F/lr_divergence --title "Increasing LR up to divergence (40.96M tokens)" --ymax 8 --ymin 1.4 \
  lr=2e-3:$S/lr2e-3 lr=1e-2:$S/lr1e-2 lr=3e-2:$S/lr3e-2 lr=1e-1:$S/lr1e-1 $( [ -d $S/lr3e-2_noclip ] && echo "lr=3e-2,no-clip:$S/lr3e-2_noclip" ) $( [ -d $S/lr1e-1_noclip ] && echo "lr=1e-1,no-clip:$S/lr1e-1_noclip" )
$P --out $F/ts_full --title "TinyStories, full budget 327.68M tokens (B=32, 40k steps)" --ymax 3.0 --ymin 1.3 \
  lr=3e-3:$RUNS/ts_full_lr3e-3 lr=2e-3:$RUNS/ts_full_lr2e-3 "lr=2e-3,short(5k):$S/lr2e-3"
A=$RUNS/ablation
$P --out $F/ablation_rmsnorm --title "Ablation: no RMSNorm (40.96M tokens)" --ymax 4.5 --ymin 1.4 \
  baseline,lr=3e-3:$S/lr3e-3 baseline,lr=1e-3:$S/lr1e-3 no-RMSNorm,lr=3e-3:$A/no_rmsnorm_lr3e-3 no-RMSNorm,lr=1e-3:$A/no_rmsnorm_lr1e-3 no-RMSNorm,lr=3e-4:$A/no_rmsnorm_lr3e-4 baseline,lr=3e-4:$S/lr3e-4
$P --out $F/ablation_postnorm --title "Ablation: pre-norm vs post-norm (40.96M tokens, lr 3e-3)" --ymax 4 --ymin 1.4 \
  pre-norm:$S/lr3e-3 post-norm:$A/post_norm_lr3e-3
$P --out $F/ablation_nope --title "Ablation: RoPE vs NoPE (40.96M tokens, lr 3e-3)" --ymax 4 --ymin 1.4 \
  RoPE:$S/lr3e-3 NoPE:$A/nope_lr3e-3
$P --out $F/ablation_swiglu --title "Ablation: SwiGLU (d_ff 1344) vs SiLU (d_ff 2048) (40.96M tokens, lr 3e-3)" --ymax 4 --ymin 1.4 \
  SwiGLU:$S/lr3e-3 SiLU:$A/silu_dff2048_lr3e-3
B=$RUNS/batch
$P --out $F/batch_size --title "Batch size at a fixed 40.96M-token budget (best LR per batch)" --ymax 4 --ymin 1.4 --logx --no-train \
  B=1,lr=3e-4:$B/bs1_lr3e-4 B=1,lr=1e-3:$B/bs1_lr1e-3 B=16,lr=2e-3:$B/bs16_lr2e-3 B=32,lr=2e-3:$S/lr2e-3 B=64,lr=3e-3:$B/bs64_lr3e-3 B=64,lr=4e-3:$B/bs64_lr4e-3
O=$RUNS/owt
$P --out $F/owt --title "OpenWebText (32k vocab) vs TinyStories, same architecture and step counts" --ymax 7 --ymin 1.3 --no-train \
  "OWT 40.96M,lr=3e-3:$O/owt40M_lr3e-3" "OWT 40.96M,lr=1e-3:$O/owt40M_lr1e-3" "OWT 327.68M,lr=3e-3:$O/owt_full_lr3e-3" "TS 40.96M,lr=3e-3:$S/lr3e-3" "TS 327.68M,lr=3e-3:$RUNS/ts_full_lr3e-3"
if [ -d $RUNS/lb_probe/L4_d512 ]; then
$P --out $F/lb_probe --title "Leaderboard probe: 4.5 min wall-clock each, OWT, B=32, lr 3e-3" --ymax 7 --ymin 3.5 \
  "4L d512 (base):$RUNS/lb_probe/L4_d512" "8L d512:$RUNS/lb_probe/L8_d512" "6L d768:$RUNS/lb_probe/L6_d768"
fi
for d in $RUNS/leaderboard/lb_*; do [ -d $d ] && $P --out $F/leaderboard --title "Leaderboard run: OWT, 45 min wall-clock on one RTX 5080" --ymax 7 --ymin 3.0 "$(basename $d):$d" "OWT 327.68M baseline:$O/owt_full_lr3e-3"; done
SE=$RUNS/seeds
if [ -d $SE/lr3e-3_seed1 ]; then
$P --out $F/ablation_swiglu_seeds --title "SwiGLU vs SiLU, two seeds each (40.96M tokens, lr 3e-3)" --ymax 3 --ymin 1.5 --no-train \
  SwiGLU,seed0:$S/lr3e-3 SwiGLU,seed1:$SE/lr3e-3_seed1 SiLU,seed0:$A/silu_dff2048_lr3e-3 SiLU,seed1:$SE/silu_dff2048_lr3e-3_seed1
fi
