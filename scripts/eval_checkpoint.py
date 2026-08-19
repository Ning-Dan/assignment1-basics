"""Evaluate a checkpoint on a *whole* tokenized validation set (assignment §7).

The training loop reports validation loss on a fixed random sample of batches (fast, good
for curves).  This script gives the definitive number: mean per-token cross-entropy over
non-overlapping ``context_length`` windows covering the entire validation file, i.e. the
"validation loss (per-token)" the assignment asks to be <= 1.45 on TinyStories.

Example::

    .venv/bin/python scripts/eval_checkpoint.py --checkpoint runs/ts_full/ckpt_final.pt \\
        --val-data data/ts_valid_10k.npy --batch-size 64 --dtype bf16
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import numpy as np
import torch

from cs336_basics.data import load_token_file
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.serialization import load_checkpoint, peek_checkpoint
from cs336_basics.train import _MODEL_ABLATION_KEYS, build_model


def main(argv=None):
    p = argparse.ArgumentParser(description="Full-validation-set loss of a checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-data", required=True, help="tokenized validation set (.npy or raw uint16)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-windows", type=int, default=None, help="evaluate only the first N windows (debug)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    p.add_argument("--json", action="store_true", help="print a JSON record instead of text")
    args = p.parse_args(argv)

    meta = peek_checkpoint(args.checkpoint)
    cfg = dict(meta["extra"].get("model_config") or {})
    for k, dflt in _MODEL_ABLATION_KEYS.items():
        cfg.setdefault(k, dflt)
    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    step = load_checkpoint(args.checkpoint, model, optimizer=None)
    model.eval()

    data = load_token_file(args.val_data)
    m = cfg["context_length"]
    n_windows = (len(data) - 1) // m
    if args.max_windows:
        n_windows = min(n_windows, args.max_windows)
    autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if (args.dtype == "bf16" and device.type == "cuda") else torch.autocast(device_type="cpu", enabled=False)
    total_loss, total_tokens = 0.0, 0
    t0 = time.perf_counter()
    with torch.no_grad(), autocast:
        for start in range(0, n_windows, args.batch_size):
            idx = np.arange(start, min(start + args.batch_size, n_windows))
            win = np.asarray(data[(idx[:, None] * m) + np.arange(m + 1)]).astype(np.int64)
            x = torch.from_numpy(win[:, :-1]).to(device)
            y = torch.from_numpy(win[:, 1:]).to(device)
            logits = model(x)
            loss = cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
            total_loss += float(loss) * y.numel()
            total_tokens += y.numel()
    dt = time.perf_counter() - t0
    mean = total_loss / total_tokens
    rec = {"checkpoint": args.checkpoint, "step": step, "val_tokens": total_tokens, "windows": n_windows,
           "val_loss": mean, "val_perplexity": math.exp(mean), "eval_s": dt}
    if args.json:
        print(json.dumps(rec))
    else:
        print(f"{args.checkpoint}: step {step} | val loss {mean:.4f} (ppl {math.exp(mean):.2f}) "
              f"over {total_tokens:,} tokens / {n_windows:,} windows in {dt:.1f}s", file=sys.stdout)


if __name__ == "__main__":
    main()
