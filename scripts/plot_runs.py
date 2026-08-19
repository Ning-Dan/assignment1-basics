"""Plot learning curves from `cs336_basics.train` run directories (assignment §7 experiment_log).

Each run directory holds `log.jsonl` (one JSON record per logged step: step, train_loss,
val_loss (on eval steps), elapsed_s, tokens_seen, lr, ...) and `config.json`.

Examples::

    # curves for several runs, step axis and wall-clock axis -> two PNGs
    .venv/bin/python scripts/plot_runs.py --out writeup/figures/lr_sweep \\
        --title "LR sweep (40M tokens)" runs/lr1e-4 runs/lr3e-4 lr=1e-3:runs/lr1e-3

    # summary table (markdown) for the experiment log
    .venv/bin/python scripts/plot_runs.py --table runs/*

A run may be given as ``label:path``; otherwise the label is the directory basename.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_run(path: str) -> dict[str, Any]:
    recs = []
    with open(os.path.join(path, "log.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    cfg = {}
    cfg_path = os.path.join(path, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    return {"path": path, "records": recs, "config": cfg}


def parse_run_arg(arg: str) -> tuple[str, str]:
    if ":" in arg and not os.path.exists(arg):
        label, path = arg.split(":", 1)
        return label, path
    return os.path.basename(os.path.normpath(arg)), arg


def series(recs, key, xkey):
    xs, ys = [], []
    for r in recs:
        if key in r and r[key] is not None:
            xs.append(r[xkey])
            ys.append(r[key])
    return xs, ys


def _finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    recs = run["records"]
    cfg = run["config"]
    train = [r for r in recs if "train_loss" in r]
    val = [r for r in recs if "val_loss" in r]
    last = recs[-1] if recs else {}
    fin_train = [r for r in train if _finite(r["train_loss"])]
    fin_val = [r for r in val if _finite(r["val_loss"])]
    diverged = bool(train) and not _finite(train[-1]["train_loss"])
    # smoothed final train loss: mean over the last 10 logged train records
    tail = [r["train_loss"] for r in fin_train[-10:]]
    return {
        "name": cfg.get("run_name") or os.path.basename(os.path.normpath(run["path"])),
        "steps": last.get("step"),
        "tokens": last.get("tokens_seen"),
        "wallclock_min": (last.get("elapsed_s") or 0) / 60,
        "final_train_loss": (sum(tail) / len(tail)) if tail else float("nan"),
        "final_val_loss": fin_val[-1]["val_loss"] if fin_val else float("nan"),
        "min_val_loss": min(r["val_loss"] for r in fin_val) if fin_val else float("nan"),
        "diverged": diverged,
        "first_nonfinite_step": next((r["step"] for r in train if not _finite(r["train_loss"])), None),
        "tokens_per_s": (sorted(r["tokens_per_s"] for r in train if _finite(r.get("tokens_per_s")))[len(train) // 2]
                         if train else float("nan")),
        "n_params": cfg.get("n_params"),
        "lr": cfg.get("lr"),
        "batch_size": cfg.get("batch_size"),
    }


def plot(runs: list[tuple[str, dict]], out: str, title: str, xkey: str, xlabel: str,
         ymax: float | None, ymin: float | None, show_train: bool, logx: bool) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, run) in enumerate(runs):
        c = colors[i % len(colors)]
        recs = run["records"]
        xs, ys = series(recs, "val_loss", xkey)
        if xkey == "elapsed_s":
            xs = [x / 60 for x in xs]
        if xs:
            ax.plot(xs, ys, "-o", ms=3, color=c, label=f"{label} (val)")
        if show_train:
            xt, yt = series(recs, "train_loss", xkey)
            if xkey == "elapsed_s":
                xt = [x / 60 for x in xt]
            yt = [y if _finite(y) else float("nan") for y in yt]
            ax.plot(xt, yt, "-", lw=0.8, alpha=0.35, color=c, label=f"{label} (train)" if not xs else None)
        # mark divergence
        s = summarize(run)
        if s["diverged"] and s["first_nonfinite_step"] is not None:
            xd = s["first_nonfinite_step"]
            if xkey == "elapsed_s":
                r = next(r for r in recs if r["step"] == xd)
                xd = r["elapsed_s"] / 60
            ax.axvline(xd, color=c, ls=":", lw=1)
            ax.text(xd, ax.get_ylim()[1] if ymax is None else ymax, f" {label} NaN", color=c, fontsize=7, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cross-entropy loss (nats/token)")
    ax.set_title(title)
    if ymax is not None or ymin is not None:
        ax.set_ylim(ymin, ymax)
    if logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


_BASELINE = {  # assignment §7.2 baseline; the table's "variant" column lists deviations from it
    "vocab_size": 10000, "context_length": 256, "d_model": 512, "num_layers": 4, "num_heads": 16, "d_ff": 1344,
    "rope_theta": 10000.0, "no_rmsnorm": False, "post_norm": False, "no_rope": False, "ffn": "swiglu",
    "tie_embeddings": False, "beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.1, "grad_clip": 1.0,
    "dtype": "bf16", "compile": True,
}


def variant(cfg: dict[str, Any]) -> str:
    diffs = []
    for k, v in _BASELINE.items():
        if k in cfg and cfg[k] != v:
            diffs.append(f"{k}={cfg[k]}")
    if cfg.get("train_data", "").find("owt") >= 0:
        diffs.append("data=OWT")
    return ", ".join(diffs) or "baseline"


def table(runs: list[tuple[str, dict]]) -> str:
    hdr = ("| run | variant | lr | batch | steps | tokens | wall-clock (min) | tok/s | "
           "final train loss | final val loss | min val loss | note |")
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    rows = [hdr, sep]
    for label, run in runs:
        s = summarize(run)
        note = f"diverged (NaN at step {s['first_nonfinite_step']})" if s["diverged"] else ""
        rows.append(
            f"| {label} | {variant(run['config'])} | {s['lr']} | {s['batch_size']} | {s['steps']} | {s['tokens']:,} | "
            f"{s['wallclock_min']:.1f} | {s['tokens_per_s']/1e3:.0f}k | {s['final_train_loss']:.3f} | "
            f"{s['final_val_loss']:.3f} | {s['min_val_loss']:.3f} | {note} |"
        )
    return "\n".join(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run dirs, optionally label:path")
    p.add_argument("--out", help="output prefix; writes <out>_steps.png and <out>_wallclock.png")
    p.add_argument("--title", default="")
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--no-train", action="store_true", help="plot only validation loss")
    p.add_argument("--logx", action="store_true")
    p.add_argument("--table", action="store_true", help="print a markdown summary table")
    args = p.parse_args(argv)

    runs = []
    for a in args.runs:
        label, path = parse_run_arg(a)
        runs.append((label, load_run(path)))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        print(plot(runs, args.out + "_steps.png", args.title, "step", "gradient step",
                   args.ymax, args.ymin, not args.no_train, args.logx))
        print(plot(runs, args.out + "_wallclock.png", args.title, "elapsed_s", "wall-clock time (min)",
                   args.ymax, args.ymin, not args.no_train, args.logx))
    if args.table or not args.out:
        print(table(runs))


if __name__ == "__main__":
    main()
