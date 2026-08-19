"""Training loop CLI (assignment §5.3, logging for §7).

Usage (from the repo root)::

    .venv/bin/python -m cs336_basics.train \\
        --train-data data/ts_train_10k.npy --val-data data/ts_valid_10k.npy \\
        --vocab-size 10000 --context-length 256 --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \\
        --batch-size 32 --max-steps 5000 --lr 3e-4 --warmup-iters 200 \\
        --out-dir runs/ts_base --device cuda --dtype bf16 --compile

Every ``--log-interval`` steps a JSON line is appended to ``<out-dir>/log.jsonl``
(and echoed to the console) with step, train_loss, val_loss (when evaluated), lr,
elapsed_s (wall-clock, survives ``--resume``), tokens_seen, ...; the same record goes
to Weights & Biases when ``--wandb`` is given.  Checkpoints go to
``<out-dir>/ckpt_step<N>.pt`` every ``--ckpt-interval`` steps and ``<out-dir>/ckpt_final.pt``
at the end.  ``--resume PATH`` restores model + optimizer + step + elapsed time.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
from typing import Any

import numpy as np
import torch

from cs336_basics.data import get_batch, load_token_file
from cs336_basics.serialization import load_checkpoint, save_checkpoint

# --------------------------------------------------------------------------- #
# Model construction (shared with scripts/generate.py)
# --------------------------------------------------------------------------- #

# CLI/config key -> TransformerLM constructor kwarg.  Base keys are the ones the
# assignment fixes; the ablation keys are only forwarded when non-default and
# the constructor actually accepts them (see §7 ablations).
_MODEL_BASE_KEYS = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta")
_MODEL_ABLATION_KEYS: dict[str, Any] = {
    # config key: default value.  Defaults are NOT forwarded, so a TransformerLM without a
    # given knob still constructs; a non-default value for a knob the constructor lacks is
    # a hard error (see build_model).  Names match cs336_basics.model.TransformerLM.
    "norm": "rmsnorm",  # "rmsnorm" | "none"   (--no-rmsnorm)
    "norm_position": "pre",  # "pre" | "post"   (--post-norm)
    "use_rope": True,  # False = NoPE          (--no-rope)
    "ffn_type": "swiglu",  # "swiglu" | "silu"  (--ffn)
    # TODO(model.py): not implemented there yet; reserved for a leaderboard experiment.
    "tie_embeddings": False,  # (--tie-embeddings)
}


def build_model(cfg: dict[str, Any]) -> torch.nn.Module:
    """Instantiate ``cs336_basics.model.TransformerLM`` from a flat config dict."""
    from cs336_basics.model import TransformerLM

    kwargs = {k: cfg[k] for k in _MODEL_BASE_KEYS}
    params = inspect.signature(TransformerLM.__init__).parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    for key, default in _MODEL_ABLATION_KEYS.items():
        val = cfg.get(key, default)
        if val == default:
            continue
        if key in params or accepts_var_kw:
            kwargs[key] = val
        else:
            raise SystemExit(
                f"model config asks for {key}={val!r} but TransformerLM.__init__ has no such parameter "
                f"(accepted: {sorted(params)}). Add the knob to cs336_basics/model.py first."
            )
    return TransformerLM(**kwargs)


def model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "norm": "none" if args.no_rmsnorm else "rmsnorm",
        "norm_position": "post" if args.post_norm else "pre",
        "use_rope": not args.no_rope,
        "ffn_type": args.ffn,
        "tie_embeddings": args.tie_embeddings,
    }


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m cs336_basics.train",
        description="Train a Transformer LM (CS336 assignment 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = p.add_argument_group("data")
    d.add_argument("--train-data", required=True, help="tokenized train set (.npy from np.save, or raw uint16 file)")
    d.add_argument("--val-data", default=None, help="tokenized validation set; omit to skip evaluation")
    d.add_argument("--data-dtype", default="uint16", help="dtype for raw (non-.npy) token files")

    m = p.add_argument_group("model")
    m.add_argument("--vocab-size", type=int, default=10000, help="tokenizer vocab size")
    m.add_argument("--context-length", type=int, default=256, help="sequence length per example")
    m.add_argument("--d-model", type=int, default=512, help="residual stream width")
    m.add_argument("--num-layers", type=int, default=4, help="number of Transformer blocks")
    m.add_argument("--num-heads", type=int, default=16, help="attention heads (d_model %% num_heads == 0)")
    m.add_argument("--d-ff", type=int, default=1344, help="FFN inner width (use 4*d_model with --ffn silu)")
    m.add_argument("--rope-theta", type=float, default=10000.0, help="RoPE base Theta")
    a = p.add_argument_group("ablations (§7.3; need matching knobs in model.py)")
    a.add_argument("--no-rmsnorm", action="store_true", help="remove all RMSNorms (layer_norm_ablation)")
    a.add_argument("--post-norm", action="store_true", help="post-norm blocks instead of pre-norm (pre_norm_ablation)")
    a.add_argument("--no-rope", action="store_true", help="NoPE: no positional information (no_pos_emb)")
    a.add_argument("--ffn", choices=("swiglu", "silu"), default="swiglu", help="FFN type (swiglu_ablation)")
    a.add_argument("--tie-embeddings", action="store_true", help="tie input embedding and LM head (leaderboard idea)")

    o = p.add_argument_group("optimizer / schedule")
    o.add_argument("--lr", type=float, default=3e-4, help="peak learning rate (alpha_max)")
    o.add_argument("--min-lr", type=float, default=None, help="final learning rate (alpha_min); default lr/10")
    o.add_argument("--warmup-iters", type=int, default=200, help="linear warmup steps (T_w)")
    o.add_argument("--cosine-iters", type=int, default=None, help="step at which cosine decay ends; default max-steps")
    o.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    o.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    o.add_argument("--eps", type=float, default=1e-8, help="AdamW epsilon")
    o.add_argument("--weight-decay", type=float, default=0.1, help="AdamW decoupled weight decay")
    o.add_argument("--grad-clip", type=float, default=1.0, help="max global grad L2 norm; <=0 disables")

    t = p.add_argument_group("training")
    t.add_argument("--batch-size", type=int, default=32, help="sequences per step")
    t.add_argument("--max-steps", type=int, default=None, help="number of optimizer steps (or use --total-tokens)")
    t.add_argument("--total-tokens", type=int, default=None, help="alternative to --max-steps: batch*ctx*steps")
    t.add_argument("--max-wallclock-min", type=float, default=None, help="stop (and checkpoint) after this many minutes")
    t.add_argument("--seed", type=int, default=0, help="torch/numpy seed")
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu | cuda | cuda:N | mps")
    t.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32", help="bf16 = autocast on cuda")
    t.add_argument("--compile", action="store_true", help="torch.compile the model")
    t.add_argument("--tf32", action="store_true", help="torch.set_float32_matmul_precision('high') (cuda only)")
    t.add_argument("--overfit-one-batch", action="store_true", help="debug: reuse the first batch every step")

    e = p.add_argument_group("eval / logging / checkpoints")
    e.add_argument("--out-dir", required=True, help="run directory (log.jsonl, checkpoints, config.json)")
    e.add_argument("--run-name", default=None, help="name for logs / wandb; default basename of --out-dir")
    e.add_argument("--log-interval", type=int, default=10, help="log train loss every N steps")
    e.add_argument("--eval-interval", type=int, default=200, help="evaluate val loss every N steps")
    e.add_argument("--eval-batches", type=int, default=20, help="fixed random val batches per evaluation")
    e.add_argument("--eval-seed", type=int, default=1234, help="seed for the (fixed) validation batches")
    e.add_argument("--ckpt-interval", type=int, default=1000, help="save every N steps; <=0 = only final")
    e.add_argument("--resume", default=None, help="checkpoint path to resume from")
    e.add_argument("--wandb", action="store_true", help="also log to Weights & Biases")
    e.add_argument("--wandb-project", default="cs336-a1", help="W&B project name")
    return p


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class JsonlLogger:
    def __init__(self, path: str, use_wandb: bool = False, wandb_run=None):
        self.path = path
        self.f = open(path, "a", buffering=1)
        self.use_wandb = use_wandb
        self.wandb_run = wandb_run

    def log(self, rec: dict[str, Any]) -> None:
        self.f.write(json.dumps(rec) + "\n")
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.log({k: v for k, v in rec.items() if isinstance(v, (int, float))}, step=rec.get("step"))

    def close(self) -> None:
        self.f.close()
        if self.use_wandb and self.wandb_run is not None:
            self.wandb_run.finish()


def _grad_global_norm(params) -> float:
    sq = 0.0
    for p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(sq)


@torch.no_grad()
def evaluate(
    model, val_data, batch_size, context_length, device, num_batches, seed, autocast_ctx, cross_entropy
) -> float:
    model.eval()
    rng = np.random.default_rng(seed)  # same batches every time -> comparable curves
    total = 0.0
    for _ in range(num_batches):
        x, y = get_batch(val_data, batch_size, context_length, device, generator=rng)
        with autocast_ctx():
            logits = model(x)
        total += float(cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)))
    model.train()
    return total / num_batches


def _fmt(rec: dict[str, Any]) -> str:
    parts = [f"step {rec['step']:>6d}"]
    if "train_loss" in rec:
        parts.append(f"loss {rec['train_loss']:.4f}")
    if "val_loss" in rec:
        parts.append(f"val {rec['val_loss']:.4f}")
    parts.append(f"lr {rec['lr']:.2e}")
    if "grad_norm" in rec:
        parts.append(f"gnorm {rec['grad_norm']:.2f}")
    if "tokens_per_s" in rec:
        parts.append(f"{rec['tokens_per_s']/1e3:.1f}k tok/s")
    parts.append(f"{rec['elapsed_s']:.0f}s")
    return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from cs336_basics.nn_utils import cross_entropy, gradient_clipping
    from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule

    # ---- resolve derived settings
    if (args.max_steps is None) == (args.total_tokens is None):
        raise SystemExit("give exactly one of --max-steps / --total-tokens")
    if args.max_steps is None:
        args.max_steps = max(1, args.total_tokens // (args.batch_size * args.context_length))
    if args.min_lr is None:
        args.min_lr = args.lr / 10
    if args.cosine_iters is None:
        args.cosine_iters = args.max_steps
    if args.run_name is None:
        args.run_name = os.path.basename(os.path.normpath(args.out_dir))
    if args.dtype == "bf16" and not str(args.device).startswith("cuda"):
        print("[warn] --dtype bf16 only autocasts on cuda; running fp32", file=sys.stderr)
        args.dtype = "fp32"
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if args.tf32 and device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    # ---- data (memory-mapped)
    train_data = load_token_file(args.train_data, dtype=np.dtype(args.data_dtype))
    val_data = load_token_file(args.val_data, dtype=np.dtype(args.data_dtype)) if args.val_data else None
    print(f"train tokens: {len(train_data):,}" + (f" | val tokens: {len(val_data):,}" if val_data is not None else ""))
    # cheap sanity check on the memmap: token ids must fit the vocab
    head = np.asarray(train_data[: min(len(train_data), 1_000_000)])
    if head.max() >= args.vocab_size:
        raise SystemExit(f"train data contains id {head.max()} >= vocab_size {args.vocab_size}; wrong dtype/vocab?")

    # ---- model / optimizer
    model_cfg = model_config_from_args(args)
    raw_model = build_model(model_cfg).to(device)
    n_params = count_params(raw_model)
    print(f"model params: {n_params/1e6:.2f}M | config: {json.dumps(model_cfg)}")
    optimizer = AdamW(
        raw_model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step, elapsed_prev, tokens_seen = 0, 0.0, 0
    if args.resume:
        start_step = load_checkpoint(args.resume, raw_model, optimizer)
        meta = torch.load(args.resume, map_location="cpu", weights_only=False).get("extra", {})
        elapsed_prev = float(meta.get("elapsed_s", 0.0))
        tokens_seen = int(meta.get("tokens_seen", start_step * args.batch_size * args.context_length))
        print(f"resumed from {args.resume} at step {start_step} (elapsed {elapsed_prev:.0f}s, {tokens_seen:,} tokens)")

    model = torch.compile(raw_model) if args.compile else raw_model
    model.train()

    if args.dtype == "bf16":
        autocast_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # noqa: E731
    else:
        import contextlib

        autocast_ctx = contextlib.nullcontext

    # ---- logging
    cfg_dump = {**vars(args), "n_params": n_params, "model_config": model_cfg}
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(cfg_dump, f, indent=1, default=str)
    wandb_run = None
    if args.wandb:
        import wandb  # optional dependency; only imported when asked for

        wandb_run = wandb.init(project=args.wandb_project, name=args.run_name, config=cfg_dump, resume="allow")
    logger = JsonlLogger(os.path.join(args.out_dir, "log.jsonl"), use_wandb=args.wandb, wandb_run=wandb_run)

    def save(step: int, name: str) -> str:
        path = os.path.join(args.out_dir, name)
        save_checkpoint(
            raw_model,
            optimizer,
            step,
            path,
            extra={
                "config": cfg_dump,
                "model_config": model_cfg,
                "elapsed_s": elapsed_prev + (time.perf_counter() - t_start),
                "tokens_seen": tokens_seen,
            },
        )
        return path

    def do_eval() -> float | None:
        if val_data is None:
            return None
        return evaluate(
            model, val_data, args.batch_size, args.context_length, device,
            args.eval_batches, args.eval_seed, autocast_ctx, cross_entropy,
        )

    # ---- loop
    fixed_batch = None
    t_start = time.perf_counter()
    t_last = t_start
    step = start_step
    stop_reason = "max_steps"
    try:
        while step < args.max_steps:
            lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup_iters, args.cosine_iters)
            for g in optimizer.param_groups:
                g["lr"] = lr

            if args.overfit_one_batch:
                if fixed_batch is None:
                    fixed_batch = get_batch(train_data, args.batch_size, args.context_length, device)
                x, y = fixed_batch
            else:
                x, y = get_batch(train_data, args.batch_size, args.context_length, device)

            with autocast_ctx():
                logits = model(x)
            loss = cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            want_norm = (step + 1) % args.log_interval == 0
            grad_norm = _grad_global_norm(raw_model.parameters()) if want_norm else None
            if args.grad_clip > 0:
                gradient_clipping(raw_model.parameters(), args.grad_clip)
            optimizer.step()

            step += 1
            tokens_seen += args.batch_size * args.context_length
            now = time.perf_counter()
            elapsed = elapsed_prev + (now - t_start)

            rec: dict[str, Any] | None = None
            if step % args.log_interval == 0 or step == args.max_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    now = time.perf_counter()
                dt = now - t_last
                t_last = now
                n_since = args.log_interval if step % args.log_interval == 0 else step % args.log_interval
                rec = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "lr": lr,
                    "elapsed_s": elapsed,
                    "tokens_seen": tokens_seen,
                    "tokens_per_s": n_since * args.batch_size * args.context_length / max(dt, 1e-9),
                    "step_ms": 1000 * dt / n_since,
                }
                if grad_norm is not None:
                    rec["grad_norm"] = grad_norm
                if not math.isfinite(rec["train_loss"]):
                    print(f"[warn] non-finite train loss at step {step}", file=sys.stderr)
            if step % args.eval_interval == 0 or step == args.max_steps:
                vl = do_eval()
                if vl is not None:
                    rec = rec or {"step": step, "lr": lr, "elapsed_s": elapsed, "tokens_seen": tokens_seen}
                    rec["val_loss"] = vl
                    rec["val_perplexity"] = math.exp(min(vl, 50))
                    t_last = time.perf_counter()  # don't bill eval time to the next step window
            if rec is not None:
                logger.log(rec)
                print(_fmt(rec), flush=True)
            if args.ckpt_interval > 0 and step % args.ckpt_interval == 0 and step < args.max_steps:
                print(f"saved {save(step, f'ckpt_step{step}.pt')}")
            if args.max_wallclock_min is not None and elapsed >= 60 * args.max_wallclock_min:
                stop_reason = "wallclock"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("interrupted; saving checkpoint", file=sys.stderr)

    final = save(step, "ckpt_final.pt")
    if stop_reason != "max_steps" and step % args.eval_interval != 0:  # loop already evaluated otherwise
        vl = do_eval()
        if vl is not None:
            rec = {
                "step": step, "lr": float(optimizer.param_groups[0]["lr"]),
                "elapsed_s": elapsed_prev + (time.perf_counter() - t_start), "tokens_seen": tokens_seen,
                "val_loss": vl, "val_perplexity": math.exp(min(vl, 50)), "final": True,
            }
            logger.log(rec)
            print(_fmt(rec))
    print(f"done ({stop_reason}) at step {step}; final checkpoint {final}")
    logger.close()


if __name__ == "__main__":
    main()
