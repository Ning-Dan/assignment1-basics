"""Sample text from a trained checkpoint (assignment §6 / §7 `generate`).

Example::

    .venv/bin/python scripts/generate.py \\
        --checkpoint runs/ts_base/ckpt_final.pt \\
        --vocab out/ts10k/vocab.json --merges out/ts10k/merges.txt \\
        --prompt "Once upon a time" --max-new-tokens 256 --temperature 0.8 --top-p 0.9

The model architecture is read from the checkpoint's stored config (written by
``cs336_basics.train``); any ``--vocab-size/--context-length/...`` flag overrides it
(needed only for checkpoints produced elsewhere).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from cs336_basics.decoding import EOT, generate
from cs336_basics.serialization import load_checkpoint, peek_checkpoint
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train import _MODEL_ABLATION_KEYS, build_model


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Generate text from a TransformerLM checkpoint.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocab", required=True, help="vocab.json of the BPE tokenizer")
    p.add_argument("--merges", required=True, help="merges.txt of the BPE tokenizer")
    p.add_argument("--prompt", default="Once upon a time", help="prompt text; '-' reads stdin")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0, help="0 = greedy")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-eos-stop", action="store_true", help="keep sampling past <|endoftext|>")
    g = p.add_argument_group("model overrides (default: from checkpoint)")
    for k in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff"):
        g.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    g.add_argument("--rope-theta", type=float, default=None)
    args = p.parse_args(argv)

    meta = peek_checkpoint(args.checkpoint)
    cfg = dict(meta["extra"].get("model_config") or {})
    for k in ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
        if k not in cfg:
            raise SystemExit(f"checkpoint has no stored model config; pass --{k.replace('_', '-')}")
    for k, dflt in _MODEL_ABLATION_KEYS.items():
        cfg.setdefault(k, dflt)

    device = torch.device(args.device)
    model = build_model(cfg).to(device)
    it = load_checkpoint(args.checkpoint, model, optimizer=None)
    model.eval()
    print(f"[loaded step {it} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M params | {cfg}]", file=sys.stderr)

    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[EOT])
    eos_id = None if args.no_eos_stop else tok.encode(EOT)[0]
    prompt = sys.stdin.read() if args.prompt == "-" else args.prompt
    prompt_ids = tok.encode(prompt)

    gen = None
    if args.seed is not None:
        gen = torch.Generator(device=device).manual_seed(args.seed)

    for i in range(args.num_samples):
        t0 = time.perf_counter()
        new_ids = generate(
            model, prompt_ids, args.max_new_tokens, temperature=args.temperature, top_p=args.top_p,
            eos_token_id=eos_id, device=device, context_length=cfg["context_length"], generator=gen,
        )
        dt = time.perf_counter() - t0
        text = tok.decode(new_ids)
        hit_eos = len(new_ids) < args.max_new_tokens
        print(f"--- sample {i+1}/{args.num_samples}: {len(new_ids)} new tokens in {dt:.1f}s"
              f" (temperature={args.temperature}, top_p={args.top_p}, stopped_at_eos={hit_eos}) ---", file=sys.stderr)
        print(prompt + text)
        print()


if __name__ == "__main__":
    main()
