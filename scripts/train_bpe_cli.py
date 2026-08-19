"""Train a byte-level BPE tokenizer on a text file and serialize vocab/merges.

Usage:
    uv run python scripts/train_bpe_cli.py --input data/TinyStoriesV2-GPT4-train.txt \
        --vocab-size 10000 --out-dir out/tinystories_bpe_10k

Prints wall-clock time, peak RSS and the longest token (assignment §2.5 deliverables).
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import resource
import time
from pathlib import Path

from cs336_basics.tokenizer import save_vocab_and_merges
from cs336_basics.train_bpe import train_bpe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--special-token", action="append", default=None, help="repeatable; default <|endoftext|>")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-processes", type=int, default=None)
    ap.add_argument("--profile", action="store_true", help="run under cProfile and print top-25 cumulative")
    args = ap.parse_args()

    specials = args.special_token or ["<|endoftext|>"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.profile:
        prof = cProfile.Profile()
        prof.enable()
    vocab, merges = train_bpe(args.input, args.vocab_size, specials, num_processes=args.num_processes)
    if args.profile:
        prof.disable()
    elapsed = time.perf_counter() - t0

    save_vocab_and_merges(vocab, merges, out_dir / "vocab.json", out_dir / "merges.txt")

    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    child_rss_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    longest = max(vocab.values(), key=len)
    print(f"vocab_size={len(vocab)} merges={len(merges)}")
    print(f"time={elapsed:.1f}s peak_rss(main)={peak_rss_mb:.0f}MB peak_rss(max child)={child_rss_mb:.0f}MB")
    print(f"longest token ({len(longest)} bytes): {longest!r}")
    print("top-10 longest:", sorted(vocab.values(), key=len, reverse=True)[:10])
    print(f"saved to {out_dir}/vocab.json and {out_dir}/merges.txt")

    if args.profile:
        pstats.Stats(prof).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
