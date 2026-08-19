"""Assignment §2.7 (tokenizer_experiments): compression ratio, throughput, dataset encoding.

Examples:
    # (a)/(b) compression ratios on 10 sampled documents
    uv run python scripts/tokenizer_experiments.py ratio \
        --vocab out/ts10k/vocab.json --merges out/ts10k/merges.txt --input data/TinyStoriesV2-GPT4-valid.txt

    # (c) throughput on a file
    uv run python scripts/tokenizer_experiments.py throughput \
        --vocab out/ts10k/vocab.json --merges out/ts10k/merges.txt --input tests/fixtures/tinystories_sample_5M.txt

    # (d) encode a whole dataset to a uint16 .npy
    uv run python scripts/tokenizer_experiments.py encode \
        --vocab out/ts10k/vocab.json --merges out/ts10k/merges.txt \
        --input data/TinyStoriesV2-GPT4-train.txt --output data/ts_train_10k.npy
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np

from cs336_basics.tokenizer import Tokenizer

EOT = "<|endoftext|>"


def sample_documents(path: str, n: int, seed: int = 0, max_bytes: int = 200_000_000) -> list[str]:
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read(max_bytes)
    docs = [d for d in text.split(EOT) if d.strip()]
    rng = random.Random(seed)
    return rng.sample(docs, min(n, len(docs)))


def cmd_ratio(args: argparse.Namespace) -> None:
    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[EOT])
    docs = sample_documents(args.input, args.num_docs, args.seed)
    total_bytes = sum(len(d.encode("utf-8")) for d in docs)
    total_tokens = sum(len(tok.encode(d)) for d in docs)
    print(f"{len(docs)} docs, {total_bytes} bytes, {total_tokens} tokens -> {total_bytes / total_tokens:.3f} bytes/token")


def cmd_throughput(args: argparse.Namespace) -> None:
    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[EOT])
    with open(args.input, encoding="utf-8", errors="ignore") as f:
        text = f.read()
    nbytes = len(text.encode("utf-8"))
    t0 = time.perf_counter()
    ids = tok.encode(text)
    dt = time.perf_counter() - t0
    bps = nbytes / dt
    print(f"{nbytes} bytes -> {len(ids)} tokens in {dt:.2f}s = {bps / 1e6:.2f} MB/s")
    pile = 825e9
    print(f"Pile (825GB) would take ~{pile / bps / 3600:.1f} h single-process")


def cmd_encode(args: argparse.Namespace) -> None:
    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[EOT])
    assert len(tok.vocab) <= 65536, "uint16 cannot hold this vocab"
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    buf: list[int] = []
    with open(args.input, encoding="utf-8", errors="ignore") as f:
        for i in tok.encode_iterable(f):
            buf.append(i)
            if len(buf) >= 1_000_000:
                chunks.append(np.asarray(buf, dtype=np.uint16))
                buf = []
    if buf:
        chunks.append(np.asarray(buf, dtype=np.uint16))
    arr = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
    np.save(args.output, arr)
    print(f"wrote {arr.shape[0]} tokens (uint16) to {args.output} in {time.perf_counter() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("ratio", cmd_ratio), ("throughput", cmd_throughput), ("encode", cmd_encode)):
        p = sub.add_parser(name)
        p.add_argument("--vocab", required=True)
        p.add_argument("--merges", required=True)
        p.add_argument("--input", required=True)
        if name == "ratio":
            p.add_argument("--num-docs", type=int, default=10)
            p.add_argument("--seed", type=int, default=0)
        if name == "encode":
            p.add_argument("--output", required=True)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
