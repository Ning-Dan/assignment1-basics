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
import os
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


_WORKER_TOK: Tokenizer | None = None


def _encode_worker_init(vocab_path: str, merges_path: str) -> None:
    global _WORKER_TOK
    _WORKER_TOK = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[EOT])


def _encode_chunk(job: tuple[str, int, int]) -> np.ndarray:
    """Encode file[start:end].  Chunk boundaries sit at the start of an EOT token (a hard
    boundary), so the concatenation equals a one-shot encode of the whole file."""
    path, start, end = job
    assert _WORKER_TOK is not None
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.asarray(_WORKER_TOK.encode(text), dtype=np.uint16)


def _iter_encoded_arrays(args: argparse.Namespace, tok: Tokenizer):
    """Yield uint16 arrays covering the whole input, in file order."""
    if args.num_processes <= 1:
        buf: list[int] = []
        with open(args.input, encoding="utf-8", errors="ignore") as f:
            for i in tok.encode_iterable(f):
                buf.append(i)
                if len(buf) >= 1_000_000:
                    yield np.asarray(buf, dtype=np.uint16)
                    buf = []
        if buf:
            yield np.asarray(buf, dtype=np.uint16)
        return

    from multiprocessing import Pool

    from cs336_basics.pretokenization import find_chunk_boundaries

    with open(args.input, "rb") as f:
        boundaries = find_chunk_boundaries(f, args.num_chunks, EOT.encode("utf-8"))
    jobs = [(args.input, s, e) for s, e in zip(boundaries[:-1], boundaries[1:]) if e > s]
    with Pool(args.num_processes, initializer=_encode_worker_init, initargs=(args.vocab, args.merges)) as pool:
        yield from pool.imap(_encode_chunk, jobs)  # ordered


def cmd_encode(args: argparse.Namespace) -> None:
    tok = Tokenizer.from_files(args.vocab, args.merges, special_tokens=[EOT])
    assert len(tok.vocab) <= 65536, "uint16 cannot hold this vocab"
    t0 = time.perf_counter()
    # Stream to a raw file first (constant memory), then wrap it in a .npy header.
    raw_path = args.output + ".raw"
    n = 0
    with open(raw_path, "wb") as raw:
        for arr in _iter_encoded_arrays(args, tok):
            raw.write(arr.tobytes())
            n += arr.shape[0]
    with open(args.output, "wb") as out:
        np.lib.format.write_array_header_1_0(out, {"descr": "<u2", "fortran_order": False, "shape": (n,)})
        with open(raw_path, "rb") as raw:
            while True:
                block = raw.read(64 << 20)
                if not block:
                    break
                out.write(block)
    os.remove(raw_path)
    print(f"wrote {n} tokens (uint16) to {args.output} in {time.perf_counter() - t0:.0f}s")


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
            p.add_argument("--num-processes", type=int, default=1, help=">1: chunk at <|endoftext|> and encode in parallel")
            p.add_argument("--num-chunks", type=int, default=512)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
